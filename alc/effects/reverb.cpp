/**
 * Ambisonic reverb engine for the OpenAL cross platform audio library
 * Copyright (C) 2008-2017 by Chris Robinson and Christopher Fitzgerald.
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Library General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 *  License along with this library; if not, write to the
 *  Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 * Or go to http://www.gnu.org/copyleft/lgpl.html
 */

#include "config.h"

#include <bit>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>

#include "alc/effects/base.h"
#include "alc/alconfig.h"
#include "core/bufferline.h"
#include "core/effects/base.h"
#include "core/filters/biquad.h"
#include "core/filters/splitter.h"
#include "core/mixer.h"
#include "core/mixer/defs.h"
#include "intrusive_ptr.h"
#include "opthelper_macros.hpp"
#include "vector.h"

#if HAVE_CXXMODULES
import common.std;

import core.ambi_utils;
import core.context;
import core.cubic_tables;
import core.device;
import core.effectslot;
import gsl;
import numeric;
import zudl;
#include "gsl_macros.hpp"

#else

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "core/ambi_utils.hpp"
#include "core/context.h"
#include "core/cubic_tables.hpp"
#include "core/device.h"
#include "core/effectslot.h"
#include "gsl/gsl"
#include "numeric.hpp"
#include "zudl.hpp"
#endif

struct BufferStorage;

namespace {

constexpr auto MinDecayTime = 0.1f;
constexpr auto MaxDecayTime = 20.0f;
constexpr auto MaxModulationTime = 4.0f;
constexpr auto DefaultModulationTime = 0.25f;
constexpr auto MaxHFReference = 20'000.0f;

#define MOD_FRACBITS 24
#define MOD_FRACONE  (1<<MOD_FRACBITS)
#define MOD_FRACMASK (MOD_FRACONE-1)


/* Max samples per process iteration. Used to limit the size needed for
 * temporary buffers. Must be a multiple of 4 for SIMD alignment.
 */
constexpr auto MAX_UPDATE_SAMPLES = 256_uz;

/* The number of spatialized lines or channels to process. Four channels allows
 * for a 3D A-Format response. NOTE: This can't be changed without taking care
 * of the conversion matrices, and a few places where the length arrays are
 * assumed to have 4 elements.
 */
constexpr auto NUM_LINES = 4_uz;


/* This coefficient is used to define the maximum frequency range controlled by
 * the modulation depth. The current value of 0.05 will allow it to swing from
 * 0.95x to 1.05x. This value must be below 1. At 1 it will cause the sampler
 * to stall on the downswing, and above 1 it will cause it to sample backwards.
 * The value 0.05 seems to be nearest to Creative hardware behavior.
 */
constexpr auto MODULATION_DEPTH_COEFF = 0.05f;


/**************************************
 *  Image-source early reflections    *
 **************************************/

/* Speed of sound in air, meters per second (matches the legacy model). */
constexpr auto SpeedOfSound = 343.3f;

/* Shoebox image-source model: reflection order 1 yields the six first-order
 * wall images (two per axis); order 2 additionally includes edge/corner
 * images, 24 in total.
 */
constexpr auto IsmMaxOrder = 2u;
constexpr auto IsmMaxTaps = 24_uz;

/* Minimum room side length in meters, guards against degenerate geometry. */
constexpr auto IsmMinRoomSize = 0.5f;

/* Average room dimension (meters) the legacy fixed taps are derived for. Used
 * as the anchor when deriving room geometry from the Density parameter.
 */
constexpr auto IsmBaseRoomSize = 5.0f;

/* One image tap: a variable-delay read of the mono input line with its own
 * gain, high-frequency loss, and arrival direction (head-relative unit
 * vector). Current/Target pairs are smoothed per update block.
 */
struct IsmTap {
    float Delay{0.0f};
    float DelayTarget{0.0f};
    float Gain{0.0f};
    float GainTarget{0.0f};
    /* One-pole low-pass state/coefficient for wall+air high-frequency loss. */
    float LpCoeff{0.0f};
    float LpCoeffTarget{0.0f};
    float LpState{0.0f};
    std::array<float,3> Dir{};
    std::array<float,3> DirTarget{};
    /* Output-channel panning gains for each first-order B-Format input
     * component (ACN 0..3 = W,Y,Z,X). Each B-Format component is mapped onto
     * the device output channels with its own gain set.
     */
    struct OutGains {
        std::array<std::array<float, MaxAmbiChannels>, NUM_LINES> Current{};
        std::array<std::array<float, MaxAmbiChannels>, NUM_LINES> Target{};

        void fill(float v) noexcept
        {
            for(auto &row : Current) row.fill(v);
            for(auto &row : Target) row.fill(v);
        }
    };
    OutGains Gains{};

    void clear() noexcept NONBLOCKING
    {
        Delay = DelayTarget = Gain = GainTarget = 0.0f;
        LpCoeff = LpCoeffTarget = LpState = 0.0f;
        Dir.fill(0.0f);
        DirTarget.fill(0.0f);
        IGNORE_FUNCTION_EFFECTS(Gains.fill(0.0f);)
    }
};

/* Configuration of the image-source model, loaded once from the [reverb]
 * config block. When mEnabled is false the legacy fixed-tap model is used.
 */
struct IsmConfig {
    bool mEnabled{false};
    unsigned int mOrder{1u};
    bool mExplicitRoom{false};
    std::array<float,3> mRoomSize{{8.0f, 6.0f, 3.0f}};
    std::array<float,3> mWallReflect{{0.9f, 0.9f, 0.9f}};
    float mSmoothing{0.5f};
};


/* The B-Format to (W-normalized) A-Format conversion matrix. This produces a
 * tetrahedral array of discrete signals (boosted by a factor of sqrt(3), to
 * reduce the error introduced in the conversion).
 */
alignas(16) constexpr std::array<std::array<float, NUM_LINES>, NUM_LINES> B2A{{
    /*   W      Y      Z      X  */
    {{ 0.5f,  0.5f,  0.5f,  0.5f }}, /* A0 */
    {{ 0.5f, -0.5f, -0.5f,  0.5f }}, /* A1 */
    {{ 0.5f,  0.5f, -0.5f, -0.5f }}, /* A2 */
    {{ 0.5f, -0.5f,  0.5f, -0.5f }}  /* A3 */
}};

/* Converts (W-normalized) A-Format to B-Format for early reflections (scaled
 * by 1/sqrt(3) to compensate for the boost in the B2A matrix).
 */
alignas(16) constexpr std::array<std::array<float, NUM_LINES>, NUM_LINES> EarlyA2B{{
    /*  A0     A1     A2     A3  */
    {{ 0.5f,  0.5f,  0.5f,  0.5f }}, /* W */
    {{ 0.5f, -0.5f,  0.5f, -0.5f }}, /* Y */
    {{ 0.5f, -0.5f, -0.5f,  0.5f }}, /* Z */
    {{ 0.5f,  0.5f, -0.5f, -0.5f }}  /* X */
}};

/* Converts (W-normalized) A-Format to B-Format for late reverb (scaled
 * by 1/sqrt(3) to compensate for the boost in the B2A matrix). The response
 * is rotated around Z (ambisonic X) so that the front lines are placed
 * horizontally in front, and the rear lines are placed vertically in back.
 */
constexpr auto InvSqrt2 = static_cast<float>(1.0/std::numbers::sqrt2);
alignas(16) constexpr std::array<std::array<float, NUM_LINES>, NUM_LINES> LateA2B{{
    /*     A0         A1         A2        A3   */
    {{     0.5f,      0.5f,      0.5f,     0.5f }}, /* W */
    {{ InvSqrt2, -InvSqrt2,      0.0f,     0.0f }}, /* Y */
    {{     0.0f,      0.0f, -InvSqrt2, InvSqrt2 }}, /* Z */
    {{     0.5f,      0.5f,     -0.5f,    -0.5f }}  /* X */
}};

/* The all-pass and delay lines have a variable length dependent on the
 * effect's density parameter, which helps alter the perceived environment
 * size. The size-to-density conversion is a cubed scale:
 *
 * density = min(1.0, pow(size, 3.0) / DensityScale);
 *
 * The line lengths scale linearly with room size, so the inverse density
 * conversion is needed, taking the cube root of the re-scaled density to
 * calculate the line length multiplier:
 *
 *     length_mult = max(1.0, cbrt(density*DensityScale));
 *
 * The density scale below will result in a max line multiplier of 10, for an
 * effective average size range of 5 to 50 meters.
 */
constexpr auto DensityScale = 1'000.0f;

/* All delay line lengths are specified in seconds.
 *
 * To approximate early reflections, we break them up into primary (those
 * arriving from the same direction as the source) and secondary (those
 * arriving from the opposite direction).
 *
 * The early taps decorrelate the 4-channel signal to approximate an average
 * room response for the primary reflections after the initial early delay.
 *
 * Given an average room dimension (d_a) and the speed of sound (c) we can
 * calculate the average reflection delay (r_a) regardless of listener and
 * source positions as:
 *
 *     r_a = d_a / c
 *     c   = 343.3
 *
 * This can be extended to find the average difference (r_d) between the
 * maximum (r_1) and minimum (r_0) reflection delays:
 *
 *     r_0 = 2 / 3 r_a
 *         = r_a - r_d / 2
 *         = r_d
 *     r_1 = 4 / 3 r_a
 *         = r_a + r_d / 2
 *         = 2 r_d
 *     r_d = 2 / 3 r_a
 *         = r_1 - r_0
 *
 * As can be determined by integrating the 1D model with a source (s) and
 * listener (l) positioned across the dimension of length (d_a):
 *
 *     r_d = int_(l=0)^d_a (int_(s=0)^d_a |2 d_a - 2 (l + s)| ds) dl / c
 *
 * The initial taps (T_(i=0)^N) are then specified by taking a power series
 * that ranges between r_0 and half of r_1 less r_0:
 *
 *     R_i = 2^(i / (2 N - 1)) r_d
 *         = r_0 + (2^(i / (2 N - 1)) - 1) r_d
 *         = r_0 + T_i
 *     T_i = R_i - r_0
 *         = (2^(i / (2 N - 1)) - 1) r_d
 *
 * Assuming an average of 5 meters, we get the following taps:
 */
constexpr std::array<float, NUM_LINES> EARLY_TAP_LENGTHS{{
    0.000000e+0f, 1.010676e-3f, 2.126553e-3f, 3.358580e-3f
}};

/* The early all-pass filter lengths are based on the early tap lengths:
 *
 *     A_i = R_i / a
 *
 * Where a is the approximate maximum all-pass cycle limit (20).
 */
constexpr std::array<float, NUM_LINES> EARLY_ALLPASS_LENGTHS{{
    4.854840e-4f, 5.360178e-4f, 5.918117e-4f, 6.534130e-4f
}};

/* The early delay lines are used to transform the primary reflections into
 * the secondary reflections.  The A-format is arranged in such a way that
 * the channels/lines are spatially opposite:
 *
 *     C_i is opposite C_(N-i-1)
 *
 * The delays of the two opposing reflections (R_i and O_i) from a source
 * anywhere along a particular dimension always sum to twice its full delay:
 *
 *     2 r_a = R_i + O_i
 *
 * With that in mind we can determine the delay between the two reflections
 * and thus specify our early line lengths (L_(i=0)^N) using:
 *
 *     O_i = 2 r_a - R_(N-i-1)
 *     L_i = O_i - R_(N-i-1)
 *         = 2 (r_a - R_(N-i-1))
 *         = 2 (r_a - T_(N-i-1) - r_0)
 *         = 2 r_a (1 - (2 / 3) 2^((N - i - 1) / (2 N - 1)))
 *
 * Using an average dimension of 5 meters, we get:
 */
constexpr std::array<float, NUM_LINES> EARLY_LINE_LENGTHS{{
    2.992520e-3f, 5.456575e-3f, 7.688329e-3f, 9.709681e-3f
}};

/* The late all-pass filter lengths are based on the late line lengths:
 *
 *     A_i = (5 / 3) L_i / r_1
 */
constexpr std::array<float, NUM_LINES> LATE_ALLPASS_LENGTHS{{
    8.091400e-4f, 1.019453e-3f, 1.407968e-3f, 1.618280e-3f
}};

/* The late lines are used to approximate the decaying cycle of recursive
 * late reflections.
 *
 * Splitting the lines in half, we start with the shortest reflection paths
 * (L_(i=0)^(N/2)):
 *
 *     L_i = 2^(i / (N - 1)) r_d
 *
 * Then for the opposite (longest) reflection paths (L_(i=N/2)^N):
 *
 *     L_i = 2 r_a - L_(i-N/2)
 *         = 2 r_a - 2^((i - N / 2) / (N - 1)) r_d
 *
 * For our 5 meter average room, we get:
 */
constexpr std::array<float, NUM_LINES> LATE_LINE_LENGTHS{{
    9.709681e-3f, 1.223343e-2f, 1.689561e-2f, 1.941936e-2f
}};

constexpr auto LateAllpassAverage = std::reduce(LATE_ALLPASS_LENGTHS.begin(),
    LATE_ALLPASS_LENGTHS.end(), 0.0f) / float{NUM_LINES};
constexpr auto LateDelayAverage = std::reduce(LATE_LINE_LENGTHS.begin(),
    LATE_LINE_LENGTHS.end(), 0.0f) / float{NUM_LINES} + LateAllpassAverage;

using ReverbUpdateLine = std::array<float, MAX_UPDATE_SAMPLES>;
using ReverbUpdateSpan = std::span<float, MAX_UPDATE_SAMPLES>;

struct DelayLineI {
    /* The delay lines use interleaved samples, with the lengths being powers
     * of 2 to allow the use of bit-masking instead of a modulus for wrapping.
     */
    std::span<float> mLine;

    /* Given the allocated sample buffer, this function updates each delay line
     * offset.
     */
    void realizeLineOffset(std::span<float> const sampleBuffer) noexcept
    { mLine = sampleBuffer; }

    /* Calculate the length of a delay line and store its mask and offset. */
    static auto calcLineLength(float const length, float const frequency, unsigned const extra)
        -> std::size_t
    {
        /* All line lengths are powers of 2, calculated from their lengths in
         * seconds, rounded up.
         */
        auto samples = al::saturating_cast<unsigned>(std::ceil(length*frequency));
        samples = std::bit_ceil(samples + extra);

        /* Return the sample count for accumulation. */
        return samples*NUM_LINES;
    }
};

struct DelayLineU {
    std::span<float> mLine;

    void realizeLineOffset(std::span<float> const sampleBuffer) noexcept
    {
        Expects(sampleBuffer.size() > 4 && !(sampleBuffer.size() & (sampleBuffer.size()-1)));
        mLine = sampleBuffer;
    }

    static auto calcLineLength(float const length, float const frequency, unsigned const extra)
        -> std::size_t
    {
        auto samples = al::saturating_cast<unsigned>(std::ceil(length*frequency));
        samples = std::bit_ceil(samples + extra);

        return samples*NUM_LINES;
    }

    [[nodiscard]]
    auto get(std::size_t const chan) const noexcept
    {
        const auto stride = mLine.size() / NUM_LINES;
        return mLine.subspan(chan*stride, stride);
    }

    void write(std::size_t offset, std::size_t const c, std::span<float const> const in) const
        noexcept
    {
        const auto stride = mLine.size() / NUM_LINES;
        const auto output = mLine.subspan(c*stride, stride);
        auto input = in.begin();
        offset &= stride-1;
        while(const auto rem = std::distance(input, in.end()))
        {
            const auto td = std::min(gsl::narrow_cast<ptrdiff_t>(stride-offset), rem);
            input = std::ranges::copy_n(input, td, (output | std::views::drop(offset)).begin()).in;
            /* Either wrapping back to 0 with more input, or it's done. */
            offset = 0;
        }
    }

    /* Writes the given input lines to the delay buffer, applying a geometric
     * reflection. This effectively applies the matrix
     *
     * [ +1/2 -1/2 -1/2 -1/2 ]
     * [ -1/2 +1/2 -1/2 -1/2 ]
     * [ -1/2 -1/2 +1/2 -1/2 ]
     * [ -1/2 -1/2 -1/2 +1/2 ]
     *
     * to the four input lines when writing to the delay buffer. The effect on
     * the B-Format signal is negating W, applying a 180-degree phase shift and
     * moving each response to its spatially opposite location.
     */
    void writeReflected(std::size_t offset, std::span<ReverbUpdateLine const, NUM_LINES> const in,
        std::size_t const count) const noexcept
    {
        const auto stride = mLine.size() / NUM_LINES;
        offset &= stride-1;
        for(auto i = 0_uz;i < count;)
        {
            auto td = std::min(stride - offset, count - i);
            do {
                const auto src = std::array{in[0][i], in[1][i], in[2][i], in[3][i]};
                ++i;

                const auto f = std::array{
                    (src[0]          - src[1] - src[2] - src[3]) * 0.5f,
                    (src[1] - src[0]          - src[2] - src[3]) * 0.5f,
                    (src[2] - src[0] - src[1]          - src[3]) * 0.5f,
                    (src[3] - src[0] - src[1] - src[2]         ) * 0.5f};
                mLine[0*stride + offset] = f[0];
                mLine[1*stride + offset] = f[1];
                mLine[2*stride + offset] = f[2];
                mLine[3*stride + offset] = f[3];
                ++offset;
            } while(--td);
            offset = 0;
        }
    }
};

struct VecAllpass {
    DelayLineI Delay;
    float Coeff{0.0f};
    std::array<std::size_t, NUM_LINES> Offset{};

    void process(std::span<ReverbUpdateLine, NUM_LINES> samples, std::size_t offset, float xCoeff,
        float yCoeff, std::size_t todo) const noexcept;
};

struct Allpass4 {
    DelayLineU Delay;
    float Coeff{0.0f};
    std::array<std::size_t, NUM_LINES> Offset{};

    void process(std::span<ReverbUpdateLine, NUM_LINES> samples, std::size_t offset,
        std::size_t todo) const noexcept;
};

struct T60Filter {
    /* Two filters are used to adjust the signal. One to control the low
     * frequencies, and one to control the high frequencies.
     */
    float mMidGain{0.0f};
    BiquadFilter mHFFilter, mLFFilter;

    void calcCoeffs(float length, float lfDecayTime, float mfDecayTime, float hfDecayTime,
        float lf0norm, float hf0norm);

    /* Applies the two T60 damping filter sections. */
    void process(std::span<float> const samples) noexcept NONBLOCKING
    { DualBiquad{mHFFilter, mLFFilter}.process(samples, samples); }

    void clear() noexcept NONBLOCKING { mHFFilter.clear(); mLFFilter.clear(); }
};

struct EarlyReflections {
    Allpass4 Allpass;

    /* An echo line is used to complete the second half of the early
     * reflections.
     */
    DelayLineU Delay;
    std::array<std::size_t, NUM_LINES> Offset{};
    float Coeff{};

    /* The gain for each output channel based on 3D panning. */
    struct OutGains {
        std::array<float, MaxAmbiChannels> Current{};
        std::array<float, MaxAmbiChannels> Target{};

        void clear() noexcept NONBLOCKING { Current.fill(0.0f); Target.fill(0.0); }
    };
    std::array<OutGains,NUM_LINES> Gains{};

    void updateLines(float density_mult, float diffusion, float decayTime, float frequency);

    void clear() noexcept NONBLOCKING
    {
        std::ranges::fill(Allpass.Delay.mLine, 0.0f);
        std::ranges::fill(Delay.mLine, 0.0f);
        IGNORE_FUNCTION_EFFECTS( /* OutGains::clear is non-blocking. */
            std::ranges::for_each(Gains, &OutGains::clear);
        )
    }
};


struct Modulation {
    /* The vibrato time is tracked with an index over a (MOD_FRACONE)
     * normalized range.
     */
    unsigned Index{0u}, Step{1u};

    /* The depth of frequency change, in samples. */
    float Depth{0.0f};

    std::array<unsigned, MAX_UPDATE_SAMPLES> ModDelays{};

    void updateModulator(float modTime, float modDepth, float frequency);

    auto calcDelays(std::size_t todo) -> std::span<unsigned const>;

    void clear() noexcept
    {
        Index = 0u;
        Step = 1u;
        Depth = 0.0f;
    }
};

struct LateReverb {
    /* A recursive delay line is used fill in the reverb tail. */
    DelayLineU Delay;
    std::array<std::size_t, NUM_LINES> Offset{};

    /* Attenuation to compensate for the modal density and decay rate of the
     * late lines.
     */
    float DensityGain{0.0f};

    /* T60 decay filters are used to simulate absorption. */
    std::array<T60Filter,NUM_LINES> T60;

    Modulation Mod;

    /* A Gerzon vector all-pass filter is used to simulate diffusion. */
    VecAllpass VecAp;

    /* The gain for each output channel based on 3D panning. */
    struct OutGains {
        std::array<float, MaxAmbiChannels> Current{};
        std::array<float, MaxAmbiChannels> Target{};

        void clear() noexcept NONBLOCKING { Current.fill(0.0f); Target.fill(0.0); }
    };
    std::array<OutGains,NUM_LINES> Gains{};

    void updateLines(float density_mult, float diffusion, float lfDecayTime, float mfDecayTime,
        float hfDecayTime, float lf0norm, float hf0norm, float frequency);

    void clear() noexcept NONBLOCKING
    {
        std::ranges::fill(VecAp.Delay.mLine, 0.0f);
        std::ranges::fill(Delay.mLine, 0.0f);
        IGNORE_FUNCTION_EFFECTS(
            std::ranges::for_each(T60, &T60Filter::clear);
        )
        Mod.clear();
        IGNORE_FUNCTION_EFFECTS(
            std::ranges::for_each(Gains, &OutGains::clear);
        )
    }
};

struct ReverbPipeline {
    /* Master effect filters */
    struct FilterPair {
        BiquadFilter Lp;
        BiquadFilter Hp;
        void clear() noexcept NONBLOCKING { Lp.clear(); Hp.clear(); }
        void process(std::span<float const> const src, std::span<float> const dst)
        { DualBiquad{Lp, Hp}.process(src, dst); }
    };
    std::array<FilterPair, NUM_LINES> mFilter;

    /* Late reverb input delay line (early reflections feed this, and late
     * reverb taps from it).
     */
    DelayLineU mLateDelayIn;

    /* Tap points for early reflection input delay. */
    std::array<std::array<std::size_t, 2>, NUM_LINES> mEarlyDelayTap{};
    std::array<float, 2> mEarlyDelayCoeff{};

    /* Tap points for late reverb feed and delay. */
    std::array<std::array<std::size_t, 2>, NUM_LINES> mLateDelayTap{};

    /* Coefficients for the all-pass and line scattering matrices. */
    float mMixX{1.0f};
    float mMixY{0.0f};

    EarlyReflections mEarly;

    LateReverb mLate;

    std::array<std::array<BandSplitter,NUM_LINES>,2> mAmbiSplitter;

    std::size_t mFadeSampleCount{1};

    /* Overall output gain for the image-source early reflections (reflection
     * gain times slot gain), cached at update time for the render path.
     */
    float mIsmOutGain{0.0f};

    /* Image-source early-reflection engine. Active only when the [reverb]
     * image-source config option is enabled. When active, the legacy fixed
     * early-reflection output is muted (its scatter into the late reverb
     * input is preserved) and these taps generate the position-aware early
     * response instead.
     */
    struct ImageSourceModel {
        /* Mono input delay line (channel 0 of a NUM_LINES-strided line). */
        DelayLineU Delay;
        std::array<IsmTap, IsmMaxTaps> Taps{};
        std::size_t TapCount{0};
        /* Last smoothed head-relative source position, held when no voice
         * contributes to the slot during an update block.
         */
        std::array<float,3> LastPos{};
        bool HasPosition{false};
        /* False until the first block with a contributing source. On that
         * block the tap geometry (delay/filter/direction) is initialized
         * directly to its target instead of being smoothed in from zero, so
         * the very first impulse gets the correct reflection delays rather
         * than a ramp of converging read positions.
         */
        bool GeometryInitialized{false};
        /* Per-tap rendered first-order B-Format (ACN/N3D: W,Y,Z,X) output.
         * Kept separate per tap so each tap's directional components use their
         * own output-channel panning gains in the final mix.
         */
        alignas(16) std::array<std::array<FloatBufferLine, NUM_LINES>, IsmMaxTaps> TapBf{};

        void clear() noexcept NONBLOCKING
        {
            std::ranges::fill(Delay.mLine, 0.0f);
            IGNORE_FUNCTION_EFFECTS(
                std::ranges::for_each(Taps, &IsmTap::clear);
                std::ranges::for_each(TapBf | std::views::join,
                    [](FloatBufferLine &line){ line.fill(0.0f); });
            )
            TapCount = 0;
            LastPos.fill(0.0f);
            HasPosition = false;
            GeometryInitialized = false;
        }
    } mIsm;

    void updateDelayLine(float gain, float earlyDelay, float lateDelay, float density_mult,
        float frequency);
    void update3DPanning(std::span<float const, 3> ReflectionsPan,
        std::span<float const, 3> LateReverbPan, float earlyGain, float lateGain, bool doUpmix,
        MixParams const *mainMix);

    void processEarly(DelayLineU const &main_delay, std::size_t offset, std::size_t samplesToDo,
        std::span<ReverbUpdateLine, NUM_LINES> tempSamples,
        std::span<FloatBufferLine, NUM_LINES> outSamples);
    void processLate(std::size_t offset, std::size_t samplesToDo,
        std::span<ReverbUpdateLine, NUM_LINES> tempSamples,
        std::span<FloatBufferLine, NUM_LINES> outSamples);

    void clear() noexcept NONBLOCKING
    {
        std::ranges::fill(mLateDelayIn.mLine, 0.0f);
        IGNORE_FUNCTION_EFFECTS( /* FilterPair::clear is non-blocking. */
            std::ranges::for_each(mFilter, &FilterPair::clear);
        )
        mEarlyDelayTap = {};
        mEarlyDelayCoeff = {};
        mLateDelayTap = {};
        mEarly.clear();
        mLate.clear();
        mIsm.clear();
        IGNORE_FUNCTION_EFFECTS( /* BandSplitter::clear is non-blocking. */
            std::ranges::for_each(mAmbiSplitter | std::views::join, &BandSplitter::clear);
        )
    }
};

struct ReverbState final : EffectState {
    /* All delay lines are allocated as a single buffer to reduce memory
     * fragmentation and management code.
     */
    al::vector<float, 16> mSampleBuffer;

    struct Params {
        /* Calculated parameters which indicate if cross-fading is needed after
         * an update.
         */
        float Density{1.0f};
        float Diffusion{1.0f};
        float DecayTime{1.49f};
        float HFDecayTime{0.83f * 1.49f};
        float LFDecayTime{1.0f * 1.49f};
        float ModulationTime{0.25f};
        float ModulationDepth{0.0f};
        float HFReference{5000.0f};
        float LFReference{250.0f};
    };
    Params mParams;

    enum PipelineState : std::uint8_t {
        DeviceClear,
        StartFade,
        Fading,
        Cleanup,
        Normal,
    };
    PipelineState mPipelineState{DeviceClear};
    bool mCurrentPipeline{false};
    /* Total length of the current pipeline cross-fade in samples, used to
     * ramp the old pipeline's image-source output gain.
     */
    std::size_t mFadeSampleTotal{1};

    /* Core delay line (early reflections tap from this). */
    DelayLineU mMainDelay;

    std::array<ReverbPipeline, 2> mPipelines;

    /* The current write offset for all delay lines. */
    std::size_t mOffset{};

    /* Temporary storage used when processing. */
    alignas(16) FloatBufferLine mTempLine{};
    alignas(16) std::array<ReverbUpdateLine, NUM_LINES> mTempSamples{};

    alignas(16) std::array<FloatBufferLine, NUM_LINES> mEarlySamples{};
    alignas(16) std::array<FloatBufferLine, NUM_LINES> mLateSamples{};

    std::array<float, MaxAmbiOrder+1> mOrderScales{};

    bool mUpmixOutput{false};

    /* Image-source model configuration, loaded on device update. */
    IsmConfig mIsmCfg{};
    /* Device sample rate, cached for image-source delay calculations. */
    float mSampleRate{48000.0f};
    /* Back-reference to the effect slot, used to read the aggregate wet
     * source position accumulated by the mixing voices each update block.
     */
    const EffectSlotBase *mSlot{nullptr};

    /* Per-block image-source inputs, cached by update() (which runs only when
     * an effect parameter changes) and consumed by process() (which runs every
     * update block). The source position itself changes without an effect
     * property update, so the tap geometry must be regenerated in process().
     */
    float mIsmDensityMult{1.0f};
    float mIsmAirLpPerSec{0.0f};
    float mIsmReflectionsDelay{0.0f};
    /* Main (device) mix params used to pan the directional B-Format taps. */
    const MixParams *mIsmMainMix{nullptr};


    static constexpr auto DoMixRow(std::span<float> const OutBuffer,
        const std::span<float const, 4> Gains,
        const std::span<FloatBufferLine const, 4> InSamples) noexcept NONBLOCKING
    {
        std::ranges::fill(OutBuffer, 0.0f);
        std::ignore = std::ranges::mismatch(Gains, InSamples,
            [OutBuffer](float const gain, FloatConstBufferSpan const inBuffer)
        {
            if(std::fabs(gain) > GainSilenceThreshold)
            {
                std::ranges::transform(OutBuffer, inBuffer, OutBuffer.begin(),
                    [gain](float const sample, float const in) noexcept
                { return sample + in*gain; });
            }
            return true;
        });
    }

    void MixOutPlain(ReverbPipeline &pipeline, std::span<FloatBufferLine> const samplesOut,
        std::size_t const todo) const
    {
        /* When not upsampling, the panning gains convert to B-Format and pan
         * at the same time.
         */
        std::ignore = std::ranges::mismatch(pipeline.mEarly.Gains, mEarlySamples,
            [samplesOut,todo](EarlyReflections::OutGains &gains,
            FloatConstBufferSpan const inBuffer)
        {
            MixSamples(inBuffer.first(todo), samplesOut, gains.Current, gains.Target, todo, 0);
            return true;
        });
        std::ignore = std::ranges::mismatch(pipeline.mLate.Gains, mLateSamples,
            [samplesOut,todo](LateReverb::OutGains &gains, FloatConstBufferSpan const inBuffer)
        {
            MixSamples(inBuffer.first(todo), samplesOut, gains.Current, gains.Target, todo, 0);
            return true;
        });
    }

    void MixOutAmbiUp(ReverbPipeline &pipeline, std::span<FloatBufferLine> const samplesOut,
        std::size_t const todo)
    {
        /* When upsampling, the B-Format conversion needs to be done separately
         * so the proper HF scaling can be applied to each B-Format channel.
         * The panning gains then pan and upsample the B-Format channels.
         */
        const auto tmpspan = std::span{mTempLine}.first(todo);
        auto hfscale = mOrderScales[0];
        auto splitter = pipeline.mAmbiSplitter[0].begin();
        std::ignore = std::ranges::mismatch(pipeline.mEarly.Gains, EarlyA2B,
            [this,samplesOut,todo,tmpspan,&splitter,&hfscale](EarlyReflections::OutGains &gains,
            std::span<float const, 4> const a2bcoeffs)
        {
            DoMixRow(tmpspan, a2bcoeffs, mEarlySamples);

            /* Apply scaling to the B-Format's HF response to "upsample" it to
             * higher-order output.
             */
            (splitter++)->processHfScale(tmpspan, hfscale);
            hfscale = mOrderScales[1];

            MixSamples(tmpspan, samplesOut, gains.Current, gains.Target, todo, 0);
            return true;
        });

        hfscale = mOrderScales[0];
        splitter = pipeline.mAmbiSplitter[1].begin();
        std::ignore = std::ranges::mismatch(pipeline.mLate.Gains, LateA2B,
            [this,samplesOut,todo,tmpspan,&splitter,&hfscale](LateReverb::OutGains &gains,
            std::span<float const, 4> const a2bcoeffs)
        {
            DoMixRow(tmpspan, a2bcoeffs, mLateSamples);

            (splitter++)->processHfScale(tmpspan, hfscale);
            hfscale = mOrderScales[1];

            MixSamples(tmpspan, samplesOut, gains.Current, gains.Target, todo, 0);
            return true;
        });
    }

    void mixOut(ReverbPipeline &pipeline, std::span<FloatBufferLine> const samplesOut,
        std::size_t const todo)
    {
        if(mUpmixOutput)
            MixOutAmbiUp(pipeline, samplesOut, todo);
        else
            MixOutPlain(pipeline, samplesOut, todo);
    }

    void allocLines(float frequency);

    /* Regenerates the image-source tap targets for the current aggregate
     * source position and the reverb parameters cached by update(), smoothing
     * the previous tap set toward them over the update block. Called every
     * block from process() so the pattern tracks a moving source. No-op when
     * the model is disabled or no mix parameters have been cached yet.
     */
    void updateImageSource(ReverbPipeline &pipeline);

    /* Renders the image-source early reflections: feeds the (already gain-
     * scaled) mono wet input into the variable-delay line and mixes each tap
     * as a directionally panned B-Format signal into the pipeline's ISM
     * output buffer, ready for the final output mix.
     */
    void processImageSource(ReverbPipeline &pipeline, std::size_t offset,
        std::size_t samplesToDo) noexcept;
    /* Mixes the rendered image-source B-Format output to the device buffers,
     * applying the tap panning gains and the overall early output gain.
     */
    void mixImageSource(ReverbPipeline &pipeline, std::span<FloatBufferLine> samplesOut,
        std::size_t samplesToDo) noexcept;

    void deviceUpdate(DeviceBase const *device, BufferStorage const *buffer) override;
    void update(ContextBase const *context, EffectSlotBase const *slot, EffectProps const *props,
        EffectTarget target) noexcept NONBLOCKING override;
    void process(std::size_t samplesToDo, std::span<FloatBufferLine const> samplesIn,
        std::span<FloatBufferLine> samplesOut) noexcept override;
};

/**************************************
 *  Device Update                     *
 **************************************/

auto CalcDelayLengthMult(float const density) -> float
{ return std::max(1.0f, std::cbrt(density*DensityScale)); }

/* Loads the image-source model configuration from the [reverb] config block.
 * Explicit room dimensions and per-axis wall reflection coefficients are
 * optional; when absent the room is derived from the effect's Density and a
 * single shared reflection coefficient is used.
 */
auto LoadIsmConfig() -> IsmConfig
{
    auto cfg = IsmConfig{};
    cfg.mEnabled = ConfigValueBool({}, "reverb", "image-source").value_or(false);
    if(!cfg.mEnabled)
        return cfg;

    if(auto v = ConfigValueU32({}, "reverb", "image-source-order"))
        cfg.mOrder = std::clamp(*v, 1u, IsmMaxOrder);
    if(auto v = ConfigValueF32({}, "reverb", "image-source-smoothing"))
        cfg.mSmoothing = std::clamp(*v, 0.0f, 0.98f);

    const auto w = ConfigValueF32({}, "reverb", "room-width");
    const auto h = ConfigValueF32({}, "reverb", "room-height");
    const auto d = ConfigValueF32({}, "reverb", "room-depth");
    if(w && h && d)
    {
        cfg.mExplicitRoom = true;
        cfg.mRoomSize = {
            std::max(IsmMinRoomSize, *w),
            std::max(IsmMinRoomSize, *h),
            std::max(IsmMinRoomSize, *d),
        };
    }

    const auto r = ConfigValueF32({}, "reverb", "wall-reflect");
    if(r)
    {
        const auto coeff = std::clamp(*r, 0.0f, 1.0f);
        cfg.mWallReflect = {coeff, coeff, coeff};
    }
    const auto rx = ConfigValueF32({}, "reverb", "wall-reflect-x");
    const auto ry = ConfigValueF32({}, "reverb", "wall-reflect-y");
    const auto rz = ConfigValueF32({}, "reverb", "wall-reflect-z");
    if(rx) cfg.mWallReflect[0] = std::clamp(*rx, 0.0f, 1.0f);
    if(ry) cfg.mWallReflect[1] = std::clamp(*ry, 0.0f, 1.0f);
    if(rz) cfg.mWallReflect[2] = std::clamp(*rz, 0.0f, 1.0f);
    return cfg;
}

/* Converts the EFX air-absorption high-frequency gain (a linear gain at the
 * 5 kHz reference) into a one-pole low-pass coefficient applied over one
 * second of propagation. Solves |H(f)|^2 = g^2 for the coefficient a:
 *
 *     a = (1-g) / (1+g) / (1 - 2g/(1+g) cos(2 pi f/fs))
 *
 * with f = 5 kHz. Returns 0 (no filtering) when the gain is at unity.
 */
auto CalcAirLpCoeff(float const airGainHF, float const frequency) -> float
{
    if(airGainHF >= 0.99999f)
        return 0.0f;
    const auto g = std::clamp(airGainHF, 1.0e-4f, 0.99999f);
    const auto w = 2.0f*std::numbers::pi_v<float>*5000.0f/frequency;
    const auto denom = (1.0f+g) - 2.0f*g*std::cos(w);
    return std::clamp((1.0f-g)/denom, 0.0f, 1.0f);
}

/* Generates the shoebox image-source taps for a head-relative source
 * position (listener at the origin). The room is centered on the listener, so
 * its walls sit at +/- L_a/2 along each axis. Image coordinates follow the
 * standard rectangular-room (Allen-Berkley) image-source relations. Per axis
 * the room spans [0, L]; with the uncentered source s = S + L/2 and listener
 * r = L/2, there are two image families for every integer p:
 *
 *     plus family:  img = 2 p L + s,  per-axis reflection order 2|p|
 *     minus family: img = 2 p L - s,  per-axis order 2p-1 (p>=1), 1-2p (p<=0)
 *
 * The centered image coordinate is img - L/2. The minus family with p = 0 is
 * the real (unreflected) source, which is excluded since only reflections are
 * wanted. Each image contributes a delay d/c, a 1/d distance attenuation
 * combined with the per-wall reflection product, and the head-relative arrival
 * direction. Taps beyond ReverbMaxReflectionsDelay are dropped and the result
 * is sorted by delay.
 */
void GenerateImageTaps(IsmConfig const &cfg, std::span<const float,3> const source,
    std::span<IsmTap, IsmMaxTaps> taps, std::size_t &tapCount, float const baseDelay,
    float const airLpPerSec)
{
    struct RawTap {
        float Delay;
        float Gain;
        float LpCoeff;
        std::array<float,3> Dir;
    };
    auto raw = std::array<RawTap, IsmMaxTaps>{};
    auto count = 0_uz;

    const auto order = static_cast<int>(cfg.mOrder);
    /* Two image families per axis: plus (even reflection order) and minus
     * (odd reflection order).
     */
    for(auto fx = 0; fx < 2; ++fx)
    for(auto fy = 0; fy < 2; ++fy)
    for(auto fz = 0; fz < 2; ++fz)
    {
        const auto family = std::array{fx, fy, fz};
        for(auto px = -order; px <= order; ++px)
        for(auto py = -order; py <= order; ++py)
        for(auto pz = -order; pz <= order; ++pz)
        {
            const auto pidx = std::array{px, py, pz};

            auto image = std::array<float,3>{};
            auto wallGain = 1.0f;
            auto reflOrder = 0u;
            for(auto a = 0; a < 3; ++a)
            {
                const auto p = pidx[a];
                const auto len = cfg.mRoomSize[a];
                const auto s = source[a] + len*0.5f;
                const auto pf = static_cast<float>(p);
                int axisOrder;
                if(family[a] == 0)
                {
                    /* Plus family: img = 2pL + s, reflection order 2|p|. */
                    image[a] = 2.0f*pf*len + s - len*0.5f;
                    axisOrder = 2*std::abs(p);
                }
                else
                {
                    /* Minus family: img = 2pL - s. The p = 0 and p = 1
                     * images are the two first-order wall reflections; the
                     * order grows by two for every further p step.
                     */
                    image[a] = 2.0f*pf*len - s - len*0.5f;
                    axisOrder = (p >= 1) ? (2*p - 1) : (1 - 2*p);
                }
                reflOrder += static_cast<unsigned>(axisOrder);
                wallGain *= std::pow(cfg.mWallReflect[a],
                    static_cast<float>(axisOrder));
            }

            /* Reflection order zero is the real (direct) source path. */
            if(reflOrder == 0u || reflOrder > cfg.mOrder)
                continue;

            const auto dx = image[0];
            const auto dy = image[1];
            const auto dz = image[2];
            const auto path = std::sqrt(dx*dx + dy*dy + dz*dz);
            if(!(path > 1.0e-4f))
                continue;

            const auto delay = path/SpeedOfSound + baseDelay;
            if(delay > ReverbMaxReflectionsDelay || count >= raw.size())
                continue;

            raw[count++] = RawTap{
                delay,
                wallGain/path,
                /* One-pole coefficient for the total path time; the
                 * per-second filter is exponentiated by travel time.
                 */
                1.0f - std::pow(1.0f - airLpPerSec, std::max(delay, 0.0f)),
                {dx/path, dy/path, dz/path},
            };
        }
    }

    std::ranges::sort(raw | std::views::take(count), {}, &RawTap::Delay);
    tapCount = count;
    for(auto i = 0_uz; i < count; ++i)
    {
        taps[i].DelayTarget = raw[i].Delay;
        taps[i].GainTarget = raw[i].Gain;
        taps[i].LpCoeffTarget = raw[i].LpCoeff;
        taps[i].DirTarget = raw[i].Dir;
    }
}

/* Calculates the delay line metrics and allocates the shared sample buffer
 * for all lines given the sample rate (frequency).
 */
void ReverbState::allocLines(float const frequency)
{
    /* Multiplier for the maximum density value, i.e. density=1, which is
     * actually the least density...
     */
    const auto multiplier = CalcDelayLengthMult(1.0f);

    /* The modulator's line length is calculated from the maximum modulation
     * time and depth coefficient, and halfed for the low-to-high frequency
     * swing.
     */
    static constexpr auto max_mod_delay = MaxModulationTime*MODULATION_DEPTH_COEFF / 2.0f;

    /* The max number of samples done per iteration of the late reverb's vector
     * all-pass.
     */
    const auto late_vecap_extra = al::saturating_cast<unsigned>(
        std::ceil(LATE_ALLPASS_LENGTHS.front() * multiplier * frequency));

    auto linelengths = std::array<std::size_t, 13>{};
    auto oidx = 0_uz;

    auto totalSamples = 0_uz;
    /* The main delay length includes the maximum early reflection delay and
     * the largest early tap width. It must also be extended by the update size
     * (BufferLineSize) for block processing.
     */
    auto length = ReverbMaxReflectionsDelay + EARLY_TAP_LENGTHS.back()*multiplier;
    auto count = mMainDelay.calcLineLength(length, frequency, BufferLineSize);
    linelengths[oidx++] = count;
    totalSamples += count;
    for(auto &pipeline : mPipelines)
    {
        static constexpr auto LateDiffAvg = (LATE_LINE_LENGTHS.back()-LATE_LINE_LENGTHS.front()) /
            float{NUM_LINES};
        length = ReverbMaxLateReverbDelay + LateDiffAvg*multiplier;
        count = pipeline.mLateDelayIn.calcLineLength(length, frequency, BufferLineSize);
        linelengths[oidx++] = count;
        totalSamples += count;

        /* The early reflection all-pass line. */
        length = EARLY_ALLPASS_LENGTHS.back() * multiplier;
        count = pipeline.mEarly.Allpass.Delay.calcLineLength(length, frequency, 0);
        linelengths[oidx++] = count;
        totalSamples += count;

        /* The early reflection delay line. */
        length = EARLY_LINE_LENGTHS.back() * multiplier;
        count = pipeline.mEarly.Delay.calcLineLength(length, frequency, MAX_UPDATE_SAMPLES);
        linelengths[oidx++] = count;
        totalSamples += count;

        /* The late vector all-pass line. */
        length = LATE_ALLPASS_LENGTHS.back() * multiplier;
        count = pipeline.mLate.VecAp.Delay.calcLineLength(length, frequency, late_vecap_extra);
        linelengths[oidx++] = count;
        totalSamples += count;

        /* The late delay lines are calculated from the largest maximum density
         * line length, and the maximum modulation delay. Four additional
         * samples are needed for resampling the modulator delay.
         */
        length = LATE_LINE_LENGTHS.back()*multiplier + max_mod_delay;
        count = pipeline.mLate.Delay.calcLineLength(length, frequency, 4);
        linelengths[oidx++] = count;
        totalSamples += count;

        /* The image-source early-reflection delay line. It must hold the
         * longest image path (bounded by ReverbMaxReflectionsDelay), plus two
         * samples of headroom for the fractional-delay interpolation.
         */
        if(mIsmCfg.mEnabled)
        {
            length = ReverbMaxReflectionsDelay;
            count = pipeline.mIsm.Delay.calcLineLength(length, frequency, 2);
            linelengths[oidx++] = count;
            totalSamples += count;
        }
    }
    Ensures(oidx == (mIsmCfg.mEnabled ? linelengths.size() : 11_uz));

    if(totalSamples != mSampleBuffer.size())
        decltype(mSampleBuffer)(totalSamples).swap(mSampleBuffer);
    std::ranges::fill(mSampleBuffer, 0.0f);

    /* Update all delays to reflect the new sample buffer. */
    auto bufferspan = std::span{mSampleBuffer};
    oidx = 0;
    mMainDelay.realizeLineOffset(bufferspan.first(linelengths[oidx]));
    bufferspan = bufferspan.subspan(linelengths[oidx++]);
    for(auto &pipeline : mPipelines)
    {
        pipeline.mLateDelayIn.realizeLineOffset(bufferspan.first(linelengths[oidx]));
        bufferspan = bufferspan.subspan(linelengths[oidx++]);
        pipeline.mEarly.Allpass.Delay.realizeLineOffset(bufferspan.first(linelengths[oidx]));
        bufferspan = bufferspan.subspan(linelengths[oidx++]);
        pipeline.mEarly.Delay.realizeLineOffset(bufferspan.first(linelengths[oidx]));
        bufferspan = bufferspan.subspan(linelengths[oidx++]);
        pipeline.mLate.VecAp.Delay.realizeLineOffset(bufferspan.first(linelengths[oidx]));
        bufferspan = bufferspan.subspan(linelengths[oidx++]);
        pipeline.mLate.Delay.realizeLineOffset(bufferspan.first(linelengths[oidx]));
        bufferspan = bufferspan.subspan(linelengths[oidx++]);
        if(mIsmCfg.mEnabled)
        {
            pipeline.mIsm.Delay.realizeLineOffset(bufferspan.first(linelengths[oidx]));
            bufferspan = bufferspan.subspan(linelengths[oidx++]);
        }
    }
    Ensures(oidx == (mIsmCfg.mEnabled ? linelengths.size() : 11_uz));
}

void ReverbState::deviceUpdate(DeviceBase const *device, BufferStorage const*)
{
    const auto frequency = static_cast<float>(device->mSampleRate);
    mSampleRate = frequency;

    /* Load the optional image-source model configuration before allocating
     * the delay lines, since the model adds its own per-pipeline line.
     */
    mIsmCfg = LoadIsmConfig();

    /* Allocate the delay lines. */
    allocLines(frequency);

    std::ranges::for_each(mPipelines, &ReverbPipeline::clear);
    mPipelineState = DeviceClear;

    /* Reset offset base. */
    mOffset = 0;

    if(device->mAmbiOrder > 1)
    {
        mUpmixOutput = true;
        mOrderScales = AmbiScale::GetHFOrderScales(1, device->mAmbiOrder, device->m2DMixing);
    }
    else
    {
        mUpmixOutput = false;
        mOrderScales.fill(1.0f);
    }

    const auto splitter = BandSplitter{device->mXOverFreq / frequency};
    std::ranges::for_each(mPipelines, [splitter](ReverbPipeline &pipeline)
    {
        pipeline.mAmbiSplitter[0].fill(splitter);
        pipeline.mAmbiSplitter[1].fill(splitter);
    });
}

void ReverbState::updateImageSource(ReverbPipeline &pipeline)
{
    if(!mIsmCfg.mEnabled || mIsmMainMix == nullptr)
        return;

    auto &ism = pipeline.mIsm;
    const auto density_mult = mIsmDensityMult;
    const auto mainMix = mIsmMainMix;

    /* Resolve the room geometry. Without explicit dimensions, derive an
     * average side length from the Density room-size multiplier (anchored on
     * the same 5 m average the legacy taps use).
     */
    auto roomSize = mIsmCfg.mRoomSize;
    if(!mIsmCfg.mExplicitRoom)
        roomSize = {
            IsmBaseRoomSize*density_mult,
            IsmBaseRoomSize*density_mult,
            IsmBaseRoomSize*density_mult,
        };
    auto cfg = mIsmCfg;
    cfg.mRoomSize = roomSize;

    /* Consume the slot's energy-weighted aggregate source position. When no
     * voice contributed this block, hold the last resolved position.
     */
    if(mSlot != nullptr && mSlot->mWetPosWeight > GainSilenceThreshold)
    {
        const auto inv = 1.0f/mSlot->mWetPosWeight;
        ism.LastPos = {
            mSlot->mWetPosAccum[0]*inv,
            mSlot->mWetPosAccum[1]*inv,
            mSlot->mWetPosAccum[2]*inv,
        };
        ism.HasPosition = true;
    }

    /* Keep the source inside the room (with a small margin) so image geometry
     * stays well conditioned even if a source is placed beyond a wall.
     */
    auto source = ism.LastPos;
    for(auto a = 0; a < 3; ++a)
    {
        const auto bound = roomSize[a]*0.5f - IsmMinRoomSize*0.1f;
        source[a] = std::clamp(source[a], -bound, bound);
    }

    GenerateImageTaps(cfg, source, ism.Taps, ism.TapCount, mIsmReflectionsDelay,
        mIsmAirLpPerSec);
    if(!ism.HasPosition)
    {
        /* No source position yet: keep the generated tap geometry but leave
         * the gains silent.
         */
        for(auto i = 0_uz; i < ism.TapCount; ++i)
            ism.Taps[i].GainTarget = 0.0f;
    }

    /* Normalize tap gains so the total early energy is independent of room
     * geometry and tap count. The slot's ReflectionsGain is applied to the
     * whole early stage by the caller through the shared early output gain.
     */
    auto energy = 0.0f;
    for(auto i = 0_uz; i < ism.TapCount; ++i)
        energy += ism.Taps[i].GainTarget*ism.Taps[i].GainTarget;
    if(energy > std::numeric_limits<float>::epsilon())
    {
        const auto norm = 1.0f/std::sqrt(energy);
        for(auto i = 0_uz; i < ism.TapCount; ++i)
            ism.Taps[i].GainTarget *= norm;
    }

    /* One-pole smoothing for tap delay/gain/filter/direction updates. On the
     * first block with a real source position, snap the geometry (delay,
     * filter coefficient, direction and panning) to its target so the opening
     * impulse is rendered with the correct reflection timing; the gain still
     * ramps up from zero to avoid a click.
     */
    const auto a = 1.0f - cfg.mSmoothing;
    const auto snapGeometry = ism.HasPosition && !ism.GeometryInitialized;
    for(auto i = 0_uz; i < ism.TapCount; ++i)
    {
        auto &tap = ism.Taps[i];
        if(snapGeometry)
        {
            tap.Delay = tap.DelayTarget;
            tap.LpCoeff = tap.LpCoeffTarget;
            tap.Dir = tap.DirTarget;
        }
        else
        {
            tap.Delay = std::lerp(tap.Delay, tap.DelayTarget, a);
            tap.LpCoeff = std::lerp(tap.LpCoeff, tap.LpCoeffTarget, a);
            for(auto c = 0; c < 3; ++c)
                tap.Dir[c] = std::lerp(tap.Dir[c], tap.DirTarget[c], a);
        }
        /* Gain always eases toward its target (from zero while silent), which
         * also provides the fade-in when taps first become audible.
         */
        tap.Gain = std::lerp(tap.Gain, tap.GainTarget, a);

        /* Compute output-channel panning gains for each first-order B-Format
         * (ACN/N3D) component this tap produces. The tap's encoded signal is
         * W plus the directional Y/Z/X components, so each component is mapped
         * through the device's ambisonic channel map independently.
         */
        auto ambi = std::array<float, MaxAmbiChannels>{};
        ambi[0] = 1.0f;                 /* ACN 0: W (omnidirectional) */
        ComputePanGains(mainMix, ambi, tap.Gain, tap.Gains.Target[0]);
        /* ACN 1..3: directional components in N3D, converted from the
         * head-relative OpenAL arrival direction (negate X and Z).
         */
        const auto dirCoeff = std::array{-tap.Dir[0], tap.Dir[1], -tap.Dir[2]};
        for(auto c = 0_uz; c < 3_uz; ++c)
        {
            ambi.fill(0.0f);
            ambi[c+1] = dirCoeff[c];
            ComputePanGains(mainMix, ambi, tap.Gain, tap.Gains.Target[c+1]);
        }
        if(snapGeometry)
            tap.Gains.Current = tap.Gains.Target;
    }
    if(ism.HasPosition)
        ism.GeometryInitialized = true;

    /* Retire taps beyond the current count. */
    for(auto i = ism.TapCount; i < IsmMaxTaps; ++i)
        ism.Taps[i].Gains.fill(0.0f);
}

void ReverbState::processImageSource(ReverbPipeline &pipeline, std::size_t const offset,
    std::size_t const samplesToDo) noexcept
{
    if(!mIsmCfg.mEnabled)
        return;

    auto &ism = pipeline.mIsm;
    const auto line = ism.Delay.get(0);
    const auto stride = line.size();

    for(auto t = 0_uz; t < ism.TapCount; ++t)
    {
        auto &tap = ism.Taps[t];
        /* Clear this tap's B-Format block. */
        for(auto c = 0_uz; c < NUM_LINES; ++c)
            std::ranges::fill(std::span{ism.TapBf[t][c]}.first(samplesToDo), 0.0f);
        if(tap.Gain <= GainSilenceThreshold && tap.GainTarget <= GainSilenceThreshold)
            continue;

        /* Fractional delay (samples) with linear interpolation, bounded by the
         * allocated line length.
         */
        const auto delaySamples = std::clamp(tap.Delay*mSampleRate, 1.0f,
            static_cast<float>(stride-3));
        const auto d0 = static_cast<std::size_t>(delaySamples);
        const auto frac = delaySamples - static_cast<float>(d0);
        const auto lpA = tap.LpCoeff;

        auto gain = tap.Gain;
        const auto gainStep = (tap.GainTarget - tap.Gain)
            / static_cast<float>(samplesToDo);

        auto lpState = tap.LpState;
        for(auto i = 0_uz; i < samplesToDo; ++i)
        {
            const auto pos = (offset + i - d0) & (stride-1);
            const auto s0 = line[pos];
            const auto s1 = line[(pos+1) & (stride-1)];
            /* Wall/air high-frequency loss as a one-pole low-pass. */
            lpState = s0 + lpA*(lpState - s0);
            const auto delayed = std::lerp(s0, s1, frac);
            const auto wet = delayed*gain;
            gain += gainStep;
            /* First-order B-Format (N3D). ACN 0 = W (omni); the directional
             * components negate OpenAL X and Z.
             */
            ism.TapBf[t][0][i] = wet;
            ism.TapBf[t][1][i] = wet*(-tap.Dir[0]);
            ism.TapBf[t][2][i] = wet*( tap.Dir[1]);
            ism.TapBf[t][3][i] = wet*(-tap.Dir[2]);
        }
        tap.LpState = lpState;
        tap.Gain = gain;
    }
}

void ReverbState::mixImageSource(ReverbPipeline &pipeline,
    std::span<FloatBufferLine> const samplesOut, std::size_t const samplesToDo) noexcept
{
    if(!mIsmCfg.mEnabled)
        return;

    auto &ism = pipeline.mIsm;
    const auto outGain = pipeline.mIsmOutGain;
    if(outGain <= GainSilenceThreshold)
        return;

    const auto nOut = std::min(samplesOut.size(), MaxAmbiChannels);
    for(auto t = 0_uz; t < ism.TapCount; ++t)
    {
        auto &tap = ism.Taps[t];
        if(tap.Gain <= GainSilenceThreshold && tap.GainTarget <= GainSilenceThreshold)
            continue;

        /* Interpolate each B-Format component's output panning gains over the
         * block while mixing its rendered signal.
         */
        for(auto bf = 0_uz; bf < NUM_LINES; ++bf)
        {
            auto gains = tap.Gains.Current[bf];
            auto steps = std::array<float, MaxAmbiChannels>{};
            for(auto c = 0_uz; c < MaxAmbiChannels; ++c)
                steps[c] = (tap.Gains.Target[bf][c] - tap.Gains.Current[bf][c])
                    / static_cast<float>(samplesToDo);

            for(auto i = 0_uz; i < samplesToDo; ++i)
            {
                const auto wet = ism.TapBf[t][bf][i]*outGain;
                for(auto c = 0_uz; c < nOut; ++c)
                {
                    samplesOut[c][i] += wet*gains[c];
                    gains[c] += steps[c];
                }
            }
            tap.Gains.Current[bf] = gains;
        }
    }
}

/**************************************
 *  Effect Update                     *
 **************************************/

/* Calculate a decay coefficient given the length of each cycle and the time
 * until the decay reaches -60 dB.
 */
auto CalcDecayCoeff(float const length, float const decayTime) -> float
{ return std::pow(ReverbDecayGain, length/decayTime); }

/* Calculate a decay length from a coefficient and the time until the decay
 * reaches -60 dB.
 */
auto CalcDecayLength(float const coeff, float const decayTime) -> float
{
    constexpr auto log10_decaygain = -3.0f/*std::log10(ReverbDecayGain)*/;
    return std::log10(coeff) * decayTime / log10_decaygain;
}

/* Calculate an attenuation to be applied to the input of any echo models to
 * compensate for modal density and decay time.
 */
auto CalcDensityGain(float const a) -> float
{
    /* The energy of a signal can be obtained by finding the area under the
     * squared signal.  This takes the form of Sum(x_n^2), where x is the
     * amplitude for the sample n.
     *
     * Decaying feedback matches exponential decay of the form Sum(a^n),
     * where a is the attenuation coefficient, and n is the sample.  The area
     * under this decay curve can be calculated as:  1 / (1 - a).
     *
     * Modifying the above equation to find the area under the squared curve
     * (for energy) yields:  1 / (1 - a^2).  Input attenuation can then be
     * calculated by inverting the square root of this approximation,
     * yielding:  1 / sqrt(1 / (1 - a^2)), simplified to: sqrt(1 - a^2).
     */
    return std::sqrt(1.0f - a*a);
}

/* Calculate the scattering matrix coefficients given a diffusion factor. */
void CalcMatrixCoeffs(float const diffusion, float *const x, float *const y)
{
    /* The matrix is of order 4, so n is sqrt(4 - 1). */
    constexpr auto n = std::numbers::sqrt3_v<float>;
    const auto t = diffusion * std::atan(n);

    /* Calculate the first mixing matrix coefficient. */
    *x = std::cos(t);
    /* Calculate the second mixing matrix coefficient. */
    *y = std::sin(t) / n;
}

/* Calculate the limited HF ratio for use with the late reverb low-pass
 * filters.
 */
auto CalcLimitedHfRatio(float const hfRatio, float const airAbsorptionGainHF,
    float const decayTime) -> float
{
    /* Find the attenuation due to air absorption in dB (converting delay time
     * to meters using the speed of sound). Then reversing the decay equation,
     * solve for HF ratio. The delay length is cancelled out of the equation,
     * so it can be calculated once for all lines.
     */
    const auto limitRatio = 1.0f / SpeedOfSoundMetersPerSec /
        CalcDecayLength(airAbsorptionGainHF, decayTime);

    /* Using the limit calculated above, apply the upper bound to the HF ratio. */
    return std::min(limitRatio, hfRatio);
}


/* Calculates the 3-band T60 damping coefficients for a particular delay line
 * of specified length, using a combination of two shelf filter sections given
 * decay times for each band split at two reference frequencies.
 */
void T60Filter::calcCoeffs(float const length, float const lfDecayTime,
    float const mfDecayTime, float const hfDecayTime, float const lf0norm,
    float const hf0norm)
{
    const auto mfGain = CalcDecayCoeff(length, mfDecayTime);
    const auto lfGain = CalcDecayCoeff(length, lfDecayTime) / mfGain;
    const auto hfGain = CalcDecayCoeff(length, hfDecayTime) / mfGain;

    mMidGain = mfGain;
    mLFFilter.setParamsFromSlope(BiquadType::LowShelf, lf0norm, lfGain, 1.0f);
    mHFFilter.setParamsFromSlope(BiquadType::HighShelf, hf0norm, hfGain, 1.0f);
}

/* Update the early reflection line lengths and gain coefficients. */
void EarlyReflections::updateLines(float const density_mult, float const diffusion,
    float const decayTime, float const frequency)
{
    /* Calculate the all-pass feed-back/forward coefficient. */
    Allpass.Coeff = diffusion*diffusion * InvSqrt2;

    /* Calculate the delay length of each all-pass line. */
    std::ranges::transform(EARLY_ALLPASS_LENGTHS, Allpass.Offset.begin(),
        [density_mult,frequency](float const length) -> unsigned
    { return al::saturating_cast<unsigned>(length * density_mult * frequency); });

    /* Calculate the delay length of each delay line. */
    std::ranges::transform(EARLY_LINE_LENGTHS, Offset.begin(),
        [density_mult,frequency](float const length) -> unsigned
    { return al::saturating_cast<unsigned>(length * density_mult * frequency); });

    /* Calculate the gain (coefficient) for the secondary reflections based on
     * the average delay and decay time.
     */
    const auto length = std::reduce(EARLY_LINE_LENGTHS.begin(), EARLY_LINE_LENGTHS.end(), 0.0f)
        / float{EARLY_LINE_LENGTHS.size()} * density_mult;
    Coeff = CalcDecayCoeff(length, decayTime);
}

/* Update the EAX modulation step and depth. Keep in mind that this kind of
 * vibrato is additive and not multiplicative as one may expect. The downswing
 * will sound stronger than the upswing.
 */
void Modulation::updateModulator(float const modTime, float const modDepth, float const frequency)
{
    /* Modulation is calculated in two parts.
     *
     * The modulation time effects the sinus rate, altering the speed of
     * frequency changes. An index is incremented for each sample with an
     * appropriate step size to generate an LFO, which will vary the feedback
     * delay over time.
     */
    Step = std::max(al::saturating_round<unsigned>(float{MOD_FRACONE} / (frequency*modTime)), 1u);

    /* The modulation depth effects the amount of frequency change over the
     * range of the sinus. It needs to be scaled by the modulation time so that
     * a given depth produces a consistent change in frequency over all ranges
     * of time. Since the depth is applied to a sinus value, it needs to be
     * halved once for the sinus range and again for the sinus swing in time
     * (half of it is spent decreasing the frequency, half is spent increasing
     * it).
     */
    if(modTime >= DefaultModulationTime)
    {
        /* To cancel the effects of a long period modulation on the late
         * reverberation, the amount of pitch should be varied (decreased)
         * according to the modulation time. The natural form is varying
         * inversely, in fact resulting in an invariant.
         */
        Depth = MODULATION_DEPTH_COEFF / 4.0f * DefaultModulationTime * modDepth * frequency;
    }
    else
        Depth = MODULATION_DEPTH_COEFF / 4.0f * modTime * modDepth * frequency;
}

/* Update the late reverb line lengths and T60 coefficients. */
void LateReverb::updateLines(float const density_mult, float const diffusion,
    float const lfDecayTime, float const mfDecayTime, float const hfDecayTime,
    float const lf0norm, float const hf0norm, float const frequency)
{
    /* Scaling factor to convert the normalized reference frequencies from
     * representing 0...freq to 0...max_reference.
     */
    const auto norm_weight_factor = frequency / MaxHFReference;

    /* To compensate for changes in modal density and decay time of the late
     * reverb signal, the input is attenuated based on the maximal energy of
     * the outgoing signal.  This approximation is used to keep the apparent
     * energy of the signal equal for all ranges of density and decay time.
     *
     * The average length of the delay lines is used to calculate the
     * attenuation coefficient.
     */

    /* The density gain calculation uses an average decay time weighted by
     * approximate bandwidth. This attempts to compensate for losses of energy
     * that reduce decay time due to scattering into highly attenuated bands.
     */
    const auto decayTimeWeighted =
        lf0norm*norm_weight_factor*lfDecayTime +
        (hf0norm - lf0norm)*norm_weight_factor*mfDecayTime +
        (1.0f - hf0norm*norm_weight_factor)*hfDecayTime;
    DensityGain = CalcDensityGain(CalcDecayCoeff(LateDelayAverage*density_mult,
        decayTimeWeighted));

    /* Calculate the all-pass feed-back/forward coefficient. */
    VecAp.Coeff = diffusion*diffusion * InvSqrt2;

    /* Calculate the delay length of each all-pass line. */
    std::ranges::transform(LATE_ALLPASS_LENGTHS, VecAp.Offset.begin(),
        [density_mult,frequency](float const length) -> unsigned
    { return al::saturating_cast<unsigned>(length*density_mult * frequency); });

    /* Calculate the length of each feedback delay line. A cubic resampler is
     * used for modulation on the feedback delay, which includes one sample of
     * delay. Reduce by one to compensate.
     */
    auto lengths = std::array<float, NUM_LINES>{};
    std::ranges::transform(LATE_LINE_LENGTHS, lengths.begin(),
        [density_mult](float const length) noexcept -> float { return length * density_mult; });

    std::ranges::transform(lengths, Offset.begin(), [frequency](float const length) -> unsigned
    { return std::max(al::saturating_cast<unsigned>(length*frequency + 0.5f), 1u) - 1u; });

    /* Approximate the absorption that the vector all-pass would exhibit given
     * the current diffusion so we don't have to process a full T60 filter for
     * each of its four lines. Also include the average modulation delay (depth
     * is half the max delay in samples).
     */
    std::ranges::transform(LATE_ALLPASS_LENGTHS, lengths, lengths.begin(),
        [density_mult,diffusion,moddepth=Mod.Depth/frequency](float const length,
            float const curlength) -> float
        {
            return al::lerp(length, LateAllpassAverage, diffusion)*density_mult + moddepth
                + curlength;
        });

    /* Calculate the T60 damping coefficients for each line. */
    std::ignore = std::ranges::mismatch(T60, lengths,
        [lfDecayTime,mfDecayTime,hfDecayTime,lf0norm,hf0norm](T60Filter &filter, float const length)
    {
        filter.calcCoeffs(length, lfDecayTime, mfDecayTime, hfDecayTime, lf0norm, hf0norm);
        return true;
    });
}


/* Update the offsets for the main effect delay line. */
void ReverbPipeline::updateDelayLine(float const gain, float const earlyDelay,
    float const lateDelay, float const density_mult, float const frequency)
{
    /* Early reflection taps are decorrelated by means of an average room
     * reflection approximation described above the definition of the taps.
     * This approximation is linear and so the above density multiplier can
     * be applied to adjust the width of the taps.  A single-band decay
     * coefficient is applied to simulate initial attenuation and absorption.
     *
     * Late reverb taps are based on the late line lengths to allow a zero-
     * delay path and offsets that would continue the propagation naturally
     * into the late lines.
     */
    mEarlyDelayCoeff[1] = gain;
    std::ranges::transform(EARLY_TAP_LENGTHS, (mEarlyDelayTap | std::views::elements<1>).begin(),
        [earlyDelay,frequency,density_mult](float const length) -> unsigned
    { return al::saturating_cast<unsigned>((length*density_mult + earlyDelay) * frequency); });

    std::ranges::transform(LATE_LINE_LENGTHS, (mLateDelayTap | std::views::elements<1>).begin(),
        [lateDelay,frequency,density_mult](float length) noexcept -> unsigned
    {
        /* Reduce the late delay tap by the shortest late delay line length to
         * compensate for the late line input being fed by the delayed early
         * output.
         */
        length -= LATE_LINE_LENGTHS.front();
        length = length*float{1.0f/NUM_LINES}*density_mult + lateDelay;
        return al::saturating_cast<unsigned>(length * frequency);
    });
}

/* Creates a transform matrix given a reverb vector. The vector pans the reverb
 * reflections toward the given direction, using its magnitude (up to 1) as a
 * focal strength. This function results in a B-Format transformation matrix
 * that spatially focuses the signal in the desired direction.
 */
using Array4x4 = std::array<std::array<float, 4>, 4>;
auto GetTransformFromVector(const std::span<float const, 3> vec) -> Array4x4
{
    /* Normalize the panning vector according to the N3D scale, which has an
     * extra sqrt(3) term on the directional components. Converting from OpenAL
     * to B-Format also requires negating X (ACN 1) and Z (ACN 3). Note however
     * that the reverb panning vectors use left-handed coordinates, unlike the
     * rest of OpenAL which use right-handed. This is fixed by negating Z,
     * which cancels out with the B-Format Z negation.
     */
    auto norm = std::array{vec[0], vec[1], vec[2]};
    auto mag = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    if(mag > 1.0f)
    {
        const auto scale = std::numbers::sqrt3_v<float> / mag;
        norm[0] *= -scale;
        norm[1] *= scale;
        norm[2] *= scale;
        mag = 1.0f;
    }
    else
    {
        /* If the magnitude is less than or equal to 1, just apply the sqrt(3)
         * term. There's no need to renormalize the magnitude since it would
         * just be reapplied in the matrix.
         */
        norm[0] *= -std::numbers::sqrt3_v<float>;
        norm[1] *= std::numbers::sqrt3_v<float>;
        norm[2] *= std::numbers::sqrt3_v<float>;
    }

    /* NOTE: This is transposed from a typical transform matrix. */
    return Array4x4{{
        {{1.0f, norm[0], norm[1], norm[2]}},
        {{0.0f,    1.0f-mag,  0.0f,  0.0f}},
        {{0.0f,    0.0f,  1.0f-mag,  0.0f}},
        {{0.0f,    0.0f,  0.0f,  1.0f-mag}}
    }};
}

/* Update the early and late 3D panning gains. */
void ReverbPipeline::update3DPanning(std::span<float const, 3> const ReflectionsPan,
    std::span<float const, 3> const LateReverbPan, float const earlyGain, float const lateGain,
    bool const doUpmix, MixParams const *const mainMix)
{
    /* Create matrices that transform a B-Format signal according to the
     * panning vectors.
     */
    auto const earlymat = GetTransformFromVector(ReflectionsPan);
    auto const latemat = GetTransformFromVector(LateReverbPan);

    auto get_coeffs = [doUpmix](std::span<std::array<float, 4> const, 4> const a2bmatrix,
        std::span<std::array<float, 4> const, 4> const matrix)
    {
        auto res = std::array<std::array<float, MaxAmbiChannels>, NUM_LINES>{};
        if(doUpmix)
        {
            /* When upsampling, combine the early and late transforms with the
             * first-order upsample matrix. This results in panning gains that
             * apply the panning transform to first-order B-Format, which is
             * then upsampled.
             */
            const auto mtx2 = std::span{AmbiScale::FirstOrderUp};

            for(const auto i : std::views::iota(0_uz, matrix.size()))
            {
                const auto dst = std::span{res[i]};
                static_assert(dst.size() >= std::tuple_size_v<decltype(mtx2)::element_type>);
                for(const auto j : std::views::iota(0_uz, matrix[i].size()))
                {
                    auto const a = matrix[i][j];
                    std::ranges::transform(mtx2[j], dst, dst.begin(),
                        [a](float const in, float const out) -> float { return a*in + out; });
                }
            }
        }
        else
        {
            /* When not upsampling, combine the A-to-B-Format conversion with
             * its respective transform. This results in panning gains that
             * convert A-Format to B-Format, which is then panned.
             */
            for(const auto i : std::views::iota(0_uz, a2bmatrix[0].size()))
            {
                const auto dst = std::span{res[i]};
                static_assert(dst.size() >= std::tuple_size_v<decltype(matrix)::element_type>);
                for(const auto j : std::views::iota(0_uz, a2bmatrix.size()))
                {
                    auto const a = a2bmatrix[j][i];
                    std::ranges::transform(matrix[j], dst, dst.begin(),
                        [a](float const in, float const out) -> float { return a*in + out; });
                }
            }
        }
        return res;
    };

    std::ignore = std::ranges::mismatch(get_coeffs(EarlyA2B, earlymat), mEarly.Gains,
        [mainMix,earlyGain](const auto &coeffs, EarlyReflections::OutGains &earlygains)
    {
        ComputePanGains(mainMix, coeffs, earlyGain, earlygains.Target);
        return true;
    });

    std::ignore = std::ranges::mismatch(get_coeffs(LateA2B, latemat), mLate.Gains,
        [mainMix,lateGain](const auto &coeffs, LateReverb::OutGains &lategains)
    {
        ComputePanGains(mainMix, coeffs, lateGain, lategains.Target);
        return true;
    });
}

void ReverbState::update(ContextBase const *const context, EffectSlotBase const *const slot,
    EffectProps const *const props, EffectTarget const target) noexcept NONBLOCKING
{
    auto &reverbprops = IGNORE_FUNCTION_EFFECTS(std::get<ReverbProps>(*props));
    const auto &device = context->mDevice;
    const auto frequency = static_cast<float>(device.mSampleRate);

    /* Keep the slot reference for per-block source-position queries. */
    mSlot = slot;

    /* If the HF limit parameter is flagged, calculate an appropriate limit
     * based on the air absorption parameter.
     */
    auto hfRatio = reverbprops.DecayHFRatio;
    if(reverbprops.DecayHFLimit && reverbprops.AirAbsorptionGainHF < 1.0f)
        hfRatio = CalcLimitedHfRatio(hfRatio, reverbprops.AirAbsorptionGainHF,
            reverbprops.DecayTime);

    /* Calculate the LF/HF decay times. */
    const auto lfDecayTime = std::clamp(reverbprops.DecayTime*reverbprops.DecayLFRatio,
        MinDecayTime, MaxDecayTime);
    const auto hfDecayTime = std::clamp(reverbprops.DecayTime*hfRatio, MinDecayTime, MaxDecayTime);

    /* Determine if a full update is required. */
    const auto fullUpdate = mPipelineState == DeviceClear ||
        /* Density is essentially a master control for the feedback delays, so
         * changes the offsets of many delay lines.
         */
        mParams.Density != reverbprops.Density ||
        /* Diffusion and decay times influences the decay rate (gain) of the
         * late reverb T60 filter.
         */
        mParams.Diffusion != reverbprops.Diffusion ||
        mParams.DecayTime != reverbprops.DecayTime ||
        mParams.HFDecayTime != hfDecayTime ||
        mParams.LFDecayTime != lfDecayTime ||
        /* Modulation time and depth both require fading the modulation delay. */
        mParams.ModulationTime != reverbprops.ModulationTime ||
        mParams.ModulationDepth != reverbprops.ModulationDepth ||
        /* HF/LF References control the weighting used to calculate the density
         * gain.
         */
        mParams.HFReference != reverbprops.HFReference ||
        mParams.LFReference != reverbprops.LFReference;
    if(fullUpdate)
    {
        mParams.Density = reverbprops.Density;
        mParams.Diffusion = reverbprops.Diffusion;
        mParams.DecayTime = reverbprops.DecayTime;
        mParams.HFDecayTime = hfDecayTime;
        mParams.LFDecayTime = lfDecayTime;
        mParams.ModulationTime = reverbprops.ModulationTime;
        mParams.ModulationDepth = reverbprops.ModulationDepth;
        mParams.HFReference = reverbprops.HFReference;
        mParams.LFReference = reverbprops.LFReference;

        mPipelineState = (mPipelineState != DeviceClear) ? StartFade : Normal;
        mCurrentPipeline = !mCurrentPipeline;

        auto &oldpipeline = mPipelines[!mCurrentPipeline];
        oldpipeline.mEarlyDelayCoeff[1] = 0.0f;
    }
    auto &pipeline = mPipelines[mCurrentPipeline];

    /* The density-based room size (delay length) multiplier. */
    const auto density_mult = CalcDelayLengthMult(reverbprops.Density);

    /* Update the main effect delay and associated taps. */
    pipeline.updateDelayLine(reverbprops.Gain, reverbprops.ReflectionsDelay,
        reverbprops.LateReverbDelay, density_mult, frequency);

    /* Update early and late 3D panning. When the image-source model is active
     * it replaces the legacy fixed-tap early output, so the legacy early
     * panning gains are muted (the legacy early stage still feeds the late
     * reverb input through its scatter path, preserving the tail response).
     */
    mOutTarget = target.Main->Buffer;
    const auto gain = slot->Gain * ReverbBoost;
    const auto earlyGain = mIsmCfg.mEnabled ? 0.0f : reverbprops.ReflectionsGain*gain;
    pipeline.update3DPanning(reverbprops.ReflectionsPan, reverbprops.LateReverbPan,
        earlyGain, reverbprops.LateReverbGain*gain, mUpmixOutput, target.Main);

    /* Cache the position-aware image-source inputs. The tap geometry itself is
     * regenerated every update block in process(), because the source position
     * moves without an effect property change; update() only runs when an
     * effect parameter changes. The reflection gain is applied to the ISM
     * output separately so it matches the legacy early level when enabled.
     */
    if(mIsmCfg.mEnabled)
    {
        pipeline.mIsmOutGain = reverbprops.ReflectionsGain*gain;
        mIsmDensityMult = density_mult;
        mIsmAirLpPerSec = CalcAirLpCoeff(reverbprops.AirAbsorptionGainHF, frequency);
        mIsmReflectionsDelay = reverbprops.ReflectionsDelay;
        mIsmMainMix = target.Main;
    }

    /* Calculate the master filters */
    auto const hf0norm = std::min(reverbprops.HFReference/frequency, 0.49f);
    auto const lf0norm = std::min(reverbprops.LFReference/frequency, 0.49f);
    pipeline.mFilter[0].Lp.setParamsFromSlope(BiquadType::HighShelf, hf0norm, reverbprops.GainHF,
        1.0f);
    pipeline.mFilter[0].Hp.setParamsFromSlope(BiquadType::LowShelf, lf0norm, reverbprops.GainLF,
        1.0f);
    std::ranges::for_each(pipeline.mFilter | std::views::drop(1),
        [&filter=pipeline.mFilter[0]](ReverbPipeline::FilterPair &targetpair) noexcept
    {
        targetpair.Lp.copyParamsFrom(filter.Lp);
        targetpair.Hp.copyParamsFrom(filter.Hp);
    });

    if(fullUpdate)
    {
        /* Update the early lines. */
        pipeline.mEarly.updateLines(density_mult, reverbprops.Diffusion, reverbprops.DecayTime,
            frequency);

        /* Get the mixing matrix coefficients. */
        CalcMatrixCoeffs(reverbprops.Diffusion, &pipeline.mMixX, &pipeline.mMixY);

        /* Update the modulator rate and depth. */
        pipeline.mLate.Mod.updateModulator(reverbprops.ModulationTime, reverbprops.ModulationDepth,
            frequency);

        /* Update the late lines. */
        pipeline.mLate.updateLines(density_mult, reverbprops.Diffusion, lfDecayTime,
            reverbprops.DecayTime, hfDecayTime, lf0norm, hf0norm, frequency);
    }

    /* Calculate the gain at the start of the late reverb stage, and the gain
     * difference from the decay target (0.001, or -60dB).
     */
    const auto decayBase = slot->Gain * reverbprops.Gain * reverbprops.LateReverbGain;
    const auto decayDiff = ReverbDecayGain / std::max(decayBase, ReverbDecayGain);

    /* Given the DecayTime (the amount of time for the late reverb to decay by
     * -60dB), calculate the time to decay to -60dB from the start of the late
     * reverb.
     *
     * Otherwise, if the late reverb already starts at -60dB or less, only
     * include the time to get to the late reverb.
     */
    const auto diffTime = !(decayDiff < 1.0f) ? 0.0f
        : (std::log10(decayDiff)*(20.0f / -60.0f) * reverbprops.DecayTime);

    const auto decaySamples = (reverbprops.ReflectionsDelay+reverbprops.LateReverbDelay+diffTime)
        * frequency;
    /* Limit to 100,000 samples (a touch over 2 seconds at 48khz) to avoid
     * excessive double-processing.
     */
    pipeline.mFadeSampleCount = static_cast<std::size_t>(std::min(decaySamples, 100'000.0f));
    mFadeSampleTotal = std::max(pipeline.mFadeSampleCount, std::size_t{1});
}


/**************************************
 *  Effect Processing                 *
 **************************************/

/* Applies a scattering matrix to the 4-line (vector) input.  This is used
 * for both the below vector all-pass model and to perform modal feed-back
 * delay network (FDN) mixing.
 *
 * The matrix is derived from a skew-symmetric matrix to form a 4D rotation
 * matrix with a single unitary rotational parameter:
 *
 *     [  d,  a,  b,  c ]          1 = a^2 + b^2 + c^2 + d^2
 *     [ -a,  d,  c, -b ]
 *     [ -b, -c,  d,  a ]
 *     [ -c,  b, -a,  d ]
 *
 * The rotation is constructed from the effect's diffusion parameter,
 * yielding:
 *
 *     1 = x^2 + 3 y^2
 *
 * Where a, b, and c are the coefficient y with differing signs, and d is the
 * coefficient x.  The final matrix is thus:
 *
 *     [  x,  y, -y,  y ]          n = sqrt(matrix_order - 1)
 *     [ -y,  x,  y,  y ]          t = diffusion_parameter * atan(n)
 *     [  y, -y,  x,  y ]          x = cos(t)
 *     [ -y, -y, -y,  x ]          y = sin(t) / n
 *
 * Any square orthogonal matrix with an order that is a power of two will
 * work (where ^T is transpose, ^-1 is inverse):
 *
 *     M^T = M^-1
 *
 * Using that knowledge, finding an appropriate matrix can be accomplished
 * naively by searching all combinations of:
 *
 *     M = D + S - S^T
 *
 * Where D is a diagonal matrix (of x), and S is a triangular matrix (of y)
 * whose combination of signs are being iterated.
 */
auto VectorPartialScatter(std::array<float, NUM_LINES> const &in, float const xCoeff,
    float const yCoeff) noexcept -> std::array<float, NUM_LINES>
{
    return std::array{
        xCoeff*in[0] + yCoeff*(          in[1] + -in[2] + in[3]),
        xCoeff*in[1] + yCoeff*(-in[0]          +  in[2] + in[3]),
        xCoeff*in[2] + yCoeff*( in[0] + -in[1]          + in[3]),
        xCoeff*in[3] + yCoeff*(-in[0] + -in[1] + -in[2]        )
    };
}

/* Applies the above to the given number of samples. */
void VectorScatter(float const xCoeff, float const yCoeff,
    std::span<ReverbUpdateLine, NUM_LINES> const samples, std::size_t const count) noexcept
{
    ASSUME(count > 0);

    for(auto i = 0_uz;i < count;++i)
    {
        auto src = std::array{samples[0][i], samples[1][i], samples[2][i], samples[3][i]};

        src = VectorPartialScatter(src, xCoeff, yCoeff);
        samples[0][i] = src[0];
        samples[1][i] = src[1];
        samples[2][i] = src[2];
        samples[3][i] = src[3];
    }
}

/* Same as the above, but also applies a line-based reflection on the input
 * channels (swapping 0<->3 and 1<->2).
 */
void VectorScatterRev(float const xCoeff, float const yCoeff,
    std::span<ReverbUpdateLine, NUM_LINES> const samples, std::size_t const count) noexcept
{
    ASSUME(count > 0);

    for(auto i = 0_uz;i < count;++i)
    {
        auto src = std::array{samples[0][i], samples[1][i], samples[2][i], samples[3][i]};

        src = VectorPartialScatter(std::array{src[3], src[2], src[1], src[0]}, xCoeff, yCoeff);
        samples[0][i] = src[0];
        samples[1][i] = src[1];
        samples[2][i] = src[2];
        samples[3][i] = src[3];
    }
}

/* This applies a Gerzon multiple-in/multiple-out (MIMO) vector all-pass
 * filter to the 4-line input.
 *
 * It works by vectorizing a regular all-pass filter and replacing the delay
 * element with a scattering matrix (like the one above) and a diagonal
 * matrix of delay elements.
 */
void VecAllpass::process(const std::span<ReverbUpdateLine,NUM_LINES> samples, std::size_t offset,
    float const xCoeff, float const yCoeff, std::size_t const todo) const noexcept
{
    const auto delaymask = std::size_t{Delay.mLine.size()/NUM_LINES - 1_uz};
    const auto delaybuf = Delay.mLine;
    const auto feedCoeff = Coeff;

    ASSUME(todo > 0);

    for(auto base = 0_uz;base < todo;)
    {
        auto vap_offset = std::array<std::size_t, NUM_LINES>{};
        std::ranges::transform(Offset, vap_offset.begin(),
            [offset,delaymask](std::size_t const delay) noexcept -> std::size_t
        { return (offset-delay) & delaymask; });
        offset &= delaymask;

        const auto maxoff = std::max(std::ranges::max(vap_offset), offset);
        /* Limit the number of samples to the minimum line delay, so that each
         * line's all-pass can be processed individually before scattering for
         * the feedback.
         */
        const auto td = std::min(Offset.front(), std::min(delaymask+1_uz - maxoff, todo - base));

        for(const auto c : std::views::iota(0_uz, NUM_LINES))
        {
            const auto sampleline = std::span{samples[c]}.subspan(base, td);
            auto out_offset = vap_offset[c];
            auto in_offset = offset;
            std::ranges::transform(sampleline, sampleline.begin(),
                [delaybuf,feedCoeff,c,&out_offset,&in_offset](float const input)
            {
                const auto out = delaybuf[(out_offset++)*NUM_LINES + c] - feedCoeff*input;
                delaybuf[(in_offset++)*NUM_LINES + c] = input + feedCoeff*out;
                return out;
            });
            vap_offset[c] = out_offset;
        }

        auto delayIn = delaybuf.begin();
        std::advance(delayIn, offset*NUM_LINES);
        for(const auto j [[maybe_unused]] : std::views::iota(0_uz, td))
        {
            const auto f = VectorPartialScatter({delayIn[0], delayIn[1], delayIn[2], delayIn[3]},
                xCoeff, yCoeff);
            delayIn = std::ranges::copy(f, delayIn).out;
        }

        offset += td;
        base += td;
    }
}

/* This applies a more typical all-pass to each line, without the scattering
 * matrix.
 */
void Allpass4::process(std::span<ReverbUpdateLine, NUM_LINES> const samples,
    std::size_t const offset, std::size_t const todo) const noexcept
{
    const auto delay = Delay;
    const auto feedCoeff = Coeff;

    ASSUME(todo > 0);

    for(const auto j : std::views::iota(0_uz, NUM_LINES))
    {
        auto lineio = std::span{samples[j]}.first(todo);
        const auto delaybuf = delay.get(j);
        auto dstoffset = offset;
        auto vap_offset = offset - Offset[j];
        while(!lineio.empty())
        {
            vap_offset &= delaybuf.size()-1;
            dstoffset &= delaybuf.size()-1;

            const auto maxoff = std::max(dstoffset, vap_offset);
            const auto td = std::min(lineio.size(), delaybuf.size()-maxoff);

            std::ranges::transform(lineio.first(td), lineio.begin(),
                [delaybuf,feedCoeff,&vap_offset,&dstoffset](float const x) -> float
            {
                const auto y = delaybuf[vap_offset++] - feedCoeff*x;
                delaybuf[dstoffset++] = x + feedCoeff*y;
                return y;
            });
            lineio = lineio.subspan(td);
        }
    }
}


/* This generates early reflections.
 *
 * This is done by obtaining the primary reflections (those arriving from the
 * same direction as the source) from the main delay line.  These are
 * attenuated and all-pass filtered (based on the diffusion parameter).
 *
 * The early lines are then reflected about the origin to create the secondary
 * reflections (those arriving from the opposite direction as the source).
 *
 * The early response is then completed by combining the primary reflections
 * with the delayed and attenuated output from the early lines.
 *
 * Finally, the early response is reflected, scattered (based on diffusion),
 * and fed into the late reverb section of the main delay line.
 */
void ReverbPipeline::processEarly(DelayLineU const &main_delay, std::size_t offset,
    std::size_t const samplesToDo, std::span<ReverbUpdateLine, NUM_LINES> const tempSamples,
    std::span<FloatBufferLine, NUM_LINES> const outSamples)
{
    const auto early_delay = mEarly.Delay;
    const auto in_delay = main_delay;
    const auto mixX = mMixX;
    const auto mixY = mMixY;

    ASSUME(samplesToDo <= BufferLineSize);

    for(auto base = 0_uz;base < samplesToDo;)
    {
        const auto todo = std::min(samplesToDo-base, MAX_UPDATE_SAMPLES);

        /* First, load decorrelated samples from the main delay line as the
         * primary reflections, applying the main room/hf/lf gains.
         */
        const auto fadeStep = 1.0f / static_cast<float>(todo);
        const auto earlycoeff0 = mEarlyDelayCoeff[0];
        const auto earlycoeff1 = mEarlyDelayCoeff[1];
        mEarlyDelayCoeff[0] = mEarlyDelayCoeff[1];
        for(const auto j : std::views::iota(0_uz, NUM_LINES))
        {
            const auto input = in_delay.get(j);
            auto early_delay_tap0 = offset - mEarlyDelayTap[j][0];
            auto early_delay_tap1 = offset - mEarlyDelayTap[j][1];
            mEarlyDelayTap[j][0] = mEarlyDelayTap[j][1];
            auto fadeCount = 0.0f;

            auto tmp = tempSamples[j].begin();
            for(auto i = 0_uz;i < todo;)
            {
                early_delay_tap0 &= input.size()-1;
                early_delay_tap1 &= input.size()-1;
                const auto max_tap = std::max(early_delay_tap0, early_delay_tap1);
                const auto td = std::min(input.size()-max_tap, todo-i);

                tmp = std::ranges::transform(input.subspan(early_delay_tap0, td),
                    input.subspan(early_delay_tap1, td), tmp,
                    [earlycoeff0,earlycoeff1,fadeStep,&fadeCount](float const in0, float const in1)
                        noexcept -> float
                {
                    auto const ret = al::lerp(in0*earlycoeff0, in1*earlycoeff1,
                        fadeStep*fadeCount);
                    fadeCount += 1.0f;
                    return ret;
                }).out;
                early_delay_tap0 += td;
                early_delay_tap1 += td;
                i += td;
            }

            /* Band-pass the incoming samples. */
            mFilter[j].process(std::span{tempSamples[j]}.first(todo), tempSamples[j]);
        }

        /* Apply an all-pass, to help color the initial reflections. */
        mEarly.Allpass.process(tempSamples, offset, todo);

        /* Apply a reflection and delay to generate secondary reflections.*/
        early_delay.writeReflected(offset, tempSamples, todo);
        const auto delay_coeff = mEarly.Coeff;
        for(const auto j : std::views::iota(0_uz, NUM_LINES))
        {
            const auto delaybuf = early_delay.get(j);
            auto delay_tap = std::size_t{offset - mEarly.Offset[j]};
            auto out = outSamples[j].begin() + base;
            auto tmp = std::span{tempSamples[j]}.first(todo);
            while(!tmp.empty())
            {
                delay_tap &= delaybuf.size()-1;

                const auto delaySrc = delaybuf | std::views::drop(delay_tap)
                    | std::views::take(tmp.size());

                /* Combine the main input with the attenuated delayed echo for
                 * the early output.
                 */
                out = std::ranges::transform(delaySrc, tmp, out,
                    [delay_coeff](float const delayspl, float const mainspl) noexcept -> float
                { return delayspl*delay_coeff + mainspl; }).out;

                tmp = tmp.subspan(delaySrc.size());
                delay_tap += delaySrc.size();
            }
        }

        /* Finally, apply a scatter to the main input to improve the initial
         * diffusion in the late reverb, writing the result to the late delay
         * line input.
         */
        VectorScatter(mixX, mixY, tempSamples, todo);
        std::ignore = std::ranges::mismatch(std::views::iota(0_uz, NUM_LINES), tempSamples,
            [late_in=mLateDelayIn,offset,todo](std::size_t const idx, ReverbUpdateSpan const tmpline)
        {
            late_in.write(offset, idx, tmpline.first(todo));
            return true;
        });

        base += todo;
        offset += todo;
    }
}

auto Modulation::calcDelays(std::size_t const todo) -> std::span<unsigned const>
{
    auto idx = Index;
    const auto step = Step;
    const auto depth = Depth * float{gCubicTable.sTableSteps};
    const auto delays = std::span{ModDelays}.first(todo);
    std::ranges::generate(delays, [step,depth,&idx]
    {
        const auto x = static_cast<float>(idx&MOD_FRACMASK) * (1.0f/MOD_FRACONE);
        /* Approximate sin(x*2pi). As long as it roughly fits a sinusoid shape
         * and stays within [-1...+1], it needn't be perfect.
         */
        const auto lfo = !(idx&(MOD_FRACONE>>1))
            ? ((-16.0f * x * x) + (8.0f * x))
            : ((16.0f * x * x) + (-8.0f * x) + (-16.0f * x) + 8.0f);
        idx += step;
        return al::saturating_cast<unsigned>((lfo+1.0f) * depth);
    });
    Index = idx;
    return delays;
}


/* This generates the reverb tail using a modified feed-back delay network
 * (FDN).
 *
 * Results from the early reflections are mixed with the output from the
 * modulated late delay lines.
 *
 * The late response is then completed by T60 and all-pass filtering the mix.
 *
 * Finally, the lines are reversed (so they feed their opposite directions)
 * and scattered with the FDN matrix before re-feeding the delay lines.
 */
void ReverbPipeline::processLate(std::size_t offset, std::size_t const samplesToDo,
    std::span<ReverbUpdateLine, NUM_LINES> const tempSamples,
    std::span<FloatBufferLine, NUM_LINES> const outSamples)
{
    const auto late_delay = mLate.Delay;
    const auto in_delay = mLateDelayIn;
    const auto mixX = mMixX;
    const auto mixY = mMixY;

    ASSUME(samplesToDo <= BufferLineSize);

    for(auto base = 0_uz;base < samplesToDo;)
    {
        const auto todo = std::min(std::min(mLate.Offset[0],MAX_UPDATE_SAMPLES), samplesToDo-base);
        ASSUME(todo > 0);

        /* First, calculate the modulated delays for the late feedback. */
        const auto delays = mLate.Mod.calcDelays(todo);

        /* Now load samples from the feedback delay lines. Filter the signal to
         * apply its frequency-dependent decay.
         */
        for(const auto j : std::views::iota(0_uz, NUM_LINES))
        {
            const auto input = late_delay.get(j);
            const auto midGain = mLate.T60[j].mMidGain;
            auto late_feedb_tap = std::size_t{offset - mLate.Offset[j]};

            std::ranges::transform(delays, tempSamples[j].begin(),
                [input,midGain,&late_feedb_tap](std::size_t const idelay) -> float
            {
                /* Calculate the read sample offset and sub-sample offset
                 * between it and the next sample.
                 */
                const auto delay = late_feedb_tap - (idelay>>gCubicTable.sTableBits);
                const auto delayoffset = std::size_t{idelay & gCubicTable.sTableMask};
                ++late_feedb_tap;

                /* Get the samples around the delayed offset, interpolated for
                 * output.
                 */
                const auto out0 = input[(delay  ) & (input.size()-1)];
                const auto out1 = input[(delay-1) & (input.size()-1)];
                const auto out2 = input[(delay-2) & (input.size()-1)];
                const auto out3 = input[(delay-3) & (input.size()-1)];

                const auto out = out0*gCubicTable.getCoeff0(delayoffset)
                    + out1*gCubicTable.getCoeff1(delayoffset)
                    + out2*gCubicTable.getCoeff2(delayoffset)
                    + out3*gCubicTable.getCoeff3(delayoffset);
                return out * midGain;
            });

            mLate.T60[j].process(std::span{tempSamples[j]}.first(todo));
        }

        /* Next load decorrelated samples from the main delay lines. */
        const auto fadeStep = 1.0f / static_cast<float>(todo);
        for(const auto j : std::views::iota(0_uz, NUM_LINES))
        {
            const auto input = in_delay.get(j);
            auto late_delay_tap0 = std::size_t{offset - mLateDelayTap[j][0]};
            auto late_delay_tap1 = std::size_t{offset - mLateDelayTap[j][1]};
            mLateDelayTap[j][0] = mLateDelayTap[j][1];
            const auto densityGain = mLate.DensityGain;
            const auto densityStep = late_delay_tap0 != late_delay_tap1
                ? densityGain*fadeStep : 0.0f;
            auto fadeCount = 0.0f;

            auto samples = std::span{tempSamples[j]}.first(todo);
            while(!samples.empty())
            {
                late_delay_tap0 &= input.size()-1;
                late_delay_tap1 &= input.size()-1;
                const auto max_delay_tap = std::max(late_delay_tap0, late_delay_tap1);
                const auto td = std::min(samples.size(), input.size()-max_delay_tap);

                std::ranges::transform(samples.first(td), samples.begin(),
                    [input,densityGain,densityStep,&late_delay_tap0,&late_delay_tap1,&fadeCount]
                    (float const sample) noexcept -> float
                {
                    const auto fade0 = densityGain - densityStep*fadeCount;
                    const auto fade1 = densityStep*fadeCount;
                    fadeCount += 1.0f;
                    return input[late_delay_tap0++]*fade0 + input[late_delay_tap1++]*fade1
                        + sample;
                });
                samples = samples.subspan(td);
            }
        }

        /* Apply a vector all-pass to improve micro-surface diffusion, and
         * write out the results for mixing.
         */
        mLate.VecAp.process(tempSamples, offset, mixX, mixY, todo);
        std::ignore = std::ranges::mismatch(tempSamples, outSamples,
            [base,todo](ReverbUpdateLine &tmpline, FloatBufferSpan const outline)
        {
            std::ranges::copy(tmpline | std::views::take(todo),
                (outline | std::views::drop(base)).begin());
            return true;
        });

        /* Finally, scatter and bounce the results to refeed the feedback buffer. */
        VectorScatterRev(mixX, mixY, tempSamples, todo);
        std::ignore = std::ranges::mismatch(std::views::iota(0_uz, NUM_LINES), tempSamples,
            [late_delay,offset,todo](std::size_t const idx, ReverbUpdateLine const &tmpline)
        {
            late_delay.write(offset, idx, std::span{tmpline}.first(todo));
            return true;
        });

        base += todo;
        offset += todo;
    }
}

void ReverbState::process(std::size_t const samplesToDo,
    std::span<FloatBufferLine const> const samplesIn, std::span<FloatBufferLine> const samplesOut)
    noexcept NONBLOCKING
{
    const auto offset = mOffset;

    ASSUME(samplesToDo <= BufferLineSize);

    auto &oldpipeline = mPipelines[!mCurrentPipeline];
    auto &pipeline = mPipelines[mCurrentPipeline];

    /* Regenerate the position-aware image taps every update block. This runs
     * here rather than in update() because source/listener movement updates the
     * voice parameters without an effect property change, and the reflection
     * pattern must track that movement in real time. The active pipeline is
     * refreshed before its delay line is read below. During a cross-fade the
     * old pipeline keeps the tap geometry it faded in with and only ramps out.
     */
    if(mIsmCfg.mEnabled && (mPipelineState == Normal || mPipelineState == DeviceClear
        || mPipelineState == StartFade))
        updateImageSource(pipeline);

    /* Convert B-Format to A-Format for processing. */
    const auto numInput = std::min(samplesIn.size(), NUM_LINES);
    for(const auto c : std::views::iota(0_uz, NUM_LINES))
    {
        const auto tmpspan = std::span{std::assume_aligned<16>(mTempLine.data()), samplesToDo};
        std::ranges::fill(tmpspan, 0.0f);
        for(const auto i : std::views::iota(0_uz, numInput))
        {
            std::ranges::transform(tmpspan, samplesIn[i], tmpspan.begin(),
                [gain=B2A[c][i]](float const sample, float const in) noexcept -> float
            { return sample + in*gain; });
        }

        mMainDelay.write(offset, c, tmpspan);
    }

    /* Feed the image-source delay lines with the omnidirectional (W) wet
     * input, which is the B-Format W (ACN 0) channel of the slot's wet
     * buffer. Both pipelines are fed so a cross-fade keeps the position-aware
     * response continuous across parameter changes.
     */
    if(mIsmCfg.mEnabled && !samplesIn.empty())
    {
        const auto wIn = std::span{samplesIn[0]}.first(samplesToDo);
        pipeline.mIsm.Delay.write(offset, 0, wIn);
        if(mPipelineState != Normal)
            oldpipeline.mIsm.Delay.write(offset, 0, wIn);
    }

    mPipelineState = std::max(Fading, mPipelineState);

    /* Process reverb for these samples. and mix them to the output. */
    pipeline.processEarly(mMainDelay, offset, samplesToDo, mTempSamples, mEarlySamples);
    pipeline.processLate(offset, samplesToDo, mTempSamples, mLateSamples);
    if(mIsmCfg.mEnabled)
    {
        processImageSource(pipeline, offset, samplesToDo);
        mixImageSource(pipeline, samplesOut, samplesToDo);
    }
    mixOut(pipeline, samplesOut, samplesToDo);

    if(mPipelineState != Normal)
    {
        if(mPipelineState == Cleanup)
        {
            oldpipeline.clear();
            mPipelineState = Normal;
        }
        else
        {
            /* If this is the final mix for this old pipeline, set the target
             * gains to 0 to ensure a complete fade out, and set the state to
             * Cleanup so the next invocation cleans up the delay buffers and
             * filters.
             */
            if(samplesToDo >= oldpipeline.mFadeSampleCount)
            {
                std::ranges::fill(oldpipeline.mEarly.Gains
                    | std::views::transform(&EarlyReflections::OutGains::Target)
                    | std::views::join, 0.0f);
                std::ranges::fill(oldpipeline.mLate.Gains
                    | std::views::transform(&LateReverb::OutGains::Target)
                    | std::views::join, 0.0f);
                oldpipeline.mFadeSampleCount = 0;
                mPipelineState = Cleanup;
            }
            else
                oldpipeline.mFadeSampleCount -= samplesToDo;

            /* Process the old reverb for these samples. */
            oldpipeline.processEarly(mMainDelay, offset, samplesToDo, mTempSamples, mEarlySamples);
            oldpipeline.processLate(offset, samplesToDo, mTempSamples, mLateSamples);
            if(mIsmCfg.mEnabled)
            {
                /* Fade the old pipeline's image-source output out alongside
                 * its legacy early/late gains. Its tap gains stay fixed while
                 * the overall ISM output gain ramps to zero over the fade.
                 */
                const auto fadeGain = oldpipeline.mIsmOutGain
                    * (static_cast<float>(oldpipeline.mFadeSampleCount)
                       / static_cast<float>(mFadeSampleTotal));
                processImageSource(oldpipeline, offset, samplesToDo);
                const auto saved = oldpipeline.mIsmOutGain;
                oldpipeline.mIsmOutGain = fadeGain;
                mixImageSource(oldpipeline, samplesOut, samplesToDo);
                oldpipeline.mIsmOutGain = saved;
            }
            mixOut(oldpipeline, samplesOut, samplesToDo);
        }
    }

    mOffset = offset + samplesToDo;
}


struct ReverbStateFactory final : EffectStateFactory {
    al::intrusive_ptr<EffectState> create() override
    { return al::intrusive_ptr<EffectState>{new ReverbState{}}; }
};

} // namespace

auto ReverbStateFactory_getFactory() -> EffectStateFactory&
{
    static auto ReverbFactory = ReverbStateFactory{};
    return ReverbFactory;
}
