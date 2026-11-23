/*
 * Empirical verification of EFX direct-filter vs send-filter routing.
 *
 * Renders a multi-tone source through a NULL-effect aux slot via the
 * ALC_SOFT_loopback device and compares the wet-only output (full render
 * minus dry-only render) between configurations:
 *
 *   A: source direct filter = strong low-pass (GainHF=0.01), send filter = none
 *   B: source direct filter = none,                    send filter = none
 *   C: source direct filter = strong low-pass,         send filter = strong LP
 *   D: source direct filter = none,                    send filter = strong LP
 *
 * If the send/wet path were cascaded after the source's direct filter,
 * the wet signal in A would be low-passed and differ from B. If the paths
 * are independent parallel branches (EFX-correct), wet A == wet B.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>
#include <AL/efx.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr ALCsizei SR = 48000;
constexpr ALCsizei NumSamples = SR; /* 1 second */

struct RenderResult {
    std::vector<float> frames; /* interleaved stereo */
};

void check_al(const char* where)
{
    ALenum e = alGetError();
    if(e != AL_NO_ERROR)
    {
        std::fprintf(stderr, "OpenAL error after %s: 0x%x %s\n", where, e,
            alGetString(e));
        std::exit(1);
    }
}

/* 1 s mono buffer: equal mix of 500 Hz and 8 kHz tones so a high-shelf
 * low-pass at 5 kHz produces a clear level difference at 8 kHz.
 */
ALuint make_tone_buffer()
{
    std::vector<ALshort> data(SR);
    for(ALCsizei i = 0; i < SR; ++i)
    {
        double const t = static_cast<double>(i) / SR;
        double const s = 0.5 * (std::sin(2.0*M_PI*500.0*t)
                              + std::sin(2.0*M_PI*8000.0*t));
        data[i] = static_cast<ALshort>(s * 0.5 * 32767.0);
    }
    ALuint buf = 0;
    alGenBuffers(1, &buf);
    alBufferData(buf, AL_FORMAT_MONO16, data.data(),
        static_cast<ALsizei>(data.size() * sizeof(ALshort)), SR);
    check_al("alBufferData");
    return buf;
}

ALuint make_lowpass(float gainhf)
{
    ALuint f = 0;
    alGenFilters(1, &f);
    alFilteri(f, AL_FILTER_TYPE, AL_FILTER_LOWPASS);
    check_al("alFilteri type");
    alFilterf(f, AL_LOWPASS_GAIN, 1.0f);
    alFilterf(f, AL_LOWPASS_GAINHF, gainhf);
    check_al("alFilterf gainhf");
    return f;
}

struct Config {
    bool direct_filter;  /* strong low-pass on AL_DIRECT_FILTER */
    bool send_filter;    /* strong low-pass on AL_AUXILIARY_SEND_FILTER */
    bool send_attached;  /* route to aux slot */
};

RenderResult render(ALCdevice* dev, ALCint const* attrs, Config const& cfg)
{
    ALCcontext* ctx = alcCreateContext(dev, attrs);
    if(!ctx || !alcMakeContextCurrent(ctx))
    {
        std::fprintf(stderr, "context creation failed: 0x%x\n",
            (unsigned)alcGetError(dev));
        std::exit(1);
    }

    ALuint const buffer = make_tone_buffer();

    ALuint slot = 0, effect = 0;
    if(cfg.send_attached)
    {
        /* Use a reverb configured for near-transparent passthrough:
         * full-band gain, no diffusion/density, shortest room/decay. The
         * wet signal then closely tracks the send input. AL_EFFECT_NULL
         * can't be used because its process() outputs silence by design.
         */
        alGenEffects(1, &effect);
        alEffecti(effect, AL_EFFECT_TYPE, AL_EFFECT_REVERB);
        check_al("alEffecti reverb");
        alEffectf(effect, AL_REVERB_GAIN, 1.0f);
        alEffectf(effect, AL_REVERB_GAINHF, 1.0f);
        alEffectf(effect, AL_REVERB_DENSITY, 0.0f);
        alEffectf(effect, AL_REVERB_DIFFUSION, 0.0f);
        alEffectf(effect, AL_REVERB_DECAY_TIME, AL_REVERB_MIN_DECAY_TIME);
        check_al("reverb params");

        alGenAuxiliaryEffectSlots(1, &slot);
        alAuxiliaryEffectSloti(slot, AL_EFFECTSLOT_EFFECT,
            static_cast<ALint>(effect));
        check_al("effect slot attach");
        alAuxiliaryEffectSlotf(slot, AL_EFFECTSLOT_GAIN, 1.0f);
    }

    ALuint src = 0;
    alGenSources(1, &src);
    alSourcei(src, AL_BUFFER, static_cast<ALint>(buffer));
    alSource3f(src, AL_POSITION, 0.0f, 0.0f, 0.0f);
    alSourcef(src, AL_GAIN, 1.0f);
    check_al("source setup");

    if(cfg.direct_filter)
    {
        ALuint lp = make_lowpass(0.01f);
        alSourcei(src, AL_DIRECT_FILTER, static_cast<ALint>(lp));
        check_al("AL_DIRECT_FILTER");
    }
    if(cfg.send_attached)
    {
        if(cfg.send_filter)
        {
            ALuint slp = make_lowpass(0.01f);
            alSource3i(src, AL_AUXILIARY_SEND_FILTER, static_cast<ALint>(slot), 0,
                static_cast<ALint>(slp));
        }
        else
            alSource3i(src, AL_AUXILIARY_SEND_FILTER, static_cast<ALint>(slot), 0,
                AL_FILTER_NULL);
        check_al("AL_AUXILIARY_SEND_FILTER");
    }

    alSourcePlay(src);

    RenderResult res;
    res.frames.resize(static_cast<size_t>(NumSamples) * 2);
    alcRenderSamplesSOFT(dev, res.frames.data(), NumSamples);

    ALint state = 0;
    alGetSourcei(src, AL_SOURCE_STATE, &state);
    if(state != AL_PLAYING && state != AL_STOPPED)
        std::fprintf(stderr, "warning: unexpected source state %d\n", state);

    alcMakeContextCurrent(nullptr);
    alcDestroyContext(ctx);
    return res;
}

/* Coherent tone level via quadrature demodulation (amplitude estimate). */
double tone_level(std::vector<float> const& frames, ALCsizei from, ALCsizei to,
    double freq)
{
    double acc_i = 0.0, acc_q = 0.0;
    size_t n = 0;
    for(ALCsizei i = from; i < to; ++i)
    {
        double const t = static_cast<double>(i) / SR;
        double const m = (frames[static_cast<size_t>(i)*2]
                        + frames[static_cast<size_t>(i)*2 + 1]) * 0.5;
        acc_i += m * std::cos(2.0*M_PI*freq*t);
        acc_q += m * std::sin(2.0*M_PI*freq*t);
        ++n;
    }
    acc_i /= static_cast<double>(n);
    acc_q /= static_cast<double>(n);
    return 2.0 * std::sqrt(acc_i*acc_i + acc_q*acc_q);
}

double rms(std::vector<float> const& frames, ALCsizei from, ALCsizei to)
{
    double acc = 0.0;
    size_t n = 0;
    for(ALCsizei i = from; i < to; ++i)
    {
        double const m = (frames[static_cast<size_t>(i)*2]
                        + frames[static_cast<size_t>(i)*2 + 1]) * 0.5;
        acc += m * m;
        ++n;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

} // namespace

int main()
{
    if(!alcIsExtensionPresent(nullptr, "ALC_SOFT_loopback"))
    {
        std::fprintf(stderr, "ALC_SOFT_loopback missing\n");
        return 1;
    }

    ALCint attrs[] = {
        ALC_FORMAT_CHANNELS_SOFT, ALC_STEREO_SOFT,
        ALC_FORMAT_TYPE_SOFT, ALC_FLOAT_SOFT,
        ALC_FREQUENCY, SR,
        0
    };
    ALCdevice* dev = alcLoopbackOpenDeviceSOFT(nullptr);
    if(!dev) { std::fprintf(stderr, "loopback open failed\n"); return 1; }
    if(!alcIsRenderFormatSupportedSOFT(dev, SR, ALC_STEREO_SOFT, ALC_FLOAT_SOFT))
    {
        std::fprintf(stderr, "float stereo loopback unsupported\n");
        return 1;
    }
    if(!alcResetDeviceSOFT(dev, attrs))
    {
        std::fprintf(stderr, "loopback reset failed\n");
        return 1;
    }
    {
        ALCcontext* probe = alcCreateContext(dev, attrs);
        alcMakeContextCurrent(probe);
        if(!alcIsExtensionPresent(dev, "ALC_EXT_EFX"))
        {
            std::fprintf(stderr, "ALC_EXT_EFX missing\n");
            return 1;
        }
        alcMakeContextCurrent(nullptr);
        alcDestroyContext(probe);
    }

    /* Each render uses its own context; wet = full - dry with identical
     * filter setup apart from the absent send in the dry render.
     */
    RenderResult rA_full = render(dev, attrs, {true,  false, true});
    RenderResult rA_dry  = render(dev, attrs, {true,  false, false});
    RenderResult rB_full = render(dev, attrs, {false, false, true});
    RenderResult rB_dry  = render(dev, attrs, {false, false, false});
    RenderResult rC_full = render(dev, attrs, {true,  true,  true});
    RenderResult rC_dry  = render(dev, attrs, {true,  false, false});
    RenderResult rD_full = render(dev, attrs, {false, true,  true});
    RenderResult rD_dry  = render(dev, attrs, {false, false, false});

    alcCloseDevice(dev);

    constexpr ALCsizei F = SR/4, T = SR*3/4;

    auto wet_of = [](RenderResult full, RenderResult const& dry) {
        for(size_t i = 0; i < full.frames.size(); ++i)
            full.frames[i] -= dry.frames[i];
        return full;
    };
    RenderResult wetA = wet_of(rA_full, rA_dry); /* direct LP, no send LP */
    RenderResult wetB = wet_of(rB_full, rB_dry); /* no filters           */
    RenderResult wetC = wet_of(rC_full, rC_dry); /* direct LP + send LP  */
    RenderResult wetD = wet_of(rD_full, rD_dry); /* send LP only         */

    std::printf("Analysis window: samples %d..%d (steady state)\n\n",
        static_cast<int>(F), static_cast<int>(T));
    std::printf("%-34s %10s %12s %12s\n", "render", "rms", "500Hz", "8kHz");
    auto report = [&](char const* name, std::vector<float> const& f) {
        std::printf("%-34s %10.6f %12.6f %12.6f\n", name, rms(f, F, T),
            tone_level(f, F, T, 500.0), tone_level(f, F, T, 8000.0));
    };
    report("wet A (direct LP only)", wetA.frames);
    report("wet B (no filters)", wetB.frames);
    report("wet C (direct LP + send LP)", wetC.frames);
    report("wet D (send LP only)", wetD.frames);
    report("dry A (direct LP only)", rA_dry.frames);
    report("dry B (no filter)", rB_dry.frames);

    double const wetA_8k = tone_level(wetA.frames, F, T, 8000.0);
    double const wetB_8k = tone_level(wetB.frames, F, T, 8000.0);
    double const wetD_8k = tone_level(wetD.frames, F, T, 8000.0);
    double const wetC_8k = tone_level(wetC.frames, F, T, 8000.0);
    double const dryA_8k = tone_level(rA_dry.frames, F, T, 8000.0);
    double const dryB_8k = tone_level(rB_dry.frames, F, T, 8000.0);

    std::printf("\n--- 8 kHz tone level ratios (dB) ---\n");
    std::printf("wet  A/B (direct LP must NOT alter wet)  = %+8.3f dB\n",
        20.0 * std::log10((wetA_8k + 1e-12) / (wetB_8k + 1e-12)));
    std::printf("wet  D/B (send LP must attenuate wet)    = %+8.3f dB\n",
        20.0 * std::log10((wetD_8k + 1e-12) / (wetB_8k + 1e-12)));
    std::printf("wet  C/A (adding send LP)                = %+8.3f dB\n",
        20.0 * std::log10((wetC_8k + 1e-12) / (wetA_8k + 1e-12)));
    std::printf("dry  A/B (direct LP must attenuate dry)  = %+8.3f dB\n",
        20.0 * std::log10((dryA_8k + 1e-12) / (dryB_8k + 1e-12)));

    bool const wet_independent = std::abs(wetA_8k - wetB_8k) < 0.02 * wetB_8k;
    bool const send_works = wetD_8k < 0.1 * wetB_8k;
    bool const direct_works = dryA_8k < 0.1 * dryB_8k;

    std::printf("\n--- Verdict ---\n");
    std::printf("direct filter attenuates dry path:       %s\n",
        direct_works ? "PASS" : "FAIL");
    std::printf("send filter attenuates wet path:         %s\n",
        send_works ? "PASS" : "FAIL");
    std::printf("wet unaffected by direct filter (A~=B):  %s\n",
        wet_independent ? "PASS (parallel, independent paths)"
                        : "FAIL (direct filter leaks into wet path)");

    return (wet_independent && send_works && direct_works) ? 0 : 2;
}
