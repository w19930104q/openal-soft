/*
 * Offline loopback test for the image-source method (ISM) early reflections.
 *
 * Renders the wet reverb impulse response (IR) for several fixed source
 * positions and for a block-by-block moving source, using ALC_SOFT_loopback
 * so no audio hardware is required. Float samples are written to binary
 * files for offline analysis:
 *
 *   <prefix>_ir_pos<k>.f32   per fixed-position wet IR (interleaved stereo)
 *   <prefix>_moving.f32      moving-source wet render (interleaved stereo)
 *   <prefix>_meta.txt        positions and render parameters
 *
 * The ISM behavior is controlled by the alsoft.conf given via ALSOFT_CONF
 * ([reverb] image-source = true/false).
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef AL_ALEXT_PROTOTYPES
#define AL_ALEXT_PROTOTYPES
#endif

#include "AL/al.h"
#include "AL/alc.h"
#include "AL/alext.h"
#include "AL/efx.h"

#define SR 48000
#define NCH 2
#define IMPULSE_SAMPLES 1024
#define PERIODIC_IMPULSE_SAMPLES 2048

/* Source positions (listener stays at the origin). Spread across the room so
 * the first-order image patterns differ clearly between positions.
 */
static const ALfloat kPositions[][3] = {
    { 1.0f, 0.0f,  1.0f},
    { 3.0f, 0.0f,  1.0f},
    { 3.0f, 0.0f, -1.0f},
    {-2.5f, 0.0f,  1.0f},
};
static const int kNumPositions
    = (int)(sizeof(kPositions)/sizeof(kPositions[0]));

static void check_al(const char *where)
{
    ALenum err = alGetError();
    if(err != AL_NO_ERROR)
    {
        fprintf(stderr, "AL error after %s: 0x%x %s\n", where, err,
            alGetString(err));
        exit(1);
    }
}

static ALCdevice *open_loopback(ALCcontext **out_ctx)
{
    if(!alcIsExtensionPresent(NULL, "ALC_SOFT_loopback"))
    {
        fprintf(stderr, "ALC_SOFT_loopback unavailable\n");
        exit(1);
    }
    ALCdevice *dev = alcLoopbackOpenDeviceSOFT(NULL);
    if(!dev){ fprintf(stderr, "loopback open failed\n"); exit(1); }

    ALCint attrs[] = {
        ALC_FORMAT_CHANNELS_SOFT, ALC_STEREO_SOFT,
        ALC_FORMAT_TYPE_SOFT, AL_FLOAT_SOFT,
        ALC_FREQUENCY, SR,
        0
    };
    if(alcIsRenderFormatSupportedSOFT(dev, SR, ALC_STEREO_SOFT, AL_FLOAT_SOFT)
        == AL_FALSE)
    {
        fprintf(stderr, "float stereo loopback not supported\n");
        exit(1);
    }
    ALCcontext *ctx = alcCreateContext(dev, attrs);
    if(!ctx || alcMakeContextCurrent(ctx) == ALC_FALSE)
    {
        fprintf(stderr, "context create failed\n");
        exit(1);
    }
    *out_ctx = ctx;
    return dev;
}

static ALuint make_impulse_buffer(void)
{
    ALfloat data[IMPULSE_SAMPLES];
    memset(data, 0, sizeof(data));
    data[0] = 1.0f;
    ALuint buf = 0;
    alGenBuffers(1, &buf);
    alBufferData(buf, AL_FORMAT_MONO_FLOAT32, data, sizeof(data), SR);
    check_al("buffer data");
    return buf;
}

static ALuint make_periodic_impulse_buffer(int total_frames)
{
    ALfloat *data = (ALfloat*)calloc((size_t)total_frames, sizeof(ALfloat));
    for(int i = 0;i < total_frames;i += PERIODIC_IMPULSE_SAMPLES)
        data[i] = 1.0f;
    ALuint buf = 0;
    alGenBuffers(1, &buf);
    alBufferData(buf, AL_FORMAT_MONO_FLOAT32, data,
        (ALsizei)(total_frames*sizeof(float)), SR);
    check_al("periodic buffer");
    free(data);
    return buf;
}

static ALuint setup_reverb_source(ALuint buf, ALuint *out_slot)
{
    ALuint effect = 0, slot = 0, source = 0;
    alGenEffects(1, &effect);
    alEffecti(effect, AL_EFFECT_TYPE, AL_EFFECT_REVERB);
    check_al("effect type");

    /* Bright, moderately live room; reflections emphasized over the tail. */
    alEffectf(effect, AL_REVERB_DENSITY, 1.0f);
    alEffectf(effect, AL_REVERB_DIFFUSION, 0.6f);
    alEffectf(effect, AL_REVERB_GAIN, 0.5f);
    alEffectf(effect, AL_REVERB_DECAY_TIME, 0.6f);
    alEffectf(effect, AL_REVERB_REFLECTIONS_GAIN, 1.0f);
    alEffectf(effect, AL_REVERB_REFLECTIONS_DELAY, 0.0f);
    alEffectf(effect, AL_REVERB_LATE_REVERB_GAIN, 0.4f);
    alEffectf(effect, AL_REVERB_LATE_REVERB_DELAY, 0.02f);
    alEffectf(effect, AL_REVERB_AIR_ABSORPTION_GAINHF, 0.995f);
    check_al("effect params");

    alGenAuxiliaryEffectSlots(1, &slot);
    alAuxiliaryEffectSloti(slot, AL_EFFECTSLOT_EFFECT, (ALint)effect);
    check_al("slot effect");

    alGenSources(1, &source);
    alSourcei(source, AL_BUFFER, (ALint)buf);
    /* Mute the dry path so the render is the pure reverb IR. */
    ALuint dryfilter = 0;
    alGenFilters(1, &dryfilter);
    alFilteri(dryfilter, AL_FILTER_TYPE, AL_FILTER_LOWPASS);
    alFilterf(dryfilter, AL_LOWPASS_GAIN, 0.0f);
    check_al("dry filter");
    alSourcef(source, AL_REFERENCE_DISTANCE, 1.0f);
    alSourcef(source, AL_ROLLOFF_FACTOR, 0.0f);
    alSourcei(source, AL_DIRECT_FILTER, (ALint)dryfilter);
    alSource3i(source, AL_AUXILIARY_SEND_FILTER, (ALint)slot, 0,
        AL_FILTER_NULL);
    check_al("source setup");

    *out_slot = slot;
    return source;
}

static void write_f32(const char *path, const float *data, size_t frames)
{
    FILE *f = fopen(path, "wb");
    if(!f){ perror(path); exit(1); }
    if(fwrite(data, sizeof(float), frames*NCH, f) != frames*NCH)
    { fprintf(stderr, "short write %s\n", path); exit(1); }
    fclose(f);
}

/* Number of frames rendered (and discarded) after setting a fixed position
 * but before recording. This lets the effect's per-block tap smoothing settle
 * on the new geometry so the recorded impulse reflects steady-state image
 * delays rather than the convergence transient of a position step.
 */
#define PRIME_FRAMES 2048

static void render_fixed(ALCdevice *dev, ALuint source, const ALfloat pos[3],
    int total_frames, const char *path)
{
    alSource3f(source, AL_POSITION, pos[0], pos[1], pos[2]);
    check_al("source position");
    alSourcePlay(source);
    check_al("source play");

    /* Prime the effect so its smoothed taps converge on this position. The
     * impulse buffer is only 1024 frames and plays once, so rewind it after
     * priming to place a fresh impulse at the start of the recorded render.
     */
    float scratch[PRIME_FRAMES*NCH];
    int primed = 0;
    while(primed < PRIME_FRAMES)
    {
        int chunk = PRIME_FRAMES - primed;
        alcRenderSamplesSOFT(dev, scratch, (ALCsizei)chunk);
        primed += chunk;
    }
    alSourceStop(source);
    alSourceRewind(source);
    check_al("rewind after prime");
    alSourcePlay(source);

    float *out = (float*)calloc((size_t)total_frames*NCH, sizeof(float));
    int done = 0;
    while(done < total_frames)
    {
        int chunk = total_frames - done;
        if(chunk > 1024) chunk = 1024;
        alcRenderSamplesSOFT(dev, out + (size_t)done*NCH, chunk);
        done += chunk;
    }
    write_f32(path, out, (size_t)total_frames);
    free(out);
    alSourceStop(source);
}

static void render_moving(ALCdevice *dev, ALuint source, int total_frames,
    const char *path)
{
    const ALfloat *p0 = kPositions[0];
    const ALfloat *p1 = kPositions[kNumPositions-1];
    alSourcePlay(source);
    check_al("moving play");

    float *out = (float*)calloc((size_t)total_frames*NCH, sizeof(float));
    int done = 0;
    const int chunk = 256;
    while(done < total_frames)
    {
        int n = total_frames - done;
        if(n > chunk) n = chunk;
        float t = (float)done / (float)total_frames;
        ALfloat pos[3];
        for(int a = 0;a < 3;++a)
            pos[a] = p0[a] + (p1[a]-p0[a])*t;
        alSource3f(source, AL_POSITION, pos[0], pos[1], pos[2]);
        alcRenderSamplesSOFT(dev, out + (size_t)done*NCH, n);
        done += n;
    }
    write_f32(path, out, (size_t)total_frames);
    free(out);
    alSourceStop(source);
}

int main(int argc, char **argv)
{
    const char *prefix = (argc > 1) ? argv[1] : "ism";
    float seconds = (argc > 2) ? (float)atof(argv[2]) : 1.0f;
    int total_frames = (int)(seconds * SR);

    ALCcontext *ctx = NULL;
    ALCdevice *dev = open_loopback(&ctx);

    ALuint buf = make_impulse_buffer();
    ALuint slot = 0;
    ALuint source = setup_reverb_source(buf, &slot);

    char meta[512], path[600];
    snprintf(meta, sizeof(meta), "%s_meta.txt", prefix);
    FILE *mf = fopen(meta, "w");
    fprintf(mf, "sr=%d nch=%d frames=%d impulse_samples=%d positions=%d\n",
        SR, NCH, total_frames, IMPULSE_SAMPLES, kNumPositions);
    for(int k = 0;k < kNumPositions;++k)
        fprintf(mf, "pos%d=%.4f,%.4f,%.4f\n", k,
            kPositions[k][0], kPositions[k][1], kPositions[k][2]);
    fclose(mf);

    for(int k = 0;k < kNumPositions;++k)
    {
        snprintf(path, sizeof(path), "%s_ir_pos%d.f32", prefix, k);
        render_fixed(dev, source, kPositions[k], total_frames, path);
        printf("rendered %s\n", path);
    }

    ALuint pbuf = make_periodic_impulse_buffer(total_frames);
    ALuint mslot = 0;
    ALuint msource = setup_reverb_source(pbuf, &mslot);
    alSourcei(msource, AL_LOOPING, AL_TRUE);
    check_al("looping");
    snprintf(path, sizeof(path), "%s_moving.f32", prefix);
    render_moving(dev, msource, total_frames, path);
    printf("rendered %s\n", path);

    alSourceStop(msource);
    alDeleteSources(1, &msource);
    alDeleteBuffers(1, &pbuf);
    alDeleteSources(1, &source);
    alDeleteAuxiliaryEffectSlots(1, &slot);
    alDeleteBuffers(1, &buf);
    alcMakeContextCurrent(NULL);
    alcDestroyContext(ctx);
    alcCloseDevice(dev);
    return 0;
}
