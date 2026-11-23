/* Sweep: across many direct-filter GainHF values, wet 8kHz level must stay
 * constant, while dry 8kHz follows the direct filter; send filter applies
 * independently on top. */
#define AL_ALEXT_PROTOTYPES
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
constexpr ALCsizei SR=48000, N=SR;
void chk(const char*w){ALenum e=alGetError();if(e){std::fprintf(stderr,"err after %s: 0x%x\n",w,e);std::exit(1);}}
ALuint tonebuf(){
  std::vector<ALshort> d(SR);
  for(ALCsizei i=0;i<SR;++i){double t=double(i)/SR;
    d[i]=ALshort(0.25*(std::sin(2*M_PI*500*t)+std::sin(2*M_PI*8000*t))*32767);}
  ALuint b;alGenBuffers(1,&b);
  alBufferData(b,AL_FORMAT_MONO16,d.data(),d.size()*2,SR);chk("buf");return b;
}
ALuint lp(float g){ALuint f;alGenFilters(1,&f);alFilteri(f,AL_FILTER_TYPE,AL_FILTER_LOWPASS);
  alFilterf(f,AL_LOWPASS_GAINHF,g);chk("lp");return f;}
struct RR{std::vector<float> f;};
RR render(ALCdevice* dev, ALCint const* attrs, float directG, float sendG, bool send){
  ALCcontext* c=alcCreateContext(dev,attrs);alcMakeContextCurrent(c);
  ALuint b=tonebuf();
  ALuint slot=0,fx=0;
  if(send){
    alGenEffects(1,&fx);alEffecti(fx,AL_EFFECT_TYPE,AL_EFFECT_REVERB);
    alEffectf(fx,AL_REVERB_GAIN,1.0f);alEffectf(fx,AL_REVERB_GAINHF,1.0f);
    alEffectf(fx,AL_REVERB_DENSITY,0.0f);alEffectf(fx,AL_REVERB_DIFFUSION,0.0f);
    alEffectf(fx,AL_REVERB_DECAY_TIME,AL_REVERB_MIN_DECAY_TIME);chk("fx");
    alGenAuxiliaryEffectSlots(1,&slot);
    alAuxiliaryEffectSloti(slot,AL_EFFECTSLOT_EFFECT,ALint(fx));chk("slot");
  }
  ALuint s;alGenSources(1,&s);alSourcei(s,AL_BUFFER,ALint(b));
  alSource3f(s,AL_POSITION,0,0,0);
  if(directG<0.999f) alSourcei(s,AL_DIRECT_FILTER,ALint(lp(directG)));
  if(send) alSource3i(s,AL_AUXILIARY_SEND_FILTER,ALint(slot),0,
                      ALint(sendG<0.999f?lp(sendG):0));
  chk("src");
  alSourcePlay(s);
  RR r;r.f.resize(size_t(N)*2);alcRenderSamplesSOFT(dev,r.f.data(),N);
  alcMakeContextCurrent(nullptr);alcDestroyContext(c);return r;
}
double lvl8(std::vector<float>const& f, ALCsizei a, ALCsizei z){
  double ii=0,qq=0;
  for(ALCsizei i=a;i<z;++i){double t=double(i)/SR,m=(f[2*i]+f[2*i+1])*0.5;
    ii+=m*std::cos(2*M_PI*8000*t);qq+=m*std::sin(2*M_PI*8000*t);}
  size_t n=z-a;ii/=n;qq/=n;return 2*std::sqrt(ii*ii+qq*qq);
}
}
int main(){
  ALCint attrs[]={ALC_FORMAT_CHANNELS_SOFT,ALC_STEREO_SOFT,ALC_FORMAT_TYPE_SOFT,
    ALC_FLOAT_SOFT,ALC_FREQUENCY,SR,0};
  ALCdevice* dev=alcLoopbackOpenDeviceSOFT(nullptr);
  alcResetDeviceSOFT(dev,attrs);
  {
    ALCcontext* c=alcCreateContext(dev,attrs);alcMakeContextCurrent(c);
    if(!alcIsExtensionPresent(dev,"ALC_EXT_EFX")){std::fprintf(stderr,"no EFX\n");return 1;}
    alcMakeContextCurrent(nullptr);alcDestroyContext(c);
  }
  float gains[]={1.0f,0.7f,0.4f,0.1f,0.03f,0.01f};
  std::printf("%-9s %-9s | %-14s %-14s %-14s\n","dirGHF","sndGHF","dry 8kHz","wet 8kHz","wet (no send LP)");
  double baselineWet=0;
  for(float dg:gains){
    RR full=render(dev,attrs,dg,0.01f,true);
    RR dry =render(dev,attrs,dg,0.01f,false);
    RR full2=render(dev,attrs,dg,1.0f,true);
    RR dry2 =render(dev,attrs,dg,1.0f,false);
    for(size_t i=0;i<full.f.size();++i){full.f[i]-=dry.f[i];full2.f[i]-=dry2.f[i];}
    double w=lvl8(full.f,SR/4,SR*3/4), w2=lvl8(full2.f,SR/4,SR*3/4);
    double d=lvl8(dry.f,SR/4,SR*3/4);
    if(dg==1.0f) baselineWet=w2;
    std::printf("%-9.3f %-9.3f | %-14.6f %-14.6f %-14.6f\n",dg,0.01f,d,w,w2);
  }
  std::printf("\nwet-no-send-LP column must be constant regardless of direct filter "
              "(baseline %.6f)\n",baselineWet);
  return 0;
}
