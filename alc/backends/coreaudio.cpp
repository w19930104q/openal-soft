/**
 * OpenAL cross platform audio library
 * Copyright (C) 1999-2007 by authors.
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

#include <cinttypes>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

#include "alc/events.h"
#include "core/converter.h"

#include <AudioUnit/AudioUnit.h>
#include <AudioToolbox/AudioToolbox.h>

#if TARGET_OS_IOS || TARGET_OS_TV || TARGET_OS_VISION
#define CAN_ENUMERATE 0
#else
#include <IOKit/audio/IOAudioTypes.h>
#define CAN_ENUMERATE 1
#endif

#if HAVE_CXXMODULES
import common.std;

import alstring;
import backends.base;
import backends.exception;
import core.device;
import core.logging;
import format;
import gsl;
import numeric;
import ringbuffer;
import zudl;

#else

#include <cmath>
#include <optional>
#include <string>

#include "alc/backends/base.hpp"
#include "alc/backends/exception.hpp"
#include "alformat.hpp"
#include "alstring.h"
#include "core/device.h"
#include "core/logging.h"
#include "gsl/gsl"
#include "numeric.hpp"
#include "ringbuffer.hpp"
#include "zudl.hpp"
#endif

namespace {

constexpr auto OutputElement = 0;
constexpr auto InputElement = 1;

// These following arrays should always be defined in ascending AudioChannelLabel value order
constexpr std::array<AudioChannelLabel, 1> MonoChanMap { kAudioChannelLabel_Mono };
constexpr std::array<AudioChannelLabel, 2> StereoChanMap { kAudioChannelLabel_Left, kAudioChannelLabel_Right};
constexpr std::array<AudioChannelLabel, 4> QuadChanMap {
        kAudioChannelLabel_Left, kAudioChannelLabel_Right,
        kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround
};
constexpr std::array<AudioChannelLabel, 6> X51ChanMap {
        kAudioChannelLabel_Left, kAudioChannelLabel_Right,
        kAudioChannelLabel_Center, kAudioChannelLabel_LFEScreen,
        kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround
};
constexpr std::array<AudioChannelLabel, 6> X51RearChanMap {
        kAudioChannelLabel_Left, kAudioChannelLabel_Right,
        kAudioChannelLabel_Center, kAudioChannelLabel_LFEScreen,
        kAudioChannelLabel_RearSurroundLeft, kAudioChannelLabel_RearSurroundRight
};
constexpr std::array<AudioChannelLabel, 7> X61ChanMap {
        kAudioChannelLabel_Left, kAudioChannelLabel_Right,
        kAudioChannelLabel_Center, kAudioChannelLabel_LFEScreen,
        kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround,
        kAudioChannelLabel_CenterSurround
};
constexpr std::array<AudioChannelLabel, 8> X71ChanMap {
        kAudioChannelLabel_Left, kAudioChannelLabel_Right,
        kAudioChannelLabel_Center, kAudioChannelLabel_LFEScreen,
        kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround,
        kAudioChannelLabel_RearSurroundLeft, kAudioChannelLabel_RearSurroundRight
};
constexpr std::array<AudioChannelLabel, 12> X714ChanMap {
        kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_Center, kAudioChannelLabel_LFEScreen,
    kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround,
    kAudioChannelLabel_LeftTopFront, kAudioChannelLabel_RightTopFront,
    kAudioChannelLabel_RearSurroundLeft, kAudioChannelLabel_RearSurroundRight,
    kAudioChannelLabel_LeftTopRear, kAudioChannelLabel_RightTopRear
};

struct ChannelMap {
    DevFmtChannels fmt;
    std::span<const AudioChannelLabel> map;
};

constexpr std::array<ChannelMap, 8> chanmaps{{
    { DevFmtX714, X714ChanMap },
    { DevFmtX71, X71ChanMap },
    { DevFmtX61, X61ChanMap },
    { DevFmtX51, X51ChanMap },
    { DevFmtX51, X51RearChanMap },
    { DevFmtQuad, QuadChanMap },
    { DevFmtStereo, StereoChanMap },
    { DevFmtMono, MonoChanMap }
}};


/* Converts a CoreAudio channel label to the OpenAL logical channel it feeds
 * for the given output format. Surround labels are ambiguous: whether they
 * feed a "side" or "back" OpenAL channel depends on the speaker set the
 * format's panning decoder actually renders. Returns NumChannels for labels
 * OpenAL has no output slot for.
 */
constexpr auto LabelToChannel(AudioChannelLabel label, DevFmtChannels fmt) noexcept -> Channel
{
    switch(label)
    {
    case kAudioChannelLabel_Left:                 return FrontLeft;
    case kAudioChannelLabel_Right:                return FrontRight;
    case kAudioChannelLabel_Center:               return FrontCenter;
    case kAudioChannelLabel_LFEScreen:            return LFE;
    case kAudioChannelLabel_Mono:                 return FrontCenter;

    case kAudioChannelLabel_LeftSurround:
    case kAudioChannelLabel_LeftSurroundDirect:
        /* Quad renders to back speakers; every other multichannel format uses
         * these labels for its side speakers.
         */
        return (fmt == DevFmtQuad) ? BackLeft : SideLeft;
    case kAudioChannelLabel_RightSurround:
    case kAudioChannelLabel_RightSurroundDirect:
        return (fmt == DevFmtQuad) ? BackRight : SideRight;

    case kAudioChannelLabel_RearSurroundLeft:
        /* 5.1/6.1 have no separate back pair, so a rear speaker plays the
         * surround (side) channel. 7.1 renders distinct back channels.
         */
        return (fmt == DevFmtX71 || fmt == DevFmtX714) ? BackLeft : SideLeft;
    case kAudioChannelLabel_RearSurroundRight:
        return (fmt == DevFmtX71 || fmt == DevFmtX714) ? BackRight : SideRight;

    case kAudioChannelLabel_CenterSurround:
    case kAudioChannelLabel_CenterSurroundDirect: return BackCenter;
    case kAudioChannelLabel_LeftTopFront:         return TopFrontLeft;
    case kAudioChannelLabel_RightTopFront:        return TopFrontRight;
    case kAudioChannelLabel_LeftTopRear:          return TopBackLeft;
    case kAudioChannelLabel_RightTopRear:         return TopBackRight;
    }
    return NumChannels;
}

/* Finds the OpenAL channel configuration whose speaker set is covered by the
 * given (unsorted) CoreAudio channel labels. The labels may be reported in any
 * order, and may include extra labels for speakers OpenAL doesn't render.
 */
template<std::size_t N>
constexpr auto MatchChannelFormat(const std::array<AudioChannelLabel,N> &labels) noexcept
    -> std::optional<DevFmtChannels>
{
    auto sorted = labels;
    std::ranges::sort(sorted);
    for(const ChannelMap &chanmap : chanmaps)
    {
        if(std::ranges::includes(sorted, chanmap.map))
            return chanmap.fmt;
    }
    return std::nullopt;
}

constexpr auto MatchChannelFormat(std::span<const AudioChannelLabel> labels) noexcept
    -> std::optional<DevFmtChannels>
{
    if(labels.empty()) return std::nullopt;
    auto sorted = std::vector<AudioChannelLabel>(labels.begin(), labels.end());
    std::ranges::sort(sorted);
    for(const ChannelMap &chanmap : chanmaps)
    {
        if(std::ranges::includes(sorted, chanmap.map))
            return chanmap.fmt;
    }
    return std::nullopt;
}

/* Builds a map of OpenAL logical channel -> device stream position from the
 * CoreAudio channel labels, which are given in the order the device expects
 * its interleaved frames. Only usable when the device exposes exactly as many
 * recognizable labels as the stream has channels; returns std::nullopt
 * otherwise so the caller can keep the default WFX order.
 */
constexpr auto MakeChannelOrder(std::span<const AudioChannelLabel> devLabels,
    unsigned numchans, DevFmtChannels fmt) noexcept
    -> std::optional<std::array<std::uint8_t, NumChannels>>
{
    if(devLabels.size() != numchans)
        return std::nullopt;

    auto order = std::array<std::uint8_t, NumChannels>{};
    order.fill(InvalidChannelIndex);

    for(unsigned i{0};i < numchans;++i)
    {
        const Channel ch{LabelToChannel(devLabels[i], fmt)};
        if(ch == NumChannels || order[ch] != InvalidChannelIndex)
            return std::nullopt;
        order[ch] = gsl::narrow_cast<std::uint8_t>(i);
    }
    return order;
}

struct FourCCPrinter {
    char mString[sizeof(UInt32) + 1]{};

    explicit constexpr FourCCPrinter(UInt32 code) noexcept
    {
        for(const auto i : std::views::iota(0_uz, sizeof(UInt32)))
        {
            const auto ch = gsl::narrow_cast<char>(code & 0xff);
            /* If this breaks early it'll leave the first byte null, to get
             * read as a 0-length string.
             */
            if(ch <= 0x1f || ch >= 0x7f)
                break;
            mString[sizeof(UInt32)-1-i] = ch;
            code >>= 8;
        }
    }
    explicit constexpr FourCCPrinter(OSStatus code) noexcept
        : FourCCPrinter{gsl::narrow_cast<UInt32>(code)}
    { }

    constexpr auto c_str() const noexcept -> gsl::czstring { return mString; }
};

#if CAN_ENUMERATE
struct DeviceEntry {
    AudioDeviceID mId;
    std::string mName;
};

std::vector<DeviceEntry> PlaybackList;
std::vector<DeviceEntry> CaptureList;


OSStatus GetHwProperty(AudioHardwarePropertyID propId, UInt32 dataSize, void *propData)
{
    const AudioObjectPropertyAddress addr{propId, kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain};
    return AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr, &dataSize,
        propData);
}

OSStatus GetHwPropertySize(AudioHardwarePropertyID propId, UInt32 *outSize)
{
    const AudioObjectPropertyAddress addr{propId, kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain};
    return AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, nullptr, outSize);
}

OSStatus GetDevProperty(AudioDeviceID devId, AudioDevicePropertyID propId, bool isCapture,
    UInt32 elem, UInt32 dataSize, void *propData)
{
    static const AudioObjectPropertyScope scopes[2]{kAudioDevicePropertyScopeOutput,
        kAudioDevicePropertyScopeInput};
    const AudioObjectPropertyAddress addr{propId, scopes[isCapture], elem};
    return AudioObjectGetPropertyData(devId, &addr, 0, nullptr, &dataSize, propData);
}

OSStatus GetDevPropertySize(AudioDeviceID devId, AudioDevicePropertyID inPropertyID,
    bool isCapture, UInt32 elem, UInt32 *outSize)
{
    static const AudioObjectPropertyScope scopes[2]{kAudioDevicePropertyScopeOutput,
        kAudioDevicePropertyScopeInput};
    const AudioObjectPropertyAddress addr{inPropertyID, scopes[isCapture], elem};
    return AudioObjectGetPropertyDataSize(devId, &addr, 0, nullptr, outSize);
}


std::string GetDeviceName(AudioDeviceID devId)
{
    std::string devname;
    CFStringRef nameRef;

    /* Try to get the device name as a CFString, for Unicode name support. */
    OSStatus err{GetDevProperty(devId, kAudioDevicePropertyDeviceNameCFString, false, 0,
        sizeof(nameRef), &nameRef)};
    if(err == noErr)
    {
        const CFIndex propSize{CFStringGetMaximumSizeForEncoding(CFStringGetLength(nameRef),
            kCFStringEncodingUTF8)};
        devname.resize(gsl::narrow_cast<std::size_t>(propSize)+1, '\0');

        CFStringGetCString(nameRef, &devname[0], propSize+1, kCFStringEncodingUTF8);
        CFRelease(nameRef);
    }
    else
    {
        /* If that failed, just get the C string. Hopefully there's nothing bad
         * with this.
         */
        UInt32 propSize{};
        if(GetDevPropertySize(devId, kAudioDevicePropertyDeviceName, false, 0, &propSize))
            return devname;

        devname.resize(propSize+1, '\0');
        if(GetDevProperty(devId, kAudioDevicePropertyDeviceName, false, 0, propSize, &devname[0]))
        {
            devname.clear();
            return devname;
        }
    }

    /* Clear extraneous nul chars that may have been written with the name
     * string, and return it.
     */
    while(!devname.empty() && !devname.back())
        devname.pop_back();
    return devname;
}

auto GetDeviceChannelCount(AudioDeviceID devId, bool isCapture) -> UInt32
{
    auto propSize = UInt32{};
    auto err = GetDevPropertySize(devId, kAudioDevicePropertyStreamConfiguration, isCapture, 0,
        &propSize);
    if(err)
    {
        ERR("kAudioDevicePropertyStreamConfiguration size query failed: '{}' ({})",
            FourCCPrinter{err}.c_str(), err);
        return 0;
    }

    auto buflist_data = std::make_unique<char[]>(propSize);
    /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */
    auto *buflist = reinterpret_cast<AudioBufferList*>(buflist_data.get());

    err = GetDevProperty(devId, kAudioDevicePropertyStreamConfiguration, isCapture, 0, propSize,
        buflist);
    if(err)
    {
        ERR("kAudioDevicePropertyStreamConfiguration query failed: '{}' ({})",
            FourCCPrinter{err}.c_str(), err);
        return 0;
    }

    auto numChannels = UInt32{0};
    for(auto i=0_uz;i < buflist->mNumberBuffers;++i)
        numChannels += buflist->mBuffers[i].mNumberChannels;
    return numChannels;
}


void EnumerateDevices(std::vector<DeviceEntry> &list, bool isCapture)
{
    UInt32 propSize{};
    if(auto err = GetHwPropertySize(kAudioHardwarePropertyDevices, &propSize))
    {
        ERR("Failed to get device list size: {}", err);
        return;
    }

    auto devIds = std::vector<AudioDeviceID>(propSize/sizeof(AudioDeviceID), kAudioDeviceUnknown);
    if(auto err = GetHwProperty(kAudioHardwarePropertyDevices, propSize, devIds.data()))
    {
        ERR("Failed to get device list: '{}' ({})", FourCCPrinter{err}.c_str(), err);
        return;
    }

    std::vector<DeviceEntry> newdevs;
    newdevs.reserve(devIds.size());

    AudioDeviceID defaultId{kAudioDeviceUnknown};
    GetHwProperty(isCapture ? kAudioHardwarePropertyDefaultInputDevice :
        kAudioHardwarePropertyDefaultOutputDevice, sizeof(defaultId), &defaultId);

    if(defaultId != kAudioDeviceUnknown)
    {
        newdevs.emplace_back(DeviceEntry{defaultId, GetDeviceName(defaultId)});
        const auto &entry = newdevs.back();
        TRACE("Got device: {} = ID {}", entry.mName, entry.mId);
    }
    for(const AudioDeviceID devId : devIds)
    {
        if(devId == kAudioDeviceUnknown)
            continue;

        auto match = std::ranges::find(newdevs, devId, &DeviceEntry::mId);
        if(match != newdevs.end()) continue;

        auto numChannels = GetDeviceChannelCount(devId, isCapture);
        if(numChannels > 0)
        {
            newdevs.emplace_back(DeviceEntry{devId, GetDeviceName(devId)});
            const auto &entry = newdevs.back();
            TRACE("Got device: {} = ID {}", entry.mName, entry.mId);
        }
    }

    if(newdevs.size() > 1)
    {
        /* Rename entries that have matching names, by appending '#2', '#3',
         * etc, as needed.
         */
        for(auto curitem = newdevs.begin()+1;curitem != newdevs.end();++curitem)
        {
            const auto subrange = std::span{newdevs.begin(), curitem};
            if(std::ranges::find(subrange, curitem->mName, &DeviceEntry::mName) != subrange.end())
            {
                auto name = std::string{};
                auto count = 1_uz;
                do {
                    name = al::format("{} #{}", curitem->mName, ++count);
                } while(std::ranges::find(subrange, name, &DeviceEntry::mName) != subrange.end());
                curitem->mName = std::move(name);
            }
        }
    }

    newdevs.shrink_to_fit();
    newdevs.swap(list);
}

struct DeviceHelper {
    DeviceHelper()
    {
        AudioObjectPropertyAddress addr{kAudioHardwarePropertyDefaultOutputDevice,
            kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
        OSStatus status = AudioObjectAddPropertyListener(kAudioObjectSystemObject, &addr, DeviceListenerProc, nil);
        if (status != noErr)
            ERR("AudioObjectAddPropertyListener fail: {}", status);
    }
    ~DeviceHelper()
    {
        AudioObjectPropertyAddress addr{kAudioHardwarePropertyDefaultOutputDevice,
            kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
        OSStatus status = AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &addr, DeviceListenerProc, nil);
        if (status != noErr)
            ERR("AudioObjectRemovePropertyListener fail: {}", status);
    }

    static OSStatus DeviceListenerProc(AudioObjectID /*inObjectID*/, UInt32 inNumberAddresses,
        const AudioObjectPropertyAddress *inAddresses, void* /*inClientData*/)
    {
        for(UInt32 i = 0; i < inNumberAddresses; ++i)
        {
            switch(inAddresses[i].mSelector)
            {
            case kAudioHardwarePropertyDefaultOutputDevice:
            case kAudioHardwarePropertyDefaultSystemOutputDevice:
                alc::Event(alc::EventType::DefaultDeviceChanged, alc::DeviceType::Playback,
                    "Default playback device changed: "+std::to_string(inAddresses[i].mSelector));
                break;
            case kAudioHardwarePropertyDefaultInputDevice:
                alc::Event(alc::EventType::DefaultDeviceChanged, alc::DeviceType::Capture,
                    "Default capture device changed: "+std::to_string(inAddresses[i].mSelector));
                break;
            }
        }
        return noErr;
    }
};

static std::optional<DeviceHelper> sDeviceHelper;

#else

static constexpr char ca_device[] = "CoreAudio Default";
#endif


/* Extracts the ordered channel labels from a CoreAudio channel layout.
 * Description-based layouts keep the stream order of the descriptions; bitmap
 * layouts are expanded in bit order, which is the stream order CoreAudio uses
 * for bitmap layouts; tag-based layouts are expanded through AudioFormat.
 */
auto GetLayoutLabels(AudioChannelLayoutTag layoutTag, AudioChannelBitmap bitmap,
    std::span<const AudioChannelDescription> descs) -> std::vector<AudioChannelLabel>
{
    auto labels = std::vector<AudioChannelLabel>{};

    if(layoutTag == kAudioChannelLayoutTag_UseChannelDescriptions)
    {
        labels.reserve(descs.size());
        std::ranges::transform(descs, std::back_inserter(labels),
            &AudioChannelDescription::mChannelLabel);
        return labels;
    }

    if(layoutTag == kAudioChannelLayoutTag_UseChannelBitmap)
    {
        struct BitmapEntry { AudioChannelBitmap bit; AudioChannelLabel label; };
        static constexpr std::array<BitmapEntry, 18> bitmapMap{{
            { kAudioChannelBit_Left,                 kAudioChannelLabel_Left },
            { kAudioChannelBit_Right,                kAudioChannelLabel_Right },
            { kAudioChannelBit_Center,               kAudioChannelLabel_Center },
            { kAudioChannelBit_LFEScreen,            kAudioChannelLabel_LFEScreen },
            { kAudioChannelBit_LeftSurround,         kAudioChannelLabel_LeftSurround },
            { kAudioChannelBit_RightSurround,        kAudioChannelLabel_RightSurround },
            { kAudioChannelBit_LeftCenter,           kAudioChannelLabel_LeftCenter },
            { kAudioChannelBit_RightCenter,          kAudioChannelLabel_RightCenter },
            { kAudioChannelBit_CenterSurround,       kAudioChannelLabel_CenterSurround },
            { kAudioChannelBit_TopCenterSurround,    kAudioChannelLabel_TopCenterSurround },
            { kAudioChannelBit_VerticalHeightLeft,   kAudioChannelLabel_LeftTopFront },
            { kAudioChannelBit_VerticalHeightCenter, kAudioChannelLabel_CenterTopFront },
            { kAudioChannelBit_VerticalHeightRight,  kAudioChannelLabel_RightTopFront },
            { kAudioChannelBit_TopBackLeft,          kAudioChannelLabel_LeftTopRear },
            { kAudioChannelBit_TopBackCenter,        kAudioChannelLabel_CenterTopRear },
            { kAudioChannelBit_TopBackRight,         kAudioChannelLabel_RightTopRear },
            { kAudioChannelBit_RearSurroundLeft,     kAudioChannelLabel_RearSurroundLeft },
            { kAudioChannelBit_RearSurroundRight,    kAudioChannelLabel_RearSurroundRight }
        }};
        for(const BitmapEntry &entry : bitmapMap)
        {
            if((bitmap & entry.bit) != 0)
                labels.emplace_back(entry.label);
        }
        return labels;
    }

    /* A fixed tag names an ordered speaker set. Ask AudioFormat to expand it
     * into channel descriptions so the same code path handles every layout.
     */
    auto propSize = UInt32{};
    OSStatus err{AudioFormatGetPropertyInfo(kAudioFormatProperty_ChannelLayoutForTag,
        sizeof(layoutTag), &layoutTag, &propSize)};
    if(err != noErr || propSize < sizeof(AudioChannelLayout))
    {
        WARN("Failed to get channel layout info for tag {:#010x}: '{}' ({})",
            static_cast<UInt32>(layoutTag), FourCCPrinter{err}.c_str(), err);
        return labels;
    }

    auto layout_data = std::make_unique<char[]>(propSize);
    /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */
    auto *tagLayout = reinterpret_cast<AudioChannelLayout*>(layout_data.get());
    err = AudioFormatGetProperty(kAudioFormatProperty_ChannelLayoutForTag, sizeof(layoutTag),
        &layoutTag, &propSize, tagLayout);
    if(err != noErr)
    {
        WARN("Failed to get channel layout for tag {:#010x}: '{}' ({})",
            static_cast<UInt32>(layoutTag), FourCCPrinter{err}.c_str(), err);
        return labels;
    }

    const auto tagDescs = std::span{std::data(tagLayout->mChannelDescriptions),
        tagLayout->mNumberChannelDescriptions};
    labels.reserve(tagDescs.size());
    std::ranges::transform(tagDescs, std::back_inserter(labels),
        &AudioChannelDescription::mChannelLabel);
    return labels;
}

auto GetLayoutLabels(const AudioChannelLayout &layout) -> std::vector<AudioChannelLabel>
{
    const auto descs = std::span{std::data(layout.mChannelDescriptions),
        layout.mNumberChannelDescriptions};
    return GetLayoutLabels(layout.mChannelLayoutTag, layout.mChannelBitmap, descs);
}

/* Queries the output AudioUnit's current channel layout, returning the labels
 * in the device's stream order. Returns an empty vector when the device
 * doesn't expose a usable layout.
 */
auto GetAudioUnitChannelLabels(AudioUnit audioUnit) -> std::vector<AudioChannelLabel>
{
    auto propSize = UInt32{};
    auto writable = Boolean{};
    OSStatus err{AudioUnitGetPropertyInfo(audioUnit, kAudioUnitProperty_AudioChannelLayout,
        kAudioUnitScope_Output, OutputElement, &propSize, &writable)};
    if(err != noErr || propSize < sizeof(AudioChannelLayout))
    {
        WARN("Failed to get audio channel layout info: '{}' ({})",
            FourCCPrinter{err}.c_str(), err);
        return {};
    }

    auto layout_data = std::make_unique<char[]>(propSize);
    /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */
    auto *layout = reinterpret_cast<AudioChannelLayout*>(layout_data.get());
    err = AudioUnitGetProperty(audioUnit, kAudioUnitProperty_AudioChannelLayout,
        kAudioUnitScope_Output, OutputElement, layout, &propSize);
    if(err != noErr)
    {
        WARN("Failed to get audio channel layout: '{}' ({})", FourCCPrinter{err}.c_str(),
            err);
        return {};
    }

    auto labels = GetLayoutLabels(*layout);
    std::erase_if(labels, [](const AudioChannelLabel label) noexcept
    {
        /* Unknown/unused/coordinate-based slots can't be matched to an OpenAL
         * speaker; drop them so they don't poison the format match.
         */
        return label == kAudioChannelLabel_Unknown || label == kAudioChannelLabel_Unused
            || label == kAudioChannelLabel_UseCoordinates;
    });
    return labels;
}


struct CoreAudioPlayback final : BackendBase {
    using BackendBase::BackendBase;
    ~CoreAudioPlayback() override;

    OSStatus MixerProc(AudioUnitRenderActionFlags *ioActionFlags,
        const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames,
        AudioBufferList *ioData) noexcept;

    void open(std::string_view name) override;
    bool reset() override;
    void start() override;
    void stop() override;

    AudioUnit mAudioUnit{};

    unsigned mFrameSize{0u};
    AudioStreamBasicDescription mFormat{}; // This is the OpenAL format as a CoreAudio ASBD
};

CoreAudioPlayback::~CoreAudioPlayback()
{
    AudioUnitUninitialize(mAudioUnit);
    AudioComponentInstanceDispose(mAudioUnit);
}


OSStatus CoreAudioPlayback::MixerProc(AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32,
    UInt32, AudioBufferList *ioData) noexcept NONBLOCKING
{
    for(auto i=0_uz;i < ioData->mNumberBuffers;++i)
    {
        auto &buffer = ioData->mBuffers[i];
        mDevice.renderSamples(buffer.mData, buffer.mDataByteSize/mFrameSize,
            buffer.mNumberChannels);
    }
    return noErr;
}


void CoreAudioPlayback::open(std::string_view name)
{
#if CAN_ENUMERATE
    AudioDeviceID audioDevice{kAudioDeviceUnknown};
    if(name.empty())
        GetHwProperty(kAudioHardwarePropertyDefaultOutputDevice, sizeof(audioDevice),
            &audioDevice);
    else
    {
        if(PlaybackList.empty())
            EnumerateDevices(PlaybackList, false);

        auto devmatch = std::ranges::find(PlaybackList, name, &DeviceEntry::mName);
        if(devmatch == PlaybackList.end())
            throw al::backend_exception{al::backend_error::NoDevice,
                "Device name \"{}\" not found", name};

        audioDevice = devmatch->mId;
    }
#else
    if(name.empty())
        name = ca_device;
    else if(name != ca_device)
        throw al::backend_exception{al::backend_error::NoDevice, "Device name \"{}\" not found",
            name};
#endif

    /* open the default output unit */
    AudioComponentDescription desc{};
    desc.componentType = kAudioUnitType_Output;
#if CAN_ENUMERATE
    desc.componentSubType = (audioDevice == kAudioDeviceUnknown) ?
        kAudioUnitSubType_DefaultOutput : kAudioUnitSubType_HALOutput;
#else
    desc.componentSubType = kAudioUnitSubType_RemoteIO;
#endif
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    desc.componentFlags = 0;
    desc.componentFlagsMask = 0;

    AudioComponent comp{AudioComponentFindNext(NULL, &desc)};
    if(comp == nullptr)
        throw al::backend_exception{al::backend_error::NoDevice, "Could not find audio component"};

    AudioUnit audioUnit{};
    OSStatus err{AudioComponentInstanceNew(comp, &audioUnit)};
    if(err != noErr)
        throw al::backend_exception{al::backend_error::NoDevice,
            "Could not create component instance: '{}' ({})", FourCCPrinter{err}.c_str(), err};

#if CAN_ENUMERATE
    if(audioDevice != kAudioDeviceUnknown)
        AudioUnitSetProperty(audioUnit, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, OutputElement, &audioDevice, sizeof(AudioDeviceID));
#endif

    err = AudioUnitInitialize(audioUnit);
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not initialize audio unit: '{}' ({})", FourCCPrinter{err}.c_str(), err};

    /* WARNING: I don't know if "valid" audio unit values are guaranteed to be
     * non-0. If not, this logic is broken.
     */
    if(mAudioUnit)
    {
        AudioUnitUninitialize(mAudioUnit);
        AudioComponentInstanceDispose(mAudioUnit);
    }
    mAudioUnit = audioUnit;

#if CAN_ENUMERATE
    if(!name.empty())
        mDeviceName = name;
    else
    {
        UInt32 propSize{sizeof(audioDevice)};
        audioDevice = kAudioDeviceUnknown;
        AudioUnitGetProperty(audioUnit, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, OutputElement, &audioDevice, &propSize);

        std::string devname{GetDeviceName(audioDevice)};
        if(!devname.empty()) mDeviceName = std::move(devname);
        else mDeviceName = "Unknown Device Name";
    }

    if(audioDevice != kAudioDeviceUnknown)
    {
        UInt32 type{};
        err = GetDevProperty(audioDevice, kAudioDevicePropertyDataSource, false,
            kAudioObjectPropertyElementMain, sizeof(type), &type);
        if(err != noErr)
            WARN("Failed to get audio device type: '{}' ({})", FourCCPrinter{err}.c_str(), err);
        else
        {
            TRACE("Got device type '{}'", FourCCPrinter{type}.c_str());
            mDevice.mFlags.set(DeviceFlag::DirectEar,
               (type == kIOAudioOutputPortSubTypeHeadphones));
        }
    }

#else
    mDeviceName = name;
#endif
}

bool CoreAudioPlayback::reset()
{
    OSStatus err{AudioUnitUninitialize(mAudioUnit)};
    if(err != noErr)
        ERR("AudioUnitUninitialize failed: '{}' ({})", FourCCPrinter{err}.c_str(), err);

    /* retrieve default output unit's properties (output side) */
    AudioStreamBasicDescription streamFormat{};
    UInt32 size{sizeof(streamFormat)};
    err = AudioUnitGetProperty(mAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output,
        OutputElement, &streamFormat, &size);
    if(err != noErr || size != sizeof(streamFormat))
    {
        ERR("AudioUnitGetProperty(StreamFormat) failed: '{}' ({})", FourCCPrinter{err}.c_str(),
            err);
        return false;
    }

    /* Use the sample rate from the output unit's current parameters, but reset
     * everything else.
     */
    if(mDevice.mSampleRate != streamFormat.mSampleRate)
    {
        mDevice.mBufferSize = gsl::narrow_cast<unsigned>(mDevice.mBufferSize
            *streamFormat.mSampleRate/mDevice.mSampleRate + 0.5);
        mDevice.mSampleRate = gsl::narrow_cast<unsigned>(streamFormat.mSampleRate);
    }

    /* Get the device's actual channel layout. The labels are kept in the
     * device's stream order so they can both select the OpenAL channel
     * configuration and remap the mixer output to the physical speakers.
     */
    const auto devLabels = GetAudioUnitChannelLabels(mAudioUnit);

    if(!mDevice.mFlags.test(DeviceFlag::ChannelsRequest))
    {
        if(const auto fmt = MatchChannelFormat(std::span{devLabels}))
        {
            TRACE("Matched device channel layout to {}", DevFmtChannelsString(*fmt));
            mDevice.FmtChans = *fmt;
        }
        else if(!devLabels.empty())
            WARN("Failed to find a matching channel configuration for {} device channels",
                devLabels.size());
    }

    streamFormat.mChannelsPerFrame = mDevice.channelsFromFmt();

    streamFormat.mFramesPerPacket = 1;
    streamFormat.mFormatFlags = kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    streamFormat.mFormatID = kAudioFormatLinearPCM;
    switch(mDevice.FmtType)
    {
    case DevFmtUByte:
        mDevice.FmtType = DevFmtByte;
        [[fallthrough]];
    case DevFmtByte:
        streamFormat.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        streamFormat.mBitsPerChannel = 8;
        break;
    case DevFmtUShort:
        mDevice.FmtType = DevFmtShort;
        [[fallthrough]];
    case DevFmtShort:
        streamFormat.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        streamFormat.mBitsPerChannel = 16;
        break;
    case DevFmtUInt:
        mDevice.FmtType = DevFmtInt;
        [[fallthrough]];
    case DevFmtInt:
        streamFormat.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        streamFormat.mBitsPerChannel = 32;
        break;
    case DevFmtFloat:
        streamFormat.mFormatFlags |= kAudioFormatFlagIsFloat;
        streamFormat.mBitsPerChannel = 32;
        break;
    }
    streamFormat.mBytesPerFrame = streamFormat.mChannelsPerFrame*streamFormat.mBitsPerChannel/8;
    streamFormat.mBytesPerPacket = streamFormat.mBytesPerFrame*streamFormat.mFramesPerPacket;

    err = AudioUnitSetProperty(mAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
        OutputElement, &streamFormat, sizeof(streamFormat));
    if(err != noErr)
    {
        ERR("AudioUnitSetProperty(StreamFormat) failed: '{}' ({})", FourCCPrinter{err}.c_str(),
            err);
        return false;
    }

    /* Set the channel order according to the device's actual layout, so each
     * OpenAL speaker is rendered to the matching physical output. If the
     * device layout can't be used (unknown labels, or not enough channels),
     * fall back to the default WFX channel order.
     */
    const auto numchans = mDevice.channelsFromFmt();
    if(const auto order = MakeChannelOrder(std::span{devLabels}, numchans, mDevice.FmtChans))
        mDevice.RealOut.ChannelIndex = *order;
    else
    {
        if(!devLabels.empty())
            WARN("Unable to remap {} channels to the device layout, using the default channel order",
                numchans);
        setDefaultWFXChannelOrder();
    }

    /* setup callback */
    mFrameSize = mDevice.frameSizeFromFmt();
    auto input = AURenderCallbackStruct{};
    input.inputProc = [](void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags,
        const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames,
        AudioBufferList *ioData) noexcept NONBLOCKING
    {
        return static_cast<CoreAudioPlayback*>(inRefCon)->MixerProc(ioActionFlags, inTimeStamp,
            inBusNumber, inNumberFrames, ioData);
    };
    input.inputProcRefCon = this;

    err = AudioUnitSetProperty(mAudioUnit, kAudioUnitProperty_SetRenderCallback,
        kAudioUnitScope_Input, OutputElement, &input, sizeof(AURenderCallbackStruct));
    if(err != noErr)
    {
        ERR("AudioUnitSetProperty(SetRenderCallback) failed: '{}' ({})",
            FourCCPrinter{err}.c_str(), err);
        return false;
    }

    /* init the default audio unit... */
    err = AudioUnitInitialize(mAudioUnit);
    if(err != noErr)
    {
        ERR("AudioUnitInitialize failed: '{}' ({})", FourCCPrinter{err}.c_str(), err);
        return false;
    }

    return true;
}

void CoreAudioPlayback::start()
{
    const OSStatus err{AudioOutputUnitStart(mAudioUnit)};
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "AudioOutputUnitStart failed: '{}' ({})", FourCCPrinter{err}.c_str(), err};
}

void CoreAudioPlayback::stop()
{
    OSStatus err{AudioOutputUnitStop(mAudioUnit)};
    if(err != noErr)
        ERR("AudioOutputUnitStop failed: '{}' ({})", FourCCPrinter{err}.c_str(), err);
}


struct CoreAudioCapture final : BackendBase {
    using BackendBase::BackendBase;
    ~CoreAudioCapture() override;

    OSStatus RecordProc(AudioUnitRenderActionFlags *ioActionFlags,
        const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber,
        UInt32 inNumberFrames, AudioBufferList *ioData) noexcept;

    void open(std::string_view name) override;
    void start() override;
    void stop() override;
    void captureSamples(std::span<std::byte> outbuffer) override;
    auto availableSamples() -> std::size_t override;

    AudioUnit mAudioUnit{0};

    unsigned mFrameSize{0u};
    AudioStreamBasicDescription mFormat{};  // This is the OpenAL format as a CoreAudio ASBD

    SampleConverterPtr mConverter;

    std::vector<std::byte> mCaptureData;

    RingBufferPtr<std::byte> mRing;
};

CoreAudioCapture::~CoreAudioCapture()
{
    if(mAudioUnit)
        AudioComponentInstanceDispose(mAudioUnit);
    mAudioUnit = 0;
}


OSStatus CoreAudioCapture::RecordProc(AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames,
    AudioBufferList*) noexcept
{
    union {
        std::byte buf[std::max(sizeof(AudioBufferList), offsetof(AudioBufferList, mBuffers[1]))];
        AudioBufferList list;
    } audiobuf{};

    audiobuf.list.mNumberBuffers = 1;
    audiobuf.list.mBuffers[0].mNumberChannels = mFormat.mChannelsPerFrame;
    audiobuf.list.mBuffers[0].mData = mCaptureData.data();
    audiobuf.list.mBuffers[0].mDataByteSize = gsl::narrow_cast<UInt32>(mCaptureData.size());

    OSStatus err{AudioUnitRender(mAudioUnit, ioActionFlags, inTimeStamp, inBusNumber,
        inNumberFrames, &audiobuf.list)};
    if(err != noErr)
    {
        ERR("AudioUnitRender capture error: '{}' ({})", FourCCPrinter{err}.c_str(), err);
        return err;
    }

    std::ignore = mRing->write(std::span{mCaptureData}
        .first(inNumberFrames*std::size_t{mFrameSize}));
    return noErr;
}


void CoreAudioCapture::open(std::string_view name)
{
#if CAN_ENUMERATE
    AudioDeviceID audioDevice{kAudioDeviceUnknown};
    if(name.empty())
        GetHwProperty(kAudioHardwarePropertyDefaultInputDevice, sizeof(audioDevice),
            &audioDevice);
    else
    {
        if(CaptureList.empty())
            EnumerateDevices(CaptureList, true);

        auto devmatch = std::ranges::find(CaptureList, name, &DeviceEntry::mName);
        if(devmatch == CaptureList.end())
            throw al::backend_exception{al::backend_error::NoDevice,
                "Device name \"{}\" not found", name};

        audioDevice = devmatch->mId;
    }
#else
    if(name.empty())
        name = ca_device;
    else if(name != ca_device)
        throw al::backend_exception{al::backend_error::NoDevice, "Device name \"{}\" not found",
            name};
#endif

    AudioComponentDescription desc{};
    desc.componentType = kAudioUnitType_Output;
#if CAN_ENUMERATE
    desc.componentSubType = (audioDevice == kAudioDeviceUnknown) ?
        kAudioUnitSubType_DefaultOutput : kAudioUnitSubType_HALOutput;
#else
    desc.componentSubType = kAudioUnitSubType_RemoteIO;
#endif
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    desc.componentFlags = 0;
    desc.componentFlagsMask = 0;

    // Search for component with given description
    AudioComponent comp{AudioComponentFindNext(NULL, &desc)};
    if(comp == NULL)
        throw al::backend_exception{al::backend_error::NoDevice, "Could not find audio component"};

    // Open the component
    OSStatus err{AudioComponentInstanceNew(comp, &mAudioUnit)};
    if(err != noErr)
        throw al::backend_exception{al::backend_error::NoDevice,
            "Could not create component instance: '{}' ({})", FourCCPrinter{err}.c_str(), err};

    // Turn off AudioUnit output
    UInt32 enableIO{0};
    err = AudioUnitSetProperty(mAudioUnit, kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Output, OutputElement, &enableIO, sizeof(enableIO));
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not disable audio unit output property: '{}' ({})", FourCCPrinter{err}.c_str(),
            err};

    // Turn on AudioUnit input
    enableIO = 1;
    err = AudioUnitSetProperty(mAudioUnit, kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Input, InputElement, &enableIO, sizeof(enableIO));
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not enable audio unit input property: '{}' ({})", FourCCPrinter{err}.c_str(),
            err};

#if CAN_ENUMERATE
    if(audioDevice != kAudioDeviceUnknown)
        AudioUnitSetProperty(mAudioUnit, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, InputElement, &audioDevice, sizeof(AudioDeviceID));
#endif

    // set capture callback
    AURenderCallbackStruct input{};
    input.inputProc = [](void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) noexcept
    { return static_cast<CoreAudioCapture*>(inRefCon)->RecordProc(ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData); };
    input.inputProcRefCon = this;

    err = AudioUnitSetProperty(mAudioUnit, kAudioOutputUnitProperty_SetInputCallback,
        kAudioUnitScope_Global, InputElement, &input, sizeof(AURenderCallbackStruct));
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not set capture callback: '{}' ({})", FourCCPrinter{err}.c_str(), err};

    // Disable buffer allocation for capture
    UInt32 flag{0};
    err = AudioUnitSetProperty(mAudioUnit, kAudioUnitProperty_ShouldAllocateBuffer,
        kAudioUnitScope_Output, InputElement, &flag, sizeof(flag));
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not disable buffer allocation property: '{}' ({})", FourCCPrinter{err}.c_str(),
            err};

    // Initialize the device
    err = AudioUnitInitialize(mAudioUnit);
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not initialize audio unit: '{}' ({})", FourCCPrinter{err}.c_str(), err};

    // Get the hardware format
    AudioStreamBasicDescription hardwareFormat{};
    UInt32 propertySize{sizeof(hardwareFormat)};
    err = AudioUnitGetProperty(mAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
        InputElement, &hardwareFormat, &propertySize);
    if(err != noErr || propertySize != sizeof(hardwareFormat))
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not get input format: '{}' ({})", FourCCPrinter{err}.c_str(), err};

    // Set up the requested format description
    auto requestedFormat = AudioStreamBasicDescription{};
    requestedFormat.mFormatFlags = kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    switch(mDevice.FmtType)
    {
    case DevFmtByte:
        requestedFormat.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        [[fallthrough]];
    case DevFmtUByte:
        requestedFormat.mBitsPerChannel = 8;
        break;
    case DevFmtShort:
        requestedFormat.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        [[fallthrough]];
    case DevFmtUShort:
        requestedFormat.mBitsPerChannel = 16;
        break;
    case DevFmtInt:
        requestedFormat.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        [[fallthrough]];
    case DevFmtUInt:
        requestedFormat.mBitsPerChannel = 32;
        break;
    case DevFmtFloat:
        requestedFormat.mFormatFlags |= kAudioFormatFlagIsFloat;
        requestedFormat.mBitsPerChannel = 32;
        break;
    }

    switch(mDevice.FmtChans)
    {
    case DevFmtMono:
        requestedFormat.mChannelsPerFrame = 1;
        break;
    case DevFmtStereo:
        requestedFormat.mChannelsPerFrame = 2;
        break;

    case DevFmtQuad:
    case DevFmtX51:
    case DevFmtX61:
    case DevFmtX71:
    case DevFmtX714:
    case DevFmtX7144:
    case DevFmtX3D71:
    case DevFmtAmbi3D:
        throw al::backend_exception{al::backend_error::DeviceError, "{} not supported",
            DevFmtChannelsString(mDevice.FmtChans)};
    }

    requestedFormat.mBytesPerFrame = requestedFormat.mChannelsPerFrame * requestedFormat.mBitsPerChannel / 8;
    requestedFormat.mBytesPerPacket = requestedFormat.mBytesPerFrame;
    requestedFormat.mSampleRate = mDevice.mSampleRate;
    requestedFormat.mFormatID = kAudioFormatLinearPCM;
    requestedFormat.mReserved = 0;
    requestedFormat.mFramesPerPacket = 1;

    // save requested format description for later use
    mFormat = requestedFormat;
    mFrameSize = mDevice.frameSizeFromFmt();

    // Use intermediate format for sample rate conversion (outputFormat)
    // Set sample rate to the same as hardware for resampling later
    AudioStreamBasicDescription outputFormat{requestedFormat};
    outputFormat.mSampleRate = hardwareFormat.mSampleRate;

    // The output format should be the requested format, but using the hardware sample rate
    // This is because the AudioUnit will automatically scale other properties, except for sample rate
    err = AudioUnitSetProperty(mAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output,
        InputElement, &outputFormat, sizeof(outputFormat));
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not set input format: '{}' ({})", FourCCPrinter{err}.c_str(), err};

    /* Calculate the minimum AudioUnit output format frame count for the pre-
     * conversion ring buffer. Ensure at least 100ms for the total buffer.
     */
    double srateScale{outputFormat.mSampleRate / mDevice.mSampleRate};
    auto FrameCount64 = std::max(
        gsl::narrow_cast<std::uint64_t>(std::ceil(mDevice.mBufferSize*srateScale)),
        gsl::narrow_cast<UInt32>(outputFormat.mSampleRate)/std::uint64_t{10});
    FrameCount64 += MaxResamplerPadding;
    if(FrameCount64 > std::numeric_limits<int>::max())
        throw al::backend_exception{al::backend_error::DeviceError,
            "Calculated frame count is too large: {}", FrameCount64};

    UInt32 outputFrameCount{};
    propertySize = sizeof(outputFrameCount);
    err = AudioUnitGetProperty(mAudioUnit, kAudioUnitProperty_MaximumFramesPerSlice,
        kAudioUnitScope_Global, OutputElement, &outputFrameCount, &propertySize);
    if(err != noErr || propertySize != sizeof(outputFrameCount))
        throw al::backend_exception{al::backend_error::DeviceError,
            "Could not get input frame count: '{}' ({})", FourCCPrinter{err}.c_str(), err};

    mCaptureData.resize(outputFrameCount * mFrameSize);

    outputFrameCount = gsl::narrow_cast<UInt32>(std::max(std::uint64_t{outputFrameCount}, FrameCount64));
    mRing = RingBuffer<std::byte>::Create(outputFrameCount, mFrameSize, false);

    /* Set up sample converter if needed */
    if(outputFormat.mSampleRate != mDevice.mSampleRate)
        mConverter = SampleConverter::Create(mDevice.FmtType, mDevice.FmtType,
            mFormat.mChannelsPerFrame, gsl::narrow_cast<unsigned>(hardwareFormat.mSampleRate),
            mDevice.mSampleRate, Resampler::FastBSinc24);

#if CAN_ENUMERATE
    if(!name.empty())
        mDeviceName = name;
    else
    {
        UInt32 propSize{sizeof(audioDevice)};
        audioDevice = kAudioDeviceUnknown;
        AudioUnitGetProperty(mAudioUnit, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, InputElement, &audioDevice, &propSize);

        std::string devname{GetDeviceName(audioDevice)};
        if(!devname.empty()) mDeviceName = std::move(devname);
        else mDeviceName = "Unknown Device Name";
    }
#else
    mDeviceName = name;
#endif
}


void CoreAudioCapture::start()
{
    OSStatus err{AudioOutputUnitStart(mAudioUnit)};
    if(err != noErr)
        throw al::backend_exception{al::backend_error::DeviceError,
            "AudioOutputUnitStart failed: '{}' ({})", FourCCPrinter{err}.c_str(), err};
}

void CoreAudioCapture::stop()
{
    OSStatus err{AudioOutputUnitStop(mAudioUnit)};
    if(err != noErr)
        ERR("AudioOutputUnitStop failed: '{}' ({})", FourCCPrinter{err}.c_str(), err);
}

void CoreAudioCapture::captureSamples(std::span<std::byte> outbuffer)
{
    if(!mConverter)
    {
        std::ignore = mRing->read(outbuffer);
        return;
    }

    auto rec_vec = mRing->getReadVector();
    const void *src0 = rec_vec[0].data();
    auto src0len = gsl::narrow_cast<unsigned>(rec_vec[0].size() / mFrameSize);
    auto got = mConverter->convert(&src0, &src0len, outbuffer.data(),
        gsl::narrow_cast<unsigned>(outbuffer.size()/mFrameSize));
    auto total_read = rec_vec[0].size()/mFrameSize - src0len;
    if(got < outbuffer.size()/mFrameSize && !src0len && !rec_vec[1].empty())
    {
        outbuffer = outbuffer.subspan(got*mFrameSize);
        const void *src1 = rec_vec[1].data();
        auto src1len = gsl::narrow_cast<unsigned>(rec_vec[1].size()/mFrameSize);
        std::ignore = mConverter->convert(&src1, &src1len, outbuffer.data(),
            gsl::narrow_cast<unsigned>(outbuffer.size()/mFrameSize));
        total_read += rec_vec[1].size()/mFrameSize - src1len;
    }

    mRing->readAdvance(total_read);
}

auto CoreAudioCapture::availableSamples() -> std::size_t
{
    if(!mConverter) return mRing->readSpace();
    return mConverter->availableOut(gsl::narrow_cast<unsigned>(mRing->readSpace()));
}

} // namespace

BackendFactory &CoreAudioBackendFactory::getFactory()
{
    static CoreAudioBackendFactory factory{};
    return factory;
}

bool CoreAudioBackendFactory::init() 
{ 
#if CAN_ENUMERATE
    sDeviceHelper.emplace();
#endif
    return true; 
}

bool CoreAudioBackendFactory::querySupport(BackendType type)
{ return type == BackendType::Playback || type == BackendType::Capture; }

auto CoreAudioBackendFactory::enumerate(BackendType type) -> std::vector<std::string>
{
    std::vector<std::string> outnames;
#if CAN_ENUMERATE
    auto append_name = [&outnames](const DeviceEntry &entry) -> void
    { outnames.emplace_back(entry.mName); };

    switch(type)
    {
    case BackendType::Playback:
        EnumerateDevices(PlaybackList, false);
        outnames.reserve(PlaybackList.size());
        std::for_each(PlaybackList.cbegin(), PlaybackList.cend(), append_name);
        break;
    case BackendType::Capture:
        EnumerateDevices(CaptureList, true);
        outnames.reserve(CaptureList.size());
        std::for_each(CaptureList.cbegin(), CaptureList.cend(), append_name);
        break;
    }

#else

    switch(type)
    {
    case BackendType::Playback:
    case BackendType::Capture:
        outnames.emplace_back(ca_device);
        break;
    }
#endif
    return outnames;
}

auto CoreAudioBackendFactory::createBackend(DeviceBase &device, BackendType type) -> BackendPtr
{
    if(type == BackendType::Playback)
        return BackendPtr{new CoreAudioPlayback{device}};
    if(type == BackendType::Capture)
        return BackendPtr{new CoreAudioCapture{device}};
    return nullptr;
}

alc::EventSupport CoreAudioBackendFactory::queryEventSupport(alc::EventType eventType, BackendType)
{
    switch(eventType)
    {
    case alc::EventType::DefaultDeviceChanged:
        return alc::EventSupport::FullSupport;

    case alc::EventType::DeviceAdded:
    case alc::EventType::DeviceRemoved:
        break;
    }
    return alc::EventSupport::NoSupport;
}
