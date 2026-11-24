#!/usr/bin/env python3
"""Independent image-source method (ISM) reference oracle.

Clean reimplementation of the shoebox image-source model used by the C++
reverb effect. It does not import any openal-soft code. The room is centered
on the listener (walls at +/- L_a/2), matching the C++ implementation which
works in head-relative coordinates with the listener at the origin.

Image source for index p along axis a (S_a given in centered coordinates,
s = S_a + L_a/2 the uncentered coordinate). Two Allen-Berkley families:

    plus family:  img = 2 p L_a + s,  per-axis order 2|p|
    minus family: img = 2 p L_a - s,  per-axis order 2p-1 (p>=1), 1-2p (p<=0)
    image_a = img - L_a/2
    path  d = ||image - listener||
    delay t = d / c
    gain  g = prod(R_a**axis_order) / d

The minus family p=0 is the direct (unreflected) source and is excluded.

Run with `python3 -I ism_reference.py ...`.
"""
import argparse
import math
import struct

SPEED_OF_SOUND = 343.3


def generate_taps(room, reflect, source, listener=(0.0, 0.0, 0.0),
                  max_order=1, base_delay=0.0, max_delay=0.3):
    taps = []
    for fx in (0, 1):
        for fy in (0, 1):
            for fz in (0, 1):
                family = (fx, fy, fz)
                for px in range(-max_order, max_order + 1):
                    for py in range(-max_order, max_order + 1):
                        for pz in range(-max_order, max_order + 1):
                            pidx = (px, py, pz)
                            image = [0.0, 0.0, 0.0]
                            wall_gain = 1.0
                            order = 0
                            for a in range(3):
                                p = pidx[a]
                                length = room[a]
                                s = source[a] + length * 0.5
                                if family[a] == 0:
                                    image[a] = 2 * p * length + s - length * 0.5
                                    axis_order = 2 * abs(p)
                                else:
                                    image[a] = 2 * p * length - s - length * 0.5
                                    # p=0 and p=1 are the two first-order wall
                                    # reflections; order grows by two per step.
                                    axis_order = (2 * p - 1) if p >= 1 \
                                        else (1 - 2 * p)
                                order += axis_order
                                wall_gain *= reflect[a] ** axis_order
                            if order == 0 or order > max_order:
                                continue
                            delta = [image[a] - listener[a] for a in range(3)]
                            path = math.sqrt(sum(d * d for d in delta))
                            if not path > 1e-4:
                                continue
                            delay = path / SPEED_OF_SOUND + base_delay
                            if delay > max_delay:
                                continue
                            gain = wall_gain / path
                            direction = tuple(d / path for d in delta)
                            taps.append((delay, gain, direction, order))
    taps.sort(key=lambda t: t[0])
    return taps


def read_f32_interleaved(path, nch=2):
    with open(path, "rb") as fh:
        raw = fh.read()
    vals = struct.unpack("<%df" % (len(raw) // 4), raw)
    return [vals[i:i + nch] for i in range(0, len(vals), nch)]


def mono_energy(frames):
    return [math.sqrt(sum(ch * ch for ch in f) / len(f)) for f in frames]


def pick_peaks(curve, sr, max_s=0.08, min_sep_s=0.0015, thresh=0.03):
    peak = max(curve[:int(max_s * sr)] or [1.0]) or 1.0
    min_sep = int(min_sep_s * sr)
    limit = min(int(max_s * sr), len(curve) - 1)
    out = []
    i = 1
    while i < limit:
        if curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1] \
                and curve[i] >= thresh * peak:
            if not out or (i - out[-1][0]) >= min_sep:
                out.append((i, curve[i]))
            i += min_sep
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="8,6,3")
    ap.add_argument("--reflect", default="0.9,0.9,0.9")
    ap.add_argument("--source", default="1,0,1")
    ap.add_argument("--listener", default="0,0,0")
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--base-delay", type=float, default=0.0)
    ap.add_argument("--ir")
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()

    room = [float(x) for x in args.room.split(",")]
    reflect = [float(x) for x in args.reflect.split(",")]
    source = tuple(float(x) for x in args.source.split(","))
    listener = tuple(float(x) for x in args.listener.split(","))

    taps = generate_taps(room, reflect, source, listener, args.order,
                         args.base_delay)
    print("# %d image taps (order<=%d) room=%s src=%s" %
          (len(taps), args.order, room, source))
    print("# idx order delay(ms) delay(samp) gain direction")
    for i, (delay, gain, direction, order) in enumerate(taps):
        print("%3d %4d %9.4f %10.2f %.5f (%.3f,%.3f,%.3f)" %
              (i, order, delay * 1000, delay * args.sr, gain, *direction))

    if args.ir:
        frames = read_f32_interleaved(args.ir)
        peaks = pick_peaks(mono_energy(frames), args.sr)
        print("\n# peaks from %s" % args.ir)
        for samp, amp in peaks[:20]:
            print("%9d %9.4f ms %.5f" % (samp, 1000.0 * samp / args.sr, amp))
        pred = sorted(t[0] for t in taps)
        obs = [p[0] / args.sr for p in peaks]
        print("\n# predicted-vs-observed (zero pipeline offset: ISM taps add "
              "no latency beyond reflection time)")
        tol = 0.002
        matched = sum(
            1 for ps in pred
            if min(abs(ps - r) for r in obs) <= tol)
        print("matched %d/%d taps" % (matched, len(pred)))


if __name__ == "__main__":
    main()
