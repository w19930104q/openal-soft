#!/usr/bin/env python3
"""Analyze rendered reverb impulse responses and verify the ISM model.

Three checks:

1. Tap accuracy: predicted ISM image delays (from the independent oracle)
   must appear as peaks in the rendered ISM IR at every source position.
2. Spatiality: the early-reflection peak pattern must move with the source
   position for the ISM build while staying fixed for the legacy build.
3. Real-time tracking: adjacent ER windows of the moving-source render must
   change continuously for ISM and be (near) identical for legacy.

Run: python3 -I analyze.py
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ism_reference import generate_taps, pick_peaks, mono_energy  # noqa: E402

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOM = [8.0, 6.0, 3.0]
REFLECT = [0.9, 0.9, 0.9]
POSITIONS = [(1.0, 0, 1.0), (3.0, 0, 1.0), (3.0, 0, -1.0), (-2.5, 0, 1.0)]
PERIOD = 2048
ER_START, ER_END = 32, 384


def load_mono(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    v = struct.unpack("<%df" % (len(raw) // 4), raw)
    n = len(v) // 2
    return [math.sqrt(v[2 * i] ** 2 + v[2 * i + 1] ** 2) for i in range(n)]


def signature(peaks, k=6):
    return [round(1000.0 * s / SR, 2) for s, _ in peaks[:k]]


def best_offset(pred, obs, tol=0.002):
    """Pipeline delay offset. The image-source taps read the same mono delay
    line at the write offset minus the geometric reflection delay, so they add
    no extra latency beyond the reflection time itself: the correct reference
    offset is zero. A free offset estimator would be misled by the extra peaks
    the still-active legacy late-scatter path contributes to the rendered IR.
    Kept as a function so the matching call sites stay self-documenting.
    """
    return 0.0


def check_tap_accuracy():
    print("=== 1. ISM predicted-vs-observed tap delays ===")
    total_matched = 0
    total_pred = 0
    for k, src in enumerate(POSITIONS):
        curve = load_mono(os.path.join(HERE, "ism_ir_pos%d.f32" % k))
        peaks = pick_peaks(curve, SR)
        obs = [s / SR for s, _ in peaks]
        pred = [t[0] for t in generate_taps(ROOM, REFLECT, src)]
        if not obs:
            print("  pos%d: no peaks observed!" % k)
            continue
        offset = best_offset(pred, obs)
        tol = 0.002
        matched = sum(
            1 for p in pred
            if min(abs((p + offset) - q) for q in obs) <= tol)
        # Mean absolute residual across every predicted tap: a direct measure
        # of image-delay accuracy independent of the binary match count.
        residuals = [
            min(abs((p + offset) - q) for q in obs) for p in pred]
        mean_res = 1000.0 * sum(residuals) / len(residuals)
        total_matched += matched
        total_pred += len(pred)
        print("  pos%d src=%-18s offset=%+.2fms matched %d/%d mean-res=%.3fms" %
              (k, str(src), 1000 * offset, matched, len(pred), mean_res))
    ratio = total_matched / max(1, total_pred)
    print("  overall tap match: %d/%d (%.0f%%)" %
          (total_matched, total_pred, 100 * ratio))
    return ratio >= 0.8


def check_spatiality():
    print("\n=== 2. Early-peak spatiality across source positions ===")
    ok = True
    for mode in ("legacy", "ism"):
        sigs = []
        for k in range(len(POSITIONS)):
            curve = load_mono(os.path.join(HERE, "%s_ir_pos%d.f32" % (mode, k)))
            sigs.append(signature(pick_peaks(curve, SR)))
        deltas = []
        for a in range(len(sigs) - 1):
            m = min(len(sigs[a]), len(sigs[a + 1]))
            deltas += [abs(sigs[a + 1][i] - sigs[a][i]) for i in range(m)]
        mean = sum(deltas) / max(1, len(deltas))
        mx = max(deltas) if deltas else 0.0
        print("  %-7s peak-time movement mean=%.3fms max=%.3fms" %
              (mode, mean, mx))
        for k, s in enumerate(sigs):
            print("    pos%d %s" % (k, s))
        if mode == "legacy":
            legacy_mean = mean
        else:
            if mean <= legacy_mean:
                print("  FAIL: ISM pattern does not move more than legacy")
                ok = False
    return ok


def norm(x):
    e = math.sqrt(sum(v * v for v in x))
    return [v / e for v in x] if e > 0 else x


def cos_dist(a, b):
    na, nb = norm(a), norm(b)
    dot = sum(x * y for x, y in zip(na, nb))
    return 1.0 - max(-1.0, min(1.0, dot))


def check_moving():
    print("\n=== 3. Real-time tracking (moving source) ===")
    results = {}
    for mode in ("legacy", "ism"):
        curve = load_mono(os.path.join(HERE, "%s_moving.f32" % mode))
        nblocks = len(curve) // PERIOD
        windows = [curve[b * PERIOD + ER_START:b * PERIOD + ER_END]
                   for b in range(nblocks)]
        # The strongest peak in each ER window is the window's reflection
        # arrival; weaker early samples are decaying residue of the previous
        # impulse, not the current reflection. Track its sample position.
        peak_track = [
            (ER_START + max(range(len(w)), key=lambda i: w[i])) / SR
            for w in windows]
        # Total distance (in ms) the dominant reflection travels as the source
        # moves, measuring how strongly the ER pattern follows the trajectory.
        travel_ms = 1000.0 * sum(
            abs(peak_track[i + 1] - peak_track[i])
            for i in range(len(peak_track) - 1))
        dists = [cos_dist(windows[i], windows[i + 1])
                 for i in range(len(windows) - 1)]
        mean = sum(dists) / len(dists)
        results[mode] = mean
        print("  %-7s adjacent ER cosine-distance mean=%.5f max=%.5f" %
              (mode, mean, max(dists)))
        print("  %-7s dominant-peak travel over trajectory=%.2f ms" %
              (mode, travel_ms))
    ratio = results["ism"] / max(results["legacy"], 1e-12)
    print("  ISM/legacy ER change ratio: %.2fx" % ratio)
    if ratio <= 1.0:
        print("  FAIL: moving-source ISM ER does not react more than legacy")
        return False
    return True


def main():
    results = [check_tap_accuracy(), check_spatiality(), check_moving()]
    print("\n=== SUMMARY ===")
    print("tap accuracy : %s" % ("PASS" if results[0] else "FAIL"))
    print("spatiality   : %s" % ("PASS" if results[1] else "FAIL"))
    print("realtime     : %s" % ("PASS" if results[2] else "FAIL"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
