#!/usr/bin/env bash
# Builds the ISM loopback test against the local static openal-soft and
# renders legacy / ISM (order 1) / ISM (order 2) impulse responses.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD="$ROOT/build"

export PATH=/home/liiing/bin:$PATH
CC=/home/liiing/bin/g++

echo "== Building test binary =="
$CC -O2 -std=c++20 -I"$ROOT/include" "$HERE/ism_ir_test.c" \
    "$BUILD/libopenal.a" -lm -lpthread -ldl \
    -o "$HERE/ism_ir_test"

SECONDS_ARG="${1:-1.0}"

echo "== Rendering LEGACY (fixed-tap) reverb IR =="
ALSOFT_CONF="$HERE/alsoft-legacy.conf" \
    "$HERE/ism_ir_test" "$HERE/legacy" "$SECONDS_ARG"

echo "== Rendering ISM order-1 reverb IR =="
ALSOFT_CONF="$HERE/alsoft-ism.conf" \
    "$HERE/ism_ir_test" "$HERE/ism" "$SECONDS_ARG"

echo "== Rendering ISM order-2 reverb IR =="
ALSOFT_CONF="$HERE/alsoft-ism-order2.conf" \
    "$HERE/ism_ir_test" "$HERE/ism2" "$SECONDS_ARG"

echo "== Done =="
ls -1 "$HERE"/*.f32
