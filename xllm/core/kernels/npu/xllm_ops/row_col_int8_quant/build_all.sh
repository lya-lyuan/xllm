#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source env.sh
mkdir -p logs
cmake --preset=default
cmake --build build_out --target binary package -j "${JOBS:-8}"
mkdir -p "$ROWCOL_PREFIX"
bash build_out/custom_opp_ubuntu_aarch64.run --quiet --install-path="$ROWCOL_PREFIX"
source env.sh
cmake -S direct -B build_direct -DCMAKE_BUILD_TYPE=Release
cmake --build build_direct -j "${JOBS:-8}"
MAX_JOBS="${JOBS:-2}" python3.12 python/build_binding.py
cmake -S examples -B build_examples -DCMAKE_BUILD_TYPE=Release
cmake --build build_examples -j 2
