#!/bin/bash
# Build libhsa-runtime64 (ROCr) with the AsyncEventsLoop poll backoff (ROCm/rocm-systems#7898) for WSL2, so that a
# process that only initialises the GPU no longer spins two threads at 100% CPU (ROCm/librocdxg#60).
#
# The image's runtime says "1.18.0-local-build-af70edc": it was built from that rocm-systems commit, not from the
# rocm-7.2.3 tag (the tag lacks hsa_amd_signal_create_v2, which the image's libamdhip64 needs). So this builds exactly
# that commit, applies the backport (cut for 7.2.0; it applies cleanly here) and refuses to produce a library unless it
# exports the same symbols as the one it replaces. Any failure leaves OUT empty and the image on the stock runtime.
#
#   WORK=/tmp/rocr OUT=/opt/hq/rocr bash rocm/rocr/build_rocr.sh
# Needs git, cmake, a C++ compiler, libelf/libdrm/libnuma headers, ROCm's clang and llvm-objcopy, and network.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT_ID=af70edc91f1dc5d228fd4c076df99e65e49c4521   # "1.18.0-local-build-af70edc"
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
WORK=${WORK:-/tmp/rocr-build}
OUT=${OUT:-/opt/hq/rocr}
PATCH="$HERE/async-events-poll-backoff.patch"
STOCK="$ROCM_ROOT/lib/libhsa-runtime64.so.1"

# Only for the runtime this commit produced.
BUILT_FROM="$(strings "$STOCK" | grep -o 'local-build-[0-9a-f]*' | head -1 | sed 's/local-build-//')"
[[ "$BUILT_FROM" == "${COMMIT_ID:0:${#BUILT_FROM}}" && -n "$BUILT_FROM" ]] || {
  echo "rocr: the image's libhsa-runtime64 comes from '${BUILT_FROM:-unknown}', not ${COMMIT_ID:0:7}: not building"; exit 10; }

rm -rf "$WORK" "$OUT"; mkdir -p "$WORK" "$OUT"
cd "$WORK"
git init -q src && cd src
git remote add origin https://github.com/ROCm/rocm-systems.git
git sparse-checkout set projects/rocr-runtime
git fetch -q --depth 1 --filter=blob:none origin "$COMMIT_ID"
git -c advice.detachedHead=false checkout -q FETCH_HEAD
[[ "$(git rev-parse HEAD)" == "$COMMIT_ID" ]] || { echo "rocr: wrong checkout"; exit 11; }
git apply --check "$PATCH" && git apply "$PATCH"

# ROCm's LLVM ships no CMake package and ROCr only needs two imported executables from it.
SHIM="$WORK/llvm-shim"; mkdir -p "$SHIM/clang" "$SHIM/llvm"
cat > "$SHIM/clang/ClangConfig.cmake" <<EOF
if(NOT TARGET clang)
  add_executable(clang IMPORTED GLOBAL)
  set_target_properties(clang PROPERTIES IMPORTED_LOCATION "$ROCM_ROOT/llvm/bin/clang")
endif()
EOF
cat > "$SHIM/llvm/LLVMConfig.cmake" <<EOF
if(NOT TARGET llvm-objcopy)
  add_executable(llvm-objcopy IMPORTED GLOBAL)
  set_target_properties(llvm-objcopy PROPERTIES IMPORTED_LOCATION "$ROCM_ROOT/llvm/bin/llvm-objcopy")
endif()
EOF
CLEAN=(env -i HOME=/root PATH=/usr/local/bin:/usr/bin:/bin)
"${CLEAN[@]}" cmake -S "$WORK/src/projects/rocr-runtime" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT" -DClang_DIR="$SHIM/clang" -DLLVM_DIR="$SHIM/llvm" -DCMAKE_SKIP_RPATH=ON \
  -DBUILD_ROCR_TESTS=OFF > "$WORK/configure.log" 2>&1 || { tail -30 "$WORK/configure.log"; exit 12; }
"${CLEAN[@]}" cmake --build "$WORK/build" -j"$(nproc)" > "$WORK/build.log" 2>&1 || { tail -40 "$WORK/build.log"; exit 13; }

LIB="$(find "$WORK/build" -name 'libhsa-runtime64.so.1.*' -type f | head -1)"
[[ -n "$LIB" ]] || { echo "rocr: no library produced"; exit 14; }
# Same exported symbols as the library it replaces, and every dependency resolved.
if ! diff <(nm -D --defined-only "$STOCK" | awk '{print $3}' | sort -u) <(nm -D --defined-only "$LIB" | awk '{print $3}' | sort -u) > "$WORK/symdiff.txt"; then
  echo "rocr: exported symbols differ from the stock runtime:"; head "$WORK/symdiff.txt"; exit 15; fi
[[ "$("${CLEAN[@]}" ldd "$LIB" | grep -c 'not found' || true)" == 0 ]] || { echo "rocr: unresolved dependencies"; exit 16; }

cp "$LIB" "$OUT/"
ln -sfn "$(basename "$LIB")" "$OUT/libhsa-runtime64.so.1"
echo "rocr: patched libhsa-runtime64 staged in $OUT (commit ${COMMIT_ID:0:7}, backport ROCm/rocm-systems#7898)"
rm -rf "$WORK"
