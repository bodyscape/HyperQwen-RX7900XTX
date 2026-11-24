#!/bin/bash
# Build libhsa-runtime64 (ROCr) from the rocm-systems tag matching the image, with the AsyncEventsLoop poll backoff
# (ROCm/rocm-systems#7898) applied, for WSL2 (librocdxg). Run inside the hyperqwen-rocm image:
#   docker run --rm -v <this dir>:/work --entrypoint bash hyperqwen-rocm:local /work/build_rocr.sh
# Output: /work/out/libhsa-runtime64.so.1.* and a libhsa-runtime64.so.1 link, plus /work/out/BUILD_INFO.txt
set -euo pipefail
TAG=${TAG:-rocm-7.2.3}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
WORK=/work/build-rocr
OUT=/work/out
PATCH=/work/rocr-7.2.0-async-events-poll-backoff.patch

HOST_ROCM="$(cat "$ROCM_ROOT/.info/version")"
echo "host ROCm: $HOST_ROCM, building tag: $TAG"
[[ "$TAG" == "rocm-$HOST_ROCM" ]] || { echo "tag $TAG does not match host ROCm $HOST_ROCM"; exit 1; }

rm -rf "$WORK" "$OUT"; mkdir -p "$WORK" "$OUT"
cd "$WORK"
git init -q src && cd src
git remote add origin https://github.com/ROCm/rocm-systems.git
git sparse-checkout set projects/rocr-runtime
if [[ -n "${COMMIT_ID:-}" ]]; then
  # The runtime shipped in the image says "1.18.0-local-build-<sha>": build from that exact revision, not the tag.
  git fetch -q --depth 1 --filter=blob:none origin "$COMMIT_ID"
  git -c advice.detachedHead=false checkout -q FETCH_HEAD
else
  git fetch -q --depth 1 --filter=blob:none origin "refs/tags/$TAG:refs/tags/$TAG"
  git -c advice.detachedHead=false checkout -q "$TAG"
fi
COMMIT="$(git rev-parse HEAD)"
echo "source commit: $COMMIT"

# The backport was cut against 7.2.0; apply it as is and stop if the context moved.
if git apply --check "$PATCH" 2>/tmp/apply.err; then
  git apply "$PATCH"; echo "patch applied cleanly"
else
  echo "PATCH DOES NOT APPLY to $TAG:"; cat /tmp/apply.err; exit 2
fi
git diff --stat

# ROCm's LLVM ships no CMake package and ROCr only needs two imported executables from it: shims, as in the original recipe.
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
env -i HOME=/root PATH=/usr/local/bin:/usr/bin:/bin cmake -S "$WORK/src/projects/rocr-runtime" -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
  -DClang_DIR="$SHIM/clang" -DLLVM_DIR="$SHIM/llvm" -DCMAKE_SKIP_RPATH=ON -DBUILD_ROCR_TESTS=OFF > "$WORK/configure.log" 2>&1 \
  || { echo "configure failed"; tail -30 "$WORK/configure.log"; exit 3; }
env -i HOME=/root PATH=/usr/local/bin:/usr/bin:/bin cmake --build "$WORK/build" -j"$(nproc)" > "$WORK/build.log" 2>&1 \
  || { echo "build failed"; tail -40 "$WORK/build.log"; exit 4; }

LIB="$(find "$WORK/build" -name 'libhsa-runtime64.so.1.*' -type f | head -1)"
[[ -n "$LIB" ]] || { echo "libhsa-runtime64 not produced"; exit 5; }
cp "$LIB" "$OUT/"
ln -sfn "$(basename "$LIB")" "$OUT/libhsa-runtime64.so.1"
echo "missing dependencies: $(env -i PATH=/usr/bin ldd "$OUT/libhsa-runtime64.so.1" | grep -c 'not found' || true)"
{ echo "tag=$TAG commit=$COMMIT"; echo "lib=$(basename "$LIB") sha256=$(sha256sum "$LIB" | cut -d' ' -f1)"; echo "stock=$(sha256sum "$ROCM_ROOT/lib/libhsa-runtime64.so.1" | cut -d' ' -f1)"; } > "$OUT/BUILD_INFO.txt"
cat "$OUT/BUILD_INFO.txt"; ls -l "$OUT"
