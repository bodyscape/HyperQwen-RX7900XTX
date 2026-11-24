#!/bin/bash
S=/opt/rocm/lib/libhsa-runtime64.so.1
P=/work/out/libhsa-runtime64.so.1
echo "stock exports $(nm -D --defined-only $S | wc -l) symbols, patched $(nm -D --defined-only $P | wc -l)"
echo "--- exported by stock but missing from patched:"
comm -23 <(nm -D --defined-only $S | awk '{print $3}' | sort -u) <(nm -D --defined-only $P | awk '{print $3}' | sort -u) | head -20
echo "--- exported by patched but not stock:"
comm -13 <(nm -D --defined-only $S | awk '{print $3}' | sort -u) <(nm -D --defined-only $P | awk '{print $3}' | sort -u) | head -10
echo "--- hsa_amd_signal_create_v2 in the 7.2.3 tag source:"
grep -rn "hsa_amd_signal_create_v2" /work/build-rocr/src/projects/rocr-runtime --include=*.h --include=*.cpp -l | head -5
echo "--- stock version string / build id:"
strings $S | grep -E "^ROCR_|1\.18\.[0-9]+" | head -5
readelf -n $S | grep -i "build id"
echo "--- source tree version files:"
grep -rn "HSA_VERSION\|rocr_version\|VERSION_PATCH" /work/build-rocr/src/projects/rocr-runtime/CMakeLists.txt | head -8
cd /work/build-rocr/src && git log -1 --format='%H %cd %s' | head -2
