"""Build the W4A16 GEMV op (rocm/w4a16_gemv.cu) into a shared library that registers hq::w4a16_gemv.

    PYTORCH_ROCM_ARCH=gfx1100 python rocm/build_gemv.py /opt/hq

Needs hipcc and the torch headers, not a GPU. The result is OUT_DIR/w4a16_gemv.so; the patched
rdna3_w4a16.py loads it with torch.ops.load_library (HQ_W4A16_GEMV_SO overrides the path).
"""
import os
import sys

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1100")
from torch.utils.cpp_extension import load  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "/opt/hq"
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
load(
    name="w4a16_gemv",
    sources=[os.path.join(here, "w4a16_gemv.cu")],
    build_directory=out,
    extra_cuda_cflags=["-O3"],
    is_python_module=False,
    verbose=False,
)
print(os.path.join(out, "w4a16_gemv.so"))
