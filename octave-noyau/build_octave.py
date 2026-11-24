"""Build the copied Octave decode into a library that registers hq::octave_decode (symbols renamed, gfx1100).

    PYTORCH_ROCM_ARCH=gfx1100 python build_octave.py <src_dir> <out_dir>
"""
import os
import sys

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1100")
from torch.utils.cpp_extension import load  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
RENAME = ["octave=hqoct", "octave_wmma=hqoct_wmma", "octave_decode=hq_octave_decode",
          "octave_decode_wmma=hq_octave_decode_wmma", "octave_decode_mfma=hq_octave_decode_mfma",
          "octave_reduce_mfma=hq_octave_reduce_mfma", "octave_cache_store=hq_octave_cache_store",
          "octave_decode_sparse=hq_octave_decode_sparse", "octave_window_store=hq_octave_window_store",
          "octave_rotate=hq_octave_rotate", "octave_slot_bytes=hq_octave_slot_bytes"]
load(
    name="hq_octave",
    sources=[os.path.join(src, f) for f in ("octave_attn.cu", "octave_decode_wmma_rdna3.cu", "hq_octave_binding.cu")],
    build_directory=out,
    extra_cflags=["-O3", "-fvisibility=hidden"] + ["-D" + r for r in RENAME],
    extra_cuda_cflags=["-O3", "-fvisibility=hidden"] + ["-D" + r for r in RENAME],
    extra_ldflags=["-Wl,-Bsymbolic"],
    is_python_module=False,
    verbose=False,
)
print(os.path.join(out, "hq_octave.so"))
