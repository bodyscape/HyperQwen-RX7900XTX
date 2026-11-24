/app/venv/bin/python - <<'EOF' 2>&1 | grep -vE 'UserWarning|raw_cnt|agent.cpp'
from typing import get_args
import torch
from vllm.config.cache import CacheDType
dt = [d for d in get_args(CacheDType) if "octave" in d or "turbo" in d or "int8" in d or "int4" in d or "kvarn" in d]
print("cache dtypes (octave/turbo/int8/int4/kvarn):", dt)
from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
print("backends:", sorted(b.name for b in E if "OCTAVE" in b.name or "RDNA" in b.name or "TURBO" in b.name or "TRITON" in b.name or b.name == "ROCM_ATTN"))
import vllm._C
ops = [n for n in dir(torch.ops._C) if "octave" in n.lower()]
print("octave ops in _C:", ops)
try:
    from vllm.v1.attention.backends.rocm_octave_attn import RocmOctaveBackend  # noqa
    print("rocm_octave_attn module: importable")
except Exception as e:
    print("rocm_octave_attn module:", type(e).__name__, str(e)[:120])
EOF
