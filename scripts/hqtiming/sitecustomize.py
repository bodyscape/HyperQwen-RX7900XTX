"""GPU time per category inside a vLLM decode step, from HIP events (the torch profiler has no GPU data on WSL2).

Active only with HQ_TIMING=1. Categories: every CUDAGraph.replay() (the PIECEWISE graph pieces: GEMMs, norms,
elementwise), and the eager custom ops listed in WATCH (GDN core, attention, KV-cache update). Totals are
written to /tmp/hq_timing.json every few hundred records.
"""
import os

if os.environ.get("HQ_TIMING") == "1":
    import atexit
    import json
    import time

    import torch

    WATCH = ("hq_rdna3_w4a16_gemm", "qwen_gdn_attention_core", "unified_attention_with_output", "unified_kv_cache_update", "gdn_attention")
    rec = []          # (name, start_event, end_event)
    tot = {}          # name -> [count, ms]
    t_first = [None]

    def _flush(force=False):
        keep = []
        for name, s, e in rec:
            if e.query():
                ms = s.elapsed_time(e)
                c = tot.setdefault(name, [0, 0.0])
                c[0] += 1
                c[1] += ms
            else:
                keep.append((name, s, e))
        rec[:] = keep
        out = {"wall_s": (time.time() - t_first[0]) if t_first[0] else 0, "totals": {k: {"calls": v[0], "ms": round(v[1], 3), "avg_us": round(1000 * v[1] / max(1, v[0]), 1)} for k, v in tot.items()}}
        with open("/tmp/hq_timing.json", "w") as f:
            json.dump(out, f, indent=1)

    import sys as _sys
    _patched = [False]

    def _patch_gdn_inner():
        """Time the individual kernels the GDN core calls, by replacing the names in its module globals."""
        if _patched[0]:
            return
        m = _sys.modules.get("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
        if m is None:
            return
        _patched[0] = True
        for attr, label in (("causal_conv1d_update", "gdn.conv_update"),
                            ("fused_sigmoid_gating_delta_rule_update", "gdn.gating_recurrent"),
                            ("fused_post_conv_prep", "gdn.post_conv_prep")):
            f = getattr(m, attr, None)
            if f is not None:
                setattr(m, attr, _timed_inner(label, f))

    def _timed_inner(name, fn):
        def wrapper(*a, **k):
            if torch.cuda.is_current_stream_capturing():
                return fn(*a, **k)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            r = fn(*a, **k)
            e.record()
            rec.append((name, s, e))
            return r
        return wrapper

    def _timed(name, fn):
        def wrapper(*a, **k):
            if torch.cuda.is_current_stream_capturing():
                return fn(*a, **k)
            if "gdn" in name:
                _patch_gdn_inner()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            r = fn(*a, **k)
            e.record()
            rec.append((name, s, e))
            if t_first[0] is None:
                t_first[0] = time.time()
            if len(rec) > 600:
                _flush()
            return r
        return wrapper

    _orig_replay = torch.cuda.CUDAGraph.replay

    def _replay(self, *a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = _orig_replay(self, *a, **k)
        e.record()
        rec.append(("graph_replay", s, e))
        if t_first[0] is None:
            t_first[0] = time.time()
        if len(rec) > 600:
            _flush()
        return r

    torch.cuda.CUDAGraph.replay = _replay

    _orig_impl = torch.library.Library.impl

    def _impl(self, op_name, fn, dispatch_key="", **kw):
        try:
            if getattr(self, "ns", "") == "vllm" and any(w in str(op_name) for w in WATCH):
                fn = _timed(str(op_name), fn)
        except Exception:
            pass
        return _orig_impl(self, op_name, fn, dispatch_key, **kw)

    torch.library.Library.impl = _impl

    # a reset hook: touching /tmp/hq_timing.reset clears the counters at the next record
    _orig_flush = _flush

    def _flush(force=False):  # noqa: F811
        if os.path.exists("/tmp/hq_timing.reset"):
            os.remove("/tmp/hq_timing.reset")
            rec.clear()
            tot.clear()
            t_first[0] = time.time()
            return
        _orig_flush(force)
