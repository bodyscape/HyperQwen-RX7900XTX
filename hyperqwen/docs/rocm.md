# AMD Radeon RX 7900 XTX (ROCm), on Windows through Docker Desktop

**Status: experimental. Qwen3.8-27B boots and answers on it; MTP speculation with 3 drafts and the single-read
W4A16 GEMV decodes 93-98 tok/s on the repo's real-prompt cohort at 8k, and the 3-bit Octave KV cache reaches a 150k
window; on the repo's long-context benchmark (labd) it decodes 88 tok/s at 74k and 79 tok/s at 147k, at parity
with the RTX 3090's 84.7 at ~66k and 78.5 at ~100k (one machine). The rebuilt image, defaults only, reads 97.8 /
104.9 tok/s on the single-stream cohort (default sampling / greedy).** This page says
what has been run on the card, what has not, and how to start. Every number elsewhere in this repo is from an RTX 3090
and does not carry over.

[← back to the main README](../README.md)

The 7900 XTX (gfx1100, RDNA3, 24 GB) runs vLLM through ROCm. On Windows the card reaches Linux through WSL2:
no `/dev/kfd`, a `/dev/dxg` device, and AMD's `librocdxg` bridging ROCm to the Windows driver. Docker Desktop's
WSL2 engine passes that device into a container.

## What has been checked (2026-11-19, one machine, RX 7900 XTX, Windows 11, Docker Desktop)

| Check | Result |
|---|---|
| torch in the container sees the card | `AMD Radeon RX 7900 XTX`, gfx1100, 23.94 GiB |
| fp16 matmul 8192³ / device-to-device copy | 81.5 TFLOPS / 720 GB/s (one run each, copy counts read and write) |
| vLLM 0.33.0 serves a small model | Qwen3-0.6B on `TRITON_ATTN`, piecewise and full graphs captured |
| Qwen3-0.6B decode | 368-376 tok/s one stream, 781 tok/s aggregate at 8 concurrent (256-token completions) |
| `patches/series` on the ROCm 7.2 wheel | all 65 patches apply at `--fuzz 0` (a few at an offset) |
| `Dockerfile.rocm` builds | patches, KVarN, `verify.sh --install` and the CPU tests pass |
| `start-rocm.sh` | dry run (`PRINT_ARGV=1`) gives the expected argv for `SPEC=off` and `SPEC=mtp` |
| `docker compose up -d rocm` end to end | prepare (download, requantize), `verify.sh`, then the server answers `/health`. First boot from cold caches: about 4.5 minutes after the container started |

Qwen3.8-27B (`Qwen3.8-27B-W4A16-AutoRound-fast`), `SPEC=off`, `CTX=fast`, `MAX_LEN=8192`, `GPU_UTIL=0.90`,
`TRITON_ATTN`, bf16 KV, one run on 2026-11-19 with nothing else on the GPU but the desktop:

| | |
|---|---|
| W4A16 GEMM | `RDNA3W4A16LinearKernel` selected by vLLM for `CompressedTensorsWNA16` (engine log) |
| other kernels | Triton/FLA GDN prefill; `TRITON_ATTN`; piecewise graphs captured (7 sizes, 6 s, 0.13 GiB) |
| weights / KV pool | 13.91 GiB loaded in 79 s (from a bind-mounted NTFS folder) / 6.74 GiB, 75,093 tokens at 400 tokens a block |
| answers | "La capitale du Danemark est Copenhague." and 17 × 23 = 391 (two prompts, greedy) |
| decode, one stream | 43.5-43.8 tok/s (256 tokens, three runs, `ignore_eos`) |
| decode, 2 concurrent | 64.5 tok/s aggregate (one run) |
| prefill | one 5,109-token prompt, cold: 877 tok/s (one run, includes HTTP and tokenization) |
| VRAM afterwards | Windows reports 23.77 of about 24 GiB dedicated memory in use, and 0.62 GiB of shared (system RAM) GPU memory |

These are ad-hoc single requests, not `bench/run_benchmarks.sh`, so they cannot be set against the RTX 3090 tables.

The same, with `SPEC=mtp` (4 drafts, the launcher's default for MTP; `DRAFT_ATTN_BACKEND=TRITON_ATTN`), same day:

| | `SPEC=off` | `SPEC=mtp` |
|---|---|---|
| KV pool | 75,093 tokens | 47,308 tokens |
| decode, one stream, a long story at temperature 0 | 43.7 tok/s | 37.7-40.2 tok/s, MTP acceptance 20.7% (589 of 2,840 draft tokens) |
| decode, one stream, counting 1 to 200 | not run | 100.9 tok/s, acceptance 98.6% (205 of 208) |
| decode, 2 concurrent | 64.5 tok/s | 45.3 tok/s |
| prefill, ~5.5k tokens, no prefix-cache hit | 877 tok/s (one run) | 790 and 826 tok/s; the first long prompt after boot took 20 s (250 tok/s), which is a one-time warmup |

So MTP runs and its drafts verify correctly, but with 4 drafts it was slower than no speculation on free prose.

### Why 4 drafts lost, and the fix (2026-11-20)

The cause is the W4A16 GEMM, not launch overhead. A profiler trace of a decode step (graphs replayed, FULL capture)
shows the worker's CPU busy for 3.6 ms of a 43 ms step: the step is GPU-bound. `gptq_gemm_rdna3`, the kernel vLLM
picks for this checkpoint, reads the weights once for up to 4 activation rows and again for the 5th. Microbenchmark on
the card, bf16 activations, group 128, weights rotated to stay out of the Infinity Cache:

| GEMM (K -> N) | M=1 | M=4 | M=5 | M=8 |
|---|---|---|---|---|
| gate_up 5120 -> 34816 | 127 us | 144 us | 228 us | 207 us |
| down 17408 -> 5120 | 79 us | 86 us | 124 us | 121 us |
| qkv-like 5120 -> 8192 | 38 us | 44 us | 62 us | 69 us |

Four drafts verify M=5. Three drafts verify M=4. The `gptq_gemm_rdna3_wmma` variant is flat but slow (about 2x the
M=1 time), and the hybrid HIP skinny kernel (`wvSplitK_int4_g`) refuses `down` at M>=3 (K*M over the LDS limit).
The launcher now defaults to `DRAFT_TOKENS=3`. Measured with `bench/run_benchmarks.sh single` cut to concurrency 1
(8 real prompts, 1024 tokens each, one stream):

| | decode (C/meanTPOT) | tokens per step |
|---|---|---|
| `SPEC=mtp`, 4 drafts | 64.5 tok/s (T=default), 65.9 (T=0) | 2.80 / 2.87 |
| `SPEC=mtp`, 3 drafts | 82.3 (T=default), 85.1 (T=0) | 2.60 / 2.63 |
| `SPEC=mtp`, 3 drafts, vLLM's default graph mode | 83.0, 84.3 | 2.60 / 2.63 |

The step went from about 43.5 ms to about 31 ms. Forcing `CUDAGRAPH_MODE=FULL` changed nothing (vLLM already captures
FULL plus PIECEWISE), so the launcher leaves it alone. The README's 127 tok/s (RTX 3090, same protocol) is not reached:
a kernel that reads the weights once for M up to 8 would allow more drafts, and the GEMMs run at 550-730 GB/s of the
960 GB/s the card has.

**Not checked:** `SPEC=dflash2`; KVarN on ROCm; MTP under `CUDAGRAPH_MODE=FULL` with the prefix-cache residue sweep
(`bench/residue_sweep.py`; the launcher does not force FULL); quality (no GSM8K, no perplexity); a soak; the effect of the 129 start-up lines `Attempted to load weight ...
weight_scale with dtype torch.float16 into parameter with dtype torch.bfloat16` (scales are converted on load, and
whether that moves the output is unmeasured); the CPU-pinned-memory path (`is_pin_memory_available()` printed `True` in
one run and the WSL warning appeared in another); whether the 0.62 GiB of shared GPU memory is a spill; another
machine, another driver.

## Requirements

- Windows 11, the AMD Adrenalin driver with WSL2 support (AMD documents 26.2.2 or newer; a reporter on another RDNA4
  card needed 26.8.1 or newer for long compute jobs to stay up).
- WSL2 with an Ubuntu distro, and Docker Desktop on its WSL2 engine.
- `%UserProfile%\.wslconfig` with more memory than the default: loading and requantizing the model needs about 20 GB
  of RAM and the 20 GB many machines start with is tight. `memory=40GB`, then `wsl --shutdown`.
- **Close whatever else uses the GPU.** On the checked machine Windows already held 8.5 GiB of the 24: the desktop
  (about 1.8 GiB), a speech server (4 GiB), a browser. vLLM refuses to start when the free memory is below
  `GPU_UTIL` of the total. The Qwen3.8-27B weights are about 16 GiB, so there is not much room even with the card empty.

## Run

```bash
docker compose up -d --build rocm
docker compose logs -f rocm
```

Do not copy `.env.example` as it is: it is the RTX 3090 file and sets `SPEC=dflash2`. For this card copy
`.env.rocm.example` (PowerShell: `Copy-Item .env.rocm.example .env`): it holds the measured 150k setup
(`HQ_ROCM_KV=octave`, `MAX_LEN=150000`, `MAX_SEQS=1`, `GPU_UTIL=0.86`, MTP with 3 drafts) and the knobs of the RDNA3
kernels, each commented.

**What needs a rebuild and what does not.** `.env` is read at every container start, and the `rocm` service mounts
the launcher scripts from this checkout (`start-rocm.sh`, `single-user/`, `launcher_common.sh`, `resolve_config.sh`,
`resolve_api_key.sh`, `docker/entrypoint.sh`), so a change to either takes effect with
`docker compose up -d --force-recreate rocm`. Only the compiled parts need `docker compose build rocm`: the kernels in
`rocm/` and the vLLM patches in `patches/`.

Name the service. A bare `docker compose --profile rocm up` also starts `prepare`, which is the CUDA image.

The first start builds `Dockerfile.rocm` (it pulls vLLM's ROCm 7.2 image, about 11 GB compressed), downloads the model
(about 20 GB) and requantizes it on the CPU, as the NVIDIA profiles do, into `./models`. The server listens on
**port 19020**: Windows reserves 17931-18030 for Hyper-V on many machines (`netsh int ipv4 show excludedportrange
protocol=tcp`), and the usual 18020 then fails to bind. Set `PORT` in `.env` to change it.

Defaults of `start-rocm.sh`, all unmeasured on the 27B and meant to be raised one at a time:

| | default | note |
|---|---|---|
| `SPEC` | `mtp` | `SPEC=off` for no speculation. `dflash2` needs the drafter (`prepare`) and has not been run |
| `DRAFT_TOKENS` | 3 | the RDNA3 W4A16 kernel holds 4 rows per pass; see "Why 4 drafts lost" above |
| `HQ_ROCM_KV` | `bf16` | `octave` selects the 3-bit `octave_k3v3_compact` cache on `ROCM_OCTAVE`, target and drafter |
| `MAX_LEN` / `MAX_SEQS` | 8192 / 2 | the KV pool is whatever `GPU_UTIL` leaves after the weights: 75,093 tokens in the run above (bf16, `SPEC=off`) |
| `GPU_UTIL` | 0.86 | leaves ~1 GiB for the Windows desktop on the same card; at 0.90 long prefills filled the card, WDDM paged GPU memory to RAM and the PC froze (see Troubleshooting). The 3090 byte pins (`KV_MEM`) are not used |
| attention | `TRITON_ATTN`, bf16 KV | named explicitly, for the target and the drafter (`DRAFT_ATTN_BACKEND`) |
| `SPEC_ATTN` | 0 | the split-KV verify kernel is untested on ROCm |
| `HQ_ROCM_W4A16_KERNEL` | `auto` | `exllama` disables the Conch kernel so vLLM falls to its exllama GEMM |

## What does not run on ROCm

`start-rocm.sh` refuses these rather than boot something else under their name.

| Setting | Why |
|---|---|
| `INT8_ACT`, `PREFILL_ATTN` | the Marlin W4A8 int8 GEMM and the int8-QK prefill path are CUDA kernels tuned for sm86 |
| `PTH_FA` | `pthfa/` is inline PTX (`mma.sync`, `ldmatrix`, `cp.async`) and has no AMD equivalent |
| `FA2_FP8KV` | `fa2/` is CUTLASS/CuTe FlashAttention-2 for CUDA, and gfx11 has no FP8 hardware |
| `LSA` | rides the `CUSTOM` backend that `PTH_FA` takes; untested here |
| `EMBED_UVA`, `KV_OFFLOAD_GB` | CUDA UVA and pinned-copy paths; untested on ROCm under WSL2 |
| FlashInfer | not built for ROCm. `VLLM_USE_FLASHINFER_SAMPLER=0` is already the launchers' default |

The patches for those features are still in the image. Each one changes nothing unless its env knob is set, and the
launcher never sets it. Do not set `--kv-cache-dtype fp8` by hand: gfx11 converts fp8 in software and vLLM issue
#56992 measures it slower than bf16 on this card.

## Long context

`CTX=fast` only, with `HQ_ROCM_KV=octave MAX_LEN=150000 MAX_SEQS=1` for long windows. That runs vLLM's `ROCM_OCTAVE`
3-bit backend (HIP kernels for gfx1100, head size 256 with 64 rotary dims, which is this model's attention shape);
`GPU_UTIL=0.90` needs the card's other GPU users closed (a 4 GB process left 18.2 GiB free and the boot refused).
Measured with `SPEC=mtp`, 3 drafts, `temperature=0`, a synthetic document with one needle, 256 new tokens, the whole
prompt but the last ~1k tokens read from the prefix cache (so the figures carry that short prefill; the pure decode is
about 20-30% higher). The needle was found at every depth:

| prompt | cold prefill | decode | MTP acceptance | tokens per step | step |
|---|---|---|---|---|---|
| 29.7k tokens | 963 tok/s (31 s) | 40.0 tok/s | 28% | 1.85 | ~35 ms |
| 70.4k tokens | 832 tok/s (85 s) | 44.5 tok/s | 38% | 2.13 | ~41 ms |
| 129.3k tokens | 700 tok/s (185 s) | 35.4 tok/s | 35% | 2.04 | ~48 ms |

Those figures carry the short prefill of the uncached tail and the free-form text MTP predicts poorly, so they are low.
A cleaner measure subtracts the prefill: the same cached prompt at `max_tokens` 64 and 320, `ignore_eos`, and the
difference over the extra 256 tokens. With 192 KV splits (below), `SPEC=mtp`, 3 drafts, `temperature=0`, before the
W4A16 GEMV below. The text is still the free-form explanation, so these tok/s are not comparable with the RTX 3090
numbers, which come from `bench/labd_bench.py`; the table after the GEMV section has that comparison:

| prompt | pure decode | tokens per step | step |
|---|---|---|---|
| 70.4k tokens | 63.0 tok/s | 2.29 | 36.3 ms |
| 129.3k tokens | 44.7 tok/s | 1.80 | 40.1 ms |

The step grows from ~31 ms at 8k to ~40 ms at 129k. The Octave backend already runs a split-KV decode kernel
(`octave_decode`, a block per KV head and split, consecutive query rows of one request sharing each K/V read), so
nothing had to be ported for the verify step. Two measurements of that kernel on the card (one layer, 4 KV heads,
head 256, compact 3-bit cache at 196 bytes per token and head):

- **Splits.** vLLM's `tq_max_kv_splits_for_cuda_graph` defaults to 32, too few to fill 48 CUs with one sequence.
  At 125k, 4 verify rows: 573 us at 32 splits, ~400 us at 192 or more. One draft row: 429 us at 32, 210 us at 192+.
  The WMMA tier caps the split count at 192 (`kSqWmmaSplits`), so `HQ_ROCM_KV=octave` sets 192. No loss at short
  contexts (1k: 58 -> 71 us, 8k: 62 -> 53 us). A server A/B was attempted but another GPU process (whisper-server)
  was likely running during it (cold prefill fell from 700 to 638 tok/s), so no end-to-end gain is claimed.
- **Rows per pass.** The kernel reads K/V once for up to 2 query rows: at 125k, 1 row 210 us, 2 rows 249 us, 3 or 4
  rows ~420 us (two passes). A variant holding 4 rows (24 attention rows, two WMMA tiles sharing each dequantized K/V
  tile) would take the 4-row verify from ~400 to ~250 us per layer, about 2.7 ms of a 40 ms step at 125k. Not done.

Without speculation the same backend decoded an estimated 40 / 35 / 29 tok/s at 30k / 68k / 125k. KVarN and the
`int8_per_token_head` cache have not been run on ROCm.

### The single-read GEMV (2026-11-20)

`rocm/w4a16_gemv.cu` is a W4A16 GEMV for 1 to 8 activation rows, bf16, on the same weight layout as
`gptq_gemm_rdna3` (exllama-shuffled int32 `[K/8, N]`, `qzeros`, bf16 scales, group 128), so no second copy of the
weights. Every weight word is read once for all rows. A whole 128-wide group is accumulated as
`sum((128 + nibble) * a)` with `v_dot2_f32_bf16`, and the scale and zero-point correction are applied once per
group: about 4 dot2 per weight word and row, against ~2 FMAs per row and weight before. A block is 8 waves that
split K and reduce through LDS; narrow outputs also split K across blocks (fixed-order second pass, so the result is
deterministic). The wave index is passed through `readfirstlane` so the activation reads become scalar loads, which
is what made M=6 to 8 faster than the old kernel instead of slower.
`patches/rocm-w4a16-gemv.patch` routes `RDNA3W4A16LinearKernel` to it for M<=8 and bf16 (as a registered op, so
`torch.compile` and HIP graphs see it); `Dockerfile.rocm` builds it into `/opt/hq/w4a16_gemv.so` with
`rocm/build_gemv.py` (hipcc and the torch headers, no GPU). `HQ_W4A16_GEMV=0` turns it off, and a missing library
or fp16 activations fall back to the original kernel.

Correctness: against an independent fp32 dequantize-and-matmul, the new kernel and `gptq_gemm_rdna3` have the same
error (0.011-0.014 of the mean |output|, which is bf16 rounding of the output) for M=1,4,5,8 on four layer shapes
with random zero points. Speed, one layer, us per call, `gptq_gemm_rdna3` -> new:

| M | gate_up | down | qkv-like | o |
|---|---|---|---|---|
| 1 | 115 -> 108 | 70 -> 56 | 35 -> 31 | 27 -> 14 |
| 4 | 141 -> 121 | 86 -> 64 | 45 -> 42 | 37 -> 19 |
| 5 | 221 -> 125 | 125 -> 67 | 59 -> 46 | 47 -> 22 |
| 8 | 212 -> 157 | 123 -> 95 | 64 -> 64 | 55 -> 39 |

Whole layer at M=4: 308 -> 246 us; at M=5: 452 -> 260 us. In the server (`SPEC=mtp`, 3 drafts, one stream, the
repo's real-prompt cohort): **93.1 tok/s at the default temperature and 98.5 at T=0**, against 83.0 and 84.3
with the old kernel; the step went from ~31 to ~26 ms. 4 drafts give 94.7 and 95.5, no better, so 3 stays.
With `HQ_ROCM_KV=octave` (256 requested, 192 used, 3 drafts), pure decode at depth on the free-form text, step time
in brackets: 70k 59.0 tok/s (33.7 ms, was 36.3), 125k 51.6 tok/s (37.3 ms, was 40.1). The needle is still found at
both depths and a short chat answers correctly.

**Against the RTX 3090, same benchmark.** `bench/labd_bench.py --ctx 66000` (six greedy tasks on a frozen copy of this
repo's docs, decode streamed so prefill is excluded, tokens per step from the server's metrics), `HQ_ROCM_KV=octave`,
192 splits, 3 drafts, the GEMV, `GPU_UTIL=0.90`, nothing else on the GPU. The corpus is built from this checkout's docs,
so it is not byte-identical to the one the 3090 numbers used; the prompt came out at 74.4k tokens:

| | decode | tokens per step |
|---|---|---|
| this card (copy / code / edit / quote / summary / qa) | 118.4 / 107.7 / 79.6 / 81.6 / 81.8 / 72.8 tok/s | 3.79 / 3.44 / 2.50 / 2.55 / 2.60 / 2.31 |
| this card, total | **88.1 tok/s** | 2.78 |
| RTX 3090, pth-flash-attn int8 (~66k, docs/long-context.md) | 84.7 tok/s | not given |
| RTX 3090, Triton int8 (~66k) | 75.5 tok/s | not given |

Deeper, same setup, `--ctx 121000 --corpus ~/bench/labd_corpus_long.txt` (the long corpus of
`bench/make_long_corpus.py`), prompt 130.4k tokens: **85.0 tok/s**, 2.79 tokens per step (copy 114.6, code 105.5,
edit 79.9, quote 77.6, summary 76.2, qa 69.6), cold prefill 175.5 s (~743 tok/s). At the window's edge, `--ctx 140000`, prompt 146.9k tokens (the 150k window
minus the 512-token answer): **79.0 tok/s**, 2.79 tokens per step (copy 106.6, code 95.1, edit 77.1, quote 73.9,
summary 74.6, qa 61.5), cold prefill 232.7 s (~631 tok/s). The 3090 has no labd number that deep; its deepest is
78.5 tok/s at ~100k.

**Against exllamav3-rocm on the same benchmark.** [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm)
(commit 360c902, an RDNA3 port of exllamav3 that publishes 107 tok/s at 99k with DFlash2, on a synthetic code
filler), built in this image's ROCm 7.2.3 container with two local changes: `rsqrtf(dim)` -> `1.0f / sqrtf(dim)` in
two host-side lines of `attention.cu` (ROCm 7.2.3's clang rejects the device-only call), and
`EXL3_EXPANDABLE_SEGMENTS=0` (expandable segments fail with `hipErrorInvalidValue` under WSL2). Weights
`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` and its DFlash2 EXL3 5.0 bpw draft; Q8 KV, draft KV Q4. The six labd tasks
on the same 74,369-token document, greedy, same chat template, decode timed from the first token, through the
Python API (not TabbyAPI):

| | decode total | per task (copy / code / edit / quote / summary / qa) | cold prefill |
|---|---|---|---|
| vLLM here (MTP 3, Octave, GEMV) | **88.1 tok/s** | 118.4 / 107.7 / 79.6 / 81.6 / 81.8 / 72.8 | 98.6 s |
| exllamav3-rocm, MTP | 57.9 tok/s | 64.9 / 88.3 / 59.0 / 47.8 / 54.4 / 48.1 | 88.3 s |
| exllamav3-rocm, DFlash2 | 47.4 tok/s | 74.2 / 77.1 / 45.8 / 37.5 / 38.4 / 38.6 | 82.0 s |

On this workload the vLLM stack is ahead on decode; exllamav3's prefill is ~10-17% faster. DFlash2 accepted 22% of
its drafts here (its drafter sees a 2,048-token window), against the high acceptance of the repeated code filler its
published numbers use. One run each.

Long-context decode is at parity with the 3090 on its own benchmark (one run per depth here, longer prompts). Cold prefill is
not: 98.6 s for 74.4k tokens (~755 tok/s), against ~75 s at ~66k on the 3090. The free-form numbers above
understate decode because MTP accepts ~2 tokens per step on that text against ~2.8 here.

### 5 to 32 rows: a WMMA kernel for concurrent requests (2026-11-21)

With `MAX_SEQS` above 1 a speculative verify step carries 4 rows per request (3 drafts + 1), so 3 requests mean 12
rows, 4 mean 16, 8 mean 32. Above 8 rows the GEMV is compute-bound (dot2 on the vector units) and vLLM's own kernels
stay near 340 GB/s (12-16 rows) or fall to 170-200 GB/s (17-32 rows, weights read twice), while the card can stream
~800 GB/s. `hq::w4a16_wmma` (same file, `rocm/w4a16_gemv.cu`) reads the weights once through the matrix units: B
fragments come straight from the int4 words (`nibble | 0x4300` is the bf16 number 128 + nibble), a second WMMA against
an all-ones B gives each row's activation sum, and scale and zero point are applied per 128-wide group in fp32. A wave
owns 64 output columns; waves split K and reduce through LDS. One layer, us per call at 16 rows: gate_up 263 -> 131,
down 192 -> 83, qkv 77 -> 48, o 64 -> 38; at 32 rows gate_up 510 -> 206. Error against an fp32 reference equals the
GEMV's (output bf16 rounding, ~1.3% of the mean on the worst element), and is below vLLM's kernel at some sizes.

Routing: 1-4 rows GEMV (a little faster there), 5-32 rows WMMA, 33-1023 vLLM's kernel, 1024+ dequant + hipBLASLt.
`HQ_W4A16_WMMA=0` turns the WMMA path off, `HQ_W4A16_WMMA_MIN_M` moves the switch (default 4).

Server, `MAX_SEQS=8`, 32k window, cohort of 8 real 1024-token prompts, end-to-end tok/s (default sampling):

| Concurrent requests | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| before | 99.3 | 145.7 | 170.3 | 184.3 |
| with the WMMA kernel | 103.3 | 166.1 | 295.0 | 277.4 |

Total decode rate at 8 requests 263 -> 641 tok/s; the time to first token at 8 requests went from 3.5 s to 7.2 s (all
eight prompts are admitted at once), so the gain is in throughput, not first-token latency. Single stream also gains a
little (labd 45k: 101.7 -> 108.4 tok/s decode, cold prefill 44.8 -> 41.9 s). GSM8K (100 questions) 95.0%, unchanged.

### Experiments that did not pay (2026-11-21)

- `--prefix-match-unit 128 --enable-mamba-shared-prefix-checkpoint`: the delay with the document cached stays 1.6-1.7 s;
  the cached prefix is still cut at a multiple of the 2176-token block (the 3090 has 816-token blocks because its
  int8 KV cache is bigger per token, which lowers the block size the hybrid model needs).
- `--max-num-batched-tokens 4096` (needs `GPU_UTIL=0.78` at 150k): cold prefill at 45k tokens 44.8 -> 42.4 s before the
  WMMA kernel; with it, 2048 already gives 41.9 s and 4096 gives 42.3 s. Not worth the extra VRAM; 2048 stays.

### Where a cold prefill goes (2026-11-21, 54k tokens, 56 s, ~980 tok/s)

HIP events around the eager custom ops (`scripts/hqtiming`, `scripts/prefill_probe.py`; the hook now skips recording
during graph capture): the W4A16 GEMM op 39.7 s (71%: dequantize + hipBLASLt, ~68 TFLOPs effective over all layers),
full attention 11.4 s (20%, ~25 ms per layer and 2048-token chunk at ~27k context, roughly 54 TFLOPs), GDN core 3.5 s
(6%), the rest (norms, elementwise, KV update) ~2 s.

Hardware ceilings measured on this card (`scripts/gemm/peak.py`, `wmma_rate.cu`, `dq_bench.py`): hipBLASLt bf16
matmul at M=1024-4096 reaches 85-95 TFLOPs in short runs (80 sustained); the raw WMMA peak is ~108 TFLOPs for bf16 and
only ~114-134 TOPS for int8 (about 1.1x, not 2x), so an int8 (W4A8) GEMM like the 3090's would not be faster here; the
dequantization kernel already runs at ~660 GB/s and costs ~7% of the GEMM time. Prefill is therefore bounded by the
matrix units; what is left is the attention kernel (a faster one could save a few seconds of the 11) and nothing in
the GEMMs.

#### The prefill attention kernel (2026-11-21)

`torch.ops._C.octave_prefill` (compiled into the wheel, no source in the image) measured alone with
`scripts/gemm/attn_bench.py` (24 query heads, 4 KV heads, head 256, 2048-token chunk, packed 3-bit prefix): 21 TFLOPs
with no prefix, 49 / 52 / 55 / 54 TFLOPs at 4k / 13k / 26k / 52k tokens of prefix, i.e. ~25 ms per layer and chunk at
27k tokens of context, matching what the server timing showed. PyTorch's SDPA on the same shapes with the prefix
already dequantized to fp16 (`scripts/gemm/sdpa_bench.py`, flash and memory-efficient backends, expanded or native GQA)
gives 44-52 TFLOPs, so there is no faster kernel to plug in. Dequantizing the prefix once per chunk would cost ~1% of
the attention time, so the packed format is not what holds the kernel back. A new kernel would have to be a
from-scratch head-256 WMMA flash-attention (the fp32 output tile alone takes ~128 registers per wave, Q fragments
another 128), against a raw WMMA peak of ~108 TFLOPs; reaching 70 TFLOPs would cut the 11 s of attention in a 54k
prefill by ~2.5 s (about 5% of the prefill). Not started: the expected gain is small for the effort and risk.

### GDN core in a decode step: real GPU time (2026-11-21)

`scripts/gdn_bench.py` replays 48 back-to-back calls from a captured graph (no host launch cost): one layer, one
request, 4 tokens: conv update 4.3 us + gated delta rule update 16.9 us = ~21 us, i.e. ~1.0 ms per step for 48 layers
(1.7 ms with 4 requests). The 74 us per layer quoted above came from eager timing and was inflated by host launches.
So the GDN kernels are not where the remaining ~4 ms against the 3090 step sit. Decode GEMMs on the real shapes
(`scripts/gemm/shapes.py`, M=4, in a graph): ~900 GB/s on lm_head, ~800 on gate_up, 670-740 on qkvz/qkv/down,
~530 on the two 16 MB output projections (fixed ~12 us per small kernel); about 18.7 ms of the step.

### Prefill: dequantize, then hipBLASLt (2026-11-20)

At prefill chunk sizes the GEMM is compute-bound. `gptq_gemm_rdna3` already runs at 52-60 TFLOPS there, and
hipBLASLt's bf16 GEMM at 59-75 with the weights already in bf16. `rocm/w4a16_gemv.cu` adds `hq::w4a16_dequant`
(the layer to bf16 `[K, N]`, one read of the int4 words and one write: 669 us for gate_up), and the patched kernel
file routes 1024+ rows to dequant + `torch` matmul. Same error as the original kernel against an fp32 reference.
Four layer shapes together: 6.25 -> 6.15 ms at 512 rows (no gain, hence the threshold), 12.5 -> 11.5 ms at 1024,
24.3 -> 22.0 ms at 2048. In the server, `HQ_ROCM_KV=octave`, labd at 74.4k tokens: **cold prefill 98.6 -> 87.9 s**
(~755 -> ~846 tok/s). Decode read 94.4 tok/s in that run against 88.1 before; this path does not touch decode
(at most 8 rows), so that difference is run-to-run variance. `HQ_W4A16_DQ_MIN_M=0` turns it off.

Rebuilt image (this path, the GEMV with non-temporal loads), `HQ_ROCM_KV=octave MAX_LEN=150000 MAX_SEQS=1`,
`GPU_UTIL=0.86`, `.wslconfig` `memory=24GB` with `autoMemoryReclaim=gradual`, sampled every ~1.5 s through the run:
single-stream cohort **99.9 / 103.4 tok/s** (default sampling / greedy; at 0.90 in the same session), and labd at
74.4k **95.6 tok/s** decode with a **84.7 s** cold prefill (~878 tok/s). System RAM in use 33.9 -> 34.0 GB, page file
unchanged at 1.5 GB, disk at 0-3%, shared GPU memory at most 0.71 GB. An earlier labd run of the same image at
`GPU_UTIL=0.90` and `memory=40GB` froze the PC for seconds and read 63.7 tok/s (copy task 28.4); it is not counted.

### Where a decode step goes now (2026-11-20)

`SPEC=mtp`, 3 drafts, one stream, ~26 ms per step with the GEMV. HIP events around each piece (a `sitecustomize`
that wraps `CUDAGraph.replay` and the eager custom ops; the torch profiler has no GPU data on WSL2), 406 steps in
`PIECEWISE` mode, ms per step: graph pieces (GEMMs, norms, elementwise) 28.6, GDN core 3.6 (74 us x 48 layers),
attention 0.9, KV-cache update 0.5. `PIECEWISE` adds ~7 ms of gaps over the `FULL` graph the server normally runs, so
read the shares, not the sum. Budget for the 26 ms: W4A16 GEMMs ~18 (14 GB of weights at ~760 GB/s), GDN core 3.6,
attention and cache 1.4, the rest elementwise.

Checked and ruled out: HIP-graph overhead per kernel is ~2.6 us (1,000 tiny kernels replay in 2.6 ms), the same order
as on NVIDIA; the unquantized GDN `in_proj_ba` GEMM is already routed to the skinny `wvSplitK` (8 us at M=4, against 31 us
for plain `F.linear`); changing the Triton launch parameters of the GDN conv update (`BLOCK_N` 256 -> 64, 2 warps)
and of the gated-delta-rule update (`BV` 32 -> 16, 2 warps) did not move their measured time (23 and 22 us), but that
measurement ran eager in `PIECEWISE` with extra events around each kernel (the GDN core read 147 us instead of 74
with them), so it may have timed host launches rather than the kernels: inconclusive, not a refutation.
The 3090 reference step is ~21.7 ms (121 tok/s at 2.63 tokens per step), so the remaining gap is ~4.5 ms, spread over
GDN (a fused conv + gating + recurrence kernel, ~2.4 ms at best), the GEMM bandwidth (~1.5 ms at best) and glue.

### What the 3090 numbers are

The README's 127 tok/s single stream is `SPEC=dflash2` (7 drafts, 15 tokens verified per step); its older MTP path is
121 tok/s, and the ~95-100 tok/s of row D rides the fp8 + FlashInfer arm, which has no AMD equivalent (the int8 pth-flash-attn
arm is 84.7 tok/s at ~66k and 78.5 at ~100k, the Triton int8 arm 75.5 and 68.8). Against the MTP path: 85 tok/s at 8k here
is about 70% of 121. `SPEC=dflash2` runs on the card (drafter prepared, TRITON_ATTN, bf16 KV, 8k) but is slower
than MTP with 3 drafts: 69.5 tok/s, 3.23 tokens per step, ~46 ms per step. Its verify step multiplies M=8 rows, and the RDNA3
W4A16 kernel reads the weights once per 4 rows (206 us on gate_up at M=8, against 127 at M=1). No existing vLLM W4A16
kernel on this card beats `gptq_gemm_rdna3` at M=4 by more than ~10 us (skinny HIP: 130 us, Triton: 350 us on gate_up),
and the skinny one refuses `down` at M>=3. A new GEMV that reads the weights once for M up to 8 at ~850 GB/s (the
current ones reach 550-730) is what would move both modes: it would cut the ~24 ms the weights take per MTP step and
make DFlash2's 8-row verify cost little more than a 4-row one.

Re-measured with the WMMA kernel (2026-11-21; `SPEC=dflash2`, 7 drafts, bf16 KV, 8k, `MAX_SEQS=4`, greedy, repo cohort):
decode 108.2 tok/s at one stream (3.33 tokens per step) and 262.6 at four (3.22), against 107.3 and ~320 for MTP with 3
drafts. The drafter accepts more per step, but its step costs as much, so it ties at one stream and loses at four. Not
worth the long-context work (it would need its own cache backend at 150k); MTP stays.

### A draft vocabulary that covers French (2026-11-21)

The shipped 40,960-id MTP draft list (English, Danish, code; gotcha 61) holds **75.95%** of the tokens of French text
(`build_draft_vocab.py --stats`, 103 French answers written by the model itself, 42k tokens; the same tool gives 92-96% on
English docs and code). A token outside the list can never be drafted and ends the chain. One stream, greedy, eight French
prompts on other topics than the corpus (`scripts/fr_bench.py`), same server, only the head changed:

| draft list | tokens/step | acceptance by position | decode |
|---|---|---|---|
| shipped, 40,960 ids | 2.01 | 0.57 / 0.30 / 0.14 | 71.5 tok/s |
| + 2,852 French ids, padded to 43,904 | **2.43** | 0.73 / 0.45 / 0.25 | **89-92 tok/s (+24-29%)** |

The English cohort is unchanged (2.61 tokens/step, 98.4 / 99.3 against 97.2 / 101.5 tok/s with the old head, run twice
each). The list is the shipped one plus the corpus' most frequent missing ids (`scripts/fr_vocab.py`, padded with Latin
tokens to a multiple of 128 so the head stays aligned for the W4A16 kernels); the head is rebuilt with
`prepare/build_draft_vocab.py <model> --ids <list>`, which rewrites `model_extra_tensors.safetensors`,
`mtp_draft_vocab_ids.pt` and the index of the `-fast` checkpoint (originals kept as `*.bak-fr`). The head grows by 7 MB.
Exact either way (the drafter only proposes). Quality, checked rather than assumed (2026-11-21): GSM8K greedy, no
thinking, **96.0% (192/200) with the old head and 95.5% (191/200) with the French one**, 374 and 376 tokens per answer
(on the first 100 questions: 95.0% and 94.0%; one question of difference is inside the noise of a 200-question run).
At T=0 the eight French answers differ between the two heads (each head is deterministic: two runs of the old one give
eight identical texts), because different drafts mean different rows verified together and bf16 rounding then tips
near-ties. At each divergence the model's own next-token probabilities of the two competing words were within a few
points of each other (0.500 against 0.500, 0.365 against 0.365, 0.472 against 0.417, 0.446 against 0.347): two valid
choices from the same distribution, not a degradation (`scripts/fr_dump.py`, `scripts/fr_ties.py`). The list comes from a small corpus (held-out coverage 96%): a bigger one,
ideally your own traffic, would cover more.

### Four drafts again, with the WMMA kernel (2026-11-21)

`DRAFT_TOKENS=4` (verify M=5), same session, `MAX_LEN=100000`: English cohort one stream greedy 97.8 tok/s at 2.93
tokens per step (3 drafts: 98.4 and 99.3 at 2.61), four streams 241.8; French 83.2 tok/s at 2.58 tokens per step (3 drafts
the same hour: 86.7 at 2.43). One more token per step does not pay for the extra draft pass and the wider verify, so 3
stays. Note that a boot at `MAX_LEN=150000` with 4 drafts needs ~2.45 GiB of KV cache against 2.1-2.8 available: it
fails intermittently at `GPU_UTIL=0.75`.

## Troubleshooting

- **The process dies with exit 139, and the log says `GetExportAddress failed ... librocdxg.so: undefined symbol:
  hsaKmtGetDefaultHostGpu`.** The ROCm runtime and `librocdxg` come from different builds. This is why the image is
  based on vLLM's ROCm 7.2 build with `librocdxg` 1.2.2, and not on the default ROCm 10.0 tag, whose bundled 1.1.0
  has that mismatch. Do not mount other copies of these libraries over the image's.
- **`bind: An attempt was made to access a socket in a way forbidden`.** The port is in a range Windows reserves.
  Pick another `PORT`.
- **`Free memory on device ... is less than desired GPU memory utilization`.** Something else holds VRAM. Check
  Task Manager, GPU tab, "Dedicated GPU memory", and close it, or lower `GPU_UTIL`.
- **A boot that works and then runs several times slower than it should.** WDDM can back an overcommitted card with
  system RAM without any error (docs/gotchas.md, WSL2 entries). Watch "Shared GPU memory" in Task Manager and lower
  `GPU_UTIL` or `MAX_LEN`.
- **The whole PC freezes for several seconds and the disk runs at 100% during long prompts.** Seen at `GPU_UTIL=0.90`
  with `memory=40GB` in `.wslconfig` on a 64 GB machine: during a 74k-token prefill the card sat at 23.9 of 24 GiB,
  system RAM in use rose by ~23 GB and the Windows page file peaked at ~25 GB. WSL also keeps the page cache of every
  model file it reads (15-37 GB) until it hits its cap. The defaults are now `GPU_UTIL=0.86`, and the setup here runs
  `.wslconfig` with `memory=24GB` and, under `[experimental]`, `autoMemoryReclaim=gradual`, which returns that cache to
  Windows (`wsl --shutdown` and restart Docker Desktop after changing it).
- **Idle CPU.** With the server up and no request running, `docker stats` showed about 410-420% CPU and the Windows GPU
  engines were at 0%. Four threads sat at 100%: two in the API server process and two in the engine process, all of
  which have `libhsa-runtime64` and `librocdxg` mapped. Confirmed with a bare process (`scripts/idle_cpu_probe.py` in
  the work folder): `import torch` alone costs 0.00 cores, one `torch.zeros(1, device="cuda")` then sleeping costs
  2.2 cores, as two unnamed threads at ~1.08 cores each in user mode with no system calls (a pure spin);
  `HSA_ENABLE_INTERRUPT=0` or `=1` changes nothing. It is ROCm's `rocr::core::Runtime::AsyncEventsLoop`: under WSL2
  `librocdxg` provides no interrupt-backed signal events, so the loop polls without sleeping (ROCm/librocdxg#60).
  The upstream fix, a polling nap that escalates from 20 us to 2 ms, is ROCm/rocm-systems#7898 (merged 2026-08-18,
  idle 182% -> 1%); it is in ROCm 10.1 and not in the 7.2.x runtime this image uses. Moving to the 10.1 runtime is
  not free: for a gfx1201 card an upstream report (ROCm/rocm-systems#13006) measures decode 31-45% slower there
  because host-visible allocations move to system memory, and a 10.x `libhsa` needs a newer `librocdxg` (the ROCm
  10.0 image died with `hsaKmtGetDefaultHostGpu`, see above). So the image rebuilds its own `libhsa-runtime64` with
  the patch backported (`rocm/rocr/`, built in `Dockerfile.rocm` into `/opt/hq/rocr`, put first on `LD_LIBRARY_PATH` by
  `start-rocm.sh`; `HQ_ROCR_PATCH=0` uses the stock one). The runtime in the image reports
  `1.18.0-local-build-af70edc`, a commit slightly past the `rocm-7.2.3` tag (the tag lacks
  `hsa_amd_signal_create_v2`, which the image's `libamdhip64` needs: first attempt, ImportError), so the script builds
  that exact commit, applies the 7.2.0 backport (it applies cleanly) and refuses to produce a library unless it exports
  the same 263 symbols. Measured, whisper stopped, `GPU_UTIL=0.75`, `HQ_ROCM_KV=octave`, 150k window:

  | | patched runtime | stock runtime |
  |---|---|---|
  | idle CPU of the container | **3.3%** | 423% |
  | bare process, idle, `idle_cpu_patched.py` | 0.01 cores | 2.18 cores |
  | single-stream cohort (default / greedy) | 101.3 / 105.4 tok/s | 99.9 / 103.4 tok/s |
  | VRAM taken by vLLM (Windows `vmwp`) | 18.24 GB | 18.54 GB |
  | KV pool reported at 0.75 | 170,689 tokens | 189,655 tokens |

  Decode speed and sync latency (0.9 ms for a 2048 matmul + synchronize) are unchanged; the patched runtime leaves
  about 0.3 GiB less for the KV pool (14.5 vs 14.2 GiB "consumed memory"). The nap delays only the async-event threads
  (HIP host callbacks, async signal handlers), not `hipStreamSynchronize`. The API server process initialises the GPU
  runtime itself (the log says "Failed to get total memory via amdsmi, falling back to torch.cuda"), which is why
  there were two processes spinning two threads each.
- The `amdsmi` warning at startup (`Can't initialize amdsmi - Error code: 34`) is expected under WSL2.

## Native Linux

Not set up here. The image would need `/dev/kfd` and `/dev/dri` (`devices`, `group_add: [video]`) in place of
`/dev/dxg` and the `/usr/lib/wsl` mount; `librocdxg` is inert there. Everything else is the same.

## Files

[`Dockerfile.rocm`](../Dockerfile.rocm) · [`start-rocm.sh`](../start-rocm.sh) · the `rocm` and `prepare-rocm` services in
[`docker-compose.yml`](../docker-compose.yml) · the `rocm` command in [`docker/entrypoint.sh`](../docker/entrypoint.sh) ·
`verify.sh` skips its FlashInfer check on a ROCm build of torch · `single-user/start_qwen.sh` reads `DRAFT_ATTN_BACKEND`.
