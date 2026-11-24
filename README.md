# HyperQwen-7900XTX: working folder

The whole port of HyperQwen to the RX 7900 XTX, in `E:\git\HyperQwen-7900XTX`. The repository is in `hyperqwen\`: it
holds the shipped code (launcher `start-rocm.sh`, `Dockerfile.rocm`, kernels in `rocm/`, patch
`patches/rocm-w4a16-gemv.patch`, doc `docs/rocm.md`, example `.env.rocm.example`) and the models (`models\`).

## Best results

Qwen3.8-27B (W4A16 AutoRound "fast") on an RX 7900 XTX (24 GB), Windows 11 + Docker Desktop (WSL2), vLLM 0.33.0 on
ROCm 7.2.3. Settings: MTP with 3 drafts, 3-bit Octave KV cache, 150k window, `MAX_SEQS=4`, `GPU_UTIL=0.75`, French
draft list, `VISION=1`. One stream unless stated otherwise.

**All figures in the "Decode", "Prefill", "Parallel requests" and "Quality" tables come from a single series of
measurements, taken on 2026-11-24, on a single build** (the one above: WMMA verify kernel, French draft list, vision
enabled), server warmed up before each series, one pass per measurement (the server was stopped and restarted once
during the series: the 147k run and the needles come after the restart). Exceptions are flagged in the row or
paragraph concerned: the comparison with the RTX 3090 (3090 not re-measured), vision (measured on 2026-11-23 on the
same build) and dated reference points. Details and protocols are in
[`hyperqwen/docs/rocm.md`](hyperqwen/docs/rocm.md).

**Decode, one stream** (tok/s)

| Case | Result |
|---|---|
| Short chat, English, default temperature / T=0 (repo cohort, 8 real prompts) | 98.2 / 99.6 |
| Short chat, French (8 prompts outside the corpus, T=0) | **92.4** (2.45 tokens per step) |
| ~45k-token document | 97.7 (2.84 tokens per step) |
| 74k-token document | 92.6 (2.80) |
| 130k-token document | 82.2 (2.76) |
| 147k-token document (edge of the 150k window) | 77.2 (2.79) |

Older reference points, from a different state of the build or other settings, for context: French before the French
draft list 71.5 tok/s (2.01 tokens per step, 2026-11-21); no speculation (`SPEC=off`, 8k, bf16 KV cache, first try on
2026-11-19) ~44 tok/s.

**Cold prefill** (reading a prompt with no cache; every measurement redoes the whole read)

| Prompt size | Duration | Throughput |
|---|---|---|
| ~45k tokens | 46.5 s | ~970 tok/s |
| 74k tokens | 85.0 s | ~875 tok/s |
| 115k tokens (3 needles) | 152.0 s | ~760 tok/s |
| 130k tokens | 174.5 s | ~747 tok/s |
| 147k tokens | 210 to 222 s | ~680 tok/s |

With the document already cached, the answer starts in about 0.7 s (needles at 115k).

**Parallel requests** (repo cohort: 8 real prompts of 1024 tokens, 150k window, `MAX_SEQS=4`, tok/s, default temperature / T=0)

| Simultaneous requests | 1 | 2 | 4 |
|---|---|---|---|
| Decode (total across streams) | 98.2 / 99.6 | 169.1 / 172.7 | 295.4 / 310.6 |
| End-to-end throughput | 95.0 / 96.7 | 152.7 / 157.1 | 257.3 / 272.3 |

**Quality**

| Test | Result |
|---|---|
| GSM8K, 100 questions, greedy, no thinking | 95.0% (same as the RTX 3090) |
| 3 needles in 115,534 tokens | 3/3 (same as the RTX 3090) |

Older reference point (2026-11-21): GSM8K on 200 questions, 96.0% with the old draft list and 95.5% with the new one.
Speculative decoding is exact (the full model checks every token): the French draft list changes speed, not the
answers, except for a few near-ties rounded differently in bf16.

**Against the original RTX 3090.** The 3090 was not re-measured: its figures come from the same protocol, measured on
2026-11-21 (3090 at 350 W, its server with 8 slots); the 7900 XTX figures are today's.

| | 7900 XTX | RTX 3090 | Ratio |
|---|---|---|---|
| Short chat, decode | 98.2 tok/s | 127.2 tok/s | 77% |
| ~45k document, decode | 97.7 tok/s | 122.3 tok/s | 80% |
| ~45k document, cold prefill | 46.5 s | 26.0 s | 56% |
| Cold prefill, 115k tokens | 152.0 s | 88.8 s | 58% |

At long context, the 7900 XTX decode exceeds the published reference points for the 3090 (92.6 tok/s at 74k against
84.7 at ~66k; 82.2 at 130k against 78.5 at ~100k): both reference points come from the original repo, at 250 W, with a
different protocol. Cold prefill and the time to first token without a cache remain clearly slower than on the 3090.

**Vision** (measured on 2026-11-23, same build): images accepted (one per request, 1 megapixel at most, about 1,024
image tokens). A real photo is described correctly in 7.4 s for 350 output tokens; text decode stays at ~96 tok/s; the
VRAM margin stays at 1.6 GB on the 24 GB card (Windows desktop included). Setting: `VISION=1` in `.env`.

**Limits**: one pass per measurement; GSM8K on 100 questions (±2 points of noise); absolute speed varies by a few
percent from one moment to the next depending on machine load; decode is 1 to 10% lower than in our measurements of
2026-11-21 to 22 (clearest at ~45k: 97.7 against 108.4 tok/s; different settings: `GPU_UTIL` 0.86 to 0.90 back then,
0.75 today, and vision was not enabled), without the cause being isolated; long-context quality is checked only by the
needles and by GSM8K at short context. **VRAM margin**: at 150k it can get very thin (a peak of 23.7 GB out of 23.94
was recorded during the 115k and 147k runs, with other programs open on the machine); close other programs that use
the GPU before launching.

| Folder | Contents | Size |
|---|---|---|
| `hyperqwen\` | the HyperQwen repository with the ROCm port, and `models\` (the prepared model and the DFlash2 draft) | ~37 GB |
| `llama-vulkan\` | llama.cpp b12556 for Windows (Vulkan) and `models\Qwen3.8-27B-UD-Q4_K_M.gguf`. First try outside vLLM (42 / 24.9 / 15.3 tok/s at 30k / 68k / 125k). | ~15.6 GB |
| `exllamav3\` | The ROCm port of ExLlamaV3 (`exllamav3-rocm\`, commit 360c902, with the `rsqrtf` fix in `attention.cu`), its Python environment (`venv\`), the EXL3 3.5 bpw weights and the DFlash2 draft (`models\`), the bench corpus (`labd_corpus.txt`), the scripts `build.sh`, `download.sh`, `run_labd.sh`, `labd_exl3.py` and the logs. Result: 57.9 tok/s (MTP) and 47.4 (DFlash2) against 88.1 for our vLLM, same bench. | ~16.3 GB |
| `octave-noyau\` | The sources of vLLM v0.33.0's Octave attention kernel, rebuilt separately (`build_octave.py`, `hq_octave_binding.cu`) and `test_octave.py`, which checks that we reproduce vLLM bit for bit. Starting point for 4-row attention, a line of work not pursued. | <1 MB |
| `scripts\` | The measurement scripts from the session (see below). | <1 MB |

## Scripts (`scripts\`)

| Script | Purpose |
|---|---|
| `mem_monitor.ps1` | runs `labd_bench` in the container and records VRAM (dedicated and shared), RAM, page file, WSL memory and disk |
| `cpu_monitor.ps1` | CPU, RAM, VRAM at idle, in chat and at 70k |
| `decode_diff.ps1` | pure decode at long context (same prompt at 64 then 320 tokens, so the prefill cancels out) |
| `decode_depth.ps1`, `depth.ps1` | needle in a document + decode, at several depths |
| `mtp_chat.ps1`, `mtp_run.ps1` | short chat with MTP acceptance rate |
| `ollama_bench.ps1` | Ollama with CPU / RAM / VRAM readings |
| `llama_req.ps1` | llama.cpp (llama-server), chat and long context |
| `kbench.py`, `kbench2.py`, `prefill_gemm.py` | microbenchmarks of the W4A16 GEMMs (vLLM kernels, skinny, Triton, hipBLASLt) |
| `gemm\` | GEMV tests (`test_gemv.py`, `test_dq.py`, `gemv_variant.py`, `run_variants.sh`) and a copy of `w4a16_gemv.cu` |
| `octave_bench.py`, `octave_sweep.py`, `octave_short.py` | microbenchmarks of the Octave kernel (splits, rows per pass) |
| `graph_gap.py` | per-kernel overhead of a HIP graph |
| `hqtiming\sitecustomize.py`, `ovr_timing.yml` | instrumentation: GPU time per category in a decode step |
| `tune\`, `ovr_tune.yml`, `tune_run.ps1` | Triton tuning of the GDN (inconclusive attempt) |
| `analyze_trace.py`, `timeline.py` | reading a torch profiler trace |

The `ovr*.yml` files mounted working copies on top of the old image. They point to old temporary paths and are no
longer used: the `hyperqwen-rocm:local` image now contains the GEMV and the prefill path.

## Restarting HyperQwen

From `E:\git\HyperQwen-7900XTX\hyperqwen` (the Docker project name stays `hyperqwen`, so the build cache
`hyperqwen_qwen-cache` and the image `hyperqwen-rocm:local` are reused):

```powershell
Copy-Item .env.rocm.example .env
docker compose up -d --force-recreate rocm
```

With another program on the GPU (whisper, Ollama), lower `GPU_UTIL` in `.env`. 150k of context needs ~18 GB for vLLM
(`GPU_UTIL` ~0.76): everything else has to fit in ~5 GB, Windows desktop included.
