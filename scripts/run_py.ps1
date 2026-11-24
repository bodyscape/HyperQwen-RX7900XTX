# usage: run_py.ps1 <script relative to scripts\> [args...]   (runs it in a throwaway container with the GPU)
param([Parameter(Mandatory=$true)][string]$Script, [Parameter(ValueFromRemainingArguments=$true)]$Rest)
docker run --rm --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -v "E:\git\HyperQwen-RX7900XTX\scripts:/work" -e PYTORCH_ROCM_ARCH=gfx1100 -e HSA_ENABLE_DXG_DETECTION=1 -e ROCPROFILER_REGISTER_ENABLED=0 --entrypoint /app/venv/bin/python hyperqwen-rocm:local -I "/work/$Script" @Rest 2>&1 | Select-String -NotMatch 'Warning|warn|raw_cnt|preprocessed|pin_memory'
