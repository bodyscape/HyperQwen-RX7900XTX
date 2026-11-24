# Essaie une configuration de test: ajoute des lignes a hyperqwen\.env, redemarre le serveur, execute un banc, puis remet .env.
# Usage: .\try_config.ps1 -ExtraArgs '--prefix-match-unit 128' -Tag cacheA [-Bench labd|none]
param([string]$ExtraArgs = '', [string]$Tag = 'try', [string]$Bench = 'labd', [int]$Ctx = 40000, [string]$GpuUtil = '', [string[]]$EnvLines = @())
$Root = 'E:\git\HyperQwen-RX7900XTX'
$f = "$Root\hyperqwen\.env"
$orig = Get-Content $f
try {
    $lines = $orig | Where-Object { $_ -notmatch '^\s*EXTRA_ARGS' -and -not ($GpuUtil -and $_ -match '^\s*GPU_UTIL') }
    if ($GpuUtil) { $lines += "GPU_UTIL=$GpuUtil" }
    foreach ($e in $EnvLines) { $k = $e.Split('=')[0]; $lines = @($lines | Where-Object { $_ -notmatch ('^\s*' + $k + '=') }) + $e }
    if ($ExtraArgs) { $lines += "EXTRA_ARGS=$ExtraArgs" }
    Set-Content -Path $f -Value $lines -Encoding ascii
    & "$Root\stop-hyperqwen.ps1" -NoWhisper | Out-Null
    & "$Root\start-hyperqwen.ps1" -NoWhisper | Select-String 'PRET|insuffisante|plante|VRAM:'
    if ($LASTEXITCODE -ne 0) { "demarrage echoue"; return }
    docker logs hyperqwen-rocm-1 2>&1 | Select-String 'GPU KV cache size|attention block size|prefix_match_unit|Invalid' | Select-Object -Last 3 | ForEach-Object { $t = $_.Line; $i = [math]::Max(0, $t.IndexOf(']') + 2); '  vLLM: ' + $t.Substring($i, [math]::Min(170, $t.Length - $i)) }
    if ($Bench -eq 'cohort') {
        docker exec hyperqwen-rocm-1 bash -lc "cd /app && export PATH=/app/venv/bin:`$PATH && HOST=127.0.0.1 PORT=19020 OUT=/tmp/pa timeout 900 bash bench/run_benchmarks.sh single 2>&1 | grep -E '^ROW|rror'"
    }
    if ($Bench -eq 'c14') {
        docker exec hyperqwen-rocm-1 bash -lc "cd /app && export PATH=/app/venv/bin:`$PATH && sed -e 's/for C in 1 2 4 8/for C in 1 4/' -e 's/for T in default 0/for T in 0/' bench/run_benchmarks.sh > bench/rb_c14.sh && HOST=127.0.0.1 PORT=19020 OUT=/tmp/pa timeout 900 bash bench/rb_c14.sh single 2>&1 | grep -E '^ROW|rror'"
    }
    if ($Bench -eq 'labd') {
        docker exec -e VLLM_API=http://127.0.0.1:19020 hyperqwen-rocm-1 bash -lc "cd /app && timeout 560 venv/bin/python bench/labd_bench.py $Tag --ctx $Ctx 2>&1 | grep -E '^LABD|rror'"
    }
}
finally { Set-Content -Path $f -Value $orig -Encoding ascii; "--- .env remis" }
