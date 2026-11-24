# Compare la VRAM prise par vLLM avec le runtime ROCm patche (defaut) et avec HQ_ROCR_PATCH=0, whisper arrete.
# Deux demarrages complets (~2,5 min chacun). Remet .env dans son etat d'origine a la fin.
$Root = 'E:\git\HyperQwen-7900XTX'
$Repo = "$Root\hyperqwen"
$envFile = "$Repo\.env"
$orig = Get-Content $envFile

function Get-Vram {
    $c = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
    $vm = 0.0; $tot = 0.0
    foreach ($x in $c) {
        $tot += $x.CookedValue
        if ($x.InstanceName -match 'pid_(\d+)_' -and (Get-Process -Id $Matches[1] -ErrorAction SilentlyContinue).Name -eq 'vmwp') { $vm += $x.CookedValue }
    }
    $sh = ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Maximum).Maximum
    $ad = ((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Maximum).Maximum
    [pscustomobject]@{ vllm = $vm / 1GB; adapter = $ad / 1GB; shared = $sh / 1GB }
}

function Run-Variant($label, $patch) {
    "=== $label"
    $lines = $orig | Where-Object { $_ -notmatch '^\s*HQ_ROCR_PATCH' }
    if ($patch -eq '0') { $lines += 'HQ_ROCR_PATCH=0' }
    Set-Content -Path $envFile -Value $lines -Encoding ascii
    & "$Root\start-hyperqwen.ps1" -NoWhisper | Select-String 'PRET|VRAM|plante|insuffisante|Erreur'
    if ($LASTEXITCODE -ne 0) { "  demarrage echoue (code $LASTEXITCODE)"; return }
    Start-Sleep 20
    $a = Get-Vram
    "  au repos          : vLLM {0:N2} Go | carte {1:N2} Go | partagee {2:N2} Go" -f $a.vllm, $a.adapter, $a.shared
    $b = @{ model = 'qwen3.8-27b'; max_tokens = 120; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false }; messages = @(@{ role = 'user'; content = 'Explique TCP et UDP.' }) } | ConvertTo-Json -Depth 6
    try { [void](Invoke-RestMethod -Uri 'http://127.0.0.1:19020/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 120) } catch { "  requete: $($_.Exception.Message)" }
    Start-Sleep 3
    $c = Get-Vram
    "  apres une requete : vLLM {0:N2} Go | carte {1:N2} Go | partagee {2:N2} Go" -f $c.vllm, $c.adapter, $c.shared
    docker stats --no-stream --format '  CPU du conteneur  : {{.CPUPerc}}' hyperqwen-rocm-1
    docker logs hyperqwen-rocm-1 2>&1 | Select-String 'Actual usage is|GPU KV cache size' | Select-Object -Last 2 | ForEach-Object {
        $l = $_.Line; $i = [math]::Max(0, $l.IndexOf('GPU KV')); if ($l -match 'Actual usage') { $i = $l.IndexOf('Actual usage') }
        '  vLLM: ' + $l.Substring($i, [math]::Min(150, $l.Length - $i)) }
    & "$Root\stop-hyperqwen.ps1" -NoWhisper | Select-String 'Arrete'
}

try {
    "whisper-server en marche: $([bool](Get-Process whisper-server -ErrorAction SilentlyContinue))"
    $base = Get-Vram; "VRAM de base (rien de lance): carte {0:N2} Go" -f $base.adapter
    Run-Variant 'runtime ROCm PATCHE (defaut)' '1'
    Run-Variant 'runtime ROCm d''origine (HQ_ROCR_PATCH=0)' '0'
}
finally { Set-Content -Path $envFile -Value $orig -Encoding ascii; "--- .env remis dans son etat d'origine" }
