param([int[]]$Lines = @(), [switch]$Chat, [int]$Tokens = 300, [int]$Count = 4, [string]$Model = 'qwen3.8_27b-q4_K_M_128k:latest',
       [string]$Base = 'http://127.0.0.1:11434', [string]$Csv = '')
$ErrorActionPreference = 'Stop'
$totalRam = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
$script:rows = @()

function Snap($prev) {
  $ps = Get-Process | Where-Object { $_.Name -like 'ollama*' -or $_.Name -eq 'llama-server' }
  $pids = @($ps | ForEach-Object { $_.Id })
  $cpu = ($ps | ForEach-Object { $_.TotalProcessorTime.TotalSeconds } | Measure-Object -Sum).Sum
  $ws = ($ps | ForEach-Object { $_.WorkingSet64 } | Measure-Object -Sum).Sum / 1GB
  $paths = '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage',
           '\GPU Engine(*)\Utilization Percentage'
  $c = (Get-Counter -Counter $paths -ErrorAction SilentlyContinue).CounterSamples
  $sys = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
  $avail = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
  $ded = ($c | Where-Object { $_.Path -like '*dedicated usage*' } | Measure-Object CookedValue -Maximum).Maximum / 1GB
  $shr = ($c | Where-Object { $_.Path -like '*shared usage*' } | Measure-Object CookedValue -Maximum).Maximum / 1GB
  $gpu = 0.0
  foreach ($x in ($c | Where-Object { $_.Path -like '*gpu engine*' })) {
    foreach ($id in $pids) { if ($x.InstanceName -like "pid_${id}_*") { $gpu += $x.CookedValue; break } }
  }
  [pscustomobject]@{ t = Get-Date; cpuSec = $cpu; ollamaWS = $ws; sysCpu = $sys; usedRam = $totalRam - $avail
                     vram = $ded; shared = $shr; gpu = $gpu }
}

function Summarise($name, $samples, $seconds) {
  $cores = @()
  for ($i = 1; $i -lt $samples.Count; $i++) {
    $dt = ($samples[$i].t - $samples[$i-1].t).TotalSeconds
    if ($dt -gt 0) { $cores += 100 * ($samples[$i].cpuSec - $samples[$i-1].cpuSec) / $dt }
  }
  $a = { param($v) if ($v.Count) { ($v | Measure-Object -Average).Average } else { 0 } }
  $m = { param($v) if ($v.Count) { ($v | Measure-Object -Maximum).Maximum } else { 0 } }
  $o = [pscustomobject]@{
    phase = $name; secs = [math]::Round($seconds, 1)
    ollamaCpuAvg = [math]::Round((& $a $cores)); ollamaCpuMax = [math]::Round((& $m $cores))
    sysCpuAvg = [math]::Round((& $a @($samples.sysCpu))); sysCpuMax = [math]::Round((& $m @($samples.sysCpu)))
    ramUsedGB = [math]::Round((& $m @($samples.usedRam)), 1); ollamaRamGB = [math]::Round((& $m @($samples.ollamaWS)), 1)
    vramGB = [math]::Round((& $m @($samples.vram)), 2); sharedGB = [math]::Round((& $m @($samples.shared)), 2)
    gpuUtilAvg = [math]::Round((& $a @($samples.gpu))); gpuUtilMax = [math]::Round((& $m @($samples.gpu)))
  }
  $script:rows += $o
  Write-Host ("   [res] CPU ollama avg {0}% max {1}% (100% = 1 coeur) | CPU systeme avg {2}% max {3}% | RAM utilisee max {4} Go (ollama {5} Go) | VRAM max {6} Go (partagee {7} Go) | GPU util avg {8}% max {9}%" -f `
    $o.ollamaCpuAvg, $o.ollamaCpuMax, $o.sysCpuAvg, $o.sysCpuMax, $o.ramUsedGB, $o.ollamaRamGB, $o.vramGB, $o.sharedGB, $o.gpuUtilAvg, $o.gpuUtilMax)
}

# Sends one /api/chat request in a job while sampling the machine every ~1s.
function Send($name, $content, $max, $temp, $think = $false) {
  $b = @{ model = $Model; stream = $false; think = $think; keep_alive = '1h'
          options = @{ num_predict = $max; temperature = $temp }
          messages = @(@{ role = 'user'; content = $content }) }
  if ($temp -gt 0) { $b.options.top_p = 0.95; $b.options.top_k = 20 }
  $json = $b | ConvertTo-Json -Depth 6
  $bytes = [Text.Encoding]::UTF8.GetBytes($json)
  $job = Start-Job -ScriptBlock {
    param($u, $bytes)
    Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 5000
  } -ArgumentList "$Base/api/chat", $bytes
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $samples = @(Snap $null)
  while ($job.State -eq 'Running') { $samples += Snap $null }
  $sw.Stop()
  $r = Receive-Job $job -ErrorAction Stop; Remove-Job $job
  $script:last = $samples; $script:lastSecs = $sw.Elapsed.TotalSeconds
  Summarise $name $samples $sw.Elapsed.TotalSeconds
  $r
}

function Fmt($r) {
  "pp {0,6:N0} tok/s ({1} tok) | tg {2,5:N1} tok/s ({3} tok) | load {4:N1}s" -f `
    ($r.prompt_eval_count / [math]::Max(1e-9, $r.prompt_eval_duration / 1e9)), $r.prompt_eval_count,
    ($r.eval_count / [math]::Max(1e-9, $r.eval_duration / 1e9)), $r.eval_count, ($r.load_duration / 1e9)
}

"--- baseline (modele decharge, 6 s)"
$s = @(); 1..5 | ForEach-Object { $s += Snap $null }
Summarise 'baseline-idle' $s 6

if ($Chat) {
  "--- chargement + warm-up"
  $r = Send 'load-warmup' 'Dis bonjour en une phrase.' 20 0
  "   $(Fmt $r)"
  "--- repos apres chargement (6 s, aucune requete)"
  $s = @(); 1..5 | ForEach-Object { $s += Snap $null }
  Summarise 'idle-loaded' $s 6
  "--- 4 prompts chat, 300 tokens, temp 1.0 top_p 0.95 top_k 20"
  $prompts = @(
    'Explique la difference entre TCP et UDP, avec un exemple concret pour chacun.',
    'Ecris une fonction Python qui renvoie les nombres premiers inferieurs a n avec le crible d Eratosthene, avec une docstring et deux tests.',
    'Traduis en anglais : Le projet consiste a servir un grand modele de langage sur une carte graphique grand public, en combinant quantification, decodage speculatif et cache clef-valeur compresse.',
    'Donne trois conseils pour bien dormir et explique pourquoi chacun fonctionne.')
  $tg = @(); $i = 0
  foreach ($p in ($prompts | Select-Object -First $Count)) {
    $i++
    $r = Send "chat-$i" $p $Tokens 1.0
    "   $(Fmt $r)   [$($p.Substring(0,36))]"
    $tg += ($r.eval_count / ($r.eval_duration / 1e9))
  }
  "   mean tg: {0:N1} tok/s" -f (($tg | Measure-Object -Average).Average)
  if ($Count -ge 4) {
    "--- comptage 1..200 (sortie previsible), temp 0"
    $r = Send 'counting' 'Compte de 1 a 200, un nombre par ligne, sans rien d autre.' 256 0
    "   $(Fmt $r)"
  }
}

$rng = [Random]::new(7)
foreach ($n in $Lines) {
  $sb = [Text.StringBuilder]::new()
  for ($i = 0; $i -lt $n; $i++) {
    [void]$sb.Append("Fait numero $($i + $n * 10): la valeur de la cle k$($rng.Next(100000,999999)) est $($rng.Next(100000,999999)) dans le registre $($rng.Next(10,99)).`n")
    if ($i -eq 5) { [void]$sb.Append("NOTE IMPORTANTE: le code secret du coffre est 7391-ALPHA.`n") }
  }
  $doc = $sb.ToString()
  "--- contexte long: $n lignes"
  $r1 = Send "prefill-$n" ($doc + "`nResume ce document en un mot.") 1 0
  "   [$n lignes] prefill a froid: {0:N1}s mur | {1}" -f $script:lastSecs, (Fmt $r1)
  $r2 = Send "decode-$n" ($doc + "`nQuel est le code secret du coffre ? Reponds par le code, puis ecris une longue explication.") 128 0
  "   [$n lignes] aiguille trouvee: {0} | {1}" -f ($r2.message.content -match '7391'), (Fmt $r2)
}

if ($Csv) { $script:rows | Export-Csv -Path $Csv -NoTypeInformation -Encoding utf8 }
"--- ollama ps"
ollama ps
