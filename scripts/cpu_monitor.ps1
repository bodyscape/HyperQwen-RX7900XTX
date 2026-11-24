param([int]$Lines = 1850, [string]$C = 'hyperqwen-rocm-1', [int]$Port = 19020)
$u = "http://127.0.0.1:$Port/v1/chat/completions"
$totalRam = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
function Snap {
  $d = (docker stats --no-stream --format '{{.CPUPerc}}|{{.MemUsage}}' $C) -split '\|'
  $cpuC = [double](($d[0] -replace '%','') -replace ',','.')
  $memC = ($d[1] -split '/')[0].Trim()
  $sys = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
  $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
  $ded = ((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1GB
  [pscustomobject]@{ cpuC = $cpuC; memC = $memC; sys = $sys; used = $totalRam - $free; vram = $ded }
}
function Phase($name, $job) {
  $s = @()
  if ($job) { while ($job.State -eq 'Running') { $s += Snap } } else { 1..5 | ForEach-Object { $s += Snap } }
  $avg = ($s.cpuC | Measure-Object -Average).Average; $max = ($s.cpuC | Measure-Object -Maximum).Maximum
  "{0,-28} container CPU avg {1,5:N0}% max {2,5:N0}% (100% = 1 coeur) | CPU Windows avg {3,3:N0}% max {4,3:N0}% | RAM conteneur {5} | RAM systeme max {6:N1} Go | VRAM max {7:N2} Go | {8} echantillons" -f `
    $name, $avg, $max, ($s.sys | Measure-Object -Average).Average, ($s.sys | Measure-Object -Maximum).Maximum, $s[-1].memC, ($s.used | Measure-Object -Maximum).Maximum, ($s.vram | Measure-Object -Maximum).Maximum, $s.Count
}
function Start-Req($content, $max) {
  $b = @{ model='qwen3.8-27b'; max_tokens=$max; temperature=0; chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content=$content}) } | ConvertTo-Json -Depth 6
  Start-Job -ScriptBlock { param($u,$bytes) Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 3000 } -ArgumentList $u, ([Text.Encoding]::UTF8.GetBytes($b))
}
function Done($j) { $r = Receive-Job $j; Remove-Job $j; $r }

Phase 'repos (modele charge)' $null
$j = Start-Req 'Explique la difference entre TCP et UDP, avec un exemple concret pour chacun.' 300
Phase 'decodage chat court (300 tok)' $j; [void](Done $j)
$j = Start-Req 'Ecris une fonction Python qui renvoie les nombres premiers inferieurs a n avec le crible d Eratosthene.' 300
Phase 'decodage chat court (bis)' $j; [void](Done $j)

$rng = [Random]::new($Lines)
$sb = [Text.StringBuilder]::new()
for ($i = 0; $i -lt $Lines; $i++) {
  [void]$sb.Append("Fait numero $($i + $Lines * 10): la valeur de la cle k$($rng.Next(100000,999999)) est $($rng.Next(100000,999999)) dans le registre $($rng.Next(10,99)).`n")
  if ($i -eq 5) { [void]$sb.Append("NOTE IMPORTANTE: le code secret du coffre est 7391-ALPHA.`n") }
}
$q = $sb.ToString() + "`nQuel est le code secret du coffre ? Reponds par le code, puis ecris une longue explication."
$j = Start-Req $q 1
Phase "prefill a froid (~70k tokens)" $j; [void](Done $j)
$j = Start-Req $q 256
Phase 'decodage a 70k (256 tok)' $j; $r = Done $j
"   needle trouvee: $($r.choices[0].message.content -match '7391')"
Phase 'repos apres charge' $null
