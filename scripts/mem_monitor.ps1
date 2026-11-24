param([string]$Tag = 'mon74', [int]$Ctx = 66000)
# Run labd inside the container while sampling VRAM (dedicated/shared), RAM, page file, WSL memory and disk activity.
$job = Start-Job -ScriptBlock {
  param($tag, $ctx)
  docker exec -e VLLM_API=http://127.0.0.1:19020 hyperqwen-rocm-1 bash -lc "cd /app && timeout 560 venv/bin/python bench/labd_bench.py $tag --ctx $ctx 2>&1 | grep -E 'LABD|Error'"
} -ArgumentList $Tag, $Ctx
$total = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
$s = @()
$t0 = Get-Date
while ($job.State -eq 'Running') {
  $c = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
  $ded = ($c | Where-Object { $_.Path -like '*dedicated*' } | Measure-Object CookedValue -Maximum).Maximum / 1GB
  $shr = ($c | Where-Object { $_.Path -like '*shared*' } | Measure-Object CookedValue -Maximum).Maximum / 1GB
  $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
  $pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object CurrentUsage -Sum).Sum / 1024
  $vm = ((Get-Process vmmemWSL -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum) / 1GB
  $disk = (Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk -Filter "Name='_Total'").PercentDiskTime
  $s += [pscustomobject]@{ t = ((Get-Date) - $t0).TotalSeconds; ded = $ded; shr = $shr; used = $total - $free; pf = $pf; vm = $vm; disk = $disk }
  Start-Sleep -Milliseconds 1500
}
Receive-Job $job; Remove-Job $job
"--- {0} samples over {1:N0} s" -f $s.Count, $s[-1].t
"VRAM dedicated max {0:N2} GB | shared max {1:N2} GB" -f ($s.ded | Measure-Object -Maximum).Maximum, ($s.shr | Measure-Object -Maximum).Maximum
"RAM used: start {0:N1} GB, max {1:N1} GB | page file: start {2:N2} GB, max {3:N2} GB | WSL (vmmemWSL) max {4:N1} GB" -f $s[0].used, ($s.used | Measure-Object -Maximum).Maximum, $s[0].pf, ($s.pf | Measure-Object -Maximum).Maximum, ($s.vm | Measure-Object -Maximum).Maximum
"disk busy: avg {0:N0}% max {1:N0}% | samples above 90%: {2}" -f ($s.disk | Measure-Object -Average).Average, ($s.disk | Measure-Object -Maximum).Maximum, @($s | Where-Object { $_.disk -gt 90 }).Count
