param([string]$Conv_BN = '256', [string]$Conv_W = '4', [string]$BV = '32', [string]$GW = '4', [string]$Label = 'baseline')
$S = "<SCRATCHPAD>"
Set-Location E:\git\hyperqwen
@"
SPEC=mtp
MAX_LEN=8192
MAX_SEQS=1
GPU_UTIL=0.90
DRAFT_TOKENS=3
CUDAGRAPH_MODE=PIECEWISE
HQ_CONV_BLOCK_N=$Conv_BN
HQ_CONV_WARPS=$Conv_W
HQ_GDN_BV=$BV
HQ_GDN_WARPS=$GW
"@ | Set-Content -Encoding ascii E:\git\hyperqwen\.env
docker compose -f docker-compose.yml -f "$S\ovr_tune.yml" up -d --force-recreate rocm 2>&1 | Select-Object -Last 1
$t = 0
while ($t -lt 420) {
  try { $r = Invoke-WebRequest http://127.0.0.1:19020/health -UseBasicParsing -TimeoutSec 3; if ($r.StatusCode -eq 200) { "healthy after ${t}s"; break } } catch {}
  $st = docker inspect -f '{{.State.Status}}' hyperqwen-rocm-1; if ($st -match 'exited|restarting') { "container: $st"; docker logs hyperqwen-rocm-1 2>&1 | Select-String 'Error|Traceback' | Select-Object -Last 5; return }
  Start-Sleep 5; $t += 5
}
function Req($p, $n) { $b = @{ model = 'qwen3.8-27b'; max_tokens = $n; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false }; messages = @(@{ role = 'user'; content = $p }) } | ConvertTo-Json -Depth 6; Invoke-RestMethod -Uri http://127.0.0.1:19020/v1/chat/completions -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 300 }
[void](Req 'Dis bonjour.' 20); [void](Req 'Explique TCP et UDP.' 200)
docker exec hyperqwen-rocm-1 bash -c 'rm -f /tmp/hq_timing.json; touch /tmp/hq_timing.reset'
Start-Sleep 1
$sw = [Diagnostics.Stopwatch]::StartNew(); $tok = 0
foreach ($p in 'Explique la difference entre TCP et UDP avec un exemple concret pour chacun.', 'Ecris une fonction Python qui renvoie les nombres premiers inferieurs a n avec une docstring.', 'Donne trois conseils pour bien dormir et explique pourquoi chacun fonctionne.') {
  $r = Req $p 300; $tok += $r.usage.completion_tokens
}
$sw.Stop(); Start-Sleep 2
"[{0}] conv BN={1} w={2} | gdn BV={3} w={4}: {5} tokens in {6:N1}s = {7:N1} tok/s (PIECEWISE, instrumented)" -f $Label, $Conv_BN, $Conv_W, $BV, $GW, $tok, $sw.Elapsed.TotalSeconds, ($tok / $sw.Elapsed.TotalSeconds)
docker exec hyperqwen-rocm-1 python3 -c "import json; d=json.load(open('/tmp/hq_timing.json'))['totals']; [print('   %-34s %6.1f us avg  (%d calls)' % (k, v['avg_us'], v['calls'])) for k, v in sorted(d.items())]"
