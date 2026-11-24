param([int[]]$Lines = @(1850), [int]$Port = 19020)
# Pure decode rate at depth: same cached prompt, max_tokens 64 vs 320; the prefill and the first 64 tokens cancel.
$u = "http://127.0.0.1:$Port/v1/chat/completions"
function Send($content, $max) {
  $b = @{ model='qwen3.8-27b'; max_tokens=$max; temperature=0; ignore_eos=$true
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content=$content}) }
  $json = $b | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 3000
}
function Spec {
  $m = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/metrics" -UseBasicParsing -TimeoutSec 10).Content -split "`n"
  $h = @{}
  foreach ($n in 'num_drafts','num_draft_tokens','num_accepted_tokens') {
    $l = $m | Where-Object { $_ -match "^vllm:spec_decode_${n}_total" } | Select-Object -First 1
    $h[$n] = if ($l) { [double](($l -split ' ')[-1]) } else { 0 }
  }
  $h
}
function Timed($q, $max) {
  $a0 = Spec; $sw = [Diagnostics.Stopwatch]::StartNew(); $r = Send $q $max; $sw.Stop(); $a1 = Spec
  [pscustomobject]@{ t = $sw.Elapsed.TotalSeconds; n = $r.usage.completion_tokens; drafts = $a1.num_drafts - $a0.num_drafts; acc = $a1.num_accepted_tokens - $a0.num_accepted_tokens; cached = $r.usage.prompt_tokens_details.cached_tokens; pt = $r.usage.prompt_tokens }
}
foreach ($n in $Lines) {
  $rng = [Random]::new($n)
  $sb = [Text.StringBuilder]::new()
  for ($i = 0; $i -lt $n; $i++) {
    [void]$sb.Append("Fait numero $($i + $n * 10): la valeur de la cle k$($rng.Next(100000,999999)) est $($rng.Next(100000,999999)) dans le registre $($rng.Next(10,99)).`n")
    if ($i -eq 5) { [void]$sb.Append("NOTE IMPORTANTE: le code secret du coffre est 7391-ALPHA.`n") }
  }
  $q = $sb.ToString() + "`nQuel est le code secret du coffre ? Reponds par le code, puis ecris une longue explication."
  $sw = [Diagnostics.Stopwatch]::StartNew(); $r0 = Send $q 1; $sw.Stop()
  "[{0} lines] prompt {1} tokens, cold prefill {2:N1}s = {3:N0} tok/s" -f $n, $r0.usage.prompt_tokens, $sw.Elapsed.TotalSeconds, ($r0.usage.prompt_tokens / $sw.Elapsed.TotalSeconds)
  [void](Send $q 8)
  $short = @(); $long = @()
  foreach ($rep in 1..2) { $short += Timed $q 64; $long += Timed $q 320 }
  $ts = ($short | Measure-Object t -Average).Average; $tl = ($long | Measure-Object t -Average).Average
  $dn = ($long[0].n - $short[0].n)
  $dd = ($long[0].drafts - $short[0].drafts); $da = ($long[0].acc - $short[0].acc)
  "[{0} lines] pure decode: {1} extra tokens in {2:N2}s = {3:N1} tok/s | tok/step {4:N2} | step {5:N1} ms | cached {6}/{7}" -f $n, $dn, ($tl - $ts), ($dn / ($tl - $ts)), (1 + $da / [math]::Max(1, $dd)), (1000 * ($tl - $ts) / [math]::Max(1, $dd)), $long[0].cached, $long[0].pt
}
