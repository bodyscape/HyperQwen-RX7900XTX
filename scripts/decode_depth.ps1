param([int[]]$Lines = @(800), [int]$Tokens = 256, [int]$Port = 19020)
$u = "http://127.0.0.1:$Port/v1/chat/completions"
function Send($content, $max) {
  $b = @{ model='qwen3.8-27b'; max_tokens=$max; temperature=0
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
foreach ($n in $Lines) {
  $rng = [Random]::new($n)
  $sb = [Text.StringBuilder]::new()
  for ($i = 0; $i -lt $n; $i++) {
    [void]$sb.Append("Fait numero $($i + $n * 10): la valeur de la cle k$($rng.Next(100000,999999)) est $($rng.Next(100000,999999)) dans le registre $($rng.Next(10,99)).`n")
    if ($i -eq 5) { [void]$sb.Append("NOTE IMPORTANTE: le code secret du coffre est 7391-ALPHA.`n") }
  }
  $doc = $sb.ToString()
  $q = $doc + "`nQuel est le code secret du coffre ? Reponds par le code, puis ecris une longue explication."
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r0 = Send $q 1
  $sw.Stop()
  $pt = $r0.usage.prompt_tokens
  "[{0} lines] prompt {1} tokens, cold prefill {2:N1}s = {3:N0} tok/s" -f $n, $pt, $sw.Elapsed.TotalSeconds, ($pt / $sw.Elapsed.TotalSeconds)
  [void](Send $q 8)   # settle the cache
  foreach ($rep in 1..2) {
    $a0 = Spec
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Send $q $Tokens
    $sw.Stop()
    $a1 = Spec
    $dr = $a1.num_drafts - $a0.num_drafts; $dt = $a1.num_draft_tokens - $a0.num_draft_tokens; $ac = $a1.num_accepted_tokens - $a0.num_accepted_tokens
    $cached = $r.usage.prompt_tokens_details.cached_tokens
    "[{0} lines] rep{1}: needle {2} | {3} tokens in {4:N2}s = {5:N1} tok/s | cached {6}/{7} | accept {8:P0} | tok/step {9:N2}" -f $n, $rep, ($r.choices[0].message.content -match '7391'), $r.usage.completion_tokens, $sw.Elapsed.TotalSeconds, ($r.usage.completion_tokens / $sw.Elapsed.TotalSeconds), $cached, $r.usage.prompt_tokens, ($ac / [math]::Max(1, $dt)), (1 + $ac / [math]::Max(1, $dr))
  }
}
