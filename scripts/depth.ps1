param([int[]]$Lines = @(100, 800), [int]$DecodeTokens = 128)
$u = 'http://127.0.0.1:19020/v1/chat/completions'
function Send($content, $max, $ignore) {
  $b = @{ model='qwen3.8-27b'; max_tokens=$max; temperature=0
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content=$content}) }
  if ($ignore) { $b.ignore_eos = $true }
  $json = $b | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 3000
}
$rng = [Random]::new(7)
foreach ($n in $Lines) {
  $sb = [Text.StringBuilder]::new()
  for ($i = 0; $i -lt $n; $i++) {
    [void]$sb.Append("Fait numero $($i + $n * 10): la valeur de la cle k$($rng.Next(100000,999999)) est $($rng.Next(100000,999999)) dans le registre $($rng.Next(10,99)).`n")
    if ($i -eq 5) { [void]$sb.Append("NOTE IMPORTANTE: le code secret du coffre est 7391-ALPHA.`n") }
  }
  $doc = $sb.ToString()
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r1 = Send ($doc + "`nResume ce document en un mot.") 1 $false
  $sw.Stop()
  $pt = $r1.usage.prompt_tokens
  "[{0} lines] prompt {1} tokens, cold prefill {2:N1}s = {3:N0} tok/s" -f $n, $pt, $sw.Elapsed.TotalSeconds, ($pt / $sw.Elapsed.TotalSeconds)
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r2 = Send ($doc + "`nQuel est le code secret du coffre ? Reponds par le code, puis ecris une longue explication.") $DecodeTokens $true
  $sw.Stop()
  $ans = $r2.choices[0].message.content
  $ok = $ans -match '7391'
  "[{0} lines] needle found: {1}; decode {2} tokens in {3:N2}s = {4:N1} tok/s (prefix cached, includes ~1 short prefill)" -f $n, $ok, $r2.usage.completion_tokens, $sw.Elapsed.TotalSeconds, ($r2.usage.completion_tokens / $sw.Elapsed.TotalSeconds)
  "   cached prompt tokens: " + $r2.usage.prompt_tokens_details.cached_tokens
}
