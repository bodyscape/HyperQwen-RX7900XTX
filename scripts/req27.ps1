$u = 'http://127.0.0.1:19020/v1/chat/completions'
function Send($prompt, $max, $ignoreEos) {
  $b = @{ model='qwen3.8-27b'; max_tokens=$max; temperature=0
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content=$prompt}) }
  if ($ignoreEos) { $b.ignore_eos = $true }
  Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body ($b | ConvertTo-Json -Depth 6) -TimeoutSec 600
}
"--- correctness"
$r = Send 'Quelle est la capitale du Danemark ? Reponds en une phrase.' 60 $false
"reply: " + $r.choices[0].message.content.Trim().Replace("`n",' ')
$r = Send 'Combien font 17 * 23 ? Donne seulement le nombre.' 20 $false
"17*23: " + $r.choices[0].message.content.Trim().Replace("`n",' ')
"--- single stream, 256 tokens x3 (first one includes warmup)"
1..3 | ForEach-Object {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Send 'Ecris une longue histoire sur un robot qui apprend a cuisiner.' 256 $true
  $sw.Stop()
  "{0}: {1} tokens in {2:N2}s = {3:N1} tok/s" -f $_, $r.usage.completion_tokens, $sw.Elapsed.TotalSeconds, ($r.usage.completion_tokens / $sw.Elapsed.TotalSeconds)
}
"--- prefill: ~2000-token prompt, 1 output token"
$words = (1..260 | ForEach-Object { "Le renard brun saute par-dessus le chien paresseux numero $_." }) -join ' '
$sw = [Diagnostics.Stopwatch]::StartNew()
$r = Send ($words + ' Resume en un mot.') 1 $false
$sw.Stop()
"{0} prompt tokens in {1:N2}s = {2:N0} tok/s (cold)" -f $r.usage.prompt_tokens, $sw.Elapsed.TotalSeconds, ($r.usage.prompt_tokens / $sw.Elapsed.TotalSeconds)
"--- 2 concurrent, 256 tokens each"
$sw = [Diagnostics.Stopwatch]::StartNew()
$jobs = 1..2 | ForEach-Object { Start-Job -ScriptBlock {
  param($u)
  $b = @{ model='qwen3.8-27b'; max_tokens=256; temperature=0; ignore_eos=$true
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content='Ecris une longue histoire sur un robot qui apprend a cuisiner.'}) } | ConvertTo-Json -Depth 6
  (Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 600).usage.completion_tokens
} -ArgumentList $u }
$tok = ($jobs | Wait-Job | Receive-Job | Measure-Object -Sum).Sum
$sw.Stop()
"{0} tokens in {1:N2}s = {2:N1} tok/s aggregate" -f $tok, $sw.Elapsed.TotalSeconds, ($tok / $sw.Elapsed.TotalSeconds)
