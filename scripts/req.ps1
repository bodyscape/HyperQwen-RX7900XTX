$u = 'http://127.0.0.1:19020/v1/chat/completions'
function Send($prompt, $max) {
  $b = @{ model='test'; max_tokens=$max; temperature=0; ignore_eos=$true
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content=$prompt}) } | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 120
}
$r = Send 'Quelle est la capitale de la France ? Reponds en une phrase.' 40
"reply: " + $r.choices[0].message.content.Trim().Replace("`n",' ')
"--- single stream, 256 tokens, x3"
1..3 | ForEach-Object {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Send 'Ecris une longue histoire sur un robot qui apprend a cuisiner.' 256
  $sw.Stop()
  "{0}: {1} tokens in {2:N2}s = {3:N1} tok/s" -f $_, $r.usage.completion_tokens, $sw.Elapsed.TotalSeconds, ($r.usage.completion_tokens / $sw.Elapsed.TotalSeconds)
}
"--- 8 concurrent, 256 tokens each"
$sw = [Diagnostics.Stopwatch]::StartNew()
$jobs = 1..8 | ForEach-Object { Start-Job -ScriptBlock {
  param($u)
  $b = @{ model='test'; max_tokens=256; temperature=0; ignore_eos=$true
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content='Ecris une longue histoire sur un robot qui apprend a cuisiner.'}) } | ConvertTo-Json -Depth 6
  (Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 180).usage.completion_tokens
} -ArgumentList $u }
$tok = ($jobs | Wait-Job | Receive-Job | Measure-Object -Sum).Sum
$sw.Stop()
"{0} tokens in {1:N2}s = {2:N1} tok/s aggregate" -f $tok, $sw.Elapsed.TotalSeconds, ($tok / $sw.Elapsed.TotalSeconds)
