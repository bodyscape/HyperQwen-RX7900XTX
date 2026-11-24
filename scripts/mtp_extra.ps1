$u = 'http://127.0.0.1:19020/v1/chat/completions'
function Send($prompt, $max) {
  $b = @{ model='qwen3.8-27b'; max_tokens=$max; temperature=0; ignore_eos=$true
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content=$prompt}) } | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 600
}
function Spec {
  $m = (Invoke-WebRequest -Uri http://127.0.0.1:19020/metrics -UseBasicParsing -TimeoutSec 10).Content -split "`n"
  $h = @{}
  foreach ($n in 'num_drafts','num_draft_tokens','num_accepted_tokens') {
    $l = $m | Where-Object { $_ -match "^vllm:spec_decode_${n}_total" } | Select-Object -First 1
    $h[$n] = if ($l) { [double](($l -split ' ')[-1]) } else { 0 }
  }
  $h
}
"--- prefill again, two different ~5k-token prompts (no prefix-cache hit)"
foreach ($seed in 1000, 2000) {
  $words = (1..260 | ForEach-Object { "Le renard brun saute par-dessus le chien paresseux numero $($_ + $seed)." }) -join ' '
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Send ($words + ' Resume en un mot.') 1
  $sw.Stop()
  "{0} prompt tokens in {1:N2}s = {2:N0} tok/s" -f $r.usage.prompt_tokens, $sw.Elapsed.TotalSeconds, ($r.usage.prompt_tokens / $sw.Elapsed.TotalSeconds)
}
"--- repetitive output (counting), where drafts should be easy to accept"
$a = Spec
$sw = [Diagnostics.Stopwatch]::StartNew()
$r = Send 'Compte de 1 a 200, un nombre par ligne, sans rien d autre.' 256
$sw.Stop()
$b = Spec
"{0} tokens in {1:N2}s = {2:N1} tok/s" -f $r.usage.completion_tokens, $sw.Elapsed.TotalSeconds, ($r.usage.completion_tokens / $sw.Elapsed.TotalSeconds)
$t = $b.num_draft_tokens - $a.num_draft_tokens; $acc = $b.num_accepted_tokens - $a.num_accepted_tokens
if ($t -gt 0) { "acceptance: {0:P1} ({1} of {2} draft tokens)" -f ($acc/$t), $acc, $t }
