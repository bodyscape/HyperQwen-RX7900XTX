$u = 'http://127.0.0.1:19020/v1/chat/completions'
function Spec {
  $m = (Invoke-WebRequest -Uri http://127.0.0.1:19020/metrics -UseBasicParsing -TimeoutSec 10).Content -split "`n"
  $h = @{}
  foreach ($n in 'num_drafts','num_draft_tokens','num_accepted_tokens') {
    $l = $m | Where-Object { $_ -match "^vllm:spec_decode_${n}_total" } | Select-Object -First 1
    $h[$n] = if ($l) { [double](($l -split ' ')[-1]) } else { 0 }
  }
  $h
}
$prompts = @(
  'Explique la difference entre TCP et UDP, avec un exemple concret pour chacun.',
  'Ecris une fonction Python qui renvoie les nombres premiers inferieurs a n avec le crible d Eratosthene, avec une docstring et deux tests.',
  'Traduis en anglais : Le projet consiste a servir un grand modele de langage sur une carte graphique grand public, en combinant quantification, decodage speculatif et cache clef-valeur compresse.',
  'Donne trois conseils pour bien dormir et explique pourquoi chacun fonctionne.'
)
$tot = 0; $totT = 0.0; $a0 = Spec
foreach ($p in $prompts) {
  $b = @{ model='qwen3.8-27b'; max_tokens=300; chat_template_kwargs=@{enable_thinking=$false}; messages=@(@{role='user'; content=$p}) } | ConvertTo-Json -Depth 6
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 600
  $sw.Stop()
  $n = $r.usage.completion_tokens; $tot += $n; $totT += $sw.Elapsed.TotalSeconds
  "{0,3} tokens in {1:N2}s = {2:N1} tok/s   [{3}]" -f $n, $sw.Elapsed.TotalSeconds, ($n / $sw.Elapsed.TotalSeconds), $p.Substring(0,40)
}
$a1 = Spec
$t = $a1.num_draft_tokens - $a0.num_draft_tokens; $acc = $a1.num_accepted_tokens - $a0.num_accepted_tokens; $d = $a1.num_drafts - $a0.num_drafts
"overall: {0:N1} tok/s ; acceptance {1:P1} ({2} of {3}) ; tokens per step {4:N2}" -f ($tot/$totT), ($acc/$t), $acc, $t, (1 + $acc/$d)
