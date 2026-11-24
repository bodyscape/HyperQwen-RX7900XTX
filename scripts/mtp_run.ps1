function Spec {
  $m = (Invoke-WebRequest -Uri http://127.0.0.1:19020/metrics -UseBasicParsing -TimeoutSec 10).Content -split "`n"
  $h = @{}
  foreach ($n in 'num_drafts','num_draft_tokens','num_accepted_tokens') {
    $l = $m | Where-Object { $_ -match "^vllm:spec_decode_${n}_total" } | Select-Object -First 1
    $h[$n] = if ($l) { [double](($l -split ' ')[-1]) } else { 0 }
  }
  $h
}
$a = Spec
& '<SCRATCHPAD>\req27.ps1'
$b = Spec
$d = $b.num_drafts - $a.num_drafts; $t = $b.num_draft_tokens - $a.num_draft_tokens; $acc = $b.num_accepted_tokens - $a.num_accepted_tokens
"--- MTP counters over the run above"
"drafts: $d, draft tokens: $t, accepted: $acc"
if ($t -gt 0) { "acceptance rate: {0:P1}; accepted per draft: {1:N2} (+1 sampled token per step)" -f ($acc/$t), ($acc/$d) }
