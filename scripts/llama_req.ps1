param([int[]]$Lines = @(), [int]$Port = 19030, [switch]$Chat)
$u = "http://127.0.0.1:$Port/v1/chat/completions"
function Send($content, $max, $temp, $ignore) {
  $b = @{ model='x'; max_tokens=$max; temperature=$temp
          chat_template_kwargs=@{enable_thinking=$false}
          messages=@(@{role='user'; content=$content}) }
  if ($temp -gt 0) { $b.top_p = 0.95; $b.top_k = 20 }
  if ($ignore) { $b.ignore_eos = $true }
  $json = $b | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri $u -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 3000
}
function Fmt($r) {
  $t = $r.timings
  "pp {0,6:N0} tok/s ({1} tok) | tg {2,5:N1} tok/s ({3} tok) | draft {4}/{5}" -f $t.prompt_per_second, $t.prompt_n, $t.predicted_per_second, $t.predicted_n, $t.draft_n_accepted, $t.draft_n
}
if ($Chat) {
  "--- warm-up (not counted)"
  [void](Send 'Dis bonjour en une phrase.' 20 0 $false)
  "--- 4 chat prompts, 300 tokens, temp 1.0 top_p 0.95 top_k 20 (the Reddit sampling)"
  $prompts = @(
    'Explique la difference entre TCP et UDP, avec un exemple concret pour chacun.',
    'Ecris une fonction Python qui renvoie les nombres premiers inferieurs a n avec le crible d Eratosthene, avec une docstring et deux tests.',
    'Traduis en anglais : Le projet consiste a servir un grand modele de langage sur une carte graphique grand public, en combinant quantification, decodage speculatif et cache clef-valeur compresse.',
    'Donne trois conseils pour bien dormir et explique pourquoi chacun fonctionne.')
  $tg = @(); $acc = 0; $dr = 0
  foreach ($p in $prompts) {
    $r = Send $p 300 1.0 $false
    "{0}   [{1}]" -f (Fmt $r), $p.Substring(0,36)
    $tg += $r.timings.predicted_per_second; $acc += $r.timings.draft_n_accepted; $dr += $r.timings.draft_n
  }
  "mean tg: {0:N1} tok/s ; draft acceptance {1:P1} ({2} of {3})" -f (($tg | Measure-Object -Average).Average), ($acc/[math]::Max(1,$dr)), $acc, $dr
  "--- counting 1..200 (predictable output), temp 0"
  $r = Send 'Compte de 1 a 200, un nombre par ligne, sans rien d autre.' 256 0 $false
  Fmt $r
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
  $r1 = Send ($doc + "`nResume ce document en un mot.") 1 0 $false
  $sw.Stop()
  "[{0} lines] cold prefill: {1:N1}s wall | {2}" -f $n, $sw.Elapsed.TotalSeconds, (Fmt $r1)
  $r2 = Send ($doc + "`nQuel est le code secret du coffre ? Reponds par le code, puis ecris une longue explication.") 128 0 $true
  "[{0} lines] needle found: {1} | {2}" -f $n, ($r2.choices[0].message.content -match '7391'), (Fmt $r2)
}
