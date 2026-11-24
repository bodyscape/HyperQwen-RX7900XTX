"""Liste de brouillon completee en francais: ids de base + les K tokens les plus frequents du corpus francais qui n'y sont pas.
Usage (dans le conteneur): python fr_vocab.py <model_dir> <base_ids.json> <corpus.jsonl> <out.json> [K]"""
import json, sys, collections
from tokenizers import Tokenizer

model, base_f, corpus_f, out_f = sys.argv[1:5]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
tok = Tokenizer.from_file(model.rstrip('/') + '/tokenizer.json')
base = json.load(open(base_f))
base = base['ids'] if isinstance(base, dict) else base
bset = set(base)
train, held = collections.Counter(), collections.Counter()
for j, line in enumerate(open(corpus_f, encoding='utf-8')):
    ids = tok.encode(json.loads(line)['text']).ids
    (held if j % 10 == 0 else train).update(ids)
nh = sum(held.values())
print('train tokens', sum(train.values()), 'held tokens', nh, 'base coverage held %.2f%%' % (100 * sum(c for t, c in held.items() if t in bset) / nh))
whole = lambda t: '�' not in tok.decode([t])
cand = [t for t, _ in train.most_common() if t not in bset and whole(t)]
print('candidates', len(cand))
for k in (1000, 2000, 4000, 8000, 16000):
    s = bset | set(cand[:k])
    print(f'  +{k:6d} ids -> {len(s)} ids, held coverage {100 * sum(c for t, c in held.items() if t in s) / nh:.2f}%')
ids = list(base) + cand[:K]
# La tete de brouillon passe par les noyaux W4A16 (N multiple de 8, et de 128 pour de bons rendements): on complete avec
# des tokens latins (lettres, accents) non encore listes, par id croissant.
import re
used = set(ids)
latin = re.compile(r"^[ ▁]?[A-Za-zÀ-ſ]+$")
i = 0
while len(ids) % 128:
    while i in used or not (whole(i) and latin.match(tok.decode([i]) or '')):
        i += 1
    ids.append(i); used.add(i)
print('complete a', len(ids), 'ids (multiple de 128)')
json.dump(ids, open(out_f, 'w'))
print('ecrit', out_f, len(ids), 'ids')
