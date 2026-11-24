"""Debit et acceptation MTP sur des prompts francais (autres sujets que le corpus), un flux, T=0, thinking off."""
import json, re, sys, time, urllib.request

BASE = 'http://127.0.0.1:19020'
prompts = ["Explique comment fonctionne une pompe à chaleur et quand elle est rentable.",
           "Rédige une lettre de motivation pour un poste de technicien en informatique.",
           "Raconte, en un texte suivi, l'histoire de la construction du canal de Panama.",
           "Quelles sont les différences entre un REER et un CELI ? Explique avec des exemples.",
           "Décris en détail les étapes pour préparer une pâte à crêpes réussie et ses variantes.",
           "Écris un article de blogue sur les avantages et les limites du télétravail.",
           "Explique le fonctionnement d'un moteur électrique et pourquoi il est plus efficace.",
           "Fais un plan détaillé et rédigé pour apprendre la guitare en six mois."]


def metrics():
    t = urllib.request.urlopen(BASE + '/metrics').read().decode()
    g = lambda n: sum(float(x) for x in re.findall(r'^vllm:' + n + r'\{[^}]*\} ([0-9.e+]+)$', t, re.M))
    pos = [float(x) for x in re.findall(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*\} ([0-9.e+]+)$', t, re.M)]
    return g('spec_decode_num_drafts_total'), g('spec_decode_num_accepted_tokens_total'), pos


def ask(p):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": p}], "max_tokens": 400,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(BASE + '/v1/chat/completions', body, {"Content-Type": "application/json"}), timeout=300)
    return json.load(r)["usage"]["completion_tokens"]


ask(prompts[0])  # echauffement
d0, a0, p0 = metrics()
t0 = time.time(); n = 0
for p in prompts:
    n += ask(p)
dt = time.time() - t0
d1, a1, p1 = metrics()
dd = d1 - d0
print(f'tag={sys.argv[1] if len(sys.argv) > 1 else "?"}  {n} tokens en {dt:.1f}s = {n/dt:.1f} tok/s | tokens/pas={1 + (a1 - a0) / dd:.2f} | '
      f'acceptation par position=' + ' '.join(f'{(b - a) / dd:.2f}' for a, b in zip(p0, p1)))
