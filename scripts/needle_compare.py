"""Three needles (start, middle, end) in a synthetic document; one question each, greedy, thinking off.
usage: needle_compare.py <base_url> <lines> <tag>      (the document is the same for every server: seeded)
"""
import json, random, re, sys, time, urllib.request

base, n, tag = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]
rng = random.Random(11)
needles = {5: ("A", "7391-ALPHA"), n // 2: ("B", "4820-BRAVO"), n - 7: ("C", "9156-CHARLIE")}
out = []
for i in range(n):
    out.append("Fait numero %d: la valeur de la cle k%d est %d dans le registre %d." % (i, rng.randint(100000, 999999), rng.randint(100000, 999999), rng.randint(10, 99)))
    if i in needles:
        k, code = needles[i]
        out.append("NOTE IMPORTANTE: le code secret du coffre %s est %s." % (k, code))
doc = "\n".join(out)


def ask(q, max_tokens=40):
    body = json.dumps({"model": "qwen3.8-27b", "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": doc + "\n\n" + q}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    return r, time.perf_counter() - t0


ok = 0
for pos, (k, code) in needles.items():
    r, dt = ask("Quel est le code secret du coffre %s ? Reponds uniquement par le code." % k)
    txt = r["choices"][0]["message"]["content"] or ""
    hit = code in txt
    ok += hit
    print("NEEDLE %-8s coffre %s (ligne %5d/%d) prompt=%d tokens | %s | %.1fs | reponse: %r"
          % (tag, k, pos, n, r["usage"]["prompt_tokens"], "OK" if hit else "RATE", dt, txt.strip()[:40]), flush=True)
print("NEEDLE %-8s TOTAL %d/3" % (tag, ok))
