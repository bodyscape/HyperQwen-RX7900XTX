"""A chaque point de divergence entre frdump_old et frdump_new, probabilites du modele (T=0, un token) pour les mots en concurrence."""
import json, sys, math, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
exec(open('E:/git/HyperQwen-7900XTX/scripts/fr_bench.py', encoding='utf-8').read().split('def metrics')[0])  # prompts, BASE
old = json.load(open('E:/git/HyperQwen-7900XTX/scripts/frdump_old.json', encoding='utf-8'))
new = json.load(open('E:/git/HyperQwen-7900XTX/scripts/frdump_new.json', encoding='utf-8'))
for i, (x, y) in enumerate(zip(old, new)):
    if x == y:
        print(i, 'identique'); continue
    k = next(j for j, (c, d) in enumerate(zip(x, y)) if c != d)
    cut = max(x.rfind(' ', 0, k), x.rfind('\n', 0, k))   # debut du mot en cause
    prefix = x[:cut]
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompts[i]}, {"role": "assistant", "content": prefix}],
                       "continue_final_message": True, "add_generation_prompt": False, "max_tokens": 1, "temperature": 0,
                       "logprobs": True, "top_logprobs": 4, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + '/v1/chat/completions', body, {"Content-Type": "application/json"}), timeout=300))
    top = r['choices'][0]['logprobs']['content'][0]['top_logprobs']
    s = ' | '.join(f"{t['token']!r} p={math.exp(t['logprob']):.3f}" for t in top)
    print(i, 'ancien:', repr(x[cut:cut + 22]), 'nouveau:', repr(y[cut:cut + 22]), '->', s)
