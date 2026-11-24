"""Sauvegarde les sorties greedy (T=0, 400 tokens) des prompts francais de fr_bench.py: python fr_dump.py <tag>"""
import json, sys, urllib.request
sys.argv_tag = sys.argv[1]
exec(open('E:/git/HyperQwen-RX7900XTX/scripts/fr_bench.py', encoding='utf-8').read().split('def metrics')[0])  # prompts, BASE


def ask(p):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": p}], "max_tokens": 400,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(BASE + '/v1/chat/completions', body, {"Content-Type": "application/json"}), timeout=300)
    return json.load(r)["choices"][0]["message"]["content"]


json.dump([ask(p) for p in prompts], open(f'E:/git/HyperQwen-RX7900XTX/scripts/frdump_{sys.argv[1]}.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('ok', sys.argv[1])
