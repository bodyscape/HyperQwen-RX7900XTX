import json, random, sys, time, urllib.request
N = int(sys.argv[1]) if len(sys.argv) > 1 else 45000
API = 'http://127.0.0.1:19020/v1/chat/completions'
random.seed(time.time())
words = ['alpha','beta','gamma','delta','river','stone','cloud','engine','market','garden','signal','window','copper','violet','harbor','ticket']
def post(prompt, mt):
    body = {'model': 'qwen3.8-27b', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': mt, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(API, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); return time.time() - t, r
# ~1 token per word with this vocabulary plus a random number every 8 words
text = ' '.join((random.choice(words) + (str(random.randint(0, 999)) if i % 8 == 0 else '')) for i in range(N))
dt, r = post(text + '\nSay OK.', 1)
print(f"prefill request: {dt:.1f}s prompt_tokens={r['usage']['prompt_tokens']} -> {r['usage']['prompt_tokens']/dt:.0f} tok/s")
time.sleep(1)
for _ in range(2): post('Write a long story about a cat.', 350)
