"""Genere un petit corpus francais avec le serveur local (sorties du modele, thinking off) pour mesurer la couverture
du vocabulaire de brouillon MTP. Ecrit scripts/fr_corpus.jsonl (un texte par ligne)."""
import json, sys, itertools, concurrent.futures as cf, urllib.request

URL = 'http://127.0.0.1:19020/v1/chat/completions'
topics = ["la cuisine québécoise", "le droit du travail", "la photosynthèse", "l'histoire de la Nouvelle-France",
          "un moteur à combustion", "l'apprentissage automatique", "la gestion de projet", "la météo et le climat",
          "un budget familial", "la santé cardiaque", "le hockey", "la fiscalité des particuliers",
          "l'architecture logicielle", "les réseaux informatiques", "la musique classique", "l'agriculture bio",
          "un voyage en train", "la physique quantique", "le marché immobilier", "les bases de données"]
tasks = ["Explique en détail, en français, {t}.",
         "Rédige un courriel professionnel en français au sujet de {t}.",
         "Écris un court dialogue en français entre deux personnes qui parlent de {t}.",
         "Donne dix conseils pratiques, en français, sur {t}, avec une courte explication chacun.",
         "Résume en français ce qu'un débutant doit savoir sur {t}."]
prompts = [tk.format(t=t) for t, tk in itertools.product(topics, tasks)]  # 100
prompts += ["Écris une fonction Python qui trie une liste de dictionnaires, avec des commentaires et une explication en français.",
            "Explique en français comment configurer un serveur web avec Docker, étape par étape.",
            "Corrige les fautes de ce texte et explique en français : « Je suis aller au marcher hier et j'ai acheter des pomme »."]


def ask(p):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": p}], "max_tokens": 450,
                       "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:
        print('ERR', e, file=sys.stderr)
        return None


out = open('E:/git/HyperQwen-RX7900XTX/scripts/fr_corpus.jsonl', 'w', encoding='utf-8')
n = 0
with cf.ThreadPoolExecutor(4) as ex:
    for txt in ex.map(ask, prompts):
        if txt:
            out.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
            n += 1
            if n % 10 == 0:
                print(n, 'textes', flush=True)
out.close()
print('fini', n)
