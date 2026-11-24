# Prompt pour l'agent de la RTX 3090 : liste de brouillon MTP completee en francais

Contexte : sur le port RX 7900 XTX de HyperQwen (E:\git\HyperQwen-7900XTX), un changement ne touchant QUE les donnees du modele (3 fichiers) a donne +24 a +29 % de vitesse de decodage sur du texte francais, sans perte de qualite. Ta mission : verifier si le meme gain existe sur la RTX 3090 et l'appliquer proprement. Ne modifie rien d'autre (pas de .env, pas de docker-compose.yml, pas de noyaux, pas de patches).

## 1. Le probleme

La decodage speculatif MTP de HyperQwen (Qwen3.8-27B-W4A16-AutoRound-fast, 3 brouillons, verification M=4 lignes) n'utilise pas la lm_head complete pour proposer les brouillons. Il utilise une tete reduite : "draft vocab" de 40 960 ids choisis dans le vocabulaire (livres dans `prepare/draft_vocab_ids.json`, construits surtout sur de l'anglais et du code).
- Fichiers concernes dans le dossier du modele `...-fast` : `model_extra_tensors.safetensors` (contient `mtp.draft_lm_head.*`, lignes int4 decoupees dans la lm_head), `mtp_draft_vocab_ids.pt` (liste des ids), `model.safetensors.index.json` (index qui doit pointer vers les tenseurs).
- Un token hors de cette liste ne peut JAMAIS etre propose en brouillon : la chaine de brouillons s'arrete la. Sur du francais, seulement ~76 % des tokens d'un texte etaient dans la liste (mesure sur un jeu mis de cote) -> 2,01 tokens acceptes par etape.
- Variable de repli existante : `MTP_DRAFT_VOCAB=0` utilise la lm_head complete.

## 2. Ce qui a ete fait (cote 7900 XTX)

1. Corpus francais : `scripts/fr_corpus.py` a genere `scripts/fr_corpus.jsonl` = 103 textes (175 Ko) produits par le modele lui-meme sur des prompts francais varies (une ligne JSON `{"text": ...}` par texte). Mis de cote pour la validation : 1 texte sur 10 (j % 10 == 0), les 9 autres servent a compter les frequences.
2. Candidats : `scripts/fr_vocab.py <model_dir> <base_ids.json> <corpus.jsonl> <out.json> [K]` (a lancer dans le conteneur, a besoin de `tokenizers`) :
   - lit les 40 960 ids de base ;
   - compte les tokens du corpus d'entrainement, garde les plus frequents absents de la base, en excluant les tokens qui ne decodent pas en caracteres entiers (pas de �) ;
   - ajoute ces tokens a la base, puis COMPLETE avec des tokens latins (lettres/accents, par id croissant) jusqu'a un multiple de 128. Raison : la tete de brouillon passe par des noyaux W4A16 qui veulent N multiple de 8 (et 128 pour de bons rendements). Resultat : 43 904 ids = 40 960 base + 2 852 francais + 92 remplissage latin. Couverture sur le jeu mis de cote : 76 % -> 96 %.
3. Reconstruction de la tete : `prepare/build_draft_vocab.py` (options `--ids`, `--stats`, `--corpus`) avec `--ids` pointant sur la liste ci-dessus. Il reecrit les trois fichiers (extras, ids .pt, index), publie de facon atomique et fait une sauvegarde unique `.bak-draft` (`backup_once`). Tete finale : 116 Mo (+7 Mo). Sur la 7900 XTX j'ai aussi garde des copies `*.bak-fr` des 3 fichiers d'origine.
4. Redemarrage du serveur (le conteneur est recree a chaque start).

## 3. Resultats mesures (7900 XTX, un flux, T=0)

| | avant | apres |
|---|---|---|
| tokens/etape (francais) | 2,01 | 2,43 |
| vitesse francais | 71,5 tok/s | 89-92 tok/s |
| anglais | ~98-101 tok/s | pas de regression |
| GSM8K (n=200) | 95,5 % | 96,0 % |

Qualite : le decodage speculatif est exact (le modele complet verifie chaque token), donc aucune perte par construction. Les sorties greedy different entre deux configs sur quelques prompts : ce sont des quasi-egalites (probabilites tres proches) que l'arrondi bf16 fait basculer selon les lignes verifiees ensemble ; verifie avec `scripts/fr_ties.py`. Ce n'est pas une baisse de qualite.

## 4. Ce que tu dois faire sur la 3090

Important : je n'ai PAS verifie que la 3090 utilise le meme checkpoint, ni que son noyau Marlin accepte N=43 904 pour cette tete. A verifier, pas a supposer.

1. Lecture seule d'abord : trouve le dossier du modele `-fast` utilise par le conteneur `repo-single-1` (compose dans `E:\HyperQwen\repo`). Compare les sha256 de ses `model_extra_tensors.safetensors` / `mtp_draft_vocab_ids.pt` / index avec les originaux `*.bak-fr` de la 7900 XTX. Identiques -> tu peux copier les 3 fichiers modifies. Differents -> reconstruis avec `prepare/build_draft_vocab.py` + la liste d'ids sur TON checkpoint (ne copie pas mes tenseurs).
2. Sauvegarde les 3 fichiers d'origine de la 3090 avant toute ecriture (ex. `*.bak-fr`).
3. Mesure AVANT (une seule fois) : un flux, T=0, memes prompts francais (`scripts/fr_bench.py <tag>` : tok/s, tokens/etape, acceptation par position ; adapte la variable BASE a l'URL locale) + un test anglais + si possible GSM8K.
4. Applique, redemarre (`hyperqwen-start.ps1`), mesure APRES. Attendu (non garanti) : tokens/etape francais ~2,4 contre ~2,0, anglais inchange. Le surcout memoire (+7 Mo) est negligeable ; garde la marge VRAM habituelle.
5. Si le demarrage echoue, si l'anglais baisse, ou si GSM8K baisse de facon significative : restaure les 3 fichiers sauvegardes et redemarre. Rien n'est perdu.
6. Rapporte : sha256 compares, tokens/etape et tok/s avant/apres (francais et anglais), GSM8K, et toute erreur au demarrage.

Interdits : ne touche pas au `.env`, au `docker-compose.yml`, aux patches, aux noyaux ; pas de test GPU long non demande ; previens avant un test lourd.
