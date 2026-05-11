# GPT-Naylis v1 — NaylisAttention

Modèle de langage GPT construit sur une attention hybride Token-Graph originale : **NaylisAttention**.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du modèle](#architecture-du-modèle)
3. [NaylisAttention — Innovation principale](#naylisattention--innovation-principale)
4. [Composants Core](#composants-core)
5. [Boucle d'entraînement et d'inférence](#boucle-dentraînement-et-dinférence)
6. [Backends d'attention — Détection hiérarchique](#backends-dattention--détection-hiérarchique)
7. [Optimiseur — Muon + MARS-M](#optimiseur--muon--mars-m)
8. [Scheduler WSD](#scheduler-wsd)
9. [Benchmark — lm-evaluation-harness](#benchmark--lm-evaluation-harness)
10. [Structure du projet](#structure-du-projet)
11. [Installation et requirements](#installation-et-requirements)
12. [Résumé des choix architecturaux](#résumé-des-choix-architecturaux)
13. [Licence](#licence)

---

## Vue d'ensemble

GPT-Naylis est un modèle de langage autorégressif conçu pour explorer une nouvelle forme d'attention combinant l'attention classique à scalable dot-product avec un **biais de graphe asymétrique learnable** (NaylisAttention).

---

## Architecture du modèle

```
NaylisGPT
├── token_embeddings     [vocab_size × embed_dim]   (weight-tied avec output_head)
├── NaylisBlock × num_layer
│   ├── ln1              RMSNorm(embed_dim)
│   ├── attention        NaylisAttention
│   │   ├── q_proj       [embed_dim → embed_dim]
│   │   ├── k_proj       [embed_dim → kv_dim]       (GQA)
│   │   ├── v_proj       [embed_dim → kv_dim]
│   │   ├── out_proj     [embed_dim → embed_dim]
│   │   ├── q_norm       RMSNorm(head_dim)           (QK Norm)
│   │   ├── k_norm       RMSNorm(head_dim)
│   │   ├── rope         RotaryPositionalEmbedding   (YaRN optionnel)
│   │   ├── rel_q_proj   [embed_dim → num_heads × rel_rank]   ← Naylis
│   │   ├── rel_k_proj   [embed_dim → num_heads × rel_rank]   ← Naylis
│   │   └── graph_scale  [num_heads]  init=0                  ← Naylis
│   ├── ln2              RMSNorm(embed_dim)
│   └── ffn              FeedForward (SwiGLU)
│       ├── gate_proj    [embed_dim → hidden_dim]
│       ├── up_proj      [embed_dim → hidden_dim]
│       └── down_proj    [hidden_dim → embed_dim]
├── ln_final             RMSNorm(embed_dim)
└── output_head          [embed_dim → vocab_size]    (poids partagés avec token_embeddings)
```

---

## NaylisAttention — Innovation principale

### Principe — Option 1.5

Toutes les dimensions participent à l'attention classique (dot-product standard). En parallèle, deux **projecteurs relationnels séparés** (`rel_q_proj` et `rel_k_proj`) extraient des canaux relationnels depuis la représentation complète, et produisent un **biais asymétrique** injecté directement dans les scores d'attention.

### Biais asymétrique

```
B[i,j] = <R_q(i), R_k(j)>  ≠  B[j,i]
```

Contrairement à une matrice de biais symétrique, `B[i,j] ≠ B[j,i]` — le modèle peut représenter la **directionnalité des relations** :
- `Paris → France` ≠ `France → Paris`

Les deux projecteurs sont **indépendants** : `rel_q_proj` encode la perspective "query" et `rel_k_proj` encode la perspective "key".

### graph_scale — Démarrage classique garanti

```python
self.graph_scale = nn.Parameter(torch.zeros(num_heads))
```

Un paramètre `graph_scale` par tête, initialisé à **0**. Au step 0, le biais de graphe est nul : le modèle se comporte exactement comme un transformer classique, stable. Au fil de l'entraînement, le modèle **active progressivement** les canaux relationnels selon le signal de gradient.

### Absence de normalisation intentionnelle

Pas de `F.normalize` sur les projecteurs relationnels : le gradient sur la magnitude est **préservé**, ce qui permet au modèle d'ajuster l'amplitude des relations en plus de leur direction.

### Désactivation optionnelle

Si `use_graph=False`, les projecteurs relationnels ne sont pas alloués — bascule en **transformer classique pur** avec la même architecture de base (utile pour l'ablation).

---

## Composants Core

### `Core/Attention/attention.py`

- **`RMSNorm`** — Normalisation par la racine carrée de la moyenne des carrés. Pas de biais. Standard moderne (LLaMA, Mistral, etc.).

- **`RotaryPositionalEmbedding`** — RoPE avec cache cos/sin, et support **YaRN** (Yet Another RoPE extensioN) pour l'extension de contexte sans coût additionnel à l'inférence.
  - `use_yarn=True` + `yarn_scale > 1.0` : recalcul des fréquences avec interpolation sélective par dimension.
  - Cache invalidé si la longueur de séquence ou le device change.

- **`NaylisAttention`** — L'attention principale. Voir section dédiée ci-dessus.

### `Core/FeedForward/feedforward.py`

**`FeedForward`** — Réseau feed-forward avec **SwiGLU** (option par défaut) ou GELU.

```
SwiGLU : hidden_dim = ((8/3 × embed_dim) + 63) // 64 * 64
         output = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))
```

La dimension cachée est arrondie au multiple de 64 supérieur pour l'alignement mémoire GPU. Le facteur 8/3 (formulation Shazeer) remplace le facteur 4 classique pour conserver un nombre de paramètres similaire avec deux matrices au lieu d'une.

### `Core/TransformerBlock/transformer_block.py`

**`NaylisBlock`** — Bloc transformer complet en **pré-norm** :

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Supporte le **sequence packing** via `cu_seqlens` (passé directement à l'attention).

### `Core/Model/naylisGPT.py`

**`NaylisGPT`** — Modèle complet. Fonctionnalités :

- **Weight tying** : `output_head.weight = token_embeddings.weight` — réduction du nombre de paramètres et convergence améliorée.
- **Cast BF16 automatique** : si l'entrée est `float32` sur CUDA, elle est castée en `bfloat16` au début du forward.
- **Génération avec KV Cache** : préfill d'abord, puis décodage token par token. Supporte top-k, top-p (nucleus sampling), température, et une liste de `eos_token_id`.
- **`count_parameters()`** : retourne le total, les paramètres Naylis (relationnels), et leur pourcentage.
- **`resize_token_embeddings()`** : redimensionne l'embedding et préserve le weight tying.
- **`_init_weights()`** : std=0.02 pour les linéaires et embeddings, ones pour RMSNorm.

---

## Boucle d'entraînement et d'inférence

### Schéma du forward pass complet (un NaylisBlock)

```
INPUT IDS  [B, S]
     │
     ▼
┌─────────────────────────────┐
│   token_embeddings          │  lookup [vocab_size × embed_dim]
│   → cast BF16 si CUDA       │
└─────────────┬───────────────┘
              │  x  [B, S, embed_dim]
              │
    ┌─────────▼──────────────────────────────────────────────┐
    │                  NaylisBlock  (× num_layers)           │
    │                                                        │
    │   residual = x                                         │
    │        │                                               │
    │   ┌────▼────────────────────────────────────────────┐  │
    │   │              RMSNorm  (ln1)                     │  │
    │   └────┬────────────────────────────────────────────┘  │
    │        │  x_norm                                       │
    │        │                                               │
    │   ┌────▼────────────────────────────────────────────┐  │
    │   │              NaylisAttention                    │  │
    │   │                                                 │  │
    │   │  ┌─────────────────────────────────────────┐   │  │
    │   │  │  q_proj → Q  [B, H, S, head_dim]        │   │  │
    │   │  │  k_proj → K  [B, Hkv, S, head_dim]  GQA │   │  │
    │   │  │  v_proj → V  [B, Hkv, S, head_dim]      │   │  │
    │   │  └─────────────────────────────────────────┘   │  │
    │   │                    │                            │  │
    │   │  ┌─────────────────▼───────────────────────┐   │  │
    │   │  │  QK Norm : RMSNorm(Q)  RMSNorm(K)       │   │  │
    │   │  └─────────────────────────────────────────┘   │  │
    │   │                    │                            │  │
    │   │  ┌─────────────────▼───────────────────────┐   │  │
    │   │  │  RoPE / YaRN : rotate(Q, K, positions)  │   │  │
    │   │  └─────────────────────────────────────────┘   │  │
    │   │                    │                            │  │
    │   │  ┌─────────────────▼───────────────────────┐   │  │
    │   │  │  Naylis Graph Bias (asymétrique)         │   │  │
    │   │  │  R_q = rel_q_proj(x)  [B, H, S, R]      │   │  │
    │   │  │  R_k = rel_k_proj(x)  [B, H, S, R]      │   │  │
    │   │  │  B = graph_scale × (R_q @ R_kᵀ)         │   │  │
    │   │  │  → injecté comme attn_mask               │   │  │
    │   │  └─────────────────────────────────────────┘   │  │
    │   │                    │                            │  │
    │   │  ┌─────────────────▼───────────────────────┐   │  │
    │   │  │  Attention (SDPA / FA2 / FA3 / FA4)     │   │  │
    │   │  │  scores = (Q @ Kᵀ) / √head_dim + B      │   │  │
    │   │  │  out    = softmax(scores) @ V            │   │  │
    │   │  └─────────────────────────────────────────┘   │  │
    │   │                    │                            │  │
    │   │  ┌─────────────────▼───────────────────────┐   │  │
    │   │  │  out_proj → attn_out  [B, S, embed_dim] │   │  │
    │   │  └─────────────────────────────────────────┘   │  │
    │   └────┬────────────────────────────────────────────┘  │
    │        │                                               │
    │   x = residual + attn_out   ← connexion résiduelle     │
    │        │                                               │
    │   residual = x                                         │
    │        │                                               │
    │   ┌────▼────────────────────────────────────────────┐  │
    │   │              RMSNorm  (ln2)                     │  │
    │   └────┬────────────────────────────────────────────┘  │
    │        │                                               │
    │   ┌────▼────────────────────────────────────────────┐  │
    │   │              FeedForward — SwiGLU               │  │
    │   │  gate = silu(gate_proj(x))                      │  │
    │   │  up   = up_proj(x)                              │  │
    │   │  out  = down_proj(gate ⊙ up)                    │  │
    │   └────┬────────────────────────────────────────────┘  │
    │        │                                               │
    │   x = residual + ffn_out    ← connexion résiduelle     │
    └─────────┬──────────────────────────────────────────────┘
              │  (répété × num_layers)
              │
     ┌────────▼────────┐
     │   RMSNorm final │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │   output_head   │  logits  [B, S, vocab_size]
     │  (poids tied)   │
     └─────────────────┘
```

---

### Boucle d'entraînement

```
╔══════════════════════════════════════════════════════════════╗
║                     BOUCLE TRAIN                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Pour chaque micro-batch (x, y) :                           ║
║                                                              ║
║  1. FORWARD                                                  ║
║     x  [B, S]  ──► NaylisGPT ──► logits [B, S, vocab]      ║
║     loss = cross_entropy(logits, y, ignore_index=-100)       ║
║     loss = loss / gradient_accumulation                      ║
║                                                              ║
║  2. BACKWARD                                                 ║
║     loss.backward()                                          ║
║     ↳ gradients accumulés dans .grad de chaque paramètre    ║
║                                                              ║
║  3. (après gradient_accumulation micro-batchs)               ║
║                                                              ║
║     CLIP   clip_grad_norm_(params, max_norm=1.0)             ║
║                                                              ║
║     STEP   Muon  → blocs (ndim ≥ 2)                         ║
║               Newton-Schulz(grad) → orthogonalise           ║
║               MARS-M → réduit la variance                    ║
║            AdamW → embeddings, norms (ndim < 2)             ║
║                                                              ║
║     LR     WSD scheduler.step()                             ║
║               Muon  : lr × 5.0                              ║
║               AdamW : lr                                    ║
║                                                              ║
║     optimizer.zero_grad()                                    ║
║                                                              ║
║  4. VALIDATION  (tous les N steps)                           ║
║     model.eval() → perplexité sur val set → model.train()   ║
║                                                              ║
║  5. CHECKPOINT  (tous les M steps)                           ║
║     sauvegarde atomique .pt + _info.json (os.replace)       ║
║     model_config embarqué dans le .pt (auto-détection)      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

### Boucle d'inférence (génération avec KV Cache)

```
╔══════════════════════════════════════════════════════════════╗
║                   BOUCLE INFÉRENCE                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ÉTAPE 1 — PREFILL  (une seule passe, toute la séquence)    ║
║                                                              ║
║  input_ids [B, S_prompt]                                     ║
║       │                                                      ║
║       ▼                                                      ║
║  NaylisGPT(input_ids, use_kv_cache=True)                    ║
║       │                                                      ║
║       ├──► logits   [B, S_prompt, vocab]                    ║
║       └──► past_kv  [(K_layer, V_layer)] × num_layers       ║
║                                                              ║
║  next_logits = logits[:, -1, :]   ← dernier token           ║
║                                                              ║
║ ─────────────────────────────────────────────────────────── ║
║                                                              ║
║  ÉTAPE 2 — DECODE  (token par token, KV cache réutilisé)    ║
║                                                              ║
║  Pour i = 1 → max_new_tokens :                              ║
║                                                              ║
║    SAMPLE next_token depuis next_logits                      ║
║      ├── température = 0  → argmax (greedy)                 ║
║      ├── top-k            → filtre les k meilleurs logits   ║
║      └── top-p (nucleus)  → filtre par masse de proba       ║
║                                                              ║
║    si next_token ∈ eos_ids → STOP                           ║
║                                                              ║
║    input_ids = concat(input_ids, next_token)                 ║
║                                                              ║
║    NaylisGPT(next_token [B,1], past_kv=past_kv)             ║
║      │  ← un seul token en entrée, KV cache complet         ║
║      ├──► logits   [B, 1, vocab]                            ║
║      └──► past_kv  mis à jour  (K, V concaténés)           ║
║                                                              ║
║    next_logits = logits[:, -1, :]                            ║
║                                                              ║
║  RETOUR  input_ids [B, S_prompt + nb_tokens_générés]        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Backends d'attention — Détection hiérarchique

La détection est effectuée **à l'import** du module `attention.py`, une seule fois pour tout le processus.

```
Priorité 1 : FlashAttention-4  (flash_attn >= 3.0, SM12x Blackwell)
Priorité 2 : FlashAttention-3  (flash_attn >= 3.0, SM90 Hopper)
Priorité 3 : FlashAttention-2  (flash_attn >= 2.0) + varlen (sequence packing)
Priorité 4 : SDPA PyTorch      (torch >= 2.0, natif FA sur B200/H100/A100)
Priorité 5 : Attention manuelle (masque custom, PyTorch < 2.0)
```

| Backend | Conditions | Avantage |
|---|---|---|
| FA-4 varlen | flash_attn ≥ 3.0, SM120 | Optimal Blackwell |
| FA-3 varlen | flash_attn ≥ 3.0, SM90 | Optimal Hopper |
| FA-2 varlen | flash_attn ≥ 2.0 | Sequence packing (0% padding) |
| SDPA | torch ≥ 2.0 | Natif PyTorch, pas de dépendance externe |
| Manuel | fallback | Compatible tout hardware |

Le biais de graphe Naylis est injecté via `attn_mask` en BF16 contiguous avant le calcul SDPA — compatible avec tous les backends.

### GQA (Grouped Query Attention)

Plusieurs têtes queries partagent un nombre réduit de têtes KV (`n_kv_heads < num_heads`). Réduction significative de la mémoire KV cache et de la latence à l'inférence, sans dégradation notable de la qualité.

### KV Cache

Implémenté dans le forward de `NaylisAttention` via un tuple `(k_cache, v_cache)`. En génération : le préfill traite toute la séquence d'entrée en une passe, puis chaque nouveau token est décodé en utilisant le cache accumulé.

---

## Optimiseur — Muon + MARS-M

Deux optimiseurs sur des groupes de paramètres distincts :

### Muon (blocs transformer)

Muon est un optimiseur à momentum de Nesterov avec une étape de **mise à zéro de puissance (Newton-Schulz)** du gradient avant la mise à jour. Cette opération orthogonalise le gradient et le normalise, rendant les mises à jour invariantes à l'échelle des couches.

**Newton-Schulz 5 itérations :**
```
a, b, c = (3.4445, -4.7750, 2.0315)
X ← G / ‖G‖
pour 5 itérations :
    A = X Xᵀ
    X ← a·X + (b·A + c·A²) X
```

**MARS-M intégré** (`use_mars=True`, `mars_gamma=0.025`) : correction du gradient courant par une estimation du momentum du gradient précédent, réduisant la variance.

### AdamW (embeddings, norms, têtes de sortie)

Appliqué aux paramètres de dimension < 2 (qui ne peuvent pas être orthogonalisés) : embeddings, biaises RMSNorm, `output_head`.

### Groupes de paramètres

```python
muon_params  = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
adamw_params = [p for p in model.parameters() if p.ndim < 2  and p.requires_grad]
```

Le LR de Muon est appliqué avec un facteur `× 5.0` par rapport au LR d'AdamW (pratique standard pour Muon).

---

## Scheduler WSD

**Warmup — Stable — Decay (cosinus)**

```
Phase Warmup  : steps 0 → warmup_steps        LR : 0 → max_lr  (linéaire)
Phase Stable  : steps warmup → stable_end      LR : max_lr       (constant)
Phase Decay   : steps stable_end → total       LR : max_lr → min_lr  (cosinus)
```

**Mini-warmup post-reprise** : à la reprise d'un checkpoint, une montée en LR de quelques steps absorbe l'instabilité due aux buffers de momentum préchauffés sur un premier gradient "froid", sans effacer les buffers (qui contiennent de l'information utile sur les gradients passés).

---

## Benchmark — lm-evaluation-harness

**Fichier** : `bench.py`

Évaluation via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) d'EleutherAI.

La config du modèle est **auto-détectée depuis le checkpoint** — pas besoin de spécifier l'architecture manuellement. `use_graph` est lu depuis le `.pt` (ou `_info.json` pour les anciens checkpoints).

### Tâches par mode

| Tâche | lm-eval name | Pretrain (few-shot) | SFT (few-shot) | Baseline aléatoire |
|---|---|---|---|---|
| MMLU | `mmlu` | 5-shot | 0-shot | 25% |
| HellaSwag | `hellaswag` | 10-shot | 0-shot | 25% |
| ARC-Challenge | `arc_challenge` | 25-shot | 0-shot | 25% |
| ARC-Easy | `arc_easy` | 5-shot | 0-shot | 25% |
| WinoGrande | `winogrande` | 5-shot | 0-shot | 50% |
| PIQA | `piqa` | 0-shot | 0-shot | 50% |
| TriviaQA | `triviaqa` | 0-shot | 0-shot | — |
| NaturalQuestions | `nq_open` | 1-shot | 0-shot | — |
| BoolQ | `boolq` | 0-shot | 0-shot | 50% |
| LAMBADA | `lambada_openai` | 0-shot | 0-shot | — |
| OpenBookQA | `openbookqa` | 0-shot | 0-shot | 25% |
| SciQ | `sciq` | 0-shot | 0-shot | 25% |
| COPA | `copa` | 0-shot | 0-shot | 50% |
| RACE | `race` | 0-shot | 0-shot | 25% |
| SIQA | `siqa` | 0-shot | 0-shot | 33% |

Les few-shots du mode pretrain suivent les standards de l'industrie (Qwen2/2.5/3, Gemma tech reports). Le mode SFT est évalué en 0-shot sur toutes les tâches sauf NaturalQuestions (1-shot).

### Utilisation

```bash
# Auto-détection totale de la config depuis le checkpoint
python bench.py --mode pretrain --model ./Model/naylis_pretrain.pt
python bench.py --mode sft      --model ./Model/naylis_sft.pt

# Override use_graph (si besoin d'ablation)
python bench.py --mode pretrain --model ./Model/naylis_pretrain.pt --no-use-graph

# Sélection de tâches
python bench.py --mode pretrain --tasks all --num_fewshot 5
python bench.py --mode pretrain --tasks openbookqa,sciq,copa,race,siqa
python bench.py --mode sft      --tasks piqa,mmlu --batch_size 4
```

---

## Structure du projet

```
NaylisGPT/
├── Core/
│   ├── Attention/
│   │   └── attention.py          # RMSNorm, RoPE/YaRN, NaylisAttention, backends FA
│   ├── FeedForward/
│   │   └── feedforward.py        # SwiGLU / GELU FFN
│   ├── Model/
│   │   └── naylisGPT.py          # NaylisGPT — modèle complet, génération, weight tying
│   └── TransformerBlock/
│       └── transformer_block.py  # NaylisBlock — pré-norm, residual, sequence packing
├── pretrain*.py                  # Scripts de pré-entraînement (plusieurs configs)
├── bench.py                      # Évaluation lm-evaluation-harness (EleutherAI)
├── requirements.txt              # Dépendances Python
└── LICENSE                       # GNU GPL v3
```

---

## Installation et requirements

### Prérequis système

- Python 3.10+
- CUDA 12.8+ (recommandé)
- GPU cible : B200 (Blackwell SM100) — fonctionne aussi sur H100, A100, et CPU

### Dépendances Python

```
torch>=2.6.0
torchvision>=0.21.0
transformers>=4.40.0
datasets>=2.19.0
huggingface-hub>=0.23.0
tokenizers>=0.19.0
tqdm>=4.66.0
numpy>=1.26.0
flash-linear-attention
# flash-attn==2.8.3  (optionnel — wheel précompilé, pas de compilation CUDA requise)
```

```bash
pip install -r requirements.txt
```

Pour l'évaluation :
```bash
pip install lm-eval>=0.4.3
```

---

## Résumé des choix architecturaux

| Composant | Choix | Raison |
|---|---|---|
| Attention | NaylisAttention (hybride Token-Graph) | Biais asymétrique learnable, directionnalité réelle |
| Positional Encoding | RoPE + YaRN optionnel | Extrapolation de longueur sans reparamétrage |
| Normalisation | RMSNorm | Plus stable que LayerNorm, sans biais |
| QK Norm | RMSNorm sur Q et K | Stabilité des logits d'attention |
| FFN | SwiGLU | Meilleure perplexité que GELU à iso-paramètres |
| KV partage | GQA (n_kv_heads < num_heads) | Réduction du KV cache, latence améliorée |
| Poids partagés | token_emb ↔ output_head | Moins de paramètres, convergence améliorée |
| Optimiseur blocs | Muon + MARS-M | Mise à jour orthogonale, invariante à l'échelle |
| Optimiseur scalaires | AdamW | Standard pour embeddings et norms |
| Scheduler | WSD (Warmup-Stable-Decay) | Standard émergent pour les LLM |

---

## Licence

GNU General Public License v3.0 — voir `LICENSE`.

---

*fait par larak silyan, 15 ans*
