#!/usr/bin/env python3
# bench.py — Naylis v1  (lm-eval harness)
# Évalue NaylisGPT via lm-evaluation-harness (EleutherAI).
#
# Install :
#   pip install lm-eval>=0.4.3
#
# Usage :
#   python bench.py --mode pretrain --model ./Model/naylis_pretrain.pt
#   python bench.py --mode sft      --model ./Model/naylis_sft.pt
#   python bench.py --mode pretrain --tasks all --num_fewshot 5
#   python bench.py --mode pretrain --tasks openbookqa,sciq,copa,race
#   python bench.py --mode sft      --tasks piqa,mmlu --batch_size 4

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Chemins locaux ────────────────────────────────────────────
_root = os.path.dirname(__file__)
for sub in ("Core/Model", "Core/Attention", "Core/FeedForward", "Core/TransformerBlock", ""):
    sys.path.append(os.path.join(_root, sub))

from naylisGPT import NaylisGPT

# ── lm-eval ───────────────────────────────────────────────────
try:
    from lm_eval.api.model import LM
    from lm_eval import simple_evaluate
except ImportError:
    sys.exit(
        "lm-evaluation-harness non trouvé.\n"
        "Installe-le avec : pip install lm-eval>=0.4.3"
    )

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
# Cles d'architecture a lire/ecrire dans le checkpoint
_ARCH_KEYS = [
    'vocab_size', 'embed_dim', 'num_heads', 'num_layers', 'max_seq_len',
    'dropout', 'use_rope', 'use_yarn', 'yarn_scale', 'yarn_original_max_len',
    'use_swiglu', 'n_kv_heads', 'use_qk_norm', 'soft_cap', 'use_flash_attn',
]

TOKENIZER_ID       = "HuggingFaceTB/cosmo2-tokenizer"
DEFAULT_MODEL_SFT  = "./Model/naylis_sft.pt"
DEFAULT_MODEL_PRE  = "./Model/naylis_pretrain.pt"

MODEL_CFG = dict(
    vocab_size     = None,   # rempli au runtime depuis le tokenizer
    embed_dim      = 512,
    num_heads      = 8,
    num_layers     = 12,
    max_seq_len    = 512,
    n_kv_heads     = 4,
    use_rope       = True,
    use_yarn       = False,
    use_swiglu     = True,
    use_qk_norm    = True,
    use_flash_attn = True,
    dropout        = 0.0,
)

# ─────────────────────────────────────────────────────────────
# TASK MAPS — few-shot par tâche selon le mode
#
# Mode SFT     → 0-shot sur tout  (modèle instruct, pas besoin d'exemples)
# Mode PRETRAIN → standard industrie (Qwen2/2.5/3, Gemma) :
#   MMLU        5-shot   (standard académique depuis le papier original)
#   HellaSwag  10-shot   (Qwen2 tech report)
#   ARC-C      25-shot   (Qwen2 tech report, très sensible au few-shot)
#   ARC-Easy    5-shot   (cohérent avec ARC-C, moins agressif)
#   WinoGrande  5-shot   (Qwen2/2.5 standard)
#   PIQA        0-shot   (tâche simple, peu sensible au few-shot)
#   TriviaQA    0-shot   (open-ended, génération)
#   nq_open     0-shot   (NaturalQuestions — bench perso)
#   boolq       0-shot   (bench perso)
#   lambada_openai 0-shot (bench perso)
# ─────────────────────────────────────────────────────────────

TASK_MAP_SFT = {
    "nq_open"        : ("nq_open",          0),
    "boolq"          : ("boolq",            0),
    "lambada_openai" : ("lambada_openai",   0),
    "piqa"           : ("piqa",             0),
    "mmlu"           : ("mmlu",             0),
    "arc_easy"       : ("arc_easy",         0),
    "arc_challenge"  : ("arc_challenge",    0),
    "hellaswag"      : ("hellaswag",        0),
    "winogrande"     : ("winogrande",       0),
    "triviaqa"       : ("triviaqa",         0),
    "openbookqa"     : ("openbookqa",       0),
    "sciq"           : ("sciq",             0),
    "copa"           : ("copa",             0),
    "race"           : ("race",             0),
    "commonsense_qa"  : ("commonsense_qa",   0),
}

TASK_MAP_PRETRAIN = {
    "nq_open"        : ("nq_open",          1),
    "boolq"          : ("boolq",            0),
    "lambada_openai" : ("lambada_openai",   0),
    "piqa"           : ("piqa",             0),
    "mmlu"           : ("mmlu",             5),
    "arc_easy"       : ("arc_easy",         5),
    "arc_challenge"  : ("arc_challenge",   25),
    "hellaswag"      : ("hellaswag",       10),
    "winogrande"     : ("winogrande",       5),
    "triviaqa"       : ("triviaqa",         0),
    "openbookqa"     : ("openbookqa",       0),
    "sciq"           : ("sciq",             0),
    "copa"           : ("copa",             0),
    "race"           : ("race",             0),
    "commonsense_qa"  : ("commonsense_qa",   0),
}

TASKS_ALL_PRETRAIN = list(TASK_MAP_PRETRAIN.keys())
TASKS_ALL_SFT      = list(TASK_MAP_SFT.keys())

RANDOM_BASELINES = {
    "piqa"           : 0.50,
    "triviaqa"       : 0.00,
    "mmlu"           : 0.25,
    "arc_easy"       : 0.25,
    "arc_challenge"  : 0.25,
    "hellaswag"      : 0.25,
    "winogrande"     : 0.50,
    "nq_open"        : 0.00,
    "boolq"          : 0.50,
    "lambada_openai" : 0.00,
    "openbookqa"     : 0.25,
    "sciq"           : 0.25,
    "copa"           : 0.50,
    "race"           : 0.25,
    "commonsense_qa" : 0.20,
}

# ─────────────────────────────────────────────────────────────
# WRAPPER lm-eval
# ─────────────────────────────────────────────────────────────

class NaylisLM(LM):
    """
    Wrapper lm-evaluation-harness pour NaylisGPT.
    Implémente toutes les propriétés requises par lm-eval 0.4.x.
    """

    def __init__(
        self,
        model      : NaylisGPT,
        tokenizer  : AutoTokenizer,
        device     : str,
        batch_size : int = 4,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.model           = model
        self.tokenizer       = tokenizer
        self.device          = device
        self._batch_size_val = batch_size
        self.max_seq_len     = max_seq_len
        self._dtype          = torch.bfloat16 if device == "cuda" else torch.float32

    # ── Propriétés requises par lm-eval 0.4.x ────────────────
    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    @property
    def accelerator(self):
        return None

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", TOKENIZER_ID)

    @property
    def chat_template(self) -> str:
        return ""

    def apply_chat_template(self, chat_history: list) -> str:
        return " ".join(m.get("content", "") for m in chat_history)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id or 0

    @property
    def max_length(self) -> int:
        return self.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 64

    @property
    def batch_size(self) -> int:
        return self._batch_size_val

    # ── Tokenisation ─────────────────────────────────────────
    def tok_encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tok_decode(self, tokens) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context: str, continuation: str):
        ctx_ids = self.tok_encode(context) if context else []
        con_ids = self.tok_encode(continuation)
        if not con_ids:
            con_ids = self.tok_encode(" " + continuation)
        full = ctx_ids + con_ids
        if len(full) > self.max_seq_len:
            full    = full[-self.max_seq_len:]
            ctx_len = max(1, len(full) - len(con_ids))
        else:
            ctx_len = len(ctx_ids)
        return full, ctx_len, len(con_ids)

    # ─────────────────────────────────────────────────────────
    # loglikelihood — scoring multiple-choice (batché)
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def loglikelihood(self, requests: list) -> list:
        results = []
        pad_id  = self.eot_token_id or 0

        for i in tqdm(range(0, len(requests), self._batch_size_val),
                      desc="  loglikelihood", unit="batch", dynamic_ncols=True,
                      leave=False):
            batch_reqs = requests[i : i + self._batch_size_val]
            batch_data = [self._encode_pair(*req.args) for req in batch_reqs]

            max_len   = max(len(d[0]) for d in batch_data)
            input_ids = torch.full(
                (len(batch_data), max_len), pad_id,
                dtype=torch.long, device=self.device,
            )
            for j, (full_ids, _, _) in enumerate(batch_data):
                input_ids[j, :len(full_ids)] = torch.tensor(
                    full_ids, dtype=torch.long, device=self.device)

            with torch.amp.autocast(self.device, dtype=self._dtype,
                                    enabled=(self.device == "cuda")):
                logits, _, _ = self.model(input_ids)

            log_probs = F.log_softmax(logits, dim=-1)

            for j, (full_ids, ctx_len, con_len) in enumerate(batch_data):
                start    = ctx_len - 1
                end      = min(ctx_len + con_len - 1, log_probs.shape[1])
                lp_slice = log_probs[j, start:end, :]
                tgt      = torch.tensor(
                    full_ids[ctx_len : ctx_len + con_len],
                    dtype=torch.long, device=self.device,
                )[:lp_slice.shape[0]]

                token_lp = lp_slice[range(len(tgt)), tgt]
                logprob  = token_lp.sum().item()
                greedy   = (lp_slice.argmax(dim=-1) == tgt).all().item()
                results.append((logprob, bool(greedy)))

        return results

    # ─────────────────────────────────────────────────────────
    # loglikelihood_rolling — perplexité sur texte long
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list) -> list:
        results = []
        for req in requests:
            (text,)   = req.args
            token_ids = self.tok_encode(text)
            if not token_ids:
                results.append(0.0)
                continue

            total_lp = 0.0
            stride   = self.max_seq_len

            for start in range(0, len(token_ids), stride):
                chunk = token_ids[max(0, start - 1) : start + stride]
                ids_t = torch.tensor([chunk], dtype=torch.long, device=self.device)
                x, y  = ids_t[:, :-1], ids_t[:, 1:]
                with torch.amp.autocast(self.device, dtype=self._dtype,
                                        enabled=(self.device == "cuda")):
                    logits, _, _ = self.model(x)
                lp         = F.log_softmax(logits, dim=-1)
                score_from = 1 if start > 0 else 0
                lp_tgt     = lp[0, score_from:].gather(
                    -1, y[0, score_from:].unsqueeze(-1)
                ).squeeze(-1)
                total_lp += lp_tgt.sum().item()

            results.append(total_lp)
        return results

    # ─────────────────────────────────────────────────────────
    # generate_until — génération (TriviaQA, open-ended)
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_until(self, requests: list) -> list:
        results = []

        for req in tqdm(requests, desc="  generate_until", unit="q",
                        dynamic_ncols=True):
            context, gen_kwargs = req.args
            until    = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            token_ids = self.tok_encode(context)
            if len(token_ids) > self.max_seq_len - max_toks:
                token_ids = token_ids[-(self.max_seq_len - max_toks):]

            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self.device
            )

            stop_token_ids = []
            for s in until:
                if not s:
                    continue
                ids = self.tok_encode(s)
                if len(ids) == 1:
                    stop_token_ids.append(ids[0])

            all_stop_ids = list({self.eot_token_id} | set(stop_token_ids))

            output_ids = self.model.generate(
                input_ids,
                max_new_tokens = max_toks,
                temperature    = 0.0,
                eos_token_id   = all_stop_ids,
            )
            gen_tokens = output_ids[0, input_ids.shape[1]:]
            generated  = self.tok_decode(gen_tokens.tolist())

            for stop in until:
                if stop and stop in generated:
                    generated = generated[:generated.index(stop)]

            results.append(generated.strip())
        return results


# ─────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────

def load_tokenizer(mode: str) -> AutoTokenizer:
    print(f"  Tokenizer : {TOKENIZER_ID}  [mode={mode}]")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    if mode == "pretrain":
        print("  ℹ️  Mode pretrain — tokens ChatML non ajoutés")
    else:
        im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        if im_start_id == tok.unk_token_id:
            tok.add_special_tokens({
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
            })
            print(f"  ℹ️  Tokens ChatML ajoutés → vocab={len(tok)}")

    MODEL_CFG.setdefault("vocab_size", len(tok))
    return tok


def load_model(model_path: str, device: str) -> NaylisGPT:
    print(f"\n  Chargement : {model_path}")

    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # ── Auto-détection config ─────────────────────────────────────────────
    cfg_found = {}
    cfg_src   = "MODEL_CFG (défaut)"

    if "model_config" in ckpt:
        cfg_found = ckpt["model_config"]
        cfg_src   = "checkpoint .pt"
    else:
        info_path = model_path.replace(".pt", "_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as _f:
                _info = json.load(_f)
            cfg_found = _info.get("config", {})
            cfg_src   = "_info.json"

    for k in _ARCH_KEYS:
        if k in cfg_found:
            MODEL_CFG[k] = cfg_found[k]

    # vocab_size depuis les poids (source de vérité absolue)
    emb_w = state.get("token_embeddings.weight")
    if emb_w is not None:
        MODEL_CFG["vocab_size"] = emb_w.shape[0]

    # ── Affichage ─────────────────────────────────────────────────────────
    print(f"  Config source  : {cfg_src}")
    print(f"  embed={MODEL_CFG['embed_dim']}  layers={MODEL_CFG['num_layers']}  "
          f"heads={MODEL_CFG['num_heads']}  kv={MODEL_CFG['n_kv_heads']}")
    print(f"  vocab_size={MODEL_CFG['vocab_size']}")

    # ── Création + chargement ──────────────────────────────────────────────
    model = NaylisGPT(**MODEL_CFG)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠️  Clés manquantes  : {len(missing)}")
    if unexpected:
        print(f"  ⚠️  Clés inattendues : {len(unexpected)}")

    model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Modèle chargé : {params:.1f}M params")
    return model


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Naylis Benchmark — lm-eval harness")
    parser.add_argument("--mode",        choices=["pretrain", "sft"], required=True,
                        help="Mode d'évaluation : pretrain (few-shot industrie) ou sft (0-shot)")
    parser.add_argument("--model",       default=None,
                        help="Chemin vers le .pt du modèle (défaut selon --mode)")
    parser.add_argument("--tasks",       default="all",
                        help="Tâches séparées par virgule ou 'all'")
    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="Override global du few-shot pour toutes les tâches")
    parser.add_argument("--batch_size",  type=int, default=4,
                        help="Taille de batch pour le scoring")
    parser.add_argument("--output",      default=None,
                        help="Fichier JSON de sortie (défaut : ./benchmark_<mode>_results.json)")
    parser.add_argument("--device",      default="auto",
                        help="cuda / cpu / auto")
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODEL_PRE if args.mode == "pretrain" else DEFAULT_MODEL_SFT
    if args.output is None:
        args.output = f"./benchmark_{args.mode}_results.json"

    task_map  = TASK_MAP_PRETRAIN if args.mode == "pretrain" else TASK_MAP_SFT
    tasks_all = TASKS_ALL_PRETRAIN if args.mode == "pretrain" else TASKS_ALL_SFT

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    mode_label = "PRETRAIN  [few-shot industrie]" if args.mode == "pretrain" else "SFT  [0-shot]"
    print("\n" + "="*65)
    print(f"  Naylis v1 — Benchmark Suite  [{mode_label}]")
    print("="*65)
    print(f"  Device      : {device}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM        : {vram:.1f} GB")
    print(f"  Modèle      : {args.model}")
    print(f"  max_seq_len : {MODEL_CFG['max_seq_len']}")

    if args.tasks.strip().lower() == "all":
        task_keys = tasks_all
    else:
        task_keys = [t.strip().lower() for t in args.tasks.split(",")]

    tokenizer = load_tokenizer(args.mode)
    model     = load_model(args.model, device)
    lm        = NaylisLM(model, tokenizer, device,
                         batch_size=args.batch_size,
                         max_seq_len=MODEL_CFG["max_seq_len"])

    all_results = {}
    t0_total    = time.time()

    for key in task_keys:
        if key not in task_map:
            print(f"  ⚠️  Tâche inconnue : {key} — ignorée")
            continue

        task_name, default_fs = task_map[key]
        fs = args.num_fewshot if args.num_fewshot is not None else default_fs

        print(f"\n{'─'*55}")
        print(f"  Tâche : {key}  ({fs}-shot)")
        t0 = time.time()

        try:
            results = simple_evaluate(
                model          = lm,
                tasks          = [task_name],
                num_fewshot    = fs,
                batch_size     = args.batch_size,
                log_samples    = False,
            )
            task_res = results["results"].get(task_name, {})
            acc      = task_res.get("acc,none",
                       task_res.get("acc_norm,none",
                       task_res.get("exact_match,none", None)))
            baseline = RANDOM_BASELINES.get(key, None)

            print(f"  acc      : {acc:.4f}" if acc is not None else "  acc : N/A")
            if acc is not None and baseline is not None:
                delta = acc - baseline
                print(f"  baseline : {baseline:.2f}  Δ={delta:+.4f}")
            print(f"  Temps    : {time.time() - t0:.1f}s")

            all_results[key] = task_res
        except Exception as e:
            print(f"  ERREUR : {e}")
            all_results[key] = {"error": str(e)}

    elapsed = time.time() - t0_total
    print(f"\n{'='*55}")
    print(f"  DONE — {len(all_results)} tâches en {elapsed/60:.1f}min")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Résultats → {args.output}")


if __name__ == "__main__":
    main()
