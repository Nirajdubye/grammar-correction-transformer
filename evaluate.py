# -*- coding: utf-8 -*-
"""
GEC Model Evaluation Benchmark Script.
Evaluates model predictions against gold reference corrections using word-level diffs:
- Exact Match rate
- Precision, Recall, F0.5 (weighting precision higher to penalize incorrect alterations)
- Over-correction rate (unnecessary changes to correct text)
- Under-correction rate (failure to correct faulty text)
"""

import argparse
import csv
import difflib
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from inference import GrammarCorrector


def normalize(s: str) -> str:
    return " ".join(str(s).strip().split())


def word_edit_set(src: str, tgt: str):
    """
    Returns the set of word-level edit operations needed to turn src into tgt,
    expressed as (opcode, src_span, tgt_span, src_start_idx).
    """
    src_words = src.split()
    tgt_words = tgt.split()
    sm = difflib.SequenceMatcher(None, src_words, tgt_words, autojunk=False)
    edits = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        src_span = " ".join(src_words[i1:i2])
        tgt_span = " ".join(tgt_words[j1:j2])
        edits.add((tag, src_span, tgt_span, i1))
    return edits


def edit_prf(src: str, gold: str, pred: str, beta: float = 0.5):
    """
    Computes precision, recall, and F0.5 of predicted edits against gold edits.
    """
    gold_edits = word_edit_set(src, gold)
    pred_edits = word_edit_set(src, pred)

    if not gold_edits and not pred_edits:
        return 1.0, 1.0, 1.0, 0, 0, 0

    tp = len(gold_edits & pred_edits)
    fp = len(pred_edits - gold_edits)
    fn = len(gold_edits - pred_edits)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fn == 0 else 0.0)

    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)

    return precision, recall, f_beta, tp, fp, fn


def run_evaluation(args):
    print(f"Loading benchmark dataset from: {args.csv}")
    df = pd.read_csv(args.csv)

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    corrector = GrammarCorrector(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        device=args.device
    )

    results = []
    category_stats = defaultdict(lambda: {
        "n": 0, "exact_match": 0, "tp": 0, "fp": 0, "fn": 0,
        "over_corr_eligible": 0, "over_corr_count": 0,
        "under_corr_eligible": 0, "under_corr_count": 0
    })

    print(f"\nEvaluating {len(df)} examples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src = normalize(str(row.get("input", row.iloc[0])))
        gold = normalize(str(row.get("target", row.get("gold", row.iloc[1]))))
        category = str(row.get("category", "GENERAL")).strip()

        pred = corrector.generate(src)
        exact = int(pred.lower() == gold.lower())

        p, r, f0_5, tp, fp, fn = edit_prf(src, gold, pred)

        stats = category_stats[category]
        stats["n"] += 1
        stats["exact_match"] += exact
        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn

        # Over-correction: Gold == Src, but model altered it
        if gold.lower() == src.lower():
            stats["over_corr_eligible"] += 1
            if pred.lower() != src.lower():
                stats["over_corr_count"] += 1

        # Under-correction: Gold != Src, but model did nothing
        if gold.lower() != src.lower():
            stats["under_corr_eligible"] += 1
            if pred.lower() == src.lower():
                stats["under_corr_count"] += 1

        results.append({
            "category": category,
            "input": src,
            "gold": gold,
            "prediction": pred,
            "exact_match": exact,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f0.5": round(f0_5, 4)
        })

    # Save detailed row-level results
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f"\nDetailed evaluation results saved to: {args.out}")

    # Build Summary Table
    summary_rows = []
    tot_n = tot_exact = tot_tp = tot_fp = tot_fn = tot_oc_elig = tot_oc_cnt = tot_uc_elig = tot_uc_cnt = 0

    for cat, s in sorted(category_stats.items()):
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else (1.0 if s["fp"] == 0 else 0.0)
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else (1.0 if s["fn"] == 0 else 0.0)
        f0_5 = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0

        oc_rate = s["over_corr_count"] / s["over_corr_eligible"] if s["over_corr_eligible"] > 0 else "n/a"
        uc_rate = s["under_corr_count"] / s["under_corr_eligible"] if s["under_corr_eligible"] > 0 else "n/a"

        summary_rows.append({
            "category": cat,
            "n_examples": s["n"],
            "exact_match_rate": round(s["exact_match"] / s["n"], 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f0.5": round(f0_5, 3),
            "over_correction_rate": round(oc_rate, 3) if isinstance(oc_rate, float) else oc_rate,
            "under_correction_rate": round(uc_rate, 3) if isinstance(uc_rate, float) else uc_rate
        })

        tot_n += s["n"]
        tot_exact += s["exact_match"]
        tot_tp += s["tp"]
        tot_fp += s["fp"]
        tot_fn += s["fn"]
        tot_oc_elig += s["over_corr_eligible"]
        tot_oc_cnt += s["over_corr_count"]
        tot_uc_elig += s["under_corr_eligible"]
        tot_uc_cnt += s["under_corr_count"]

    tot_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    tot_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    tot_f0_5 = (1.25 * tot_p * tot_r) / (0.25 * tot_p + tot_r) if (tot_p + tot_r) > 0 else 0.0
    tot_oc = tot_oc_cnt / tot_oc_elig if tot_oc_elig > 0 else 0.0
    tot_uc = tot_uc_cnt / tot_uc_elig if tot_uc_elig > 0 else 0.0

    summary_rows.append({
        "category": "OVERALL",
        "n_examples": tot_n,
        "exact_match_rate": round(tot_exact / tot_n, 3),
        "precision": round(tot_p, 3),
        "recall": round(tot_r, 3),
        "f0.5": round(tot_f0_5, 3),
        "over_correction_rate": round(tot_oc, 3),
        "under_correction_rate": round(tot_uc, 3)
    })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        summary_df.to_csv(args.summary, index=False)
        print(f"\nSummary table saved to: {args.summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GEC Transformer Model")
    parser.add_argument("--csv", type=str, default="data/gec_benchmark.csv", help="Benchmark dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_20.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Path to tokenizer folder")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation examples")
    parser.add_argument("--out", type=str, default="data/eval_results.csv", help="Output path for row-level results")
    parser.add_argument("--summary", type=str, default="data/eval_summary.csv", help="Output path for summary table")
    args = parser.parse_args()

    run_evaluation(args)
