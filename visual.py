# -*- coding: utf-8 -*-
"""
Visualizes per-category GEC evaluation results using Matplotlib and Seaborn.
Generates:
1. gec_eval_main_metrics.png: Precision, Recall, F0.5, Exact Match per category
2. gec_eval_f05_ranked.png: Clean F0.5 horizontal bar chart with performance tiers
3. gec_eval_over_under_correction.png: Diverging bar chart showing over vs under correction
4. gec_eval_heatmap.png: Overview heatmap of all core metrics
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", context="talk")


def load_from_csv(path="data/eval_summary.csv"):
    df = pd.read_csv(path)
    df = df[df["category"] != "OVERALL"].copy()
    df = df.rename(columns={
        "n_examples": "n",
        "exact_match_rate": "exact_match",
        "f0.5": "f0_5",
        "over_correction_rate": "over_corr",
        "under_correction_rate": "under_corr"
    })
    for col in ["over_corr", "under_corr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_main_metrics(df, save_path="assets/gec_eval_main_metrics.png"):
    df_sorted = df.sort_values("f0_5", ascending=True)
    metrics = ["precision", "recall", "f0_5", "exact_match"]
    labels = ["Precision", "Recall", "F0.5", "Exact Match"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(13, 11))
    y = np.arange(len(df_sorted))
    bar_h = 0.19

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        offset = (i - 1.5) * bar_h
        ax.barh(y + offset, df_sorted[metric], height=bar_h, label=label, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["category"], fontsize=11)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("GEC Model Evaluation — Core Metrics by Category\n(sorted by F0.5, low to high)", fontsize=15, pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"Saved: {save_path}")


def plot_f05_ranked(df, save_path="assets/gec_eval_f05_ranked.png"):
    df_sorted = df.sort_values("f0_5", ascending=True)

    def tier_color(v):
        if v < 0.15:
            return "#C44E52"   # Red - Needs Improvement
        elif v < 0.5:
            return "#DD8452"   # Orange - Weak
        elif v < 0.7:
            return "#CCB974"   # Yellow - Moderate
        else:
            return "#55A868"   # Green - Strong

    colors = [tier_color(v) for v in df_sorted["f0_5"]]
    fig, ax = plt.subplots(figsize=(11, 10))
    bars = ax.barh(df_sorted["category"], df_sorted["f0_5"], color=colors)

    for bar, val in zip(bars, df_sorted["f0_5"]):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=10)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F0.5 Score")
    ax.set_title("F0.5 Score by Category (Red = Low, Green = Strong)", fontsize=15, pad=15)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C44E52", label="< 0.15 (Challenging)"),
        plt.Rectangle((0, 0), 1, 1, color="#DD8452", label="0.15 - 0.50 (Fair)"),
        plt.Rectangle((0, 0), 1, 1, color="#CCB974", label="0.50 - 0.70 (Good)"),
        plt.Rectangle((0, 0), 1, 1, color="#55A868", label="> 0.70 (High Accuracy)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"Saved: {save_path}")


def plot_over_under_correction(df, save_path="assets/gec_eval_over_under_correction.png"):
    df2 = df.copy()
    df2["over_corr_plot"] = -df2["over_corr"].fillna(0)
    df2["under_corr_plot"] = df2["under_corr"].fillna(0)
    df2["total_bad"] = df2["over_corr"].fillna(0) + df2["under_corr"].fillna(0)
    df2 = df2.sort_values("total_bad", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(df2))

    ax.barh(y, df2["over_corr_plot"], color="#C44E52", label="Over-correction rate (modified valid text)")
    ax.barh(y, df2["under_corr_plot"], color="#4C72B0", label="Under-correction rate (missed errors)")

    ax.set_yticks(y)
    ax.set_yticklabels(df2["category"], fontsize=11)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("<- Over-correction | Under-correction ->")
    ax.set_title("Over-correction vs Under-correction by Category", fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap(df, save_path="assets/gec_eval_heatmap.png"):
    plot_df = df.set_index("category")[["exact_match", "precision", "recall", "f0_5"]]
    plot_df.columns = ["Exact Match", "Precision", "Recall", "F0.5"]
    plot_df = plot_df.sort_values("F0.5", ascending=False)

    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Score", "shrink": 0.8}, ax=ax)
    ax.set_title("Evaluation Metric Heatmap by Category", fontsize=14, pad=15)
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), fontsize=10)
    plt.subplots_adjust(left=0.4, right=0.95, top=0.94, bottom=0.1)
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GEC Evaluation Plots")
    parser.add_argument("--csv", type=str, default="data/eval_summary.csv", help="Path to eval_summary.csv")
    parser.add_argument("--out_dir", type=str, default="assets", help="Directory to save charts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_from_csv(args.csv)

    plot_main_metrics(df, os.path.join(args.out_dir, "gec_eval_main_metrics.png"))
    plot_f05_ranked(df, os.path.join(args.out_dir, "gec_eval_f05_ranked.png"))
    plot_over_under_correction(df, os.path.join(args.out_dir, "gec_eval_over_under_correction.png"))
    plot_heatmap(df, os.path.join(args.out_dir, "gec_eval_heatmap.png"))
    print("\nAll visualization charts updated.")
