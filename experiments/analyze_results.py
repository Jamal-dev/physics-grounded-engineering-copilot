"""Create publication-ready retrieval charts and a concise result summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import DEFAULT_RESULTS
from experiments.run_retrieval_benchmark import _select_configuration, _summarize

COLORS = {
    "all-minilm": "#386CB0",
    "nomic-embed-text": "#C99720",
    "embeddinggemma": "#E67E22",
    "mxbai-embed-large": "#6B7D3A",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#3B4148",
            "axes.labelcolor": "#24292F",
            "axes.titlecolor": "#24292F",
            "text.color": "#24292F",
            "xtick.color": "#57606A",
            "ytick.color": "#57606A",
            "axes.grid": True,
            "grid.color": "#D8DEE4",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFD",
        }
    )


def recall_heatmaps(summary: pd.DataFrame, output: Path) -> None:
    frame = summary[(summary["split"] == "development") & (summary["k"] == 5)]
    models = list(COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, model in zip(axes.flat, models, strict=False):
        pivot = (
            frame[frame["model"] == model]
            .pivot(index="chunk_words", columns="overlap_words", values="recall_at_k")
            .sort_index()
        )
        image = axis.imshow(pivot.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                value = pivot.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.65 else "#24292F",
                    fontweight="bold",
                )
        axis.set_title(model)
        axis.set_xlabel("Overlap (words)")
        axis.set_ylabel("Chunk size (words)")
        axis.set_xticks(range(len(pivot.columns)), pivot.columns)
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        axis.grid(False)
    fig.colorbar(image, ax=axes, label="Development evidence Recall@5", shrink=0.8)
    fig.suptitle("Retrieval recall across chunk and overlap settings", fontsize=15)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def held_out_curve(summary: pd.DataFrame, selection: dict, output: Path) -> None:
    chosen = selection["selected_configuration"]
    frame = summary[
        (summary["model"] == chosen["model"])
        & (summary["chunk_words"] == chosen["chunk_words"])
        & (summary["overlap_words"] == chosen["overlap_words"])
        & (summary["split"].isin(["development", "test"]))
    ].copy()
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for split, style, color in (
        ("development", "--", "#57606A"),
        ("test", "-", COLORS[chosen["model"]]),
    ):
        rows = frame[frame["split"] == split].sort_values("k")
        axis.errorbar(
            rows["k"],
            rows["recall_at_k"],
            yerr=[
                rows["recall_at_k"] - rows["recall_ci_low"],
                rows["recall_ci_high"] - rows["recall_at_k"],
            ],
            linestyle=style,
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{split.title()} (n={int(rows['questions'].iloc[0])})",
        )
        for _, row in rows.iterrows():
            is_test = split == "test"
            axis.annotate(
                f"{row['recall_at_k']:.0%}",
                (row["k"], row["recall_at_k"]),
                xytext=(9 if is_test else -9, 10 if is_test else -18),
                textcoords="offset points",
                ha="left" if is_test else "right",
                fontsize=9,
            )
    axis.set_ylim(0, 1.08)
    axis.set_xticks([1, 3, 5, 10])
    axis.set_xlabel("Retrieved chunks (k)")
    axis.set_ylabel("Evidence Recall@k")
    axis.set_title(
        f"Selected configuration: {chosen['model']}, "
        f"{chosen['chunk_words']} words, {chosen['overlap_words']}-word overlap"
    )
    axis.legend(frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def model_tradeoffs(summary: pd.DataFrame, output: Path) -> None:
    dev = summary[(summary["split"] == "development") & (summary["k"] == 5)].copy()
    ordered = dev.sort_values(
        ["model", "recall_at_k", "mrr_at_k", "retrieval_ms_per_question"],
        ascending=[True, False, False, True],
    ).groupby("model", as_index=False).head(1)
    test = summary[(summary["split"] == "test") & (summary["k"] == 5)]
    rows = ordered.merge(
        test,
        on=["model", "chunk_words", "overlap_words", "k"],
        suffixes=("_development", "_test"),
    ).sort_values("recall_at_k_test", ascending=True)

    labels = [
        f"{row.model}\n{int(row.chunk_words)} words / {int(row.overlap_words)} overlap"
        for row in rows.itertuples()
    ]
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.barh(
        positions,
        rows["recall_at_k_test"],
        color=[COLORS[name] for name in rows["model"]],
        edgecolor="#3B4148",
        linewidth=0.7,
    )
    axis.scatter(
        rows["recall_at_k_development"],
        positions,
        marker="D",
        facecolor="white",
        edgecolor="#24292F",
        zorder=3,
        label="Development Recall@5",
    )
    for position, value in zip(positions, rows["recall_at_k_test"], strict=False):
        axis.text(
            value / 2,
            position,
            f"{value:.0%}",
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
        )
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Evidence Recall@5")
    axis.set_title("Held-out performance of each model's development-selected chunking")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: pd.DataFrame, selection: dict, output: Path) -> None:
    chosen = selection["selected_configuration"]
    held_out = pd.DataFrame(selection["held_out_test"]).sort_values("k")
    model_rows = summary[(summary["split"] == "development") & (summary["k"] == 5)]
    leaders = (
        model_rows.sort_values(
            ["model", "recall_at_k", "mrr_at_k"], ascending=[True, False, False]
        )
        .groupby("model", as_index=False)
        .head(1)
    )
    lines = [
        "# Retrieval experiment results",
        "",
        "The configuration was selected only on the development split. The held-out test",
        "results below were reported after selection.",
        "",
        "## Selected configuration",
        "",
        f"- Embedding model: `{chosen['model']}`",
        f"- Chunk size: {chosen['chunk_words']} words",
        f"- Overlap: {chosen['overlap_words']} words",
        f"- Development Recall@5: {chosen['development_recall_at_5']:.1%}",
        f"- Development MRR@5: {chosen['development_mrr_at_5']:.3f}",
        "",
        "## Held-out result",
        "",
        "| k | Questions | Evidence Recall@k (95% CI) | Page Recall@k | MRR@k |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in held_out.itertuples():
        lines.append(
            f"| {int(row.k)} | {int(row.questions)} | {row.recall_at_k:.1%} "
            f"({row.recall_ci_low:.1%}–{row.recall_ci_high:.1%}) | "
            f"{row.page_recall_at_k:.1%} | {row.mrr_at_k:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Best development setting within each model",
            "",
            "| Model | Chunk words | Overlap words | Development Recall@5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in leaders.sort_values("model").itertuples():
        lines.append(
            f"| `{row.model}` | {int(row.chunk_words)} | {int(row.overlap_words)} | "
            f"{row.recall_at_k:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Recall tests retrieval of reviewed evidence, not full generated-answer correctness.",
            "- The 50 questions come from one engineering textbook, so results may not transfer unchanged to another corpus.",
            "- Models with a 512-token context can truncate some 400- and 800-word chunks; the result reflects deployed behavior.",
            "- Page recall is diagnostic only. Large chunks can touch a labeled page without carrying the answer, so evidence recall is the selection metric.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def analyze(results_dir: Path) -> None:
    _style()
    summary = pd.read_csv(results_dir / "retrieval_summary.csv")
    if not {"recall_ci_low", "recall_ci_high"}.issubset(summary.columns):
        details = pd.read_csv(results_dir / "retrieval_details.csv")
        summary_rows = _summarize(details.to_dict(orient="records"))
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(results_dir / "retrieval_summary.csv", index=False)
        selection = _select_configuration(summary_rows)
        (results_dir / "selected_configuration.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
    else:
        selection = json.loads(
            (results_dir / "selected_configuration.json").read_text()
        )
    recall_heatmaps(summary, results_dir / "recall_heatmaps.png")
    held_out_curve(summary, selection, results_dir / "held_out_recall_at_k.png")
    model_tradeoffs(summary, results_dir / "model_tradeoffs.png")
    write_summary(summary, selection, results_dir / "RESULTS.md")
    print(f"Wrote charts and summary to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    analyze(args.results.expanduser().resolve())


if __name__ == "__main__":
    main()
