"""Build and execute the retrieval experiment analysis notebook."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_DIR / "notebooks" / "retrieval_choices.ipynb"


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown(
            """
            # How retrieval choices influence engineering-answer reliability

            ## TL;DR

            This notebook compares four local embedding models, three word-based
            chunk sizes, three overlaps, and four top-k values on a 50-question
            page-grounded engineering benchmark. A 35-question development split
            selects the configuration; 15 held-out questions estimate final
            retrieval performance.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd

            PROJECT = Path.cwd()
            RESULTS = PROJECT / "results"
            summary = pd.read_csv(RESULTS / "retrieval_summary.csv")
            details = pd.read_csv(RESULTS / "retrieval_details.csv")
            inventory = pd.read_csv(RESULTS / "model_inventory.csv")
            selection = json.loads((RESULTS / "selected_configuration.json").read_text())
            manifest = json.loads((RESULTS / "experiment_manifest.json").read_text())
            """
        ),
        markdown(
            """
            ## Context and methods

            A retrieval failure prevents a grounded generator from seeing the
            answer, so evidence Recall@k is treated as a necessary—not
            sufficient—condition for answer reliability. A relevant result must
            overlap the reviewed source page and contain every manually checked
            evidence term. Page-only recall is retained to expose cases where a
            large chunk touches the right page but omits the answer.

            Key assumptions:

            - Questions are paraphrased from one legally held engineering text.
            - Development Recall@5 is the primary selection metric; MRR@5 is the
              first tie-breaker.
            - Test outcomes do not participate in configuration selection.
            - Ollama truncation remains enabled, matching normal deployed use.
            """
        ),
        code(
            """
            pd.DataFrame({
                "item": ["Questions", "Development", "Held-out test", "Configurations", "Summary points"],
                "value": [
                    manifest["benchmark"]["questions"],
                    manifest["benchmark"]["development"],
                    manifest["benchmark"]["test"],
                    len(manifest["models"]) * len(manifest["chunk_sizes_words"]) * len(manifest["overlap_words"]),
                    len(manifest["models"]) * len(manifest["chunk_sizes_words"]) * len(manifest["overlap_words"]) * len(manifest["top_k"]),
                ],
            })
            """
        ),
        markdown(
            """
            ## Data

            Exact local model digests make the run reproducible. Context limits
            matter because 400- and 800-word chunks can exceed the 512-token
            windows of smaller encoders.
            """
        ),
        code(
            """
            inventory[["model", "digest", "embedding_dimension", "context_tokens", "size_bytes"]]
            """
        ),
        markdown("## Results"),
        code(
            """
            selected = selection["selected_configuration"]
            pd.Series(selected, name="selected_on_development").to_frame()
            """
        ),
        code(
            """
            pd.DataFrame(selection["held_out_test"]).style.format({
                "recall_at_k": "{:.1%}",
                "recall_ci_low": "{:.1%}",
                "recall_ci_high": "{:.1%}",
                "page_recall_at_k": "{:.1%}",
                "mrr_at_k": "{:.3f}",
            })
            """
        ),
        code(
            """
            frame = summary[(summary.split == "development") & (summary.k == 5)]
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            for axis, model in zip(axes.flat, inventory.model, strict=False):
                pivot = frame[frame.model == model].pivot(
                    index="chunk_words", columns="overlap_words", values="recall_at_k"
                ).sort_index()
                image = axis.imshow(pivot, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                for row in range(pivot.shape[0]):
                    for column in range(pivot.shape[1]):
                        value = pivot.iloc[row, column]
                        axis.text(column, row, f"{value:.0%}", ha="center", va="center",
                                  color="white" if value >= .65 else "#24292F", fontweight="bold")
                axis.set(title=model, xlabel="Overlap (words)", ylabel="Chunk size (words)")
                axis.set_xticks(range(len(pivot.columns)), pivot.columns)
                axis.set_yticks(range(len(pivot.index)), pivot.index)
                axis.grid(False)
            fig.colorbar(image, ax=axes, label="Development evidence Recall@5", shrink=.8)
            fig.suptitle("Retrieval recall across chunk and overlap settings", fontsize=15)
            plt.show()
            """
        ),
        code(
            """
            leaders = (frame.sort_values(
                ["model", "recall_at_k", "mrr_at_k"], ascending=[True, False, False]
            ).groupby("model", as_index=False).head(1))
            leaders[["model", "chunk_words", "overlap_words", "recall_at_k", "mrr_at_k",
                     "document_embedding_seconds", "context_truncation_risk"]].style.format({
                         "recall_at_k": "{:.1%}", "mrr_at_k": "{:.3f}",
                         "document_embedding_seconds": "{:.1f}",
                     })
            """
        ),
        markdown(
            """
            ## Takeaways

            The selected configuration and its held-out curve above are the
            defensible conclusion of this run. Differences among development
            and test results should be treated as sampling uncertainty, not as
            proof that one model is universally superior.

            This experiment supports a concrete engineering claim: retrieval
            settings were chosen from held-out evidence-retrieval performance,
            not intuition. It does not yet prove final answer correctness.
            Answer-level factuality and citation-entailment scoring are the next
            evaluation layer.

            **Validation status: Ready to share with caveats.** The calculations
            are reproducible and the split is respected, but the sample is small
            and comes from one textbook.
            """
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (local-rag)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    notebook = build_notebook()
    if not args.no_execute:
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_DIR)}},
        ).execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Wrote {'executed ' if not args.no_execute else ''}notebook to {output}")


if __name__ == "__main__":
    main()
