"""Independently validate saved retrieval metrics and experiment invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import DEFAULT_RESULTS
from experiments.run_retrieval_benchmark import _select_configuration


def validate_results(results_dir: Path) -> dict[str, object]:
    details = pd.read_csv(results_dir / "retrieval_details.csv")
    summary = pd.read_csv(results_dir / "retrieval_summary.csv")
    selected = json.loads((results_dir / "selected_configuration.json").read_text())
    inventory = pd.read_csv(results_dir / "model_inventory.csv")

    expected_detail_rows = 4 * 3 * 3 * 4 * 50
    expected_summary_rows = 4 * 3 * 3 * 4 * 3
    assert len(details) == expected_detail_rows, (len(details), expected_detail_rows)
    assert len(summary) == expected_summary_rows, (len(summary), expected_summary_rows)
    assert details["question_id"].nunique() == 50
    assert details["model"].nunique() == 4
    assert inventory["model"].nunique() == 4

    expanded = pd.concat([details, details.assign(split="all")], ignore_index=True)
    keys = ["model", "chunk_words", "overlap_words", "k", "split"]
    recomputed = (
        expanded.groupby(keys, as_index=False)
        .agg(
            recall_check=("evidence_hit", "mean"),
            page_recall_check=("page_hit", "mean"),
            mrr_check=("reciprocal_rank", "mean"),
            questions_check=("question_id", "count"),
        )
    )
    compared = summary.merge(recomputed, on=keys, validate="one_to_one")
    tolerances = {
        "recall": np.max(np.abs(compared["recall_at_k"] - compared["recall_check"])),
        "page_recall": np.max(
            np.abs(compared["page_recall_at_k"] - compared["page_recall_check"])
        ),
        "mrr": np.max(np.abs(compared["mrr_at_k"] - compared["mrr_check"])),
    }
    assert max(tolerances.values()) < 1e-12
    assert (compared["questions"] == compared["questions_check"]).all()

    for _, rows in summary.groupby(["model", "chunk_words", "overlap_words", "split"]):
        ordered = rows.sort_values("k")
        assert (ordered["recall_at_k"].diff().fillna(0) >= -1e-12).all()
        assert (ordered["page_recall_at_k"].diff().fillna(0) >= -1e-12).all()
    assert ((summary["recall_ci_low"] <= summary["recall_at_k"]).all())
    assert ((summary["recall_at_k"] <= summary["recall_ci_high"]).all())

    independently_selected = _select_configuration(summary.to_dict("records"))
    assert independently_selected["selected_configuration"] == selected["selected_configuration"]
    assert independently_selected["selection_rule"] == selected["selection_rule"]
    assert independently_selected["selected_on"] == selected["selected_on"]
    for recomputed_row, saved_row in zip(
        independently_selected["held_out_test"], selected["held_out_test"], strict=True
    ):
        assert recomputed_row["k"] == saved_row["k"]
        assert recomputed_row["questions"] == saved_row["questions"]
        for metric in (
            "recall_at_k",
            "recall_ci_low",
            "recall_ci_high",
            "page_recall_at_k",
            "mrr_at_k",
        ):
            assert np.isclose(recomputed_row[metric], saved_row[metric], atol=1e-12)
    chosen = selected["selected_configuration"]
    held_out_at_5 = next(row for row in selected["held_out_test"] if row["k"] == 5)
    selected_test = details[
        (details["model"] == chosen["model"])
        & (details["chunk_words"] == chosen["chunk_words"])
        & (details["overlap_words"] == chosen["overlap_words"])
        & (details["split"] == "test")
        & (details["k"] == 5)
    ]
    assert int(selected_test["evidence_hit"].sum()) == 15
    assert held_out_at_5["recall_at_k"] == 1.0

    report = {
        "assessment": "Ready to share with caveats",
        "checks": {
            "detail_rows": len(details),
            "summary_rows": len(summary),
            "questions": int(details["question_id"].nunique()),
            "models": int(details["model"].nunique()),
            "aggregate_max_absolute_error": {
                key: float(value) for key, value in tolerances.items()
            },
            "recall_monotonic_in_k": True,
            "confidence_intervals_contain_estimates": True,
            "selection_recomputed": True,
            "selected_test_recall_at_5_hits": int(selected_test["evidence_hit"].sum()),
            "selected_test_questions": len(selected_test),
        },
        "required_caveats": [
            "The held-out split contains 15 questions and has wide confidence intervals.",
            "The benchmark covers one technical section of one engineering textbook.",
            "Retrieval recall is not a complete measure of generated-answer correctness.",
        ],
    }
    (results_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    report = validate_results(args.results.expanduser().resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
