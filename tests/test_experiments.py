from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.build_notebook import build_notebook
from experiments.common import parse_pages, rechunk
from experiments.run_retrieval_benchmark import (
    _evidence_relevant,
    _exclusive_run_lock,
    _select_configuration,
    _summarize,
)


class ExperimentTests(unittest.TestCase):
    def test_rechunk_preserves_page_provenance_and_overlap(self) -> None:
        records = [
            {"sequence": 0, "pages": [1], "text": "one two three four"},
            {"sequence": 1, "pages": [2], "text": "five six seven eight"},
        ]
        chunks = rechunk(records, chunk_size=4, overlap=2)
        self.assertEqual(chunks[0]["text"], "one two three four")
        self.assertEqual(chunks[1]["text"], "three four five six")
        self.assertEqual(chunks[1]["pages"], [1, 2])

    def test_page_parser_accepts_string_and_list(self) -> None:
        self.assertEqual(parse_pages("4, 2, 4"), [2, 4])
        self.assertEqual(parse_pages([3, 1]), [1, 3])

    def test_evidence_relevance_requires_page_and_terms(self) -> None:
        question = {"expected_pages": [4], "evidence_terms": ["pore pressure", "fluid"]}
        self.assertTrue(
            _evidence_relevant(
                {"pages": [4], "text": "Fluid motion changes pore pressure."},
                question,
            )
        )
        self.assertFalse(
            _evidence_relevant(
                {"pages": [3], "text": "Fluid motion changes pore pressure."},
                question,
            )
        )
        self.assertFalse(
            _evidence_relevant(
                {"pages": [4], "text": "Fluid motion is present."},
                question,
            )
        )

    def test_configuration_selection_uses_development_only(self) -> None:
        base = {
            "chunk_words": 200,
            "overlap_words": 0,
            "k": 5,
            "mrr_at_k": 0.5,
            "retrieval_ms_per_question": 1.0,
            "questions": 2,
            "page_recall_at_k": 1.0,
            "recall_ci_low": 0.0,
            "recall_ci_high": 1.0,
            "mrr_ci_low": 0.0,
            "mrr_ci_high": 1.0,
        }
        summary = [
            {**base, "model": "a", "split": "development", "recall_at_k": 0.9},
            {**base, "model": "b", "split": "development", "recall_at_k": 0.8},
            {**base, "model": "a", "split": "test", "recall_at_k": 0.1},
            {**base, "model": "b", "split": "test", "recall_at_k": 1.0},
        ]
        selected = _select_configuration(summary)
        self.assertEqual(selected["selected_configuration"]["model"], "a")

    def test_benchmark_lock_rejects_a_second_writer(self) -> None:
        with TemporaryDirectory() as directory:
            lock = Path(directory) / ".benchmark.lock"
            with _exclusive_run_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "already using"):
                    with _exclusive_run_lock(lock):
                        self.fail("a second writer acquired the benchmark lock")

    def test_summary_includes_bounded_wilson_interval(self) -> None:
        base = {
            "model": "example",
            "chunk_words": 200,
            "overlap_words": 50,
            "k": 5,
            "split": "test",
            "embedding_dimension": 128,
            "model_context_tokens": 512,
            "index_chunks": 10,
            "document_embedding_seconds": 1.0,
            "query_embedding_seconds": 0.1,
            "retrieval_ms_per_question": 1.0,
            "page_hit": True,
            "reciprocal_rank": 1.0,
        }
        rows = [{**base, "evidence_hit": hit} for hit in (True, True, False)]
        summary = _summarize(rows)[0]
        self.assertLess(summary["recall_ci_low"], summary["recall_at_k"])
        self.assertGreater(summary["recall_ci_high"], summary["recall_at_k"])
        self.assertGreaterEqual(summary["recall_ci_low"], 0.0)
        self.assertLessEqual(summary["recall_ci_high"], 1.0)

    def test_generated_notebook_code_is_dedented(self) -> None:
        notebook = build_notebook()
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        for cell in code_cells:
            first_line = next(line for line in cell.source.splitlines() if line.strip())
            self.assertEqual(first_line, first_line.lstrip())


if __name__ == "__main__":
    unittest.main()
