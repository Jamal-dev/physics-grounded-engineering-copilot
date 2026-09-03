from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from fine_tuning.comparison import VARIANTS, ComparisonRunner
from fine_tuning.config import load_config
from fine_tuning.data import (
    evaluation_examples,
    load_and_validate,
    split_prompt_collisions,
)
from fine_tuning.metrics import answer_metrics, score_records, token_f1


class FineTuningDataTests(unittest.TestCase):
    def test_config_paths_are_relative_to_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text("", encoding="utf-8")
            config_path = root / "run.yaml"
            config_path.write_text(
                """model:
  name_or_path: example/model
data:
  train_file: train.jsonl
output_dir: output
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.base_model, "example/model")
            self.assertEqual(config.train_file, root / "train.jsonl")
            self.assertEqual(config.adapter_dir, root / "output" / "final")

    def test_dataset_preserves_tools_and_extracts_held_out_prompt(self) -> None:
        row = {
            "id": "case-1",
            "family": "poisson",
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Compute a source."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "calculate"}}],
                },
                {"role": "tool", "name": "calculate", "content": "42"},
                {"role": "assistant", "content": '{"answer": 42}'},
            ],
            "tools": [{"type": "function", "function": {"name": "calculate"}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows, report = load_and_validate(path)
            [example] = evaluation_examples(rows)
        self.assertEqual(report.tool_call_examples, 1)
        self.assertEqual([m["role"] for m in example.prompt_messages], ["system", "user"])
        self.assertEqual(example.reference, '{"answer": 42}')
        self.assertTrue(example.expects_tool_call)
        self.assertEqual(example.metadata["family"], "poisson")

    def test_split_prompt_collision_check_is_exact(self) -> None:
        def row(row_id: str, query: str):
            return {
                "id": row_id,
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "answer"},
                ],
            }

        collisions = split_prompt_collisions(
            {
                "train": [row("a", "same")],
                "validation": [row("b", "same")],
                "test": [row("c", "different")],
            }
        )
        self.assertEqual(collisions["train_validation"], 1)
        self.assertEqual(collisions["train_test"], 0)


@dataclass
class _Retrieved:
    text: str
    sources: list[dict[str, object]]


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, str]] = []

    def complete(self, messages, *, tuned, tools=None):
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        self.calls.append((tuned, user))
        return f"tuned={tuned};rag={'Retrieved context:' in user}"


class ComparisonTests(unittest.TestCase):
    def test_runner_reuses_one_retrieval_for_all_four_variants(self) -> None:
        retrieval_calls: list[tuple[str, int]] = []

        def retrieve(query: str, top_k: int) -> _Retrieved:
            retrieval_calls.append((query, top_k))
            return _Retrieved("evidence", [{"source": "paper.pdf"}])

        model = _FakeModel()
        result = ComparisonRunner(model, retrieve).run("question", top_k=3)
        self.assertEqual(retrieval_calls, [("question", 3)])
        self.assertEqual([answer.variant for answer in result.answers], [v.key for v in VARIANTS])
        self.assertEqual(
            [answer.text for answer in result.answers],
            [
                "tuned=False;rag=False",
                "tuned=False;rag=True",
                "tuned=True;rag=False",
                "tuned=True;rag=True",
            ],
        )

    def test_metrics_report_factorial_fine_tuning_effect(self) -> None:
        reference = '{"status": "correct"}'
        texts = {
            "base": '{"status": "wrong"}',
            "base_rag": '{"status": "wrong"}',
            "fine_tuned": reference,
            "fine_tuned_rag": reference,
        }
        record = {
            "reference": reference,
            "expects_tool_call": False,
            "tool_names": [],
            "answers": [
                {"variant": key, "text": text, "latency_seconds": 1.0}
                for key, text in texts.items()
            ],
        }
        summary = score_records([record], json_fields=["status"])
        self.assertEqual(summary["variants"]["base"]["json_field.status"], 0.0)
        self.assertEqual(summary["variants"]["fine_tuned"]["json_field.status"], 1.0)
        self.assertEqual(
            summary["effects"]["json_field.status"]["fine_tuning_without_rag"],
            1.0,
        )
        self.assertEqual(token_f1("a a b", "a b b"), 2 / 3)

    def test_missing_reference_json_field_is_not_scored_as_correct(self) -> None:
        metrics = answer_metrics(
            '{"status": "correct"}',
            '{"status": "correct"}',
            json_fields=["status", "boundary_condition_assessment.sufficient"],
        )
        self.assertEqual(metrics["json_field.status"], 1.0)
        self.assertIsNone(
            metrics["json_field.boundary_condition_assessment.sufficient"]
        )


if __name__ == "__main__":
    unittest.main()
