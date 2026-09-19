from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_qwen_commentary.py"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_timesoccer import load_annotation  # noqa: E402
from generate_qwen_commentary import (  # noqa: E402
    PreannotationError,
    atomic_save_json,
    build_event_result,
    build_preannotation_manifest,
    calculate_context_window,
    ensure_distinct_paths,
    events_requiring_generation,
    load_or_initialize_preannotation,
    upsert_event_result,
    validate_resume_compatibility,
    video_basename_matches,
)


ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / "annotations" / "basketball_001.annotation.json"
)


def current_annotation() -> dict:
    return load_annotation(ANNOTATION_PATH, require_export_ready=False)


def expected_manifest(annotation: dict | None = None) -> dict:
    source = annotation or current_annotation()
    return build_preannotation_manifest(
        source,
        ANNOTATION_PATH,
        "Qwen/Qwen2.5-VL-7B-Instruct",
        5.0,
        5.0,
    )


def completed_result(annotation: dict, event_index: int = 0) -> dict:
    return build_event_result(
        annotation["events"][event_index],
        annotation["duration_sec"],
        5.0,
        5.0,
        candidate_commentary_zh="人工审核前的模型候选。",
    )


def failed_result(annotation: dict, event_index: int = 0) -> dict:
    return build_event_result(
        annotation["events"][event_index],
        annotation["duration_sec"],
        5.0,
        5.0,
        error="RuntimeError: simulated failure",
    )


class ContextWindowTests(unittest.TestCase):
    def test_normal_five_second_window(self) -> None:
        self.assertEqual(
            calculate_context_window(17.0, 60.0, 5.0, 5.0),
            (12.0, 22.0, 5.0),
        )

    def test_window_is_clipped_at_video_start(self) -> None:
        self.assertEqual(
            calculate_context_window(2.0, 60.0, 5.0, 5.0),
            (0.0, 7.0, 2.0),
        )

    def test_window_is_clipped_at_video_end(self) -> None:
        self.assertEqual(
            calculate_context_window(58.0, 60.0, 5.0, 5.0),
            (53.0, 60.0, 5.0),
        )


class InputSafetyTests(unittest.TestCase):
    def test_video_match_uses_basename_only(self) -> None:
        self.assertTrue(
            video_basename_matches(
                "/root/autodl-tmp/videos/basketball_001.mp4",
                "basketball_001.mp4",
            )
        )
        self.assertFalse(
            video_basename_matches(
                "/root/autodl-tmp/videos/another_video.mp4",
                "basketball_001.mp4",
            )
        )

    def test_output_path_cannot_equal_annotation_path(self) -> None:
        with self.assertRaisesRegex(PreannotationError, "must differ"):
            ensure_distinct_paths(ANNOTATION_PATH, ANNOTATION_PATH)

    def test_module_imports_without_qwen_dependencies(self) -> None:
        unavailable = {
            "torch": None,
            "transformers": None,
            "qwen_vl_utils": None,
        }
        with mock.patch.dict(sys.modules, unavailable):
            namespace = runpy.run_path(
                str(SCRIPT_PATH), run_name="qwen_logic_import_test"
            )
        self.assertIn("calculate_context_window", namespace)


class ResumeTests(unittest.TestCase):
    def test_matching_event_snapshot_can_resume(self) -> None:
        annotation = current_annotation()
        expected = expected_manifest(annotation)
        existing = copy.deepcopy(expected)
        existing["events"].append(completed_result(annotation))
        validate_resume_compatibility(existing, expected, annotation)

    def test_changed_fact_rejects_old_result(self) -> None:
        annotation = current_annotation()
        expected = expected_manifest(annotation)
        existing = copy.deepcopy(expected)
        existing["events"].append(completed_result(annotation))
        changed = copy.deepcopy(annotation)
        changed["events"][0]["fact_zh"] = "被修改的人工事实。"
        with self.assertRaisesRegex(PreannotationError, "fact_zh differs"):
            validate_resume_compatibility(existing, expected, changed)

    def test_changed_timestamp_rejects_old_result(self) -> None:
        annotation = current_annotation()
        expected = expected_manifest(annotation)
        existing = copy.deepcopy(expected)
        existing["events"].append(completed_result(annotation))
        changed = copy.deepcopy(annotation)
        changed["events"][0]["timestamp_sec"] = 8.5
        with self.assertRaisesRegex(PreannotationError, "timestamp_sec differs"):
            validate_resume_compatibility(existing, expected, changed)

    def test_changed_event_id_rejects_old_result(self) -> None:
        annotation = current_annotation()
        expected = expected_manifest(annotation)
        existing = copy.deepcopy(expected)
        existing["events"].append(completed_result(annotation))
        changed = copy.deepcopy(annotation)
        changed["events"][0]["event_id"] = "E099"
        with self.assertRaisesRegex(PreannotationError, "no longer exists"):
            validate_resume_compatibility(existing, expected, changed)

    def test_completed_event_is_skipped(self) -> None:
        annotation = current_annotation()
        preannotation = expected_manifest(annotation)
        preannotation["events"].append(completed_result(annotation))
        pending_ids = {
            event["event_id"]
            for event in events_requiring_generation(annotation, preannotation)
        }
        self.assertNotIn("E001", pending_ids)
        self.assertEqual(len(pending_ids), 11)

    def test_failed_event_is_retried(self) -> None:
        annotation = current_annotation()
        preannotation = expected_manifest(annotation)
        preannotation["events"].append(failed_result(annotation))
        pending_ids = [
            event["event_id"]
            for event in events_requiring_generation(annotation, preannotation)
        ]
        self.assertIn("E001", pending_ids)
        self.assertEqual(len(pending_ids), 12)


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_path = (
            Path(__file__).resolve().parent
            / f".preannotation_test_{os.getpid()}.json"
        )
        self.temporary_path = self.output_path.with_name(
            f".{self.output_path.name}.{os.getpid()}.tmp"
        )
        self.output_path.unlink(missing_ok=True)
        self.temporary_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.output_path.unlink(missing_ok=True)
        self.temporary_path.unlink(missing_ok=True)

    def test_atomic_save_produces_valid_json(self) -> None:
        annotation = current_annotation()
        preannotation = expected_manifest(annotation)
        upsert_event_result(
            preannotation, completed_result(annotation), annotation
        )
        atomic_save_json(self.output_path, preannotation)

        loaded = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, preannotation)
        self.assertFalse(self.temporary_path.exists())

    def test_annotation_is_unchanged_by_preannotation_persistence(self) -> None:
        before = ANNOTATION_PATH.read_bytes()
        annotation = current_annotation()
        preannotation = expected_manifest(annotation)
        upsert_event_result(
            preannotation, failed_result(annotation), annotation
        )

        atomic_save_json(self.output_path, preannotation)

        self.assertEqual(ANNOTATION_PATH.read_bytes(), before)

    def test_existing_compatible_file_is_loaded(self) -> None:
        annotation = current_annotation()
        expected = expected_manifest(annotation)
        expected["events"].append(completed_result(annotation))
        atomic_save_json(self.output_path, expected)

        loaded = load_or_initialize_preannotation(
            self.output_path,
            expected_manifest(annotation),
            annotation,
        )
        self.assertEqual(loaded, expected)


if __name__ == "__main__":
    unittest.main()
