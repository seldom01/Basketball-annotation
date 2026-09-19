from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_timesoccer import (  # noqa: E402
    AnnotationError,
    build_timesoccer_payload,
    export_annotation,
    load_annotation,
    validate_annotation,
)


ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / "annotations" / "basketball_001.annotation.json"
)

EXPECTED_FACTS = [
    "白衣球员罚球命中，同时另一名白衣球员在罚球线内投篮未命中。",
    "黑衣球员命中两分球。",
    "白衣球员命中三分球。",
    "白衣球员在三分线内投篮未命中。",
    "另一名白衣球员在三分线外投篮未命中。",
    "黑衣球员命中两分球。",
    "黑衣球员篮下投篮命中。",
    "白衣球员在三分线内投篮未命中，篮球被篮筐弹出。",
    "一名白衣球员在三分线内投篮未命中。",
    "黑衣球员在罚球线外投篮未命中。",
    "白衣球员命中两分球。",
    "白衣球员在罚球线外投篮命中，紧接着再次投篮但未命中。",
]


def ready_annotation() -> dict:
    return {
        "schema_version": "1.0",
        "video": "ready_source.mp4",
        "source_game_id": "ready_game",
        "duration_sec": 60.0,
        "events": [
            {
                "event_id": "E001",
                "timestamp_sec": 17.0,
                "fact_zh": "人工事实一。",
                "commentary_zh": "确认后的中文解说一。",
                "commentary_en": "First confirmed English commentary.",
                "status": "ready_for_export",
                "notes": "",
            },
            {
                "event_id": "E002",
                "timestamp_sec": 17.25,
                "fact_zh": "人工事实二。",
                "commentary_zh": "确认后的中文解说二。",
                "commentary_en": "Second confirmed English commentary.",
                "status": "ready_for_export",
                "notes": "",
            },
        ],
    }


class CurrentAnnotationTests(unittest.TestCase):
    def test_current_annotation_preserves_all_twelve_human_facts(self) -> None:
        annotation = load_annotation(
            ANNOTATION_PATH, require_export_ready=False
        )
        events = annotation["events"]

        self.assertEqual(len(events), 12)
        self.assertEqual(
            [event["event_id"] for event in events],
            [f"E{number:03d}" for number in range(1, 13)],
        )
        self.assertEqual([event["fact_zh"] for event in events], EXPECTED_FACTS)
        self.assertTrue(
            all(
                0.0 <= event["timestamp_sec"] <= annotation["duration_sec"]
                for event in events
            )
        )

    def test_current_annotation_is_not_export_ready(self) -> None:
        with self.assertRaisesRegex(AnnotationError, "formal export refused"):
            load_annotation(ANNOTATION_PATH, require_export_ready=True)


class ExporterTests(unittest.TestCase):
    def test_export_uses_only_timestamp_and_english_commentary_in_answer(self) -> None:
        annotation = ready_annotation()
        payload = build_timesoccer_payload(annotation)
        answer = payload[0]["QA"][0]["a"]

        self.assertIn("17.0 seconds, First confirmed English commentary.", answer)
        self.assertIn("17.25 seconds, Second confirmed English commentary.", answer)
        self.assertNotIn("E001", answer)
        self.assertNotIn("人工事实", answer)
        self.assertNotIn("ready_for_export", answer)

    def test_payload_build_does_not_modify_annotation(self) -> None:
        annotation = ready_annotation()
        before = copy.deepcopy(annotation)

        build_timesoccer_payload(annotation)

        self.assertEqual(annotation, before)

    def test_export_rejects_using_source_as_output(self) -> None:
        before = ANNOTATION_PATH.read_bytes()
        with self.assertRaisesRegex(AnnotationError, "must be different"):
            export_annotation(ANNOTATION_PATH, ANNOTATION_PATH)
        self.assertEqual(ANNOTATION_PATH.read_bytes(), before)

    def test_duplicate_event_id_is_rejected(self) -> None:
        annotation = ready_annotation()
        annotation["events"][1]["event_id"] = "E001"
        with self.assertRaisesRegex(AnnotationError, "duplicate event_id"):
            validate_annotation(annotation, require_export_ready=False)

    def test_timestamp_outside_source_duration_is_rejected(self) -> None:
        annotation = ready_annotation()
        for invalid_timestamp in (-0.1, 60.1):
            with self.subTest(timestamp=invalid_timestamp):
                changed = copy.deepcopy(annotation)
                changed["events"][0]["timestamp_sec"] = invalid_timestamp
                with self.assertRaisesRegex(AnnotationError, "outside"):
                    validate_annotation(changed, require_export_ready=False)


if __name__ == "__main__":
    unittest.main()
