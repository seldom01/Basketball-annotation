"""Export one authoritative basketball annotation to TimeSoccer JSON.

This module uses only the Python standard library. The current exporter treats the
complete source video as one training clip, so its window start is 0.0 seconds.
Future windowed exports must subtract window_start_sec from each authoritative
source-video timestamp instead of changing the annotation itself.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "1.0"
ALLOWED_STATUSES = {
    "fact_confirmed",
    "commentary_confirmed",
    "ready_for_export",
}
EVENT_ID_PATTERN = re.compile(r"^E[0-9]{3,}$")
DEFAULT_QUESTION = (
    "Provide timestamped English commentary for the basketball events in the video."
)


class AnnotationError(ValueError):
    """Raised when authoritative annotation data violates the project contract."""


def _require_exact_keys(
    value: dict[str, Any], required: set[str], context: str
) -> None:
    actual = set(value)
    missing = required - actual
    extra = actual - required
    if missing:
        raise AnnotationError(f"{context} is missing fields: {sorted(missing)}")
    if extra:
        raise AnnotationError(f"{context} has unsupported fields: {sorted(extra)}")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationError(f"{field} must be a non-empty string")
    return value


def _require_float(value: Any, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise AnnotationError(
            f"{field} must be a finite JSON decimal such as 17.0"
        )
    return value


def validate_annotation(data: Any, *, require_export_ready: bool) -> dict[str, Any]:
    """Validate one parsed authoritative annotation without mutating it."""
    if not isinstance(data, dict):
        raise AnnotationError("annotation root must be an object")

    _require_exact_keys(
        data,
        {"schema_version", "video", "source_game_id", "duration_sec", "events"},
        "annotation root",
    )
    if data["schema_version"] != SCHEMA_VERSION:
        raise AnnotationError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {data['schema_version']!r}"
        )
    _require_nonempty_string(data["video"], "video")
    _require_nonempty_string(data["source_game_id"], "source_game_id")
    duration = _require_float(data["duration_sec"], "duration_sec")
    if duration < 0.0:
        raise AnnotationError("duration_sec must be non-negative")

    events = data["events"]
    if not isinstance(events, list) or not events:
        raise AnnotationError("events must be a non-empty array")

    event_ids: set[str] = set()
    event_fields = {
        "event_id",
        "timestamp_sec",
        "fact_zh",
        "commentary_zh",
        "commentary_en",
        "status",
        "notes",
    }

    for index, event in enumerate(events):
        context = f"events[{index}]"
        if not isinstance(event, dict):
            raise AnnotationError(f"{context} must be an object")
        _require_exact_keys(event, event_fields, context)

        event_id = _require_nonempty_string(event["event_id"], f"{context}.event_id")
        if not EVENT_ID_PATTERN.fullmatch(event_id):
            raise AnnotationError(
                f"{context}.event_id must match E followed by at least three digits"
            )
        if event_id in event_ids:
            raise AnnotationError(f"duplicate event_id: {event_id}")
        event_ids.add(event_id)

        timestamp = _require_float(event["timestamp_sec"], f"{context}.timestamp_sec")
        if not 0.0 <= timestamp <= duration:
            raise AnnotationError(
                f"{event_id} timestamp_sec {timestamp} is outside 0.0..{duration}"
            )

        _require_nonempty_string(event["fact_zh"], f"{context}.fact_zh")
        for field in ("commentary_zh", "commentary_en", "notes"):
            if not isinstance(event[field], str):
                raise AnnotationError(f"{context}.{field} must be a string")

        status = event["status"]
        if status not in ALLOWED_STATUSES:
            raise AnnotationError(f"{context}.status is invalid: {status!r}")

        commentary_zh = event["commentary_zh"].strip()
        commentary_en = event["commentary_en"].strip()
        if status == "fact_confirmed" and (commentary_zh or commentary_en):
            raise AnnotationError(
                f"{event_id} is fact_confirmed, so both commentary fields must be empty"
            )
        if status == "commentary_confirmed" and (
            not commentary_zh or commentary_en
        ):
            raise AnnotationError(
                f"{event_id} commentary_confirmed requires Chinese commentary only"
            )
        if status == "ready_for_export" and (
            not commentary_zh or not commentary_en
        ):
            raise AnnotationError(
                f"{event_id} ready_for_export requires both commentary fields"
            )
        if require_export_ready and status != "ready_for_export":
            raise AnnotationError(
                f"{event_id} is {status}, not ready_for_export; formal export refused"
            )

    return data


def load_annotation(path: Path, *, require_export_ready: bool) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except FileNotFoundError as exc:
        raise AnnotationError(f"input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"invalid JSON in {path}: {exc}") from exc
    return validate_annotation(data, require_export_ready=require_export_ready)


def format_timestamp(timestamp_sec: float) -> str:
    """Preserve useful precision while always returning decimal notation."""
    text = format(Decimal(str(timestamp_sec)), "f")
    if "." not in text:
        text += ".0"
    return text


def build_timesoccer_payload(
    annotation: dict[str, Any], question: str = DEFAULT_QUESTION
) -> list[dict[str, Any]]:
    validate_annotation(annotation, require_export_ready=True)
    _require_nonempty_string(question, "question")

    ordered_events = sorted(
        annotation["events"], key=lambda event: event["timestamp_sec"]
    )
    answer_parts = [
        f"{format_timestamp(event['timestamp_sec'])} seconds, "
        f"{event['commentary_en'].strip()}"
        for event in ordered_events
    ]
    return [
        {
            "video": annotation["video"],
            "QA": [{"q": question.strip(), "a": " ".join(answer_parts)}],
            "source": "basketball_annotation",
        }
    ]


def export_annotation(
    input_path: Path, output_path: Path, question: str = DEFAULT_QUESTION
) -> None:
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    if input_resolved == output_resolved:
        raise AnnotationError("input and output paths must be different")

    annotation = load_annotation(input_resolved, require_export_ready=True)
    payload = build_timesoccer_payload(annotation, question)
    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    output_resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one complete-source basketball annotation to TimeSoccer JSON."
    )
    parser.add_argument("--input", required=True, type=Path, help="annotation JSON")
    parser.add_argument("--output", required=True, type=Path, help="output JSON")
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="TimeSoccer QA question text",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        export_annotation(args.input, args.output, args.question)
    except AnnotationError as exc:
        print(f"Export refused: {exc}")
        return 1
    print(f"Exported: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

