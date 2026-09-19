"""Generate unreviewed Qwen commentary candidates for authoritative events.

All data preparation, validation, resume, checkpoint, and FFmpeg helpers in this
module use only the Python standard library. Qwen/PyTorch dependencies are
imported lazily only when ``load_qwen_runtime`` is called on the GPU machine.

This script never writes to the authoritative annotation. Its only data output
is a separate preannotation JSON containing unreviewed candidates.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

from export_timesoccer import AnnotationError, load_annotation


PREANNOTATION_SCHEMA_VERSION = "1.0"
TASK = "constrained_commentary_expansion"
PROMPT_VERSION = "qwen_basketball_commentary_expansion_zh_v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "qwen_basketball_commentary_expansion_zh_v1.txt"
)
TOP_LEVEL_FIELDS = {
    "schema_version",
    "source_annotation_file",
    "video",
    "source_game_id",
    "duration_sec",
    "model",
    "task",
    "prompt_version",
    "context_before_sec",
    "context_after_sec",
    "events",
}
EVENT_FIELDS = {
    "event_id",
    "timestamp_sec",
    "fact_zh",
    "window_start_sec",
    "window_end_sec",
    "target_offset_sec",
    "generation_status",
    "candidate_commentary_zh",
    "error",
}
RESUME_METADATA_FIELDS = (
    "schema_version",
    "source_annotation_file",
    "video",
    "source_game_id",
    "duration_sec",
    "model",
    "task",
    "prompt_version",
    "context_before_sec",
    "context_after_sec",
)


class PreannotationError(ValueError):
    """Raised when generation inputs or an existing preannotation are unsafe."""


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreannotationError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise PreannotationError(f"{field} must be finite")
    return number


def format_seconds(value: float) -> str:
    text = format(Decimal(str(value)), "f")
    if "." not in text:
        text += ".0"
    return text


def calculate_context_window(
    timestamp_sec: float,
    duration_sec: float,
    context_before_sec: float,
    context_after_sec: float,
) -> tuple[float, float, float]:
    """Return source start/end and target offset for one local video window."""
    timestamp = _require_finite_number(timestamp_sec, "timestamp_sec")
    duration = _require_finite_number(duration_sec, "duration_sec")
    before = _require_finite_number(context_before_sec, "context_before_sec")
    after = _require_finite_number(context_after_sec, "context_after_sec")
    if duration < 0.0:
        raise PreannotationError("duration_sec must be non-negative")
    if before < 0.0 or after < 0.0:
        raise PreannotationError("context durations must be non-negative")
    if not 0.0 <= timestamp <= duration:
        raise PreannotationError(
            f"timestamp_sec {timestamp} is outside 0.0..{duration}"
        )

    window_start = max(0.0, timestamp - before)
    window_end = min(duration, timestamp + after)
    target_offset = timestamp - window_start
    return window_start, window_end, target_offset


def video_basename_matches(video_argument: str | Path, annotation_video: str) -> bool:
    """Compare only the runtime video's basename with annotation.video."""
    return Path(video_argument).name == annotation_video


def ensure_distinct_paths(annotation_path: Path, output_path: Path) -> None:
    annotation_normalized = os.path.normcase(str(annotation_path.resolve()))
    output_normalized = os.path.normcase(str(output_path.resolve()))
    if annotation_normalized == output_normalized:
        raise PreannotationError("output path must differ from annotation path")


def load_prompt_template(path: Path) -> str:
    try:
        template = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PreannotationError(f"prompt file does not exist: {path}") from exc
    required_placeholders = ("{timestamp_sec}", "{target_offset_sec}", "{fact_zh}")
    missing = [item for item in required_placeholders if item not in template]
    if missing:
        raise PreannotationError(f"prompt is missing placeholders: {missing}")
    return template


def render_prompt(
    template: str,
    timestamp_sec: float,
    target_offset_sec: float,
    fact_zh: str,
) -> str:
    return template.format(
        timestamp_sec=format_seconds(timestamp_sec),
        target_offset_sec=format_seconds(target_offset_sec),
        fact_zh=fact_zh,
    )


def build_preannotation_manifest(
    annotation: dict[str, Any],
    annotation_path: Path,
    model: str,
    context_before_sec: float,
    context_after_sec: float,
) -> dict[str, Any]:
    if not isinstance(model, str) or not model.strip():
        raise PreannotationError("model must be a non-empty string")
    before = _require_finite_number(context_before_sec, "context_before_sec")
    after = _require_finite_number(context_after_sec, "context_after_sec")
    if before < 0.0 or after < 0.0:
        raise PreannotationError("context durations must be non-negative")
    return {
        "schema_version": PREANNOTATION_SCHEMA_VERSION,
        "source_annotation_file": annotation_path.name,
        "video": annotation["video"],
        "source_game_id": annotation["source_game_id"],
        "duration_sec": annotation["duration_sec"],
        "model": model.strip(),
        "task": TASK,
        "prompt_version": PROMPT_VERSION,
        "context_before_sec": before,
        "context_after_sec": after,
        "events": [],
    }


def build_event_result(
    event: dict[str, Any],
    duration_sec: float,
    context_before_sec: float,
    context_after_sec: float,
    *,
    candidate_commentary_zh: str = "",
    error: str = "",
) -> dict[str, Any]:
    window_start, window_end, target_offset = calculate_context_window(
        event["timestamp_sec"],
        duration_sec,
        context_before_sec,
        context_after_sec,
    )
    candidate = candidate_commentary_zh.strip()
    error_text = error.strip()
    if candidate and error_text:
        raise PreannotationError("an event result cannot be completed and failed")
    if not candidate and not error_text:
        raise PreannotationError("an event result requires a candidate or an error")
    return {
        "event_id": event["event_id"],
        "timestamp_sec": event["timestamp_sec"],
        "fact_zh": event["fact_zh"],
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "target_offset_sec": target_offset,
        "generation_status": "completed" if candidate else "failed",
        "candidate_commentary_zh": candidate,
        "error": "" if candidate else error_text,
    }


def validate_preannotation_structure(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise PreannotationError("preannotation root must be an object")
    actual_fields = set(data)
    if actual_fields != TOP_LEVEL_FIELDS:
        missing = sorted(TOP_LEVEL_FIELDS - actual_fields)
        extra = sorted(actual_fields - TOP_LEVEL_FIELDS)
        raise PreannotationError(
            f"preannotation fields mismatch; missing={missing}, extra={extra}"
        )
    if not isinstance(data["events"], list):
        raise PreannotationError("preannotation events must be an array")

    event_ids: set[str] = set()
    for index, event in enumerate(data["events"]):
        if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
            raise PreannotationError(f"preannotation events[{index}] has invalid fields")
        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise PreannotationError(f"preannotation events[{index}].event_id is invalid")
        if event_id in event_ids:
            raise PreannotationError(f"duplicate preannotation event_id: {event_id}")
        event_ids.add(event_id)

        status = event["generation_status"]
        candidate = event["candidate_commentary_zh"]
        error = event["error"]
        if not isinstance(candidate, str) or not isinstance(error, str):
            raise PreannotationError(f"{event_id} candidate and error must be strings")
        if status == "completed" and (not candidate.strip() or error):
            raise PreannotationError(f"{event_id} has an invalid completed result")
        if status == "failed" and (candidate or not error.strip()):
            raise PreannotationError(f"{event_id} has an invalid failed result")
        if status not in {"completed", "failed"}:
            raise PreannotationError(f"{event_id} has an invalid generation_status")
    return data


def validate_resume_compatibility(
    existing: dict[str, Any],
    expected: dict[str, Any],
    annotation: dict[str, Any],
) -> None:
    validate_preannotation_structure(existing)
    for field in RESUME_METADATA_FIELDS:
        if existing[field] != expected[field]:
            raise PreannotationError(
                f"cannot resume: metadata field {field!r} differs "
                f"({existing[field]!r} != {expected[field]!r})"
            )

    authoritative_events = {
        event["event_id"]: event for event in annotation["events"]
    }
    for saved in existing["events"]:
        event_id = saved["event_id"]
        authoritative = authoritative_events.get(event_id)
        if authoritative is None:
            raise PreannotationError(
                f"cannot resume: {event_id} no longer exists in annotation"
            )
        for field in ("timestamp_sec", "fact_zh"):
            if saved[field] != authoritative[field]:
                raise PreannotationError(
                    f"cannot resume: {event_id} {field} differs from annotation"
                )

        expected_start, expected_end, expected_offset = calculate_context_window(
            authoritative["timestamp_sec"],
            annotation["duration_sec"],
            expected["context_before_sec"],
            expected["context_after_sec"],
        )
        expected_window = {
            "window_start_sec": expected_start,
            "window_end_sec": expected_end,
            "target_offset_sec": expected_offset,
        }
        for field, value in expected_window.items():
            if saved[field] != value:
                raise PreannotationError(
                    f"cannot resume: {event_id} {field} is incompatible"
                )


def load_or_initialize_preannotation(
    output_path: Path,
    expected: dict[str, Any],
    annotation: dict[str, Any],
) -> dict[str, Any]:
    if not output_path.exists():
        return expected
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreannotationError(f"invalid preannotation JSON: {output_path}") from exc
    validate_resume_compatibility(existing, expected, annotation)
    return existing


def events_requiring_generation(
    annotation: dict[str, Any], preannotation: dict[str, Any]
) -> list[dict[str, Any]]:
    saved = {event["event_id"]: event for event in preannotation["events"]}
    return [
        event
        for event in annotation["events"]
        if event["event_id"] not in saved
        or saved[event["event_id"]]["generation_status"] == "failed"
    ]


def upsert_event_result(
    preannotation: dict[str, Any],
    result: dict[str, Any],
    annotation: dict[str, Any],
) -> None:
    by_id = {event["event_id"]: event for event in preannotation["events"]}
    by_id[result["event_id"]] = result
    order = {
        event["event_id"]: index for index, event in enumerate(annotation["events"])
    }
    preannotation["events"] = sorted(
        by_id.values(), key=lambda item: order[item["event_id"]]
    )


def atomic_save_json(path: Path, data: dict[str, Any]) -> None:
    """Write valid JSON beside its destination and atomically replace it."""
    validate_preannotation_structure(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_ffmpeg_command(
    ffmpeg_executable: str,
    source_video: Path,
    output_clip: Path,
    window_start_sec: float,
    window_end_sec: float,
) -> list[str]:
    """Build one accurate, re-encoding FFmpeg command for a local MP4 window."""
    duration = window_end_sec - window_start_sec
    if duration <= 0.0:
        raise PreannotationError("video context window must have positive duration")
    return [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-ss",
        format_seconds(window_start_sec),
        "-t",
        format_seconds(duration),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_clip),
    ]


def create_video_clip(
    ffmpeg_executable: str,
    source_video: Path,
    output_clip: Path,
    window_start_sec: float,
    window_end_sec: float,
) -> None:
    command = build_ffmpeg_command(
        ffmpeg_executable,
        source_video,
        output_clip,
        window_start_sec,
        window_end_sec,
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:] or "unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg failed: {detail}")
    if not output_clip.is_file() or output_clip.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not create a usable video clip")


def load_qwen_runtime(model_name: str) -> tuple[Any, Any, Callable[..., Any]]:
    """Load GPU-only dependencies lazily; never called by ordinary local tests."""
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "Qwen runtime dependencies are unavailable. Run this command only in "
            "the dedicated AutoDL environment with PyTorch, Transformers, and "
            "qwen-vl-utils installed."
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, process_vision_info


def generate_qwen_candidate(
    runtime: tuple[Any, Any, Callable[..., Any]],
    video_clip: Path,
    prompt_text: str,
) -> str:
    """Run one Qwen generation and return only the decoded candidate text."""
    model, processor, process_vision_info = runtime
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_clip.resolve().as_uri()},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    candidate = decoded[0].strip() if decoded else ""
    if not candidate:
        raise RuntimeError("Qwen returned an empty candidate")
    return candidate


def _bounded_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"[:1000]


def run_generation(args: argparse.Namespace) -> int:
    annotation_path = args.annotation.resolve()
    video_path = args.video.resolve()
    output_path = args.output.resolve()
    ensure_distinct_paths(annotation_path, output_path)

    annotation_bytes_before = annotation_path.read_bytes()
    annotation = load_annotation(annotation_path, require_export_ready=False)
    if not video_path.is_file():
        raise PreannotationError(f"video file does not exist: {video_path}")
    if not video_basename_matches(video_path, annotation["video"]):
        raise PreannotationError(
            f"video basename {video_path.name!r} does not match "
            f"annotation video {annotation['video']!r}"
        )

    prompt_template = load_prompt_template(args.prompt.resolve())
    expected = build_preannotation_manifest(
        annotation,
        annotation_path,
        args.model,
        args.context_before,
        args.context_after,
    )
    preannotation = load_or_initialize_preannotation(
        output_path, expected, annotation
    )
    pending_events = events_requiring_generation(annotation, preannotation)
    if not pending_events:
        print("All events are already completed; nothing to generate.")
        return 0

    # Loading failure is global: exit before creating failed entries or a new output.
    runtime = load_qwen_runtime(args.model)
    failure_count = 0

    for event in pending_events:
        window_start, window_end, target_offset = calculate_context_window(
            event["timestamp_sec"],
            annotation["duration_sec"],
            args.context_before,
            args.context_after,
        )
        prompt_text = render_prompt(
            prompt_template,
            event["timestamp_sec"],
            target_offset,
            event["fact_zh"],
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"basketball_{event['event_id']}_"
            ) as temporary_directory:
                clip_path = Path(temporary_directory) / "context.mp4"
                create_video_clip(
                    args.ffmpeg,
                    video_path,
                    clip_path,
                    window_start,
                    window_end,
                )
                candidate = generate_qwen_candidate(
                    runtime, clip_path, prompt_text
                )
            result = build_event_result(
                event,
                annotation["duration_sec"],
                args.context_before,
                args.context_after,
                candidate_commentary_zh=candidate,
            )
            print(f"{event['event_id']}: completed")
        except Exception as exc:
            failure_count += 1
            result = build_event_result(
                event,
                annotation["duration_sec"],
                args.context_before,
                args.context_after,
                error=_bounded_error(exc),
            )
            print(f"{event['event_id']}: failed: {result['error']}")

        upsert_event_result(preannotation, result, annotation)
        atomic_save_json(output_path, preannotation)

    if annotation_path.read_bytes() != annotation_bytes_before:
        raise RuntimeError("authoritative annotation changed during generation")
    return 1 if failure_count else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate unreviewed Qwen commentary candidates without modifying "
            "the authoritative annotation."
        )
    )
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-before", type=float, default=5.0)
    parser.add_argument("--context-after", type=float, default=5.0)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_generation(args)
    except (AnnotationError, PreannotationError, OSError, RuntimeError) as exc:
        print(f"Generation refused: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
