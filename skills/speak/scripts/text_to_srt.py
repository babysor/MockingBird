#!/usr/bin/env python3
"""Convert plain text to SRT with auto-estimated timings.

Splits text into sentences, estimates duration per sentence based on
character-per-second rate, and writes a valid SRT file.
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[。！？.!?\n])\s*'
)


def split_sentences(text: str) -> List[str]:
    raw = SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def estimate_timings(
    sentences: List[str],
    chars_per_second: float,
    gap_ms: int,
    start_offset_ms: int = 0,
) -> List[Tuple[int, int, int, str]]:
    """Return list of (index, start_ms, end_ms, text)."""
    result = []
    cursor_ms = start_offset_ms
    for i, sentence in enumerate(sentences, start=1):
        char_count = len(sentence)
        duration_ms = max(500, int(char_count / chars_per_second * 1000))
        start_ms = cursor_ms
        end_ms = start_ms + duration_ms
        result.append((i, start_ms, end_ms, sentence))
        cursor_ms = end_ms + gap_ms
    return result


def ms_to_srt_time(ms: int) -> str:
    total_sec, millis = divmod(ms, 1000)
    total_min, sec = divmod(total_sec, 60)
    hour, minute = divmod(total_min, 60)
    return f"{hour:02d}:{minute:02d}:{sec:02d},{millis:03d}"


def write_srt(entries: List[Tuple[int, int, int, str]], path: Path) -> None:
    lines = []
    for idx, start_ms, end_ms, text in entries:
        lines.append(str(idx))
        lines.append(f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}")
        lines.append(text)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert text file to SRT with auto-estimated timings."
    )
    parser.add_argument("--input", required=True, help="Input text file path")
    parser.add_argument("--output", required=True, help="Output SRT file path")
    parser.add_argument(
        "--chars-per-second",
        type=float,
        default=4.0,
        help="Reading speed in characters per second (default: 4.0, good for Chinese; "
        "use ~15 for English)",
    )
    parser.add_argument(
        "--gap-ms",
        type=int,
        default=300,
        help="Gap between segments in milliseconds (default: 300)",
    )
    parser.add_argument(
        "--start-offset-ms",
        type=int,
        default=0,
        help="Timeline start offset in milliseconds (default: 0)",
    )
    args = parser.parse_args()

    try:
        text = Path(args.input).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Input text is empty.")

        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("No sentences found after splitting.")

        entries = estimate_timings(
            sentences,
            chars_per_second=args.chars_per_second,
            gap_ms=args.gap_ms,
            start_offset_ms=args.start_offset_ms,
        )
        write_srt(entries, Path(args.output))
        print(f"Done. {len(entries)} segments written to {args.output}")
        total_ms = entries[-1][2] if entries else 0
        print(f"Total duration: {ms_to_srt_time(total_ms)}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
