#!/usr/bin/env python3
"""Simple TTS via Noiz API (no timeline).

Supports direct text or text-file input, optional emotion enhancement,
voice cloning via reference audio, and emotion parameters.
Use kokoro-tts CLI directly for the Kokoro backend (no wrapper needed).
"""
import argparse
import base64
import binascii
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def normalize_api_key_base64(api_key: str) -> str:
    key = api_key.strip()
    if not key:
        return key
    padded = key + ("=" * (-len(key) % 4))
    try:
        decoded = base64.b64decode(padded, validate=True)
        canonical = base64.b64encode(decoded).decode("ascii").rstrip("=")
        if decoded and canonical == key.rstrip("="):
            return key
    except binascii.Error:
        pass
    return base64.b64encode(key.encode("utf-8")).decode("ascii")


def call_emotion_enhance(
    base_url: str, api_key: str, text: str, timeout: int
) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/emotion-enhance",
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json={"text": text},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"/emotion-enhance failed: status={resp.status_code}, body={resp.text}"
        )
    enhanced = resp.json().get("data", {}).get("emotion_enhance")
    if not enhanced:
        raise RuntimeError(f"/emotion-enhance returned no data: {resp.text}")
    return enhanced


def synthesize(
    base_url: str,
    api_key: str,
    text: str,
    voice_id: Optional[str],
    reference_audio: Optional[Path],
    output_format: str,
    speed: float,
    emo: Optional[str],
    target_lang: Optional[str],
    similarity_enh: bool,
    save_voice: bool,
    duration: Optional[float],
    timeout: int,
    out_path: Path,
) -> float:
    if duration is not None and not (0 < duration <= 36):
        raise ValueError("duration must be in range (0, 36] seconds")
    url = f"{base_url.rstrip('/')}/text-to-speech"
    data: Dict[str, str] = {
        "text": text,
        "output_format": output_format,
        "speed": str(speed),
    }
    if voice_id:
        data["voice_id"] = voice_id
    if emo:
        data["emo"] = emo
    if target_lang:
        data["target_lang"] = target_lang
    if similarity_enh:
        data["similarity_enh"] = "true"
    if save_voice:
        data["save_voice"] = "true"
    if duration is not None:
        data["duration"] = str(duration)

    files = None
    if reference_audio:
        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        files = {
            "file": (
                reference_audio.name,
                reference_audio.open("rb"),
                "application/octet-stream",
            )
        }
    elif not voice_id:
        raise ValueError("Either --voice-id or --reference-audio is required.")

    try:
        resp = requests.post(
            url,
            headers={"Authorization": api_key},
            data=data,
            files=files,
            timeout=timeout,
        )
    finally:
        if files and files["file"][1]:
            files["file"][1].close()

    if resp.status_code != 200:
        raise RuntimeError(
            f"/text-to-speech failed: status={resp.status_code}, body={resp.text}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    dur = resp.headers.get("X-Audio-Duration")
    return float(dur) if dur else -1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple TTS via Noiz API (no timeline).")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Text string to synthesize")
    g.add_argument("--text-file", help="Path to text file")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--voice-id")
    parser.add_argument("--reference-audio", help="Local audio for voice cloning")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="https://noiz.ai/v1")
    parser.add_argument("--output-format", choices=["wav", "mp3"], default="wav")
    parser.add_argument("--auto-emotion", action="store_true")
    parser.add_argument("--emo", help='Emotion JSON string, e.g. \'{"Joy":0.5}\'')
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--target-lang")
    parser.add_argument("--similarity-enh", action="store_true")
    parser.add_argument("--save-voice", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SEC",
        help="Target audio duration in seconds (0, 36], optional",
    )
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()
    args.api_key = normalize_api_key_base64(args.api_key)

    try:
        if args.text_file:
            text = Path(args.text_file).read_text(encoding="utf-8").strip()
        else:
            text = args.text

        if not text:
            raise ValueError("Input text is empty.")

        if len(text) > 5000:
            print(
                f"Warning: text is {len(text)} chars (max 5000). "
                "Consider chunking for long texts.",
                file=sys.stderr,
            )

        if args.auto_emotion:
            text = call_emotion_enhance(
                args.base_url, args.api_key, text, args.timeout_sec
            )

        ref = Path(args.reference_audio) if args.reference_audio else None
        out_duration = synthesize(
            base_url=args.base_url,
            api_key=args.api_key,
            text=text,
            voice_id=args.voice_id,
            reference_audio=ref,
            output_format=args.output_format,
            speed=args.speed,
            emo=args.emo,
            target_lang=args.target_lang,
            similarity_enh=args.similarity_enh,
            save_voice=args.save_voice,
            duration=args.duration,
            timeout=args.timeout_sec,
            out_path=Path(args.output),
        )
        print(f"Done. Output: {args.output} (duration: {out_duration}s)")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
