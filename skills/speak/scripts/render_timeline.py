#!/usr/bin/env python3
"""Timeline mode: render SRT to timeline-accurate audio.

Supports two backends:
  - kokoro (default): local CLI, uses ffmpeg atempo for duration matching
  - noiz: cloud API with server-side duration forcing, emotion, voice cloning

Parses SRT, resolves per-segment voice config from a voice-map JSON,
calls TTS for each segment, normalizes to exact duration, delays to
correct start time, and mixes into one timeline track.
"""
import argparse
import base64
import binascii
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$")


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


@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)


# ── SRT parsing ──────────────────────────────────────────────────────


def parse_timestamp_ms(value: str) -> int:
    match = TIMESTAMP_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    hh, mm, ss, ms = map(int, match.groups())
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


def parse_srt(path: Path) -> List[Cue]:
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", content.strip())
    cues: List[Cue] = []
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0])
        except ValueError:
            continue
        if "-->" not in lines[1]:
            continue
        start_raw, end_raw = [s.strip() for s in lines[1].split("-->", 1)]
        start_ms = parse_timestamp_ms(start_raw)
        end_ms = parse_timestamp_ms(end_raw)
        text = "\n".join(lines[2:]).strip()
        if text:
            cues.append(Cue(index=idx, start_ms=start_ms, end_ms=end_ms, text=text))
    if not cues:
        raise ValueError("No valid cues parsed from SRT.")
    return cues


# ── Voice map resolution ─────────────────────────────────────────────


def parse_segment_key(key: str) -> Tuple[int, int]:
    key = key.strip()
    if "-" in key:
        left, right = key.split("-", 1)
        return int(left), int(right)
    v = int(key)
    return v, v


def resolve_segment_cfg(index: int, config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config.get("default", {}))
    for key, seg_cfg in config.get("segments", {}).items():
        lo, hi = parse_segment_key(key)
        if lo <= index <= hi:
            merged.update(seg_cfg)
    return merged


# ── ffmpeg helpers ────────────────────────────────────────────────────


def _run_ff(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH.")


def probe_duration_ms(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr}")
    return float(proc.stdout.strip()) * 1000


def normalize_duration_pad_trim(inp: Path, outp: Path, target_ms: int) -> None:
    """Pad short audio then trim to exact target duration (Noiz backend)."""
    sec = target_ms / 1000.0
    _run_ff([
        "ffmpeg", "-y", "-i", str(inp),
        "-af", f"apad=pad_dur={sec:.3f}",
        "-t", f"{sec:.3f}", str(outp),
    ])


def normalize_duration_atempo(inp: Path, outp: Path, target_ms: int) -> None:
    """Use atempo to stretch/compress audio to target duration (Kokoro backend)."""
    actual_ms = probe_duration_ms(inp)
    if actual_ms <= 0:
        normalize_duration_pad_trim(inp, outp, target_ms)
        return

    ratio = actual_ms / target_ms
    # atempo accepts 0.5–100.0; chain filters for extreme ratios
    filters = []
    r = ratio
    while r > 100.0:
        filters.append("atempo=100.0")
        r /= 100.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r:.6f}")

    _run_ff([
        "ffmpeg", "-y", "-i", str(inp),
        "-af", ",".join(filters),
        "-t", f"{target_ms / 1000.0:.3f}", str(outp),
    ])


def delay_segment(inp: Path, outp: Path, start_ms: int) -> None:
    _run_ff([
        "ffmpeg", "-y", "-i", str(inp),
        "-af", f"adelay={start_ms}:all=1", str(outp),
    ])


def mix_all(inputs: List[Path], outp: Path, total_ms: int) -> None:
    if not inputs:
        raise ValueError("No segments to mix.")
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]
    cmd += [
        "-filter_complex",
        f"amix=inputs={len(inputs)}:duration=longest:dropout_transition=0",
        "-t", f"{total_ms / 1000.0:.3f}", str(outp),
    ]
    _run_ff(cmd)


# ── Noiz backend ─────────────────────────────────────────────────────


def _noiz_emotion_enhance(
    base_url: str, api_key: str, text: str, timeout: int
) -> str:
    import requests  # noqa: delayed import so kokoro path doesn't need requests

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


def _bool_form(v: Any) -> str:
    return "true" if bool(v) else "false"


def _resolve_reference_audio(ref: str, timeout: int) -> Tuple[Path, Optional[Path]]:
    """Resolve reference_audio to a path. If ref is a URL, download to temp file.
    Returns (path_to_use, temp_path_to_cleanup_or_None)."""
    if ref.startswith("http://") or ref.startswith("https://"):
        import requests
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        r = requests.get(ref, timeout=timeout)
        r.raise_for_status()
        Path(tmp.name).write_bytes(r.content)
        return Path(tmp.name), Path(tmp.name)
    p = Path(ref)
    if not p.exists():
        raise FileNotFoundError(f"reference_audio not found: {ref}")
    return p, None


def _noiz_tts(
    base_url: str,
    api_key: str,
    cue: Cue,
    cfg: Dict[str, Any],
    output_format: str,
    timeout: int,
    out_path: Path,
) -> float:
    import requests

    url = f"{base_url.rstrip('/')}/text-to-speech"
    payload: Dict[str, str] = {
        "text": cue.text,
        "duration": f"{cue.duration_ms / 1000.0:.3f}",
        "output_format": output_format,
    }
    for field in ("voice_id", "quality_preset", "speed", "target_lang"):
        if field in cfg and cfg[field] is not None:
            payload[field] = str(cfg[field])
    if "similarity_enh" in cfg:
        payload["similarity_enh"] = _bool_form(cfg["similarity_enh"])
    if "save_voice" in cfg:
        payload["save_voice"] = _bool_form(cfg["save_voice"])
    if "emo" in cfg and cfg["emo"] is not None:
        emo = cfg["emo"]
        payload["emo"] = emo if isinstance(emo, str) else json.dumps(emo)

    files = None
    ref_cleanup: Optional[Path] = None
    ref = cfg.get("reference_audio")
    if ref:
        ref_path, ref_cleanup = _resolve_reference_audio(ref, timeout)
        files = {
            "file": (
                ref_path.name,
                ref_path.open("rb"),
                "application/octet-stream",
            )
        }
    elif not cfg.get("voice_id"):
        raise ValueError(
            f"Cue {cue.index}: either voice_id or reference_audio required."
        )

    try:
        resp = requests.post(
            url, headers={"Authorization": api_key},
            data=payload, files=files, timeout=timeout,
        )
    finally:
        if files and files["file"][1]:
            files["file"][1].close()
        if ref_cleanup is not None:
            ref_cleanup.unlink(missing_ok=True)

    if resp.status_code != 200:
        raise RuntimeError(
            f"/text-to-speech cue {cue.index}: "
            f"status={resp.status_code}, body={resp.text}"
        )
    out_path.write_bytes(resp.content)
    dur_h = resp.headers.get("X-Audio-Duration")
    return float(dur_h) if dur_h else -1.0


# ── Kokoro backend ───────────────────────────────────────────────────


def _ensure_kokoro() -> None:
    if not shutil.which("kokoro-tts"):
        raise RuntimeError("kokoro-tts CLI not found.")


def _kokoro_tts(
    cue: Cue,
    cfg: Dict[str, Any],
    output_format: str,
    out_path: Path,
) -> float:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(cue.text)
        tmp_path = tmp.name

    try:
        cmd = ["kokoro-tts", tmp_path, str(out_path)]
        voice = cfg.get("voice")
        if voice:
            cmd += ["--voice", str(voice)]
        lang = cfg.get("lang")
        if lang:
            cmd += ["--lang", str(lang)]
        speed = cfg.get("speed")
        if speed is not None:
            cmd += ["--speed", str(speed)]
        cmd += ["--format", output_format]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"kokoro-tts failed for cue {cue.index}: {proc.stderr}"
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if out_path.exists():
        return probe_duration_ms(out_path) / 1000.0
    raise RuntimeError(f"kokoro-tts produced no output for cue {cue.index}")


# ── main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render timeline-accurate speech from SRT."
    )
    ap.add_argument("--srt", required=True, help="Input SRT file")
    ap.add_argument("--voice-map", required=True, help="Voice-map JSON")
    ap.add_argument(
        "--backend", choices=["kokoro", "noiz"], default="kokoro",
        help="TTS backend (default: kokoro)",
    )
    ap.add_argument("--api-key", help="API key (required for noiz backend)")
    ap.add_argument("--output", required=True, help="Output audio file")
    ap.add_argument("--base-url", default="https://noiz.ai/v1")
    ap.add_argument("--work-dir", default=".tmp/tts")
    ap.add_argument("--auto-emotion", action="store_true",
                     help="Noiz backend only: call /emotion-enhance before TTS")
    ap.add_argument("--ref-audio-track", help="Original audio track to dynamically slice as reference audio per segment")
    ap.add_argument("--output-format", choices=["wav", "mp3"], default="wav")
    ap.add_argument("--timeout-sec", type=int, default=120)
    args = ap.parse_args()

    if args.backend == "noiz" and not args.api_key:
        print("Error: --api-key is required for noiz backend.", file=sys.stderr)
        return 1
    if args.api_key:
        args.api_key = normalize_api_key_base64(args.api_key)

    try:
        ensure_ffmpeg()
        if args.backend == "kokoro":
            _ensure_kokoro()

        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        cues = parse_srt(Path(args.srt))
        voice_map = json.loads(Path(args.voice_map).read_text(encoding="utf-8"))

        delayed: List[Path] = []
        report: List[Dict[str, Any]] = []

        for cue in cues:
            cfg = resolve_segment_cfg(cue.index, voice_map)
            
            if args.ref_audio_track and not cfg.get("voice_id") and not cfg.get("reference_audio"):
                ref_slice_path = work / f"seg_{cue.index:04d}_ref.wav"
                if not ref_slice_path.exists():
                    _run_ff([
                        "ffmpeg", "-y",
                        "-ss", f"{cue.start_ms / 1000.0:.3f}",
                        "-i", str(args.ref_audio_track),
                        "-t", f"{cue.duration_ms / 1000.0:.3f}",
                        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                        str(ref_slice_path)
                    ])
                cfg["reference_audio"] = str(ref_slice_path)
            
            text = cue.text

            if args.backend == "noiz" and args.auto_emotion:
                text = _noiz_emotion_enhance(
                    args.base_url, args.api_key, cue.text, args.timeout_sec
                )

            synth_cue = Cue(cue.index, cue.start_ms, cue.end_ms, text)
            raw = work / f"seg_{cue.index:04d}_raw.{args.output_format}"
            norm = work / f"seg_{cue.index:04d}_norm.wav"
            dly = work / f"seg_{cue.index:04d}_delay.wav"

            if args.backend == "noiz":
                api_dur = _noiz_tts(
                    args.base_url, args.api_key, synth_cue,
                    cfg, args.output_format, args.timeout_sec, raw,
                )
                normalize_duration_pad_trim(raw, norm, cue.duration_ms)
            else:
                api_dur = _kokoro_tts(synth_cue, cfg, args.output_format, raw)
                normalize_duration_atempo(raw, norm, cue.duration_ms)

            delay_segment(norm, dly, cue.start_ms)
            delayed.append(dly)

            seg_report: Dict[str, Any] = {
                "index": cue.index,
                "start_ms": cue.start_ms,
                "end_ms": cue.end_ms,
                "duration_ms": cue.duration_ms,
                "raw_duration_sec": api_dur,
                "backend": args.backend,
            }
            if args.backend == "noiz":
                seg_report["voice_id"] = cfg.get("voice_id")
                seg_report["reference_audio"] = cfg.get("reference_audio")
                seg_report["emo"] = cfg.get("emo")
            else:
                seg_report["voice"] = cfg.get("voice")
                seg_report["lang"] = cfg.get("lang")
            report.append(seg_report)

        timeline_wav = work / "timeline.wav"
        total_ms = max(c.end_ms for c in cues)
        mix_all(delayed, timeline_wav, total_ms)

        out = Path(args.output)
        if out.suffix.lower() != ".wav":
            _run_ff(["ffmpeg", "-y", "-i", str(timeline_wav), str(out)])
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(timeline_wav.read_bytes())

        report_path = work / "render_report.json"
        report_path.write_text(
            json.dumps({
                "srt": args.srt,
                "output": args.output,
                "backend": args.backend,
                "total_ms": total_ms,
                "segments": report,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Done. Output: {out}")
        print(f"Report: {report_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
