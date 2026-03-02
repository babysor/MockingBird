---
name: MockingBird
description: AI voice cloning and TTS toolkit. The only active skill is `speak`.
---

# MockingBird

Voice cloning / text-to-speech toolkit built on encoder-synthesizer-vocoder pipeline.

## Available Skills

| Skill | Path | Purpose |
|-------|------|---------|
| speak | `skills/speak/SKILL.md` | Text-to-speech (Kokoro local / Noiz cloud), timeline rendering, voice cloning |

## Project Layout

- `models/` — encoder, synthesizer, vocoder, ppg2mel, ppg_extractor
- `control/` — CLI and GUI entry points (toolbox, mkgui)
- `skills/speak/` — TTS skill scripts (`tts.sh`, `noiz_tts.py`, `render_timeline.py`, `text_to_srt.py`)
- `data/ckpt/` — pretrained checkpoints

## When to Use

If the user asks for anything related to speech synthesis, TTS, voice cloning, dubbing, or audio generation, read and follow `skills/speak/SKILL.md`.
