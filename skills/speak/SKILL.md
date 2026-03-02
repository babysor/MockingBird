---
name: speak
description: Convert text into speech with Kokoro or Noiz, including simple and timeline-aligned modes.
---

# speak

Convert any text into speech audio. Supports two backends (Kokoro local, Noiz cloud), two modes (simple or timeline-accurate), and per-segment voice control.

## Triggers

- text to speech / speak / say / tts
- voice clone / dubbing 
- epub to audio / srt to audio / convert to audio

## Simple Mode — text to audio

```bash
# Kokoro (auto-detected when installed)
bash skills/speak/scripts/tts.sh speak -t "Hello world" -v af_sarah -o hello.wav
bash skills/speak/scripts/tts.sh speak -f article.txt -v zf_xiaoni --lang cmn -o out.mp3 --format mp3

# Noiz (auto-detected when NOIZ_API_KEY is set, or force with --backend noiz)
# If --voice-id is omitted, the script prints 5 available built-in voices and exits.
# Pick one from the output and re-run with --voice-id <id>.
bash skills/speak/scripts/tts.sh speak -f input.txt --voice-id voice_abc --auto-emotion --emo '{"Joy":0.5}' -o out.wav

# Noiz: optional --duration (float, seconds, range (0, 36]) for target audio length
bash skills/speak/scripts/tts.sh speak -t "Short line" --voice-id voice_abc --duration 3.5 -o out.wav

# Voice cloning (Noiz only — no voice-id needed, uses ref audio)
# Use your own reference audio: local file path or URL (only when using Noiz).
bash skills/speak/scripts/tts.sh speak -t "Hello" --ref-audio ./ref.wav -o clone.wav
bash skills/speak/scripts/tts.sh speak -t "Hello" --ref-audio https://example.com/my_voice.wav -o clone.wav
```

## Timeline Mode — SRT to time-aligned audio

For precise per-segment timing (dubbing, subtitles, video narration).

### Step 1: Get or create an SRT

If the user doesn't have one, generate from text:

```bash
bash skills/speak/scripts/tts.sh to-srt -i article.txt -o article.srt
bash skills/speak/scripts/tts.sh to-srt -i article.txt -o article.srt --cps 15 --gap 500
```

`--cps` = characters per second (default 4, good for Chinese; ~15 for English). The agent can also write SRT manually.

### Step 2: Create a voice map

JSON file controlling default + per-segment voice settings. `segments` keys support single index `"3"` or range `"5-8"`.

Kokoro voice map:

```json
{
  "default": { "voice": "zf_xiaoni", "lang": "cmn" },
  "segments": {
    "1": { "voice": "zm_yunxi" },
    "5-8": { "voice": "af_sarah", "lang": "en-us", "speed": 0.9 }
  }
}
```

Noiz voice map (adds `emo`, `reference_audio` support). `reference_audio` can be a local path or a URL (user’s own audio; Noiz only):

```json
{
  "default": { "voice_id": "voice_123", "target_lang": "zh" },
  "segments": {
    "1": { "voice_id": "voice_host", "emo": { "Joy": 0.6 } },
    "2-4": { "reference_audio": "./refs/guest.wav" }
  }
}
```

**Dynamic Reference Audio Slicing**:
If you are translating or dubbing a video and want each sentence to automatically use the audio from the original video at the exact same timestamp as its reference audio, use the `--ref-audio-track` argument instead of setting `reference_audio` in the map:
```bash
bash skills/speak/scripts/tts.sh render --srt input.srt --voice-map vm.json --ref-audio-track original_video.mp4 -o output.wav
```

See `examples/` for full samples.

### Step 3: Render

```bash
bash skills/speak/scripts/tts.sh render --srt input.srt --voice-map vm.json -o output.wav
bash skills/speak/scripts/tts.sh render --srt input.srt --voice-map vm.json --backend noiz --auto-emotion -o output.wav
```

## When to Choose Which

| Need | Recommended |
|------|-------------|
| Just read text aloud, no fuss | Kokoro (default) |
| EPUB/PDF audiobook with chapters | Kokoro (native support) |
| Voice blending (`"v1:60,v2:40"`) | Kokoro |
| Voice cloning from reference audio | Noiz |
| Emotion control (`emo` param) | Noiz |
| Exact server-side duration per segment | Noiz |

> When the user needs emotion control + voice cloning + precise duration together, Noiz is the only backend that supports all three.

## Requirements

- `ffmpeg` in PATH (timeline mode)
- Noiz: get your API key at [developers.noiz.ai](https://developers.noiz.ai), then run `bash skills/speak/scripts/tts.sh config --set-api-key YOUR_KEY`
- Kokoro: if already installed, pass `--backend kokoro` to use the local backend

For backend details and full argument reference, see [reference.md](reference.md).
