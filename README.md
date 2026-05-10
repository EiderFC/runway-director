# Runway Director — Agentic Cinema Studio

Hackathon entry for **Runway 2026 API Hackathon**.

> One sentence in. A directed cinematic sequence out.

GPT-4o writes an editable storyboard, locks the character identity across shots with an auto-generated reference portrait, Runway `seedance2` renders each chained clip, OpenAI TTS-1-HD adds a voice-over auto-trimmed to video length, and `ffmpeg` stream-copies the master with no audio re-encoding. You can re-roll any shot, A/B compare takes, generate a 12-second social trailer with GPT-edited cuts, and export the whole project as a `.zip` bundle.

## Why it's agentic, not just an API call
- **Two-stage flow** — the agent plans the storyboard, you review/edit, then it renders. The agent reasons before it spends credits.
- **Character lock** — gpt-image-1 generates a reference portrait from the AI Director's `character_bible`, embedded as a data-URI reference for every shot.
- **Self-healing prompts** — multi-phase retry: original → drop refs → rewritten → neutralized, with explicit content-block detection.
- **Trailer editor** — GPT-4o receives the rendered storyboard and returns an edit decision list (timestamps, speeds, hook, title card). `ffmpeg` assembles a 12s social cut.

## Highlights
- 🎬 Live storyboard reveal (shots appear progressively as the AI Director "writes")
- 🔒 Character Lock with auto-generated portrait reference
- 🎚 A/B takes per shot (re-roll keeps every version)
- 🎞 Master/Trailer toggle, Theater Mode, ZIP bundle export
- 🖱 Drag & drop reorder, add/delete/duplicate shots, inline-editable text
- 📤 Storyboard JSON import/export, reference-image file upload
- ⚡ ffmpeg stream-copy merge — zero audio re-encoding
- ⌨ Keyboard shortcuts (`⌘+Enter` plan/render, `T` theater, `ESC` exit)

## Run locally

**Requirements**: Python **3.10, 3.11, or 3.12** (recommended: 3.12).
Python 3.13+ is not yet recommended because Pillow 10 does not ship Windows wheels for it.

```bash
# 1. Use Python 3.12 if you have it:
py -3.12 -m venv venv          # Windows
python3.12 -m venv venv        # macOS / Linux

# 2. Activate the venv:
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install and run:
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. Paste your OpenAI key + Runway secret in the top right (cached in localStorage).

> **Don't have Python 3.12?** Download it from <https://www.python.org/downloads/release/python-3127/>.
> During install, check **"Add python.exe to PATH"**.

ffmpeg is bundled via `imageio-ffmpeg` — no system install needed.

## Tech stack
- **Backend** — Flask, OpenAI SDK (gpt-4o, gpt-image-1, tts-1-hd), Runway SDK (`seedance2`)
- **Post-production** — ffmpeg (stream-copy concat, atempo+setpts speed, drawtext title cards) with moviepy fallback
- **Frontend** — Vanilla JS + CSS, no framework. Polling-based live state.

## API endpoints
| Endpoint | Purpose |
|---|---|
| `POST /plan` | AI Director generates editable storyboard |
| `POST /update_storyboard` | Save user edits before rendering |
| `POST /add_shot` · `/delete_shot` · `/duplicate_shot` · `/reorder_shots` | Storyboard editor |
| `POST /render` | Run video generation + voice-over + master merge |
| `POST /reroll` | Re-render a specific shot (preserves continuity) |
| `POST /select_take` | Switch active take in A/B comparison |
| `POST /remerge` | Re-stitch master without re-rendering |
| `POST /regen_voiceover` | New TTS narration over existing master |
| `POST /generate_trailer` | GPT-edited 12s social trailer from existing shots |
| `POST /upload_image` | Upload reference image (auto-compressed to data URI) |
| `GET  /export_storyboard` · `POST /import_storyboard` | Project portability |
| `GET  /bundle.zip` | Download master + clips + script + VO + portrait |
| `GET  /get_status` | Live project state for the UI |
