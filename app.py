import os
import time
import threading
import json
import base64
import io
import shutil
import subprocess
import zipfile
import uuid
import requests
try:
    from imageio_ffmpeg import get_ffmpeg_exe as _get_ffmpeg
    FFMPEG = _get_ffmpeg()
except Exception:
    FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
import concurrent.futures
import re
import traceback

# --- Pillow 10+ compatibility shim for moviepy 1.0.3 ---
try:
    import PIL.Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS
except Exception:
    pass

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response
from openai import OpenAI
from runwayml import RunwayML
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
import moviepy.video.fx.all as vfx


app = Flask(__name__)

if not os.path.exists("static"):
    os.makedirs("static")


STATE = {
    "is_running": False,
    "phase": "idle",
    "status_msg": "System Idle. Ready for input.",
    "current_shot": 0,
    "total_shots": 0,
    "shots": [],
    "logs": [],
    "character_bible": "",
    "character_portrait_url": None,
    "final_video_url": None,
    "voiceover_url": None,
    "trailer_url": None,
    "started_at": None,
    "finished_at": None,
    "config": {},
    "credits_estimate": 0,
}

STATE_LOCK = threading.RLock()


def wlog(msg, msg_type="normal"):
    try:
        print(msg, flush=True)
    except Exception:
        pass
    clean_msg = re.sub(r'\033\[[0-9;]*m', '', msg).strip()
    if clean_msg:
        with STATE_LOCK:
            STATE["logs"].append({"text": clean_msg, "type": msg_type, "t": time.time()})


def set_status(msg, phase=None):
    with STATE_LOCK:
        STATE["status_msg"] = msg
        if phase:
            STATE["phase"] = phase


def reset_state(keep_keys=False):
    with STATE_LOCK:
        STATE["is_running"] = False
        STATE["phase"] = "idle"
        STATE["status_msg"] = "Ready."
        STATE["current_shot"] = 0
        STATE["total_shots"] = 0
        STATE["shots"] = []
        STATE["logs"] = []
        STATE["character_bible"] = ""
        STATE["character_portrait_url"] = None
        STATE["final_video_url"] = None
        STATE["voiceover_url"] = None
        STATE["started_at"] = None
        STATE["finished_at"] = None
        STATE["credits_estimate"] = 0
        if not keep_keys:
            STATE["config"] = {}


def estimate_credits(shots):
    # Approx: ~5 credits per second for seedance2 video
    return sum(int(s.get("duration", 5)) * 5 for s in shots)


# ---------- AI ORCHESTRATION ----------

def build_director_prompt(topic, total_duration, pacing):
    return f"""
You are an elite, award-winning film director and visual storyteller writing for an AI video model.

Brief: '{topic}'
Total duration: {total_duration} seconds.
Pacing: {pacing}.

CRITICAL RULES:
1. Maintain EXACT SAME character, clothing, and environment across all shots — describe a strict, repeatable subject in 'character_bible'.
2. Keep actions physically plausible, safe, and free of violence, weapons, blood, or destruction.
3. MATHEMATICAL REQUIREMENT: split the sequence into shots of 5 to 10 seconds. The SUM of all "duration" values MUST be EXACTLY {total_duration}.
4. Each "action" MUST be 1–2 vivid sentences, present-tense, focused on visible motion and concrete imagery (not metaphors).
5. Give every shot a short, evocative "title" (3–5 words).

Return ONLY a JSON object:
{{
  "title": "short evocative project title",
  "logline": "one-sentence summary of the sequence",
  "character_bible": "strict 1–2 sentence description of the recurring subject",
  "shots": [
    {{ "title": "...", "duration": 5, "action": "..." }}
  ]
}}
""".strip()


def director_generate(openai_client, topic, total_duration, pacing):
    chat = openai_client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": build_director_prompt(topic, total_duration, pacing)}],
    )
    return json.loads(chat.choices[0].message.content)


def _extract_complete_shots(buffer):
    """Greedy extract complete {...} shot objects from a partial JSON buffer."""
    shots = []
    # find shots array opening
    arr_start = buffer.find('"shots"')
    if arr_start == -1:
        return shots, None, None
    bracket_start = buffer.find("[", arr_start)
    if bracket_start == -1:
        return shots, None, None
    i = bracket_start + 1
    depth = 0
    obj_start = None
    while i < len(buffer):
        c = buffer[i]
        if c == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                snippet = buffer[obj_start:i+1]
                try:
                    shots.append(json.loads(snippet))
                except Exception:
                    pass
                obj_start = None
        elif c == "]" and depth == 0:
            break
        i += 1
    # try to extract title / character_bible from header
    title = None; bible = None
    m = re.search(r'"character_bible"\s*:\s*"([^"]*)"', buffer)
    if m: bible = m.group(1)
    m = re.search(r'"title"\s*:\s*"([^"]*)"', buffer)
    if m: title = m.group(1)
    return shots, title, bible


def director_generate_streaming(openai_client, topic, total_duration, pacing, on_shot=None, on_meta=None):
    """Plan storyboard then emit progressively for a 'live' visual effect.

    Note: we used to truly stream tokens here, but `response_format=json_object`
    can cause GPT-4o to buffer the whole JSON before emitting, which made the UI
    appear frozen. We now plan in one call and progressively yield shots.
    """
    plan = director_generate(openai_client, topic, total_duration, pacing)
    title = plan.get("title", "Untitled")
    bible = plan.get("character_bible", "")
    shots = plan.get("shots", [])
    if on_meta:
        try: on_meta(title, bible)
        except Exception: pass
    if on_shot:
        for s in shots:
            try: on_shot(s)
            except Exception: pass
            time.sleep(0.35)  # progressive reveal — feels alive without being slow
    return plan


def rewrite_prompt(openai_client, original_prompt):
    wlog("   [!] Triggering AI Script Doctor (safety adaptation)...", "warning")
    instruction = (
        "The following video generation prompt was REJECTED by safety filters:\n"
        f"\"{original_prompt}\"\n"
        "Rewrite it to be EXTREMELY safe, slow, calm, simple. Remove any chaos, rapid movement, "
        "violence, destruction, or risky object interactions. Return ONLY the new prompt text."
    )
    try:
        chat = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": instruction}],
        )
        new_prompt = chat.choices[0].message.content.strip()
        wlog(f"   [!] Prompt mutated: {new_prompt[:80]}...", "success")
        return new_prompt
    except Exception:
        return original_prompt


RATIO_HINTS = {
    "1470:630":  "Ultrawide 21:9 cinematic aspect ratio, anamorphic widescreen composition.",
    "1280:720":  "Standard 16:9 landscape framing.",
    "720:1280":  "Vertical 9:16 mobile composition.",
    "960:960":   "Square 1:1 centered composition.",
}


def compose_shot_prompt(character_bible, action, camera, art_style, motion, audio_prompt, ratio="1280:720"):
    ratio_hint = RATIO_HINTS.get(ratio, "")
    return (
        f"The video is fullscreen and uncropped. {ratio_hint} Subject: {character_bible}. "
        f"Action: {action} Camera: {camera}. {art_style}. Motion: {motion}. {audio_prompt}"
    )


def _try_runway(runway_client, prompt_text, ratio, duration, last_video_url, image_refs):
    extra_body = {}
    if last_video_url:
        extra_body["referenceVideos"] = [{"type": "video", "uri": last_video_url}]
    if image_refs:
        extra_body["references"] = image_refs
    task = runway_client.text_to_video.create(
        model="seedance2",
        prompt_text=(
            f"Seamlessly continue the scene from the reference video. {prompt_text}"
            if last_video_url else prompt_text
        ),
        ratio=ratio, duration=duration, extra_body=extra_body,
    )
    completed = task.wait_for_task_output()
    output_data = completed.output
    return output_data[0] if isinstance(output_data, list) else output_data


def _looks_like_content_block(err_str):
    s = (err_str or "").lower()
    return any(k in s for k in [
        "moderation", "safety", "policy", "nsfw", "violation", "blocked",
        "inappropriate", "sensitive", "content guidelines",
    ])


def runway_render_shot(runway_client, prompt_text, ratio, duration, image_refs, last_video_url, openai_client):
    """Render a single shot with explicit retry strategy.

    Strategy (in order, each step is one or more attempts):
      A. original prompt, with refs        — 2 attempts (network/transient)
      B. original prompt, no refs          — 1 attempt (refs may be unsupported)
      C. rewritten prompt, no refs         — 2 attempts
      D. heavily-rewritten prompt, no refs — 1 attempt
    """
    last_err = ""

    def attempt(p, refs):
        nonlocal last_err
        try:
            return _try_runway(runway_client, p, ratio, duration, last_video_url, refs)
        except Exception as e:
            last_err = str(e)
            return None

    # Phase A
    for i in range(2):
        wlog(f"   [→] Attempt A{i+1}: original prompt + refs", "normal")
        url = attempt(prompt_text, image_refs)
        if url: return url, prompt_text
        wlog(f"   [x] {last_err[:160]}", "error")
        time.sleep(3 * (i + 1))

    # Phase B
    if image_refs:
        wlog("   [→] Attempt B: dropping image reference", "warning")
        url = attempt(prompt_text, None)
        if url: return url, prompt_text
        wlog(f"   [x] {last_err[:160]}", "error")

    # Phase C — rewritten by GPT
    rewritten = rewrite_prompt(openai_client, prompt_text)
    for i in range(2):
        wlog(f"   [→] Attempt C{i+1}: rewritten prompt, no refs", "warning")
        url = attempt(rewritten, None)
        if url: return url, rewritten
        wlog(f"   [x] {last_err[:160]}", "error")
        time.sleep(3)

    # Phase D — last resort: very generic neutral version
    final = rewrite_prompt(openai_client, rewritten)
    wlog("   [→] Attempt D: neutralized prompt, no refs", "warning")
    url = attempt(final, None)
    if url: return url, final
    wlog(f"   [x] {last_err[:200]}", "error")

    if _looks_like_content_block(last_err):
        raise RuntimeError(
            "Runway rejected this shot for content policy reasons even after rewrites. "
            "Try softening your story concept (avoid nudity, sexual themes, violence, weapons, "
            "or copyrighted characters)."
        )
    raise RuntimeError(f"Shot generation failed: {last_err[:200] or 'unknown error'}")


# ---------- POST PRODUCTION ----------

def download_clip(url, index):
    wlog(f"   ⬇ Downloading chunk {index}...", "normal")
    response = requests.get(url, timeout=120)
    filename = f"static/temp_shot_{index}.mp4"
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename


def _ffmpeg_run(args):
    """Run ffmpeg with hidden stderr unless it fails."""
    try:
        result = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", *args],
            capture_output=True, text=True, check=True,
        )
        return True, result.stderr
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or "")[-500:]
    except FileNotFoundError as e:
        return False, str(e)


def _merge_via_ffmpeg_streamcopy(local_files, voiceover_path, output_path):
    """High-fidelity merge using ffmpeg: stream-copy video, only re-encode audio if mixing VO.

    Returns True on success, False if a fallback (re-encode) is needed.
    """
    list_path = "static/_concat_list.txt"
    abs_files = [os.path.abspath(p).replace("\\", "/") for p in local_files]
    with open(list_path, "w", encoding="utf-8") as f:
        for p in abs_files:
            f.write(f"file '{p}'\n")

    if not voiceover_path or not os.path.exists(voiceover_path):
        # Pure stream copy — zero re-encode
        ok, err = _ffmpeg_run([
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", "-movflags", "+faststart", output_path,
        ])
        if not ok:
            wlog(f"   ⚠ ffmpeg stream-copy failed: {err[-200:]}", "warning")
        try: os.remove(list_path)
        except Exception: pass
        return ok

    # With voice-over: concat (video stream copy) into temp, then mux+mix audio.
    temp_concat = "static/_concat_tmp.mp4"
    ok, err = _ffmpeg_run([
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", "-movflags", "+faststart", temp_concat,
    ])
    if not ok:
        wlog(f"   ⚠ ffmpeg concat failed: {err[-200:]}", "warning")
        try: os.remove(list_path)
        except Exception: pass
        return False

    # Probe duration of concat result
    probe = subprocess.run(
        [FFMPEG, "-i", temp_concat], capture_output=True, text=True,
    )
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", probe.stderr or "")
    video_dur = None
    if dur_match:
        h, m, s = dur_match.groups()
        video_dur = int(h) * 3600 + int(m) * 60 + float(s)

    # Mix: bg ducked at 0.4, VO at 1.1, force final duration to video length, video stream-copied
    filter_chain = "[0:a]volume=0.4[a1];[1:a]volume=1.1[a2];[a1][a2]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    args = [
        "-i", temp_concat, "-i", voiceover_path,
        "-filter_complex", filter_chain,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
    ]
    if video_dur:
        args += ["-t", f"{video_dur:.3f}"]
    args += ["-movflags", "+faststart", output_path]

    ok, err = _ffmpeg_run(args)
    if not ok:
        wlog(f"   ⚠ ffmpeg mux+mix failed: {err[-200:]}", "warning")

    for tmp in (list_path, temp_concat):
        try: os.remove(tmp)
        except Exception: pass
    return ok


def merge_videos_cinematic(urls, voiceover_path=None, output_path="static/Final_Master_Sequence.mp4"):
    wlog("\n🎞 POST-PRODUCTION: cinematic merge...", "highlight")
    local_files = []
    clips = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as ex:
            futures = [ex.submit(download_clip, url, i) for i, url in enumerate(urls)]
            for fut in concurrent.futures.as_completed(futures):
                local_files.append(fut.result())
        local_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        # FAST PATH: ffmpeg stream-copy (no audio degradation)
        wlog("   ⚡ Stream-copy merge via ffmpeg (no re-encode)...", "normal")
        if _merge_via_ffmpeg_streamcopy(local_files, voiceover_path, output_path):
            for f in local_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except Exception: pass
            wlog("✅ POST-PRODUCTION COMPLETE (stream copy)", "success")
            return "/" + output_path
        wlog("   ↳ Falling back to moviepy re-encode path...", "warning")

        wlog("   🎥 Applying fades & normalizing aspect ratio...", "normal")
        for f in local_files:
            clips.append(VideoFileClip(f))
        base_size = clips[0].size
        resized = [c.resize(newsize=base_size) for c in clips]
        if resized:
            resized[0] = resized[0].fx(vfx.fadein, 0.6)
            resized[-1] = resized[-1].fx(vfx.fadeout, 1.2)

        final_clip = concatenate_videoclips(resized, method="compose")

        if voiceover_path and os.path.exists(voiceover_path):
            wlog("   🎙 Mixing voice-over track...", "normal")
            try:
                video_duration = final_clip.duration
                vo_full = AudioFileClip(voiceover_path)
                vo_dur = vo_full.duration
                if vo_dur > video_duration:
                    wlog(f"   ✂ Trimming voice-over: {vo_dur:.1f}s → {video_duration:.1f}s (matches video).", "warning")
                    vo = vo_full.subclip(0, video_duration).volumex(1.1)
                else:
                    vo = vo_full.volumex(1.1)
                if final_clip.audio is not None:
                    bg = final_clip.audio.volumex(0.4).subclip(0, video_duration)
                    mixed = CompositeAudioClip([bg, vo]).set_duration(video_duration)
                else:
                    mixed = vo.set_duration(video_duration)
                final_clip = final_clip.set_audio(mixed).set_duration(video_duration)
            except Exception as e:
                wlog(f"   ⚠ Voice-over mix failed: {e}", "warning")

        wlog("   ⚙ Encoding H.264 master (high-quality fallback)...", "normal")
        final_clip.write_videofile(
            output_path, codec="libx264", audio_codec="aac", fps=24,
            verbose=False, logger=None, threads=4, preset="slow",
            bitrate="9000k", audio_bitrate="320k",
            ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )

        for c in clips:
            c.close()
        for c in resized:
            c.close()
        for f in local_files:
            if os.path.exists(f):
                os.remove(f)

        wlog("✅ POST-PRODUCTION COMPLETE", "success")
        return "/" + output_path
    except Exception as e:
        wlog(f"❌ Post-production error: {e}", "error")
        return None


def _save_compressed_portrait(raw_bytes, out_path="static/character_portrait.jpg"):
    """Re-encode portrait to a compact JPEG to fit reference-image size limits."""
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        # cap longest side at 768px to keep base64 well under 1MB
        max_side = 768
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(out_path, format="JPEG", quality=82, optimize=True)
        return out_path
    except Exception as e:
        wlog(f"   ⚠ Portrait compression failed ({e}); saving raw.", "warning")
        out_path = out_path.rsplit(".", 1)[0] + ".png"
        with open(out_path, "wb") as f:
            f.write(raw_bytes)
        return out_path


def generate_character_portrait(openai_client, character_bible, ratio="1280:720"):
    """Generate a reference portrait of the main subject for character lock."""
    try:
        wlog("   🎨 Generating character portrait for lock...", "normal")
        if ratio.startswith("720:1280"):
            size = "1024x1792"
        elif ratio.startswith("960:960"):
            size = "1024x1024"
        else:
            size = "1792x1024"
        prompt = (
            f"Studio-quality cinematic character reference portrait, full body, neutral background. "
            f"{character_bible}. Sharp focus, soft cinematic lighting, hyper-detailed, photoreal."
        )
        raw = None
        try:
            res = openai_client.images.generate(model="gpt-image-1", prompt=prompt, size=size, n=1)
            data = res.data[0]
            if getattr(data, "b64_json", None):
                raw = base64.b64decode(data.b64_json)
            elif getattr(data, "url", None):
                raw = requests.get(data.url, timeout=60).content
        except Exception as e1:
            wlog(f"   ⚠ gpt-image-1 unavailable ({e1}); trying dall-e-3...", "warning")
            res = openai_client.images.generate(model="dall-e-3", prompt=prompt, size=size, n=1, quality="hd")
            raw = requests.get(res.data[0].url, timeout=60).content

        if raw:
            out_path = _save_compressed_portrait(raw)
            wlog("   🎨 Character portrait locked.", "success")
            return "/" + out_path.replace("\\", "/")
    except Exception as e:
        wlog(f"   ⚠ Portrait generation failed: {e}", "warning")
    return None


def write_narration_script(openai_client, shot_actions, total_duration):
    """Use GPT to compress shot actions into a tight narration matching video duration."""
    # Target ~2 words/second (slightly slower than typical 2.5 wps for breathing room)
    target_words = max(8, int(total_duration * 2))
    instruction = (
        f"You are a film narrator writing voice-over for a {total_duration}-second cinematic sequence.\n\n"
        f"Shots:\n" + "\n".join(f"- {a}" for a in shot_actions) + "\n\n"
        f"Write a SINGLE flowing narration in English of AT MOST {target_words} words "
        f"(strict — count carefully). Evocative, present tense, poetic but punchy. "
        f"No shot labels. No 'shot 1', no markdown. Just the narration paragraph."
    )
    try:
        chat = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": instruction}],
        )
        text = chat.choices[0].message.content.strip()
        # hard cap as belt-and-suspenders
        words = text.split()
        if len(words) > target_words + 5:
            text = " ".join(words[: target_words + 5])
        return text
    except Exception as e:
        wlog(f"   ⚠ Narration script fallback ({e})", "warning")
        return " ".join(shot_actions)[: target_words * 8]


def generate_voiceover(openai_client, script_text, voice="onyx"):
    """OpenAI TTS narration."""
    try:
        wlog(f"   🎙 Generating voice-over ({len(script_text.split())} words)...", "normal")
        out_path = "static/voiceover.mp3"
        speech = openai_client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=script_text[:3500],
        )
        speech.stream_to_file(out_path)
        wlog("   🎙 Voice-over ready.", "success")
        return out_path
    except Exception as e:
        wlog(f"   ⚠ Voice-over generation failed: {e}", "warning")
        return None


# ---------- WORKERS ----------

def plan_worker(data):
    try:
        os.environ["OPENAI_API_KEY"] = data.get("openai_key", "")
        openai_client = OpenAI()

        topic = data.get("topic", "").strip()
        total_duration = int(data.get("duration", 15))
        pacing = data.get("pacing", "Balanced pacing")

        set_status("AI Director is crafting the storyboard...", phase="planning")
        wlog(f"🧠 Planning {total_duration}s sequence about: {topic}", "highlight")

        def _on_meta(title, bible):
            with STATE_LOCK:
                if title:
                    STATE["config"]["title"] = title
                if bible:
                    STATE["character_bible"] = bible
                    wlog(f"   ✦ Character: {bible[:90]}", "success")

        def _on_shot(s):
            with STATE_LOCK:
                idx = len(STATE["shots"])
                STATE["shots"].append({
                    "shot": idx + 1,
                    "title": s.get("title", f"Shot {idx+1}"),
                    "duration": int(s.get("duration", 5)),
                    "action": s.get("action", ""),
                    "prompt": "",
                    "url": None,
                    "status": "planned",
                })
                STATE["total_shots"] = len(STATE["shots"])
                STATE["credits_estimate"] = estimate_credits(STATE["shots"])
            wlog(f"   ▸ Shot {idx+1}: {s.get('title','')}", "highlight")

        ai_plan = director_generate_streaming(
            openai_client, topic, total_duration, pacing,
            on_shot=_on_shot, on_meta=_on_meta,
        )

        # Reconcile: if streaming missed final state, fill in from parsed plan
        with STATE_LOCK:
            if not STATE["character_bible"]:
                STATE["character_bible"] = ai_plan.get("character_bible", "")
            if not STATE["shots"]:
                shots_p = ai_plan.get("shots", [])
                STATE["shots"] = [{
                    "shot": i + 1,
                    "title": s.get("title", f"Shot {i+1}"),
                    "duration": int(s.get("duration", 5)),
                    "action": s.get("action", ""),
                    "prompt": "", "url": None, "status": "planned",
                } for i, s in enumerate(shots_p)]
                STATE["total_shots"] = len(STATE["shots"])
                STATE["credits_estimate"] = estimate_credits(STATE["shots"])
            if "title" not in STATE["config"]:
                STATE["config"]["title"] = ai_plan.get("title", "Untitled Sequence")
            STATE["config"]["logline"] = ai_plan.get("logline", "")
            shots = STATE["shots"]

        wlog(f"📋 Storyboard ready: {len(shots)} shots planned.", "success")

        # Optional: auto-generate character portrait for lock
        if str(data.get("char_lock", "")).lower() in ("1", "true", "on", "yes"):
            char_bible = STATE["character_bible"]
            if char_bible:
                ratio = data.get("ratio", "1280:720")
                portrait = generate_character_portrait(openai_client, char_bible, ratio=ratio)
                if portrait:
                    with STATE_LOCK:
                        STATE["character_portrait_url"] = portrait

        set_status("Storyboard ready — review and approve.", phase="awaiting_approval")
    except Exception as e:
        wlog(f"❌ Planning failed: {e}", "error")
        set_status(f"Planning error: {e}", phase="error")
    finally:
        with STATE_LOCK:
            STATE["is_running"] = False


def render_worker(data):
    try:
        with STATE_LOCK:
            STATE["is_running"] = True
            STATE["phase"] = "rendering"
            STATE["started_at"] = time.time()
            STATE["finished_at"] = None
            STATE["final_video_url"] = None

        os.environ["OPENAI_API_KEY"] = data.get("openai_key", "")
        os.environ["RUNWAYML_API_SECRET"] = data.get("runway_key", "")
        openai_client = OpenAI()
        runway_client = RunwayML()

        cfg = data
        ratio = cfg.get("ratio", "1280:720")
        camera = cfg.get("camera", "")
        art_style = cfg.get("style", "")
        motion = cfg.get("motion", "")
        base_image = (cfg.get("image") or "").strip()
        music_pref = cfg.get("music", "no_music")
        narrator_voice = cfg.get("narrator", "off")  # off / alloy / echo / fable / onyx / nova / shimmer

        audio_prompt = (
            "[AUDIO] Strictly NO music. High-quality ambient SFX and Foley only."
            if music_pref == "no_music"
            else "[AUDIO] Cinematic background music with immersive SFX."
        )

        with STATE_LOCK:
            shots = list(STATE["shots"])
            character_bible = STATE["character_bible"]
            portrait_url = STATE.get("character_portrait_url")

        # Prefer user-supplied image; otherwise use auto-generated portrait (as data URI)
        ref_uri = base_image
        if not ref_uri and portrait_url:
            try:
                local_path = portrait_url.lstrip("/")
                mime = "image/jpeg" if local_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
                with open(local_path, "rb") as f:
                    raw = f.read()
                size_kb = len(raw) // 1024
                b64 = base64.b64encode(raw).decode("ascii")
                ref_uri = f"data:{mime};base64,{b64}"
                wlog(f"   🔒 Character portrait reference attached ({size_kb} KB).", "highlight")
            except Exception as e:
                wlog(f"   ⚠ Could not load portrait as reference: {e}", "warning")
        image_refs = [{"uri": ref_uri}] if ref_uri else []
        last_url = ""
        all_urls = []

        wlog("🚀 RUNWAY ORCHESTRATION ENGINE INITIATED", "highlight")
        wlog(f"Character identity locked: {character_bible[:90]}", "success")

        for idx, shot in enumerate(shots):
            shot_num = idx + 1
            with STATE_LOCK:
                STATE["current_shot"] = shot_num
                STATE["shots"][idx]["status"] = "rendering"
            set_status(f"Rendering shot {shot_num}/{len(shots)} ({shot['duration']}s)...")

            full_prompt = compose_shot_prompt(
                character_bible, shot["action"], camera, art_style, motion, audio_prompt, ratio=ratio
            )
            wlog(f"\n--- RENDERING SHOT {shot_num}: {shot.get('title','')} ---", "highlight")
            wlog(f"Prompt: {full_prompt}", "normal")

            url, used_prompt = runway_render_shot(
                runway_client, full_prompt, ratio, shot["duration"],
                image_refs, last_url, openai_client,
            )

            with STATE_LOCK:
                STATE["shots"][idx]["url"] = url
                STATE["shots"][idx]["prompt"] = used_prompt
                STATE["shots"][idx]["status"] = "ready"
            wlog(f"✅ Shot {shot_num} ready", "success")
            with STATE_LOCK:
                STATE["shots"][idx].setdefault("takes", []).append({
                    "url": url, "prompt": used_prompt, "ts": time.time(),
                })
                STATE["shots"][idx]["active_take"] = len(STATE["shots"][idx]["takes"]) - 1
            all_urls.append(url)
            last_url = url

        # Voice-over
        voiceover_path = None
        if narrator_voice and narrator_voice != "off":
            with STATE_LOCK:
                lines = [s.get("action", "") for s in STATE["shots"]]
                total_dur = sum(int(s.get("duration", 5)) for s in STATE["shots"])
            script_text = write_narration_script(openai_client, lines, total_dur)
            wlog(f"   📝 Narration script ({len(script_text.split())} words for {total_dur}s)", "normal")
            voiceover_path = generate_voiceover(openai_client, script_text, voice=narrator_voice)
            if voiceover_path:
                with STATE_LOCK:
                    STATE["voiceover_url"] = "/" + voiceover_path

        set_status("Post-processing: merging clips...", phase="post")
        final_url = merge_videos_cinematic(all_urls, voiceover_path=voiceover_path)
        if final_url:
            with STATE_LOCK:
                STATE["final_video_url"] = final_url
            set_status("Master sequence complete.", phase="done")
            wlog("🎉 SEQUENCE COMPLETED AND MERGED SUCCESSFULLY", "success")
        else:
            set_status("Sequence generated, but merge failed.", phase="error")

    except Exception as e:
        traceback.print_exc()
        wlog(f"❌ FATAL: {e}", "error")
        set_status(f"Error: {e}", phase="error")
    finally:
        with STATE_LOCK:
            STATE["is_running"] = False
            STATE["finished_at"] = time.time()
        wlog("🏁 Render engine terminated.", "highlight")


def reroll_worker(data):
    try:
        with STATE_LOCK:
            STATE["is_running"] = True
            STATE["phase"] = "rerolling"

        os.environ["OPENAI_API_KEY"] = data.get("openai_key", "")
        os.environ["RUNWAYML_API_SECRET"] = data.get("runway_key", "")
        openai_client = OpenAI()
        runway_client = RunwayML()

        idx = int(data["shot_index"])
        new_action = data.get("action", "").strip()
        ratio = data.get("ratio", "1280:720")
        camera = data.get("camera", "")
        art_style = data.get("style", "")
        motion = data.get("motion", "")
        music_pref = data.get("music", "no_music")
        base_image = (data.get("image") or "").strip()
        audio_prompt = (
            "[AUDIO] Strictly NO music. High-quality ambient SFX and Foley only."
            if music_pref == "no_music"
            else "[AUDIO] Cinematic background music with immersive SFX."
        )

        with STATE_LOCK:
            if new_action:
                STATE["shots"][idx]["action"] = new_action
            shot = STATE["shots"][idx]
            character_bible = STATE["character_bible"]
            last_url = STATE["shots"][idx - 1]["url"] if idx > 0 else ""

        prompt = compose_shot_prompt(
            character_bible, shot["action"], camera, art_style, motion, audio_prompt, ratio=ratio
        )
        wlog(f"🔁 Re-rolling shot {idx+1}: {shot.get('title','')}", "highlight")

        image_refs = [{"uri": base_image}] if base_image else []
        url, used_prompt = runway_render_shot(
            runway_client, prompt, ratio, shot["duration"],
            image_refs, last_url, openai_client,
        )

        with STATE_LOCK:
            takes = STATE["shots"][idx].setdefault("takes", [])
            # Seed with the original take if missing
            if STATE["shots"][idx].get("url") and not takes:
                takes.append({
                    "url": STATE["shots"][idx]["url"],
                    "prompt": STATE["shots"][idx].get("prompt", ""),
                    "ts": time.time() - 1,
                })
            takes.append({"url": url, "prompt": used_prompt, "ts": time.time()})
            STATE["shots"][idx]["active_take"] = len(takes) - 1
            STATE["shots"][idx]["url"] = url
            STATE["shots"][idx]["prompt"] = used_prompt
            STATE["shots"][idx]["status"] = "ready"
        wlog(f"✅ Shot {idx+1} re-rolled (take {len(takes)})", "success")
        set_status(f"Shot {idx+1} re-rolled.", phase="awaiting_approval")
    except Exception as e:
        wlog(f"❌ Re-roll failed: {e}", "error")
        set_status(f"Re-roll error: {e}", phase="error")
    finally:
        with STATE_LOCK:
            STATE["is_running"] = False


def remerge_worker():
    try:
        with STATE_LOCK:
            STATE["is_running"] = True
            STATE["phase"] = "post"
            urls = [s["url"] for s in STATE["shots"] if s.get("url")]
            vo = STATE.get("voiceover_url")
        vo_path = vo.lstrip("/") if vo else None
        set_status("Re-merging master with updated shots...")
        final_url = merge_videos_cinematic(urls, voiceover_path=vo_path)
        if final_url:
            with STATE_LOCK:
                STATE["final_video_url"] = final_url + f"?v={int(time.time())}"
            set_status("Master re-merged.", phase="done")
    except Exception as e:
        wlog(f"❌ Remerge failed: {e}", "error")
    finally:
        with STATE_LOCK:
            STATE["is_running"] = False


# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/plan", methods=["POST"])
def plan():
    data = request.json or {}
    if not data.get("openai_key"):
        return jsonify({"error": "OpenAI key required"}), 400
    if STATE["is_running"]:
        return jsonify({"error": "Engine busy"}), 400
    reset_state()
    with STATE_LOCK:
        STATE["is_running"] = True
        STATE["phase"] = "planning"
        STATE["config"] = {k: data.get(k) for k in (
            "topic", "duration", "ratio", "pacing", "camera", "style",
            "motion", "music", "image", "narrator"
        )}
    threading.Thread(target=plan_worker, args=(data,), daemon=True).start()
    return jsonify({"status": "planning"})


@app.route("/update_storyboard", methods=["POST"])
def update_storyboard():
    data = request.json or {}
    new_shots = data.get("shots", [])
    with STATE_LOCK:
        if STATE["is_running"]:
            return jsonify({"error": "Engine busy"}), 400
        for i, s in enumerate(new_shots):
            if i < len(STATE["shots"]):
                STATE["shots"][i]["title"] = s.get("title", STATE["shots"][i]["title"])
                STATE["shots"][i]["action"] = s.get("action", STATE["shots"][i]["action"])
                STATE["shots"][i]["duration"] = int(s.get("duration", STATE["shots"][i]["duration"]))
        STATE["credits_estimate"] = estimate_credits(STATE["shots"])
    return jsonify({"status": "ok"})


@app.route("/add_shot", methods=["POST"])
def add_shot():
    data = request.json or {}
    after = data.get("after", -1)
    duration = int(data.get("duration", 5))
    action = data.get("action", "New shot — describe the action.")
    title = data.get("title", "New Shot")
    with STATE_LOCK:
        if STATE["is_running"]:
            return jsonify({"error": "Engine busy"}), 400
        new_shot = {
            "shot": 0, "title": title, "duration": duration, "action": action,
            "prompt": "", "url": None, "status": "planned",
        }
        if after < 0 or after >= len(STATE["shots"]):
            STATE["shots"].append(new_shot)
        else:
            STATE["shots"].insert(after + 1, new_shot)
        for i, s in enumerate(STATE["shots"]):
            s["shot"] = i + 1
        STATE["total_shots"] = len(STATE["shots"])
        STATE["credits_estimate"] = estimate_credits(STATE["shots"])
    return jsonify({"status": "ok"})


@app.route("/delete_shot", methods=["POST"])
def delete_shot():
    data = request.json or {}
    idx = int(data.get("index", -1))
    with STATE_LOCK:
        if STATE["is_running"]:
            return jsonify({"error": "Engine busy"}), 400
        if 0 <= idx < len(STATE["shots"]):
            STATE["shots"].pop(idx)
            for i, s in enumerate(STATE["shots"]):
                s["shot"] = i + 1
            STATE["total_shots"] = len(STATE["shots"])
            STATE["credits_estimate"] = estimate_credits(STATE["shots"])
    return jsonify({"status": "ok"})


@app.route("/duplicate_shot", methods=["POST"])
def duplicate_shot():
    data = request.json or {}
    idx = int(data.get("index", -1))
    with STATE_LOCK:
        if STATE["is_running"]:
            return jsonify({"error": "Engine busy"}), 400
        if 0 <= idx < len(STATE["shots"]):
            src = STATE["shots"][idx]
            clone = {
                "shot": 0, "title": src["title"] + " (copy)",
                "duration": src["duration"], "action": src["action"],
                "prompt": "", "url": None, "status": "planned",
            }
            STATE["shots"].insert(idx + 1, clone)
            for i, s in enumerate(STATE["shots"]):
                s["shot"] = i + 1
            STATE["total_shots"] = len(STATE["shots"])
            STATE["credits_estimate"] = estimate_credits(STATE["shots"])
    return jsonify({"status": "ok"})


@app.route("/reorder_shots", methods=["POST"])
def reorder_shots():
    data = request.json or {}
    order = data.get("order", [])  # list of original indices in new order
    with STATE_LOCK:
        if STATE["is_running"]:
            return jsonify({"error": "Engine busy"}), 400
        if len(order) != len(STATE["shots"]):
            return jsonify({"error": "Order length mismatch"}), 400
        new_shots = [STATE["shots"][i] for i in order]
        for i, s in enumerate(new_shots):
            s["shot"] = i + 1
        STATE["shots"] = new_shots
    return jsonify({"status": "ok"})


@app.route("/render", methods=["POST"])
def render():
    data = request.json or {}
    if not data.get("openai_key") or not data.get("runway_key"):
        return jsonify({"error": "API keys required"}), 400
    if STATE["is_running"]:
        return jsonify({"error": "Engine busy"}), 400
    if not STATE["shots"]:
        return jsonify({"error": "No storyboard. Run /plan first."}), 400
    threading.Thread(target=render_worker, args=(data,), daemon=True).start()
    return jsonify({"status": "rendering"})


@app.route("/reroll", methods=["POST"])
def reroll():
    data = request.json or {}
    if not data.get("openai_key") or not data.get("runway_key"):
        return jsonify({"error": "API keys required"}), 400
    if STATE["is_running"]:
        return jsonify({"error": "Engine busy"}), 400
    threading.Thread(target=reroll_worker, args=(data,), daemon=True).start()
    return jsonify({"status": "rerolling"})


@app.route("/select_take", methods=["POST"])
def select_take():
    data = request.json or {}
    idx = int(data.get("index", -1))
    take = int(data.get("take", -1))
    with STATE_LOCK:
        if STATE["is_running"]:
            return jsonify({"error": "Engine busy"}), 400
        if 0 <= idx < len(STATE["shots"]):
            takes = STATE["shots"][idx].get("takes", [])
            if 0 <= take < len(takes):
                STATE["shots"][idx]["active_take"] = take
                STATE["shots"][idx]["url"] = takes[take]["url"]
                STATE["shots"][idx]["prompt"] = takes[take].get("prompt", "")
                return jsonify({"status": "ok", "url": takes[take]["url"]})
    return jsonify({"error": "Invalid take/index"}), 400


@app.route("/remerge", methods=["POST"])
def remerge():
    if STATE["is_running"]:
        return jsonify({"error": "Engine busy"}), 400
    if not any(s.get("url") for s in STATE["shots"]):
        return jsonify({"error": "Nothing to merge"}), 400
    threading.Thread(target=remerge_worker, daemon=True).start()
    return jsonify({"status": "remerging"})


@app.route("/get_status", methods=["GET"])
def get_status():
    with STATE_LOCK:
        return jsonify(STATE)


@app.route("/reset", methods=["POST"])
def reset():
    if STATE["is_running"]:
        return jsonify({"error": "Engine busy"}), 400
    reset_state()
    return jsonify({"status": "reset"})


@app.route("/export_storyboard", methods=["GET"])
def export_storyboard():
    with STATE_LOCK:
        payload = {
            "version": 1,
            "title": STATE.get("config", {}).get("title", "Untitled"),
            "logline": STATE.get("config", {}).get("logline", ""),
            "character_bible": STATE.get("character_bible", ""),
            "shots": [
                {"title": s["title"], "duration": s["duration"], "action": s["action"]}
                for s in STATE.get("shots", [])
            ],
            "config": STATE.get("config", {}),
        }
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(
        body, mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=storyboard.json"},
    )


@app.route("/import_storyboard", methods=["POST"])
def import_storyboard():
    data = request.json or {}
    shots_in = data.get("shots", [])
    if not shots_in:
        return jsonify({"error": "No shots in payload"}), 400
    with STATE_LOCK:
        if STATE["is_running"]:
            return jsonify({"error": "Engine busy"}), 400
        STATE["shots"] = [{
            "shot": i + 1,
            "title": s.get("title", f"Shot {i+1}"),
            "duration": int(s.get("duration", 5)),
            "action": s.get("action", ""),
            "prompt": "", "url": None, "status": "planned",
        } for i, s in enumerate(shots_in)]
        STATE["total_shots"] = len(STATE["shots"])
        STATE["character_bible"] = data.get("character_bible", "")
        STATE["credits_estimate"] = estimate_credits(STATE["shots"])
        STATE["phase"] = "awaiting_approval"
        STATE["status_msg"] = "Storyboard imported — review and approve."
        if data.get("config"):
            STATE["config"].update(data["config"])
    return jsonify({"status": "ok"})


@app.route("/bundle.zip", methods=["GET"])
def bundle_zip():
    with STATE_LOCK:
        shots = list(STATE["shots"])
        master = STATE.get("final_video_url")
        vo = STATE.get("voiceover_url")
        portrait = STATE.get("character_portrait_url")
        bible = STATE.get("character_bible", "")
        title = STATE.get("config", {}).get("title", "Project")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # README
        readme = (
            f"# {title}\n\n"
            f"Generated by Runway Director.\n\n"
            f"## Character Bible\n{bible}\n\n"
            f"## Shots\n" +
            "\n".join(f"{i+1}. ({s['duration']}s) {s.get('title','')} — {s.get('action','')}"
                      for i, s in enumerate(shots))
        )
        zf.writestr("README.md", readme)

        # storyboard.json
        sb = {
            "title": title, "character_bible": bible,
            "shots": [{"title": s["title"], "duration": s["duration"], "action": s["action"]}
                      for s in shots],
        }
        zf.writestr("storyboard.json", json.dumps(sb, indent=2, ensure_ascii=False))

        # director's script
        script_txt = f"🎬 {title.upper()}\n{'='*48}\n\n"
        for i, s in enumerate(shots):
            script_txt += f"SHOT {i+1} — {s.get('title','')} ({s['duration']}s)\n{s.get('action','')}\n\n"
        zf.writestr("Directors_Script.txt", script_txt)

        # individual clips (re-download from Runway URLs)
        for i, s in enumerate(shots):
            url = s.get("url")
            if not url: continue
            try:
                r = requests.get(url, timeout=120)
                if r.ok:
                    zf.writestr(f"clips/shot_{i+1:02d}.mp4", r.content)
            except Exception:
                pass

        # master, VO, portrait — local files
        for src_url, arc in [(master, "Master.mp4"), (vo, "voiceover.mp3"),
                              (portrait, "character_portrait.jpg")]:
            if not src_url: continue
            local = src_url.split("?")[0].lstrip("/")
            if os.path.exists(local):
                with open(local, "rb") as f:
                    zf.writestr(arc, f.read())

    buf.seek(0)
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:40] or "project"
    return send_file(
        buf, mimetype="application/zip", as_attachment=True,
        download_name=f"{safe_title}_bundle.zip",
    )


@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        return jsonify({"error": "Unsupported format"}), 400
    fname = f"upload_{uuid.uuid4().hex[:10]}{ext}"
    path = os.path.join("static", fname)
    f.save(path)
    # Compress to keep base-64 small
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        max_side = 768
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        jpg_path = os.path.join("static", f"upload_{uuid.uuid4().hex[:10]}.jpg")
        img.save(jpg_path, format="JPEG", quality=85, optimize=True)
        try: os.remove(path)
        except Exception: pass
        path = jpg_path
    except Exception:
        pass
    # Return as data URI so Runway can use it without a public URL
    with open(path, "rb") as fp:
        raw = fp.read()
    mime = "image/jpeg" if path.endswith(".jpg") else "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return jsonify({
        "data_uri": f"data:{mime};base64,{b64}",
        "preview_url": "/" + path.replace("\\", "/"),
        "size_kb": len(raw) // 1024,
    })


def trailer_edit_plan(openai_client, shots, target_dur=12):
    """Ask GPT-4o for an aggressive trailer edit decision list."""
    summary = "\n".join(
        f"Shot {i+1}: {s.get('duration',5)}s — {s.get('action','')[:160]}"
        for i, s in enumerate(shots)
    )
    instruction = (
        f"You are an elite trailer editor (think A24, Vince Mancini style). "
        f"Given the shots below, design a {target_dur}-second social trailer.\n\n"
        f"{summary}\n\n"
        f"Rules:\n"
        f"- 6 to 10 cuts. Most cuts should be 0.4–1.5s. End on a slightly longer (1.5–2.5s) hero beat.\n"
        f"- Use start/end ranges within each shot's actual duration.\n"
        f"- Speed values: 1.0 default; use 0.5 for slow-mo on the climax; 1.5–2.0 for tension stacking.\n"
        f"- Provide a 4-8 word voice-over hook.\n"
        f"- Provide a 1-3 word title_card (uppercase short).\n\n"
        f"Return ONLY JSON:\n"
        f'{{"voiceover_hook": "...", "title_card": "...", '
        f'"cuts": [{{"shot": 1, "start": 0.5, "end": 1.4, "speed": 1.0}}]}}'
    )
    chat = openai_client.chat.completions.create(
        model="gpt-4o", response_format={"type": "json_object"},
        messages=[{"role": "user", "content": instruction}],
    )
    return json.loads(chat.choices[0].message.content)


def build_trailer(openai_client, shots, output_path="static/Trailer.mp4", target_dur=12, voice="onyx", ratio="1280:720"):
    plan = trailer_edit_plan(openai_client, shots, target_dur=target_dur)
    cuts = plan.get("cuts", [])
    hook = (plan.get("voiceover_hook") or "").strip()
    title = (plan.get("title_card") or "").strip().upper()
    wlog(f"   ✂ Trailer plan: {len(cuts)} cuts · hook: \"{hook}\"", "highlight")

    # Download all unique shot urls (parallel)
    urls = {}
    for c in cuts:
        s_idx = int(c.get("shot", 1)) - 1
        if 0 <= s_idx < len(shots) and shots[s_idx].get("url"):
            urls[s_idx] = shots[s_idx]["url"]
    local = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as ex:
        futs = {ex.submit(download_clip, u, k): k for k, u in urls.items()}
        for fut in concurrent.futures.as_completed(futs):
            local[futs[fut]] = fut.result()

    # Generate hook VO
    vo_path = None
    if hook:
        vo_path = generate_voiceover(openai_client, hook, voice=voice)

    # Build each cut as its own re-encoded clip (necessary for trim+speed)
    parts = []
    for i, c in enumerate(cuts):
        s_idx = int(c.get("shot", 1)) - 1
        if s_idx not in local: continue
        src = local[s_idx]
        start = max(0.0, float(c.get("start", 0)))
        end = max(start + 0.2, float(c.get("end", start + 1.0)))
        speed = max(0.25, min(3.0, float(c.get("speed", 1.0))))
        out = f"static/_t_part_{i:02d}.mp4"
        # Trim with re-encode (we MUST re-encode here because speed/trim breaks stream copy anyway)
        # setpts adjusts video speed; atempo for audio (max 2.0 each, chain if higher)
        atempo_chain = ""
        a = speed
        while a > 2.0:
            atempo_chain += "atempo=2.0,"
            a /= 2.0
        atempo_chain += f"atempo={a:.4f}"
        ok, err = _ffmpeg_run([
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
            "-filter_complex",
            f"[0:v]setpts={1/speed:.4f}*PTS[v];[0:a]{atempo_chain}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
            "-r", "24", out,
        ])
        if ok:
            parts.append(out)
        else:
            wlog(f"   ⚠ Cut {i+1} failed: {err[-160:]}", "warning")

    if not parts:
        wlog("   ⚠ Trailer build aborted: no usable cuts.", "error")
        return None

    # Title card (1s) — black with text overlay
    if title:
        tc = "static/_t_title.mp4"
        # use drawtext, fallback to plain black if font missing
        ok, _ = _ffmpeg_run([
            "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=1.0:r=24",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", f"drawtext=text='{title}':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2:alpha='if(lt(t,0.2),t/0.2,if(lt(t,0.8),1,(1-(t-0.8)/0.2)))'",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-t", "1.0", tc,
        ])
        if ok:
            parts.insert(0, tc)

    # Concat all parts (re-encode because parts have different timelines)
    list_path = "static/_t_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p).replace(chr(92), '/')}'\n")

    base_out = output_path
    if vo_path:
        intermediate = "static/_t_concat.mp4"
        ok, err = _ffmpeg_run([
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-crf", "19", "-preset", "fast",
            "-c:a", "aac", "-b:a", "256k", "-pix_fmt", "yuv420p", intermediate,
        ])
        if ok:
            ok2, err2 = _ffmpeg_run([
                "-i", intermediate, "-i", vo_path,
                "-filter_complex",
                "[0:a]volume=0.35[a1];[1:a]volume=1.2[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                "-shortest", "-movflags", "+faststart", base_out,
            ])
            try: os.remove(intermediate)
            except Exception: pass
            if not ok2:
                wlog(f"   ⚠ Trailer mux failed: {err2[-160:]}", "error"); base_out = None
        else:
            wlog(f"   ⚠ Trailer concat failed: {err[-160:]}", "error"); base_out = None
    else:
        ok, err = _ffmpeg_run([
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-crf", "19", "-preset", "fast",
            "-c:a", "aac", "-b:a", "256k", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", base_out,
        ])
        if not ok:
            wlog(f"   ⚠ Trailer concat failed: {err[-160:]}", "error"); base_out = None

    # Cleanup
    for p in parts + [list_path]:
        try: os.remove(p)
        except Exception: pass

    return ("/" + base_out) if base_out else None


def trailer_worker(data):
    try:
        with STATE_LOCK:
            STATE["is_running"] = True
            STATE["phase"] = "post"
        os.environ["OPENAI_API_KEY"] = data.get("openai_key", "")
        openai_client = OpenAI()
        target = int(data.get("duration", 12))
        voice = data.get("voice", "onyx")
        with STATE_LOCK:
            shots = list(STATE["shots"])
            ratio = STATE.get("config", {}).get("ratio", "1280:720")
        if not any(s.get("url") for s in shots):
            set_status("No rendered shots for trailer.", phase="error"); return
        wlog(f"🎬 Generating {target}s trailer from {len(shots)} shots...", "highlight")
        set_status(f"Generating {target}s trailer...")
        url = build_trailer(openai_client, shots, target_dur=target, voice=voice, ratio=ratio)
        if url:
            with STATE_LOCK:
                STATE["trailer_url"] = url + f"?v={int(time.time())}"
            wlog("🎬 Trailer ready.", "success")
            set_status("Trailer ready.", phase="done")
        else:
            set_status("Trailer build failed.", phase="error")
    except Exception as e:
        traceback.print_exc()
        wlog(f"❌ Trailer error: {e}", "error")
    finally:
        with STATE_LOCK:
            STATE["is_running"] = False


@app.route("/generate_trailer", methods=["POST"])
def generate_trailer():
    data = request.json or {}
    if not data.get("openai_key"):
        return jsonify({"error": "OpenAI key required"}), 400
    if STATE["is_running"]:
        return jsonify({"error": "Engine busy"}), 400
    if not any(s.get("url") for s in STATE["shots"]):
        return jsonify({"error": "No rendered shots"}), 400
    threading.Thread(target=trailer_worker, args=(data,), daemon=True).start()
    return jsonify({"status": "trailer"})


def regen_voiceover_worker(data):
    try:
        with STATE_LOCK:
            STATE["is_running"] = True
            STATE["phase"] = "post"
        os.environ["OPENAI_API_KEY"] = data.get("openai_key", "")
        openai_client = OpenAI()
        voice = data.get("voice", "onyx")
        custom_script = (data.get("script") or "").strip()
        with STATE_LOCK:
            lines = [s.get("action", "") for s in STATE["shots"]]
            total_dur = sum(int(s.get("duration", 5)) for s in STATE["shots"])
            urls = [s["url"] for s in STATE["shots"] if s.get("url")]
        if not urls:
            set_status("No shots to merge.", phase="error"); return
        script = custom_script or write_narration_script(openai_client, lines, total_dur)
        wlog(f"📝 New narration ({len(script.split())} words)", "highlight")
        vo_path = generate_voiceover(openai_client, script, voice=voice)
        if vo_path:
            with STATE_LOCK:
                STATE["voiceover_url"] = "/" + vo_path
        set_status("Re-merging master with new voice-over...")
        final_url = merge_videos_cinematic(urls, voiceover_path=vo_path)
        if final_url:
            with STATE_LOCK:
                STATE["final_video_url"] = final_url + f"?v={int(time.time())}"
            set_status("Voice-over regenerated.", phase="done")
    except Exception as e:
        wlog(f"❌ Voice-over regen failed: {e}", "error")
    finally:
        with STATE_LOCK:
            STATE["is_running"] = False


@app.route("/regen_voiceover", methods=["POST"])
def regen_voiceover():
    data = request.json or {}
    if not data.get("openai_key"):
        return jsonify({"error": "OpenAI key required"}), 400
    if STATE["is_running"]:
        return jsonify({"error": "Engine busy"}), 400
    if not any(s.get("url") for s in STATE["shots"]):
        return jsonify({"error": "No rendered shots"}), 400
    threading.Thread(target=regen_voiceover_worker, args=(data,), daemon=True).start()
    return jsonify({"status": "regenerating"})


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
