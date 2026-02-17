"""Entry point for the Wan2GP Timeline Editor plugin.

This file must stay at repository root so Wan2GP can import
`<plugin_folder>.plugin` during plugin installation/loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr

# Wan2GP plugin base can vary slightly across versions/import paths.
# Keep graceful fallbacks so the module remains importable.
try:
    from shared.plugins.base import WAN2GPPlugin  # type: ignore
except Exception:  # pragma: no cover - fallback path
    try:
        from shared.plugins.base_plugin import WAN2GPPlugin  # type: ignore
    except Exception:  # pragma: no cover - local/dev fallback
        class WAN2GPPlugin:  # type: ignore
            """Fallback base for static checks outside Wan2GP runtime."""

            pass


# Optional component dependency used for NLE-like timeline UX.
try:
    from gradio_vistimeline import VisTimeline
except Exception:  # pragma: no cover - handled in UI
    VisTimeline = None


# Reuse Wan2GP metadata helper if available.
try:
    from shared.utils.video_metadata import read_metadata_from_video, save_video_metadata
except Exception:  # pragma: no cover - local/dev fallback
    read_metadata_from_video = None
    save_video_metadata = None


PLUGIN_ID = "wan2gp-timeline-editor"
DEBUG = os.getenv("WAN2GP_TIMELINE_DEBUG", "0") == "1"
PREMIERE_UI = os.getenv("WAN2GP_TIMELINE_PREMIERE_UI", "1") == "1"


def _log(level: str, eid: str, msg: str, **kv: Any) -> None:
    if level == "DEBUG" and not DEBUG:
        return
    tail = " ".join([f"{k}={repr(v)}" for k, v in kv.items()])
    print(f"[{eid}] {level} {msg}" + (f" {tail}" if tail else ""))


def _sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _run_ffprobe(path: str) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return json.loads(result.stdout)


def _video_meta(path: str) -> Dict[str, Any]:
    probe = _run_ffprobe(path)
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError(f"No video stream found: {path}")

    rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    return {
        "duration": float(fmt.get("duration") or video_stream.get("duration") or 0.0),
        "fps": fps,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
    }


def _ensure_thumb(path: str, thumbs_dir: str, width: int = 260) -> str:
    os.makedirs(thumbs_dir, exist_ok=True)
    thumb_name = hashlib.sha1(path.encode("utf-8")).hexdigest() + ".jpg"
    thumb_path = os.path.join(thumbs_dir, thumb_name)
    if os.path.exists(thumb_path):
        return thumb_path

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        "0.5",
        "-i",
        path,
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-1",
        "-q:v",
        "3",
        thumb_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _log("DEBUG", "TL-MBIN-THUMB-001", "Thumbnail generation failed", path=path)
        return path
    return thumb_path


def _sec_to_frames(sec: float, fps: int) -> int:
    return int(round(sec * float(fps)))


def _frames_to_sec(frames: int, fps: int) -> float:
    return (float(frames) / float(fps)) if fps > 0 else 0.0


@dataclass
class Clip:
    id: str
    source_id: str
    track_id: str
    timeline_start: int
    timeline_end: int
    src_in: int
    src_out: int
    opacity: float = 1.0
    audio_gain: float = 1.0
    mute: bool = False


def _empty_project(fps: int = 30) -> Dict[str, Any]:
    return {
        "version": 1,
        "fps": fps,
        "resolution": {"w": 1080, "h": 1920},
        "audio_rate": 48000,
        "sources": [],
        "tracks": {
            "video_tracks": [{"id": "V1", "name": "V1"}, {"id": "V2", "name": "V2"}],
            "audio_tracks": [{"id": "A1", "name": "A1"}, {"id": "A2", "name": "A2"}],
        },
        "clips": [],
        "transitions": [],
    }


def _project_to_timeline_value(project: Dict[str, Any]) -> Dict[str, Any]:
    fps = int(project.get("fps", 30))
    groups = []
    for track in project["tracks"]["video_tracks"] + project["tracks"]["audio_tracks"]:
        groups.append({"id": track["id"], "content": track["name"]})

    items = []
    for clip in project.get("clips", []):
        items.append(
            {
                "id": clip["id"],
                "content": clip["id"],
                "group": clip["track_id"],
                "start": int(1000 * _frames_to_sec(int(clip["timeline_start"]), fps)),
                "end": int(1000 * _frames_to_sec(int(clip["timeline_end"]), fps)),
                "className": "tl-item-video" if str(clip["track_id"]).startswith("V") else "tl-item-audio",
            }
        )
    return {"groups": groups, "items": items}


def _apply_timeline_edits(project: Dict[str, Any], timeline_value: Dict[str, Any]) -> Dict[str, Any]:
    fps = int(project.get("fps", 30))
    by_id = {str(item.get("id")): item for item in (timeline_value or {}).get("items", [])}
    for clip in project.get("clips", []):
        item = by_id.get(str(clip["id"]))
        if not item:
            continue
        start = _sec_to_frames(float(item.get("start", 0)) / 1000.0, fps)
        end = _sec_to_frames(float(item.get("end", 0)) / 1000.0, fps)
        if end <= start:
            end = start + 1
        old_duration = int(clip["timeline_end"]) - int(clip["timeline_start"])
        new_duration = end - start
        clip["timeline_start"] = start
        clip["timeline_end"] = end
        clip["track_id"] = item.get("group") or clip["track_id"]
        clip["src_out"] = int(clip["src_in"]) + new_duration
        _log(
            "DEBUG",
            "TL-EDL-SYNC-001",
            "Timeline edit applied",
            clip_id=clip["id"],
            dur_old=old_duration,
            dur_new=new_duration,
        )
    return project


def _validate_v1_no_overlap(project: Dict[str, Any]) -> Tuple[bool, str]:
    v1 = [clip for clip in project.get("clips", []) if clip.get("track_id") == "V1"]
    v1.sort(key=lambda c: int(c["timeline_start"]))
    for left, right in zip(v1, v1[1:]):
        if int(right["timeline_start"]) < int(left["timeline_end"]):
            return False, f"Overlap in V1: {left['id']} overlaps {right['id']}"
    return True, "ok"


def _build_ffmpeg_concat_cmd(project: Dict[str, Any], out_path: str, preset: str) -> Tuple[List[str], str]:
    fps = int(project.get("fps", 30))
    width = int(project["resolution"]["w"])
    height = int(project["resolution"]["h"])
    audio_rate = int(project.get("audio_rate", 48000))

    if preset == "vertical_9x16_1080p":
        width, height = 1080, 1920
    elif preset == "vertical_9x16_720p":
        width, height = 720, 1280

    clips = [clip for clip in project.get("clips", []) if clip.get("track_id") == "V1"]
    clips.sort(key=lambda c: int(c["timeline_start"]))
    if not clips:
        raise ValueError("No clips in V1 to export")

    cmd: List[str] = ["ffmpeg", "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1"]
    for clip in clips:
        source = next(src for src in project["sources"] if src["id"] == clip["source_id"])
        cmd += ["-i", source["path"]]

    filters: List[str] = []
    v_labels: List[str] = []
    a_labels: List[str] = []
    for index, clip in enumerate(clips):
        start_sec = _frames_to_sec(int(clip["src_in"]), fps)
        end_sec = _frames_to_sec(int(clip["src_out"]), fps)
        filters.append(
            f"[{index}:v]trim=start={start_sec}:end={end_sec},setpts=PTS-STARTPTS,"
            f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v{index}]"
        )
        filters.append(
            f"[{index}:a]atrim=start={start_sec}:end={end_sec},asetpts=PTS-STARTPTS,"
            f"aresample={audio_rate},aformat=channel_layouts=stereo:sample_fmts=fltp[a{index}]"
        )
        v_labels.append(f"[v{index}]")
        a_labels.append(f"[a{index}]")

    transitions = project.get("transitions") or []
    if transitions and len(clips) > 1:
        transition = transitions[0]
        transition_type = transition.get("type", "dissolve")
        duration = float(transition.get("duration", 0.8))
        prev_v = "[v0]"
        prev_a = "[a0]"
        offsets: List[float] = []
        elapsed = _frames_to_sec(int(clips[0]["src_out"]) - int(clips[0]["src_in"]), fps)
        for i in range(1, len(clips)):
            offset = max(0.0, elapsed - duration)
            offsets.append(offset)
            filters.append(f"{prev_v}[v{i}]xfade=transition={transition_type}:duration={duration}:offset={offset}[vx{i}]")
            filters.append(f"{prev_a}[a{i}]acrossfade=d={duration}[ax{i}]")
            prev_v = f"[vx{i}]"
            prev_a = f"[ax{i}]"
            elapsed += _frames_to_sec(int(clips[i]["src_out"]) - int(clips[i]["src_in"]), fps) - duration
        filters.append(f"{prev_v}copy[vout]")
        filters.append(f"{prev_a}acopy[aout]")
        _log("DEBUG", "TL-XFADE-001", "xfade chain built", transition=transition_type, duration=duration, offsets=offsets)
    else:
        merged = "".join([item for pair in zip(v_labels, a_labels) for item in pair])
        filters.append(f"{merged}concat=n={len(clips)}:v=1:a=1[vout][aout]")

    filter_complex = ";".join(filters)
    cmd += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]", "-r", str(fps), out_path]
    return cmd, filter_complex


# MVP stubs for next milestones.
def _build_audio_mix_graph(project: Dict[str, Any]) -> Optional[str]:
    _ = project
    _log("DEBUG", "TL-MULTI-STUB-001", "multi-track stubs present (no-op)")
    return None


def _build_video_overlay_graph(project: Dict[str, Any]) -> Optional[str]:
    _ = project
    _log("DEBUG", "TL-MULTI-STUB-001", "multi-track stubs present (no-op)")
    return None


def _embed_project_metadata(out_path: str, project: Dict[str, Any]) -> None:
    payload = {"timeline_project": project}
    if save_video_metadata is not None:
        save_video_metadata(out_path, payload)
        return

    # Fallback: write JSON into comment tag with ffmpeg CLI if helper unavailable.
    meta_json = json.dumps(payload, ensure_ascii=False)
    temp_out = out_path + ".tmp_embed.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        out_path,
        "-c",
        "copy",
        "-metadata",
        f"comment={meta_json}",
        temp_out,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "metadata embed failed")
    os.replace(temp_out, out_path)


def _export_ffmpeg(project: Dict[str, Any], preset: str, embed_json: bool, out_dir: str) -> Generator[Tuple[Any, str, Any], None, None]:
    valid, reason = _validate_v1_no_overlap(project)
    if not valid:
        yield None, f"[TL-EXPORT-VAL-001] {reason}", None
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "timeline_export.mp4")

    try:
        cmd, filter_complex = _build_ffmpeg_concat_cmd(project, out_path, preset)
    except Exception as exc:
        yield None, f"[TL-EXPORT-ERR-001] {exc}", None
        return

    _log("DEBUG", "TL-EXPORT-CMD-001", "ffmpeg command built", filter_complex=filter_complex)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    progress_data: Dict[str, str] = {}
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        progress_data[key] = value
        if progress_data.get("progress") in {"continue", "end"}:
            status = " ".join(
                [f"{k}={progress_data[k]}" for k in ("out_time_ms", "speed", "progress") if k in progress_data]
            )
            yield None, f"[TL-EXPORT-PROG-001] {status}", None
            if progress_data.get("progress") == "end":
                break
            progress_data = {}

    return_code = process.wait()
    if return_code != 0:
        error_tail = (process.stderr.read() if process.stderr else "")[-1200:]
        _log("INFO", "TL-EXPORT-ERR-001", "ffmpeg failed", rc=return_code)
        yield None, f"[TL-EXPORT-ERR-001] ffmpeg failed rc={return_code}\n{error_tail}", None
        return

    if embed_json:
        try:
            _embed_project_metadata(out_path, project)
            _log("DEBUG", "TL-EXPORT-META-001", "project JSON embedded", out_path=out_path)
        except Exception as exc:
            _log("INFO", "TL-EXPORT-META-ERR-001", "metadata embed failed", error=str(exc))

    yield out_path, f"[TL-EXPORT-OK-001] Exported: {out_path}", out_path


def _premiere_css() -> str:
    return """
    .timeline-mode .tab-nav { display: none !important; }
    .timeline-mode .gradio-container { max-width: 100vw !important; }
    .tl-root { min-height: calc(100vh - 20px); }
    .tl-item-video { font-weight: 600; }
    .tl-item-audio { opacity: 0.85; }
    """


def _premiere_js() -> str:
    return """
    function tl_enablePremiereMode(rootId) {
      const root = document.getElementById(rootId);
      if (!root) return;
      const body = document.body;
      const obs = new IntersectionObserver((entries) => {
        for (const e of entries) {
          if (e.isIntersecting) body.classList.add('timeline-mode');
          else body.classList.remove('timeline-mode');
        }
      }, { threshold: 0.2 });
      obs.observe(root);
    }
    """


class TimelineEditorPlugin(WAN2GPPlugin):
    """Installable Wan2GP plugin adding a dedicated NLE-like timeline tab."""

    name = "Wan2GP Timeline Editor"

    def setup_ui(self, app=None):
        """Create the Timeline Editor tab with media bin, timeline, preview and export."""
        project_state = gr.State(_empty_project())
        selection_state = gr.State({"selected_source_id": None})

        thumbs_dir = os.path.join("outputs", ".timeline_thumbs")
        export_dir = os.path.join("outputs", ".timeline_exports")

        def _scan_media_paths() -> List[str]:
            roots = ["outputs"]
            if app is not None:
                server_config = getattr(app, "server_config", {}) or {}
                roots.extend([server_config.get("save_path"), server_config.get("image_save_path")])
            roots = [root for root in roots if root and os.path.isdir(root)]
            paths: List[str] = []
            for root in roots:
                for dirpath, _, filenames in os.walk(root):
                    for filename in filenames:
                        if filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                            paths.append(os.path.join(dirpath, filename))
            deduped = sorted(list(dict.fromkeys(paths)))
            _log("DEBUG", "TL-MBIN-SCAN-001", "Media scan completed", count=len(deduped))
            return deduped

        def _rebuild_media_bin(current_project: Dict[str, Any]):
            sources: List[Dict[str, Any]] = []
            gallery: List[Tuple[str, str]] = []
            for index, path in enumerate(_scan_media_paths()):
                try:
                    meta = _video_meta(path)
                except Exception:
                    continue
                source = {"id": f"src{index}", "path": path, "sha256": _sha256(path), **meta}
                sources.append(source)
                thumb = _ensure_thumb(path, thumbs_dir)
                gallery.append((thumb, f"{os.path.basename(path)} | {meta['duration']:.2f}s | {meta['width']}x{meta['height']}"))
            current_project["sources"] = sources
            return gallery, current_project

        def _import_files(files: List[Any], current_project: Dict[str, Any]):
            existing = {src["path"] for src in current_project.get("sources", [])}
            start_index = len(current_project.get("sources", []))
            if files:
                for i, file_item in enumerate(files):
                    path = getattr(file_item, "name", str(file_item))
                    if path in existing:
                        continue
                    meta = _video_meta(path)
                    current_project["sources"].append(
                        {"id": f"src{start_index + i}", "path": path, "sha256": _sha256(path), **meta}
                    )
            gallery = []
            for src in current_project.get("sources", []):
                thumb = _ensure_thumb(src["path"], thumbs_dir)
                gallery.append((thumb, f"{os.path.basename(src['path'])} | {src['duration']:.2f}s | {src['width']}x{src['height']}"))
            return gallery, current_project

        def _on_gallery_select(event: gr.SelectData, current_project: Dict[str, Any]):
            idx = int(event.index) if event and event.index is not None else -1
            if idx < 0 or idx >= len(current_project.get("sources", [])):
                return {"selected_source_id": None}
            return {"selected_source_id": current_project["sources"][idx]["id"]}

        def _append_to_v1(selection: Dict[str, Any], current_project: Dict[str, Any]):
            selected_id = selection.get("selected_source_id")
            if not selected_id:
                return current_project, _project_to_timeline_value(current_project)
            fps = int(current_project.get("fps", 30))
            v1_clips = [clip for clip in current_project.get("clips", []) if clip["track_id"] == "V1"]
            current_end = max([int(clip["timeline_end"]) for clip in v1_clips], default=0)
            source = next(src for src in current_project["sources"] if src["id"] == selected_id)
            duration_frames = _sec_to_frames(float(source["duration"]), fps)
            clip = Clip(
                id=f"clip{len(current_project['clips']) + 1}",
                source_id=selected_id,
                track_id="V1",
                timeline_start=current_end,
                timeline_end=current_end + duration_frames,
                src_in=0,
                src_out=duration_frames,
            )
            current_project["clips"].append(asdict(clip))
            _log("DEBUG", "TL-EDL-ADD-001", "clip appended", clip_id=clip.id)
            return current_project, _project_to_timeline_value(current_project)

        def _on_timeline_input(value: Dict[str, Any], current_project: Dict[str, Any]):
            updated = _apply_timeline_edits(current_project, value)
            return updated, updated

        def _on_export(current_project: Dict[str, Any], preset_value: str, embed_value: bool):
            _log("INFO", "TL-EXPORT-START-001", "Export requested", preset=preset_value, embed=embed_value)
            yield from _export_ffmpeg(current_project, preset_value, bool(embed_value), export_dir)

        with gr.Tab("Timeline Editor", id="timeline_editor"):
            if VisTimeline is None:
                gr.Markdown(
                    "⚠️ `gradio_vistimeline` est requis. Installez `pip install -r requirements.txt` puis redémarrez Wan2GP."
                )
                return

            if PREMIERE_UI:
                _log("DEBUG", "TL-UI-001", "Premiere UI injection enabled", enabled=PREMIERE_UI)
                gr.HTML(f"<style>{_premiere_css()}</style>")
                gr.HTML(f"<script>{_premiere_js()} tl_enablePremiereMode('timeline_editor_root');</script>")

            with gr.Column(elem_id="timeline_editor_root", elem_classes=["tl-root"]):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Media Bin")
                        file_input = gr.File(label="Importer des clips", file_count="multiple")
                        refresh_btn = gr.Button("Refresh Media Bin")
                        media_gallery = gr.Gallery(label="Clips", columns=2, height=420)
                        add_clip_btn = gr.Button("Add selected to V1 (append)")

                    with gr.Column(scale=6):
                        gr.Markdown("### Timeline")
                        timeline = VisTimeline(
                            value=_project_to_timeline_value(_empty_project()),
                            options={
                                "editable": True,
                                "stack": True,
                                "zoomable": True,
                                "showCurrentTime": False,
                            },
                            elem_id="timeline_editor_timeline",
                            preserve_old_content_on_value_change=True,
                        )
                        project_json = gr.JSON(label="Project (EDL JSON)", value=_empty_project())

                    with gr.Column(scale=3):
                        gr.Markdown("### Inspector")
                        gr.Markdown("MVP: drag/resize clips in timeline. Multi-track inspector bindings arrive next.")

                with gr.Row():
                    with gr.Column(scale=6):
                        preview = gr.Video(label="Last export preview")
                    with gr.Column(scale=4):
                        preset = gr.Dropdown(
                            choices=["vertical_9x16_1080p", "vertical_9x16_720p"],
                            value="vertical_9x16_1080p",
                            label="Preset",
                        )
                        embed_json = gr.Checkbox(value=True, label="Embed project JSON into output")
                        export_btn = gr.Button("Render (FFmpeg)")
                        status = gr.Textbox(label="Status", lines=6)

            refresh_btn.click(_rebuild_media_bin, inputs=[project_state], outputs=[media_gallery, project_state])
            file_input.change(_import_files, inputs=[file_input, project_state], outputs=[media_gallery, project_state])
            media_gallery.select(_on_gallery_select, inputs=[project_state], outputs=[selection_state])
            add_clip_btn.click(_append_to_v1, inputs=[selection_state, project_state], outputs=[project_state, timeline])
            timeline.input(_on_timeline_input, inputs=[timeline, project_state], outputs=[project_state, project_json])
            export_btn.click(_on_export, inputs=[project_state, preset, embed_json], outputs=[preview, status, preview])


# Optional alias some loaders/tools may expect.
Plugin = TimelineEditorPlugin
