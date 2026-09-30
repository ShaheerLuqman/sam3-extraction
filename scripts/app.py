"""
SAM 3 interactive UI (Gradio) — runs on the local native `sam3` package + sam3.pt.

Tabs:
  1. Image · concept     — type a text phrase -> every matching instance
  2. Image · interactive — draw include/exclude boxes (+ optional text) -> refine live
  3. Video · track        — text and/or box on a frame -> mask + track every instance

Launch:  .venv\\Scripts\\python.exe app.py  [--ckpt checkpoints\\sam3.pt] [--share]
"""
import argparse
import glob
import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

CKPT = None            # set from CLI
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_image_model = None    # built with interactivity; shared by tabs 1 & 2
_video_predictor = None


def _autocast():
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if _DEVICE == "cuda" else nullcontext())


def get_image_model():
    global _image_model
    if _image_model is None:
        from sam3.model_builder import build_sam3_image_model
        kw = {"checkpoint_path": CKPT, "load_from_HF": False} if CKPT else {}
        _image_model = build_sam3_image_model(device=_DEVICE, **kw)
    return _image_model


def get_video_predictor():
    global _video_predictor
    if _video_predictor is None:
        from sam3.model_builder import build_sam3_video_predictor
        # single GPU: run_video tweaks model attributes (hotstart) in-process, which
        # the multi-GPU predictor's worker processes wouldn't see.
        gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else None
        kw = {"checkpoint_path": CKPT} if CKPT else {}
        _video_predictor = build_sam3_video_predictor(gpus_to_use=gpus, **kw)
    return _video_predictor


def _color(i):
    rng = np.random.default_rng(int(i) + 12345)
    return rng.integers(60, 256, size=3).astype(np.float32)


def _video_writer(path_no_ext, fps, size):
    """Prefer browser-playable WebM/VP8; fall back to mp4v if VP8 is unavailable."""
    webm = path_no_ext + ".webm"
    vw = cv2.VideoWriter(webm, cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if vw.isOpened():
        return vw, webm
    mp4 = path_no_ext + ".mp4"
    return cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, size), mp4


def _overlay(rgb, masks, labels=None, alpha=0.5):
    out = rgb.astype(np.float32)
    for k, m in enumerate(masks):
        m = np.squeeze(np.asarray(m)).astype(bool)
        if not m.any():
            continue
        out[m] = (1 - alpha) * out[m] + alpha * _color(k if labels is None else labels[k])
    return out.clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- #
# Tab 1 — concept segmentation
# ----------------------------------------------------------------------------- #
def run_concept(image, text, box_str, threshold):
    if image is None:
        return None, "Upload an image."
    if not text and not box_str.strip():
        return image, "Enter a text phrase and/or a box (x1,y1,x2,y2)."

    from sam3.model.sam3_image_processor import Sam3Processor
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    W, H = pil.size
    proc = Sam3Processor(get_image_model(), device=_DEVICE, confidence_threshold=threshold)

    with _autocast():
        state = proc.set_image(pil)
        if text:
            state = proc.set_text_prompt(text, state)
        if box_str.strip():
            x1, y1, x2, y2 = [float(v) for v in box_str.replace(" ", "").split(",")]
            cxcywh = [((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H,
                      abs(x2 - x1) / W, abs(y2 - y1) / H]
            state = proc.add_geometric_prompt(cxcywh, True, state)

    masks = state.get("masks")
    n = 0 if masks is None else len(masks)
    canvas = _overlay(np.array(pil), [m.cpu().numpy() for m in masks] if n else [])
    if n:
        for i in range(n):
            x1, y1, x2, y2 = state["boxes"][i].cpu().tolist()
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    return canvas, f"Found {n} instance(s)."


# ----------------------------------------------------------------------------- #
# Tab 2 — interactive exemplar boxes (click two corners; include / exclude)
# ----------------------------------------------------------------------------- #
def iv_set_image(image, state):
    state = {"img": image, "boxes": [], "pending": None, "thr": 0.5}
    msg = "Click one corner, then the opposite corner, to draw a box."
    return image, state, (msg if image is not None else "Upload an image.")


def iv_click(state, mode, thr, evt: gr.SelectData):
    if not state or state.get("img") is None:
        return None, state, "Upload an image first."
    state["thr"] = thr
    x, y = float(evt.index[0]), float(evt.index[1])
    if state["pending"] is None:
        state["pending"] = (x, y)
        canvas = state["img"].copy()
        cv2.drawMarker(canvas, (int(x), int(y)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        return canvas, state, "Now click the opposite corner."
    x0, y0 = state["pending"]
    state["pending"] = None
    box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y),
           1 if mode.startswith("include") else 0)
    if abs(box[2] - box[0]) < 3 or abs(box[3] - box[1]) < 3:
        return state["img"].copy(), state, "Box too small — try again."
    state["boxes"].append(box)
    return _iv_run(state)


def iv_text(state, text, thr):
    if not state or state.get("img") is None:
        return None, state, "Upload an image first."
    state["thr"] = thr
    state["text"] = text
    return _iv_run(state)


def iv_undo(state):
    if state and state["boxes"]:
        state["boxes"].pop()
    return _iv_run(state) if state else (None, state, "")


def iv_clear(state):
    if state:
        state["boxes"], state["pending"] = [], None
        state.pop("text", None)
    return (state["img"] if state else None), state, "Cleared."


def _iv_run(state):
    from sam3.model.sam3_image_processor import Sam3Processor
    img = state["img"]
    H, W = img.shape[:2]
    text = state.get("text") or ""
    proc = Sam3Processor(get_image_model(), device=_DEVICE,
                         confidence_threshold=state.get("thr", 0.5))
    with _autocast():
        st = proc.set_image(Image.fromarray(img))
        if text:
            st = proc.set_text_prompt(text, st)
        for x1, y1, x2, y2, lab in state["boxes"]:
            cxcywh = [((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H,
                      abs(x2 - x1) / W, abs(y2 - y1) / H]
            st = proc.add_geometric_prompt(cxcywh, bool(lab), st)

    masks = st.get("masks")
    n = 0 if masks is None else len(masks)
    canvas = _overlay(img, [m.cpu().numpy() for m in masks] if n else [])
    for x1, y1, x2, y2, lab in state["boxes"]:
        c = (0, 220, 0) if lab else (255, 60, 60)
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
    tail = f' · text "{text}"' if text else ""
    return canvas, state, f"{n} instance(s) · {len(state['boxes'])} box(es){tail}"


# ----------------------------------------------------------------------------- #
# Tab 3 — video tracking
# ----------------------------------------------------------------------------- #
def _read_frames(path, max_frames):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _count_instances(out):
    if not out:
        return 0
    oids = out["out_obj_ids"].tolist()
    masks = out["out_binary_masks"]
    return sum(1 for i in range(len(oids))
              if np.squeeze(masks[i].cpu().numpy() if hasattr(masks[i], "cpu")
                            else np.asarray(masks[i])).any())


def run_video(video, text, box_str, prompt_frame, max_frames, det_thr,
              progress=gr.Progress()):
    if video is None:
        return None, "Upload a video."
    if not text and not box_str.strip():
        return None, "Enter a text phrase and/or a box."

    frames, fps = _read_frames(video, int(max_frames))
    if not frames:
        return None, "Could not read frames."
    H, W = frames[0].shape[:2]
    pf = max(0, min(int(prompt_frame), len(frames) - 1))

    predictor = get_video_predictor()
    try:
        predictor.model.score_threshold_detection = float(det_thr)
    except Exception:
        pass
    # A box with no text is a pure visual prompt (detector fires only on the prompt
    # frame); SAM3's hotstart heuristic then suppresses every instance for the whole
    # clip. Disable it in that case, restore the defaults otherwise.
    visual_only = bool(box_str.strip()) and not text
    for attr, on in (("hotstart_delay", 15), ("hotstart_unmatch_thresh", 8),
                     ("hotstart_dup_thresh", 8)):
        if hasattr(predictor.model, attr):
            setattr(predictor.model, attr, 0 if visual_only else on)
    progress(0.1, desc="starting session")
    sid = predictor.handle_request(
        request=dict(type="start_session", resource_path=video)
    )["session_id"]

    req = dict(type="add_prompt", session_id=sid, frame_index=pf)
    if text:
        req["text"] = text
    if box_str.strip():
        x1, y1, x2, y2 = [float(v) for v in box_str.replace(" ", "").split(",")]
        # predictor wants [xmin, ymin, w, h] normalised to [0, 1] w.r.t. this video
        nb = [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]
        if min(nb) < 0 or nb[0] + nb[2] > 1.001 or nb[1] + nb[3] > 1.001:
            return None, (f"Box {int(x1)},{int(y1)},{int(x2)},{int(y2)} is outside "
                          f"this video's {W}x{H} frame — are the coords from a "
                          f"different resolution?")
        req["bounding_boxes"] = [[max(0.0, min(1.0, v)) for v in nb]]
        req["bounding_box_labels"] = [1]

    per_frame = {}
    with _autocast():  # box/visual-prompt path isn't autocast-wrapped upstream
        prompt_resp = predictor.handle_request(request=req)
        n_prompt = _count_instances(prompt_resp.get("outputs"))
        progress(0.2, desc=f"prompt frame: {n_prompt} found — propagating")
        for resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=sid)
        ):
            per_frame[resp["frame_index"]] = resp["outputs"]

    vw, out_path = _video_writer(
        os.path.join(os.path.dirname(video), "sam3_tracked"), fps, (W, H))
    ids = set()
    for idx, rgb in enumerate(frames):
        out = per_frame.get(idx)
        canvas = rgb.astype(np.float32)
        if out is not None:
            oids = out["out_obj_ids"].tolist()
            for i, oid in enumerate(oids):
                m = out["out_binary_masks"][i]
                m = np.squeeze(m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(bool)
                if not m.any():
                    continue
                ids.add(oid)
                canvas[m] = 0.5 * canvas[m] + 0.5 * _color(oid)
                ys, xs = np.where(m)
                cv2.putText(canvas, f"#{oid}", (int(xs.min()), max(int(ys.min()) - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(cv2.cvtColor(canvas.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()
    predictor.handle_request(dict(type="close_session", session_id=sid))

    msg = (f"Prompt frame {pf}: **{n_prompt}** detected · "
           f"tracked **{len(ids)}** instance(s) across {len(frames)} frames.")
    if n_prompt == 0:
        msg += ("\n\nNothing matched at the prompt frame. Try: a different "
                "**prompt frame index** (where the object is clearly visible), "
                "a lower **detection threshold**, a simpler **text phrase**, or "
                "check the box is in *this* video's pixel space.")
    return out_path, msg


# ----------------------------------------------------------------------------- #
# UI
# ----------------------------------------------------------------------------- #
def build_ui():
    with gr.Blocks(title="SAM 3") as demo:
        gr.Markdown("## SAM 3 — concept segmentation, interactive masking, video tracking")

        with gr.Tab("Image · concept"):
            with gr.Row():
                with gr.Column():
                    c_img = gr.Image(label="image", type="numpy")
                    c_text = gr.Textbox(label="concept phrase", placeholder="yellow forklift")
                    c_box = gr.Textbox(label="exemplar box  x1,y1,x2,y2 (optional)")
                    c_thr = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="threshold")
                    c_run = gr.Button("segment", variant="primary")
                with gr.Column():
                    c_out = gr.Image(label="result")
                    c_status = gr.Markdown()
            c_run.click(run_concept, [c_img, c_text, c_box, c_thr], [c_out, c_status])

        with gr.Tab("Image · interactive"):
            iv_state = gr.State(None)
            with gr.Row():
                with gr.Column():
                    iv_img = gr.Image(label="click two corners to draw a box", type="numpy")
                    iv_mode = gr.Radio(
                        ["include (match this)", "exclude (not this)"],
                        value="include (match this)", label="next box is")
                    iv_txt = gr.Textbox(label="text phrase (optional)")
                    iv_thr = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="threshold")
                    with gr.Row():
                        iv_undo_b = gr.Button("undo box")
                        iv_clr = gr.Button("clear")
                with gr.Column():
                    iv_out = gr.Image(label="result")
                    iv_status = gr.Markdown()
            iv_img.upload(iv_set_image, [iv_img, iv_state], [iv_out, iv_state, iv_status])
            iv_img.select(iv_click, [iv_state, iv_mode, iv_thr], [iv_out, iv_state, iv_status])
            iv_txt.submit(iv_text, [iv_state, iv_txt, iv_thr], [iv_out, iv_state, iv_status])
            iv_thr.release(iv_text, [iv_state, iv_txt, iv_thr], [iv_out, iv_state, iv_status])
            iv_undo_b.click(iv_undo, [iv_state], [iv_out, iv_state, iv_status])
            iv_clr.click(iv_clear, [iv_state], [iv_out, iv_state, iv_status])

        with gr.Tab("Video · track"):
            with gr.Row():
                with gr.Column():
                    v_vid = gr.Video(label="video")
                    v_text = gr.Textbox(label="concept phrase", placeholder="person")
                    v_box = gr.Textbox(label="box on prompt frame  x1,y1,x2,y2 (optional)")
                    v_pf = gr.Number(value=0, label="prompt frame index", precision=0)
                    v_max = gr.Slider(10, 600, value=120, step=10, label="max frames")
                    v_dthr = gr.Slider(0.05, 0.9, value=0.5, step=0.05,
                                       label="detection threshold (lower = more objects)")
                    v_run = gr.Button("track", variant="primary")
                with gr.Column():
                    v_out = gr.Video(label="tracked")
                    v_status = gr.Markdown()
            v_run.click(run_video, [v_vid, v_text, v_box, v_pf, v_max, v_dthr],
                        [v_out, v_status])

    return demo


if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_root, "checkpoints", "sam3.pt"))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    CKPT = a.ckpt if os.path.isfile(a.ckpt) else None
    if CKPT is None:
        print(f"[warn] {a.ckpt} not found — will try HF auto-download")
    build_ui().launch(server_port=a.port, share=a.share)
