"""Run one of the research scripts in this folder, unchanged — in the *qwen* venv.

    <qwen venv>/bin/python launch.py <extract_frames|embed_frames|match_frames|vlm_classify> ARGS...

is `python <script>.py ARGS...`, exactly as qwen_vl/extraction_project ran it. The
four scripts beside this file are byte-identical copies of the research's
(scripts/*.py); do not edit them. Two opt-in hooks, both off unless the backend
sets them:

  FAWADSEG_GPU_UTIL  vLLM's gpu_memory_utilization instead of the script's own
                     (0.9 embedding, 0.92 VLM), for when the card has less free
                     memory than that. It only sizes the KV cache.
  FAWADSEG_NO_GT     the target video has no labels (a placeholder preds.json with
                     none stands in): the selected-frames video is drawn without
                     the GT / TP / FP text, which would be meaningless.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ("extract_frames", "embed_frames", "match_frames", "vlm_classify")


def _patch_gpu_util(util: float) -> None:
    import vllm

    base = vllm.LLM

    class LLM(base):
        def __init__(self, *args, **kwargs):
            kwargs["gpu_memory_utilization"] = util
            super().__init__(*args, **kwargs)

    vllm.LLM = LLM  # the scripts do `from vllm import LLM` after this


def _patch_no_gt() -> None:
    import cv2
    import match_frames

    def write_video(path, frames_dir, chosen, score_by_frame, gt_by_frame, label_by_frame, fps,
                    id2step):
        """match_frames.write_video without the ground-truth overlay."""
        first = cv2.imread(os.path.join(frames_dir, f"{chosen[0]:06d}.jpg"))
        h, w = first.shape[:2]
        tmp = path + ".tmp.mp4"
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in chosen:
            im = cv2.imread(os.path.join(frames_dir, f"{f:06d}.jpg"))
            cv2.rectangle(im, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(im, f"frame {f}  score {score_by_frame[f]:.2f}", (6, 20), 0, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(im)
        vw.release()
        rc = os.system(f'/usr/bin/ffmpeg -y -v error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -crf 23 "{path}"')
        if rc == 0:
            os.remove(tmp)
        else:
            os.replace(tmp, path)

    # vlm_classify does `from match_frames import write_video`, and match_frames.main
    # looks it up in its own module, so both see this one
    match_frames.write_video = write_video


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    name, args = sys.argv[1], sys.argv[2:]
    sys.path.insert(0, HERE)  # vlm_classify imports match_frames
    if os.environ.get("FAWADSEG_GPU_UTIL"):
        _patch_gpu_util(float(os.environ["FAWADSEG_GPU_UTIL"]))
    if os.environ.get("FAWADSEG_NO_GT"):
        _patch_no_gt()
    mod = importlib.import_module(name)
    sys.argv = [os.path.join(HERE, name + ".py"), *args]
    if name == "extract_frames":  # its __main__ block passes argv positionally
        if len(args) != 2:
            print(mod.__doc__, file=sys.stderr)
            sys.exit(1)
        mod.main(*args)
    else:
        mod.main()


if __name__ == "__main__":
    main()
