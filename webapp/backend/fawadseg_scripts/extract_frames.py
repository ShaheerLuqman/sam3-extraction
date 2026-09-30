"""Decode every readable frame of a video to JPEG and record the decodable range.

Usage:
    python extract_frames.py <video.mp4> <out_dir>

Writes <out_dir>/<frame_idx:06d>.jpg and <out_dir>/meta.json. Decoding is sequential,
so frame_idx matches the index used as key in the *_preds.json files.
"""
import json
import os
import sys

import cv2


def main(video: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    header_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    meta = {
        "video": os.path.abspath(video),
        "header_frames": header_frames,
        "decoded_frames": idx,
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
