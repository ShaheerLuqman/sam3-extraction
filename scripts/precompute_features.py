"""
Precompute the SAM 3 image-backbone feature cache for a video.

This is optional - `track_bbox.py` builds the same cache automatically on the
first run for a video. Use this when you want to build the cache ahead of time
(e.g. overnight) or with a specific --stride / location.

  python scripts/precompute_features.py --video inputs/clip.mp4
  python scripts/precompute_features.py --video inputs/long.mp4 --stride 5 \
      --out-dir "features/long s5"

Then: python scripts/track_bbox.py --video inputs/clip.mp4 --boxes-json boxes.json
(which reuses the cache under features/<video filename>/, or --features-dir).
"""
import argparse
from pathlib import Path

from track_bbox import build_feature_cache


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sam3.pt"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="cache dir (default: features/<video filename>)")
    p.add_argument("--stride", type=int, default=1,
                   help="cache every Nth frame (default 1 = all)")
    args = p.parse_args()

    out_dir = args.out_dir or (Path("features") / args.video.name)
    build_feature_cache(args.video, out_dir, args.stride, args.ckpt)


if __name__ == "__main__":
    main()
