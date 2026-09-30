"""
Merge a text_track_probe.py retry pass into the main run and write report.md.

  * copies <merge-dir>/*.mp4 over <dir>/*.mp4 (retry wins for those classes)
  * merges <merge-dir>/*_textrefs.json entries into <dir>/*_textrefs.json
  * writes <dir>/report.md: per-class prompt, SAM 3 result (instances / frames /
    ref frame / box / score) next to the YOLO detector's numbers for context

  python scripts/textprobe_report.py \
      --dir "outputs/textprobe/2026-08-03 06_51_58" \
      --merge-dir "outputs/textprobe/2026-08-03 06_51_58_retry" \
      --dets "inputs/datasets/Stellantis Station 140 (3.3)/Videos/2026-08-03 06_51_58.mp4_dets.json" \
      --phrases scripts/phrases_stellantis.json scripts/phrases_stellantis_retry.json
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="main text_track_probe out dir")
    p.add_argument("--merge-dir", type=Path, default=None, help="retry out dir to fold in")
    p.add_argument("--dets", required=True, type=Path, help="the video's *_dets.json")
    p.add_argument("--phrases", nargs="*", type=Path, default=[],
                   help="{class: phrase} maps; later files win (retry last)")
    p.add_argument("--skip", nargs="*", default=[],
                   help="class names deliberately not probed (shown as 'skipped')")
    return p.parse_args()


def load_textrefs(d: Path):
    if d is None:
        return {}
    f = next(d.glob("*_textrefs.json"), None)
    if not f:
        return {}
    return {e["name"]: e for e in json.loads(f.read_text())}


def yolo_stats(dets_path: Path):
    d = json.loads(dets_path.read_text())
    names = d["meta"]["names"]
    fc = int(d["meta"]["frame_count"])
    det = d["detections"]
    ndet, frames, peak = Counter(), defaultdict(set), defaultdict(int)
    first = {}
    for k in sorted(det, key=int):
        fi = int(k)
        pc = Counter()
        for b in det[k]:
            c = int(b[5])
            ndet[c] += 1
            frames[c].add(fi)
            pc[c] += 1
            first.setdefault(c, fi)
        for c, v in pc.items():
            peak[c] = max(peak[c], v)
    stats = {}
    for cs, nm in names.items():
        c = int(cs)
        stats[nm] = dict(id=c, dets=ndet.get(c, 0), frames=len(frames.get(c, ())),
                         peak=peak.get(c, 0), first=first.get(c))
    return names, fc, stats


def main():
    a = parse_args()
    main_dir, merge_dir = a.dir, a.merge_dir

    phrase_map = {}
    for pf in a.phrases:
        phrase_map.update(json.loads(pf.read_text()))

    names, total_frames, ystats = yolo_stats(a.dets)

    main_refs = load_textrefs(main_dir)
    retry_refs = load_textrefs(merge_dir)

    # 1) copy retry videos + merge json entries
    merged_from = []
    if merge_dir:
        for mp4 in sorted(merge_dir.glob("*.mp4")):
            shutil.copy2(mp4, main_dir / mp4.name)
            merged_from.append(mp4.stem)
        for nm, e in retry_refs.items():
            main_refs[nm] = e
    if merged_from:
        print(f"copied {len(merged_from)} videos from {merge_dir.name}: "
              f"{', '.join(merged_from)}")

    order = sorted(names.items(), key=lambda kv: int(kv[0]))
    merged_json = [main_refs[nm] for _, nm in order if nm in main_refs]
    ref_file = next(main_dir.glob("*_textrefs.json"))
    ref_file.write_text(json.dumps(merged_json, indent=2) + "\n")
    print(f"wrote {ref_file.name}  ({len(merged_json)}/{len(order)} classes with a SAM3 ref)")

    # 2) report.md
    L = []
    L.append(f"# SAM 3 text-prompt probe — `{main_dir.name}`\n")
    L.append(f"Video: **{a.dets.stem}** · {total_frames} frames · "
             f"probe = SAM 3 concept detection + tracking, one text phrase per class, "
             f"prompt added on frame 0 and propagated over the whole clip.\n")
    L.append("`ref frame` / `box` / `score` = the single highest-scoring "
             "(frame, instance) SAM 3 produced for that phrase — the reference "
             "exemplar. `box` is `[x1,y1,x2,y2]` in the video's pixels "
             f"({total_frames} frames, stitched multi-camera view).\n")
    L.append("YOLO columns are the station's trained detector on the same clip, "
             "for comparison only.\n")

    hdr = ("| id | class | prompt | SAM3 found | peak inst | frames (SAM3) | "
           "ref frame | ref box | score | det thr | YOLO dets | YOLO frames | YOLO peak |")
    sep = "|---:|---|---|:--:|--:|--:|--:|---|--:|--:|--:|--:|--:|"
    L += [hdr, sep]

    n_found = 0
    for cs, nm in order:
        y = ystats[nm]
        e = main_refs.get(nm)
        prompt = (e or {}).get("prompt") or phrase_map.get(nm) or nm.replace("_", " ")
        thr = f'{e["det_thr"]:g}' if e and "det_thr" in e else "–"
        if nm in a.skip and not e:
            found = "_skipped_"
            peak = fr = rf = box = sc = "–"
        elif e and e.get("status") == "oom":
            found = "_OOM_"
            peak = fr = rf = box = sc = "–"
        elif e:
            n_found += 1
            found = "yes"
            peak = str(e["peak_instances"])
            fr = f'{e["frames_present"]}/{e.get("total_frames", total_frames)}'
            rf = str(e["frame"])
            box = "`" + ",".join(map(str, e["box"])) + "`"
            sc = f'{e["score"]:.3f}'
        else:
            found = "**no**"
            peak = fr = rf = box = sc = "–"
        L.append(f"| {y['id']} | `{nm}` | \"{prompt}\" | {found} | {peak} | {fr} "
                 f"| {rf} | {box} | {sc} | {thr} | {y['dets']} | {y['frames']} | {y['peak']} |")

    L.append("")
    L.append(f"**{n_found}/{len(order)} classes** produced a SAM 3 detection "
             f"(`OOM` = ran out of GPU memory, not probed).\n")

    # notes
    notes = []
    for cs, nm in order:
        e = main_refs.get(nm)
        if nm in a.skip and not e:
            continue
        if e and e.get("status") == "oom":
            notes.append(f"- `{nm}` — OOM: the phrase tracks too many instances "
                         f"to fit in GPU memory over the full clip.")
        elif e and e["score"] < 0.5:
            notes.append(f"- `{nm}` — weak (score {e['score']:.2f}); "
                         f"treat the box as unreliable.")
        elif not e and ystats[nm]["dets"] > 0:
            notes.append(f"- `{nm}` — SAM 3 found nothing though YOLO has "
                         f"{ystats[nm]['dets']} detections on {ystats[nm]['frames']} frames.")
    if notes:
        L.append("## Notes\n")
        L += notes
        L.append("")

    (main_dir / "report.md").write_text("\n".join(L))
    print(f"wrote {main_dir / 'report.md'}")


if __name__ == "__main__":
    main()
