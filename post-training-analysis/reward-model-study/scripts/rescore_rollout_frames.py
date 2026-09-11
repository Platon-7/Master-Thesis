"""Re-score live IBRL rollout frames offline, to test whether the on-policy AUC
collapse is a FRAME-FORMAT artifact (raw render vs the mp4-encoded frames the
reward model trained on) rather than a real content/distribution shift.

For each dumped episode (ep*.npz with frames,label,sp), score the SAME frames
three ways with the SAME scorer the offline eval used (get_robometer_4b):
  1. RAW          — frames as-is (should reproduce the live on-policy sp/AUC)
  2. MP4-ROUNDTRIP— encode to h264 mp4 + decode (the curated-data pipeline)
  3. JPEG-ROUNDTRIP— per-frame JPEG q=90 (a lighter compression control)

If AUC jumps RAW->MP4, the live IBRL scoring path is feeding the model
out-of-distribution (uncompressed) frames and the negative result is an
artifact. If all three stay ~chance, the collapse is genuine content shift.

  python rescore_rollout_frames.py --frames-dir <dir> --model <ckpt> --out <jsonl> [--limit N]
"""
import argparse, io, json, os, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/vlm_ibrl")
sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")
from env.robometer_utils import get_robometer_4b
import imageio.v3 as iio
from PIL import Image


def auc(p, n):
    p = np.asarray(p); n = np.asarray(n)
    if not len(p) or not len(n):
        return float("nan")
    a = np.concatenate([p, n]); o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def mp4_roundtrip(frames, size=None):
    # Encode to h264/yuv420p (the curated-data codec) and decode back. If `size`
    # is given, resize first (curated metaworld frames are 240x240; live are 224).
    arr = np.stack([np.asarray(f).astype(np.uint8) for f in frames])
    if size is not None and (arr.shape[1] != size or arr.shape[2] != size):
        arr = np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.BILINEAR)) for x in arr])
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    try:
        # Encode via the imageio FFMPEG plugin (bundled ffmpeg binary) — pyav's
        # imwrite segfaults (SIGBUS) mid-run; FFMPEG is stable. Decode via pyav
        # imiter (imread trips the av-close bug; imiter does not).
        iio.imwrite(path, arr, plugin="FFMPEG", codec="libx264", pixelformat="yuv420p", fps=10)
        dec = [np.asarray(f) for f in iio.imiter(path, plugin="pyav")]
    finally:
        os.unlink(path)
    return dec


def jpeg_roundtrip(frames):
    out = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(f).astype(np.uint8)).save(buf, format="JPEG", quality=90)
        buf.seek(0)
        out.append(np.asarray(Image.open(buf).convert("RGB")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    eps = sorted(Path(args.frames_dir).glob("ep*.npz"))
    if args.limit:
        eps = eps[: args.limit]
    print(f"{len(eps)} episodes in {args.frames_dir}", flush=True)
    scorer = get_robometer_4b(model_path=args.model, max_frames=args.max_frames)
    print(f"scorer max_frames = {getattr(scorer, 'max_frames', '?')}", flush=True)

    recs = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, p in enumerate(eps):
            d = np.load(p)
            frames = [d["frames"][k] for k in range(d["frames"].shape[0])]
            label = int(d["label"]); sp_live = float(d["sp"])
            def sc(fr):
                return float(scorer([Image.fromarray(np.asarray(x).astype(np.uint8)) for x in fr],
                                    task="push the coffee cup", icl_frames=None)["success_prob"])
            rec = {"ep": p.stem, "label": label, "sp_live": sp_live,
                   "sp_raw": sc(frames),                       # raw 224 (should ~= live)
                   "sp_mp4": sc(mp4_roundtrip(frames)),        # h264 @ 224 (isolate codec)
                   "sp_cur": sc(mp4_roundtrip(frames, size=240)),  # h264 @ 240 (full curated pipeline)
                   "sp_jpg": sc(jpeg_roundtrip(frames))}       # jpeg @ 224 (lighter codec control)
            recs.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                L = np.array([r["label"] for r in recs])
                for k in ["sp_live", "sp_raw", "sp_mp4", "sp_cur", "sp_jpg"]:
                    v = np.array([r[k] for r in recs])
                    print(f"  [{i+1}/{len(eps)}] {k:7s} AUC={auc(v[L==1], v[L==0]):.3f}", flush=True)

    L = np.array([r["label"] for r in recs])
    print(f"\n==== FINAL ({len(recs)} eps, {int(L.sum())} succ / {int((L==0).sum())} fail) ====")
    for k in ["sp_live", "sp_raw", "sp_mp4", "sp_cur", "sp_jpg"]:
        v = np.array([r[k] for r in recs])
        print(f"  {k:8s}  AUC={auc(v[L==1], v[L==0]):.4f}   succ_mean={v[L==1].mean():.3f}  fail_mean={v[L==0].mean():.3f}")
    print("\n  sp_live = on-policy number (raw 224 frames, live scorer)")
    print("  sp_raw  = sanity: offline scorer on raw 224 frames (should ~= sp_live)")
    print("  sp_mp4  = h264 @ 224 — isolates the codec")
    print("  sp_cur  = h264 @ 240 — the full curated-data pipeline (codec + resolution)")
    print("  If sp_cur >> sp_raw -> the on-policy collapse is a frame-format artifact, not content shift.")


if __name__ == "__main__":
    main()
