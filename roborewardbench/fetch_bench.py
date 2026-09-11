"""Resilient fetch of the RoboRewardBench test split (teetone/RoboReward, test/).

snapshot_download swallows a 429 on the repo-info call and silently returns the
partial local dir, so this walks the file list itself, downloads what is missing,
and backs off on rate limits. Idempotent: rerun to continue.

Uses ~/.cache/huggingface/token automatically if present (HF_TOKEN also works),
which lifts the anonymous per-IP limit.
"""
import os, sys, time, random
from huggingface_hub import HfApi, hf_hub_download

REPO = "teetone/RoboReward"
SPLIT = os.environ.get("RRB_SPLIT", "test")
LOCAL = os.environ.get("RRB_LOCAL", "/scratch-shared/%s/RoboReward" % os.environ["USER"])
DEADLINE = time.time() + float(os.environ.get("RRB_MAX_SECONDS", 20000))

api = HfApi()
files = None
for a in range(20):
    try:
        files = [s.rfilename for s in api.dataset_info(REPO).siblings if s.rfilename.startswith(SPLIT + "/")]
        break
    except Exception as e:
        print(f"[list] attempt {a}: {type(e).__name__}", flush=True)
        time.sleep(min(300, 20 * (a + 1)))
if files is None:
    sys.exit("could not list repo files")
print(f"{len(files)} files in {SPLIT}/", flush=True)

missing = [f for f in files if not os.path.exists(os.path.join(LOCAL, f))]
print(f"{len(files)-len(missing)} already local, {len(missing)} to fetch", flush=True)

fails = 0
for i, f in enumerate(missing):
    if time.time() > DEADLINE:
        print("deadline reached, stopping early", flush=True); break
    for a in range(8):
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=LOCAL)
            fails = 0
            break
        except Exception as e:
            is429 = "429" in str(e) or "Too Many Requests" in str(e)
            wait = min(600, (60 if is429 else 5) * (a + 1)) + random.uniform(0, 5)
            print(f"[{i}/{len(missing)}] {type(e).__name__} {'429' if is429 else ''} -> sleep {wait:.0f}s", flush=True)
            time.sleep(wait)
    else:
        fails += 1
        if fails > 20:
            sys.exit("too many consecutive failures")
    if i % 100 == 0:
        print(f"progress {i}/{len(missing)}", flush=True)

have = sum(1 for f in files if os.path.exists(os.path.join(LOCAL, f)))
print(f"DONE have {have}/{len(files)}", flush=True)
sys.exit(0 if have == len(files) else 3)
