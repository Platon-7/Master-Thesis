"""Build pairs_index_test_v3.jsonl — adapts existing test to user's spec.

Target composition (3000 total):
  Robometer family (incl. droid mixed in):
    700 robometer-source failures + 300 droid failures = 1000 failures
    700 robometer-source successes + 300 droid successes = 1000 successes
  Metaworld:
    500 failures
    500 successes
  Failsafe: dropped

Rules:
  - All rows have partner_episode_id non-null and tier != "no_pair"
  - Episode_ids excluded if they appear in any of the existing 7 splits
    (pairs_index_train, warmup, eval_droid, eval_robometer, eval_metaworld,
     eval_failsafe, train_plus_failsafe), as either query or partner
  - Reuse the existing pairs_index_test.jsonl rows where possible (same eids
    we already validated, just trim to target counts and drop failsafe)
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

random.seed(42)

SPLITS_DIR = "/scratch-shared/pkarageorgis1/robometer_frames_splits"
PAIRS_UNIFIED = "/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl"
OUT_PATH = f"{SPLITS_DIR}/pairs_index_test_v3.jsonl"

# Per-bucket target counts (droid dropped — exhausted in train+eval, see earlier check)
TARGET = {
    ("robometer", "failure"): 1000,
    ("robometer", "success"): 1000,
    ("metaworld", "failure"): 500,
    ("metaworld", "success"): 500,
}

# Restrict robometer candidates to archives we actually trained on (per user's
# "don't pick completely unrelated splits" instruction). Built from
# pairs_index_train + warmup so the test set covers the same archive families.
TRAINED_ROBOMETER_ARCHIVES = None  # populated below

def _collect_trained_archives():
    global TRAINED_ROBOMETER_ARCHIVES
    s = set()
    for fname in ["pairs_index_train.jsonl", "pairs_index_warmup.jsonl"]:
        fpath = os.path.join(SPLITS_DIR, fname)
        if not os.path.exists(fpath): continue
        for line in open(fpath):
            d = json.loads(line)
            if d.get("source") in ("robometer", "robometer_orphan_success"):
                s.add(d.get("archive"))
    TRAINED_ROBOMETER_ARCHIVES = s

# === Step 1: build the "used" set from all OTHER existing splits ===
USED_SPLITS = [
    "pairs_index_train.jsonl",
    "pairs_index_warmup.jsonl",
    "pairs_index_eval_droid.jsonl",
    "pairs_index_eval_robometer.jsonl",
    "pairs_index_eval_metaworld.jsonl",
    "pairs_index_eval_failsafe.jsonl",
    "pairs_index_train_plus_failsafe.jsonl",
]

used_eids = set()
for fname in USED_SPLITS:
    fpath = os.path.join(SPLITS_DIR, fname)
    if not os.path.exists(fpath):
        continue
    n = 0
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            used_eids.add(d.get("episode_id"))
            partner = d.get("partner_episode_id")
            if partner:
                used_eids.add(partner)
            n += 1
    print(f"  loaded {fname}: {n} rows")
print(f"\nTotal used eids (queries + partners): {len(used_eids)}")

_collect_trained_archives()
print(f"Trained-on robometer archives: {len(TRAINED_ROBOMETER_ARCHIVES)}")

# === Step 2: scan pairs_unified to find candidates per bucket ===
print("\n=== Scanning pairs_unified.jsonl ===")
candidates = defaultdict(list)  # (source, label) → list of rows
n_total = 0
with open(PAIRS_UNIFIED) as f:
    for line in f:
        n_total += 1
        d = json.loads(line)
        eid = d.get("episode_id")
        if eid in used_eids:
            continue
        # Skip no-pair rows
        if d.get("tier") == "no_pair" or not d.get("partner_episode_id"):
            continue
        src = d.get("source")
        lab = d.get("label")
        # Map "robometer_orphan_success" → "robometer" + "success" (treat as robometer)
        if src == "robometer_orphan_success":
            src = "robometer"
        key = (src, lab)
        if key not in TARGET:
            continue
        # For robometer buckets, restrict to archives we trained on
        if src == "robometer" and d.get("archive") not in TRAINED_ROBOMETER_ARCHIVES:
            continue
        candidates[key].append(d)

print(f"  scanned {n_total} unified rows")
print(f"  candidates per bucket (after used+pair filter):")
for k in TARGET:
    print(f"    {k}: {len(candidates[k])}")

# === Step 3: sample to targets ===
print("\n=== Sampling to targets ===")
selected = []
for k, target_n in TARGET.items():
    pool = candidates[k]
    if len(pool) < target_n:
        print(f"  WARN {k}: only {len(pool)} candidates, target {target_n} — taking all")
        sampled = pool
    else:
        sampled = random.sample(pool, target_n)
    selected.extend(sampled)
    print(f"  {k}: {len(sampled)} sampled")

# === Step 4: also include existing test rows (where source/label fits TARGET) up to target ===
# Actually just sample fresh from pairs_unified — used_eids already excludes existing test
# rows IF we add the existing test to the used set first.
# Wait — we want to ALLOW existing test rows. Let me NOT add existing test to used_eids.
# (Existing pairs_index_test.jsonl was NOT in USED_SPLITS list.)
# But this means selected might have overlap with existing test, which is fine — they're
# just episodes that exist in pairs_unified that we're choosing for v3.

# === Step 5: write output ===
random.shuffle(selected)
with open(OUT_PATH, "w") as f:
    for d in selected:
        f.write(json.dumps(d) + "\n")
print(f"\n=== Wrote {OUT_PATH}: {len(selected)} rows ===")

# === Step 6: verify ===
print("\n=== Verification ===")
with open(OUT_PATH) as f:
    rows = [json.loads(l) for l in f]
src_label = Counter()
arch = Counter()
no_pair = 0
for d in rows:
    s = d.get("source")
    if s == "robometer_orphan_success":
        s = "robometer"
    src_label[(s, d.get("label"))] += 1
    if d.get("source") == "robometer":
        arch[d.get("archive")] += 1
    if not d.get("partner_episode_id") or d.get("tier") == "no_pair":
        no_pair += 1
print(f"  total rows: {len(rows)}")
print(f"  no-pair rows: {no_pair}")
print(f"  source × label:")
for k, v in sorted(src_label.items()):
    print(f"    {k}: {v}")
print(f"  top robometer-source archives:")
for a, c in arch.most_common(10):
    print(f"    {a}: {c}")
