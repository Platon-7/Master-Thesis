"""
find_split_failures.py

For split-part archives that have frames (not OXE embedding-only archives),
find which part contains index_mappings.json and count failures.

Usable split archives (confirmed to have frames/):
  - jesbu1_roboreward_rfm_roboreward_train (7 parts)
  - jesbu1_roboreward_rfm_high_res_roboreward_train (10 parts)
  - jesbu1_soar_rfm_soar_rfm (6 parts)
  - jesbu1_roboarena_0825_rfm_roboarena (2 parts)
  - jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist (4 parts)
  - jesbu1_failsafe_rfm_failsafe (3 parts)
  - anqil_rh20t_subset_rfm_rh20t_robot (10 parts) -- robot, not human
  - jesbu1_galaxea_rfm_galaxea_part1..5_r1_lite (2 parts each)
  - jesbu1_molmoact_rfm_molmoact_dataset_household (2 parts)

OXE archives are embedding-only — skip them.
"""

import os, sys, json, requests
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_URL = "https://huggingface.co/datasets/robometer/processed_datasets/resolve/main"

USABLE_SPLITS = {
    "jesbu1_roboreward_rfm_roboreward_train":
        [f"part-a{c}" for c in "abcdefg"],
    "jesbu1_roboreward_rfm_high_res_roboreward_train":
        [f"part-a{c}" for c in "abcdefghij"],
    "jesbu1_soar_rfm_soar_rfm":
        [f"part-a{c}" for c in "abcdef"],
    "jesbu1_roboarena_0825_rfm_roboarena":
        ["part-aa", "part-ab"],
    "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist":
        [f"part-a{c}" for c in "abcd"],
    "jesbu1_failsafe_rfm_failsafe":
        ["part-aa", "part-ab", "part-ac"],
    "anqil_rh20t_subset_rfm_rh20t_robot":
        [f"part-a{c}" for c in "abcdefghij"],
    "jesbu1_galaxea_rfm_galaxea_part1_r1_lite": ["part-aa", "part-ab"],
    "jesbu1_galaxea_rfm_galaxea_part2_r1_lite": ["part-aa", "part-ab"],
    "jesbu1_galaxea_rfm_galaxea_part3_r1_lite": ["part-aa", "part-ab"],
    "jesbu1_galaxea_rfm_galaxea_part4_r1_lite": ["part-aa", "part-ab"],
    "jesbu1_galaxea_rfm_galaxea_part5_r1_lite": ["part-aa", "part-ab"],
    "jesbu1_molmoact_rfm_molmoact_dataset_household": ["part-aa", "part-ab"],
}

# Also scan these single-file archives that failed with compression issues
SINGLE_COMPRESSED = [
    "jesbu1_racer_rfm_racer_train",
    "jesbu1_auto_eval_rfm_auto_eval_rfm",
    "jesbu1_h2r_rfm_h2r",
    "jesbu1_ph2d_rfm_ph2d",
    "jesbu1_roboreward_rfm_roboreward_val",
    "jesbu1_roboreward_rfm_roboreward_test",
    "abraranwar_libero_rfm_libero256_90",
    "jesbu1_molmoact_rfm_molmoact_dataset_tabletop",
    "jesbu1_oxe_rfm_eval_oxe_bc_z_eval",
    "jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval",
]

SEARCH_SIZES_MB = [200, 500, 1000]


def parse_tar_header(block):
    if len(block) < 512 or block[:512] == b"\x00" * 512:
        return None
    name = block[:100].split(b"\x00")[0].decode("utf-8", errors="replace").strip()
    prefix = block[345:500].split(b"\x00")[0].decode("utf-8", errors="replace").strip()
    if prefix:
        name = prefix + "/" + name
    size_field = block[124:136].split(b"\x00")[0].decode("ascii", errors="replace").strip()
    try:
        size = int(size_field, 8) if size_field else 0
    except ValueError:
        size = 0
    return name, size


def extract_counts_from_buffer(data, range_start):
    """Try to find and parse index_mappings.json in a raw byte buffer."""
    # First: plain text search to locate the file
    pos = data.rfind(b"index_mappings")
    if pos < 0:
        return None, "not-found"

    # Found the string — now find the tar header before it
    # Align to 512-byte boundary relative to full-file offset
    abs_offset = range_start + pos
    # Scan backwards in 512-byte blocks from pos
    align = 512 - (range_start % 512) if range_start % 512 else 0
    aligned_data = data[align:]
    aligned_base = range_start + align

    for i in range(0, len(aligned_data) - 512, 512):
        block = aligned_data[i:i+512]
        parsed = parse_tar_header(block)
        if not parsed:
            continue
        name, entry_size = parsed
        if "index_mappings.json" not in name:
            continue
        file_start = i + 512
        file_end = file_start + entry_size
        if file_end > len(aligned_data):
            return None, f"cut-off (need {entry_size/1024:.0f}KB)"
        try:
            mapping = json.loads(aligned_data[file_start:file_end])
        except json.JSONDecodeError as e:
            return None, f"json-error: {e}"
        q = mapping.get("quality_indices", {})
        failures = len(q.get("failure", []))
        total = sum(len(v) for v in q.values())
        return (failures, total), "ok"

    return None, "string-found-but-header-not-parsed"


def check_archive(session, url, label):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    rh = session.head(url, headers=headers, timeout=30, allow_redirects=True)
    if rh.status_code == 404:
        return None, "404"
    size = int(rh.headers.get("Content-Length", 0))

    for mb in SEARCH_SIZES_MB:
        fetch = min(mb * 1024 * 1024, size)
        range_start = size - fetch
        r = session.get(url, headers={**headers, "Range": f"bytes={range_start}-"}, timeout=180)
        data = r.content
        result, status = extract_counts_from_buffer(data, range_start)
        if result is not None:
            return result, f"ok@{mb}MB"
        if status == "not-found":
            continue
        return None, status  # cut-off or error — don't retry larger

    return None, f"not-found-in-{SEARCH_SIZES_MB[-1]}MB"


def main():
    session = requests.Session()
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    print(f"{'Archive':<70} {'Failures':>9} {'Total':>9} {'Rate':>6}  Status")
    print("-" * 105)

    total_failures = 0
    total_eps = 0

    # 1. Split archives (check last part first, then work backwards)
    print("\n-- SPLIT ARCHIVES (frames present) --")
    for archive, parts in USABLE_SPLITS.items():
        found = False
        for part_suffix in reversed(parts):
            url = f"{BASE_URL}/{archive}.tar.{part_suffix}"
            result, status = check_archive(session, url, f"{archive}.{part_suffix}")
            if result is not None:
                f, t = result
                total_failures += f
                total_eps += t
                rate = f"{f/t*100:.0f}%" if t else "100%"
                print(f"{archive:<70} {f:>9,} {t:>9,} {rate:>6}  {status} (in {part_suffix})")
                found = True
                break
            if status == "404":
                continue  # try next part
            if "not-found" not in status:
                print(f"{archive:<70} {'?':>9} {'?':>9} {'?':>6}  {status} (in {part_suffix})")
                found = True
                break
        if not found:
            print(f"{archive:<70} {'?':>9} {'?':>9} {'?':>6}  not-found in any part")

    # 2. Single-file archives that failed with "not-found-in-range" before
    print("\n-- SINGLE-FILE COMPRESSED ARCHIVES (retry with larger window) --")
    for archive in SINGLE_COMPRESSED:
        url = f"{BASE_URL}/{archive}.tar"
        result, status = check_archive(session, url, archive)
        if result is not None:
            f, t = result
            total_failures += f
            total_eps += t
            rate = f"{f/t*100:.0f}%" if t else "100%"
            print(f"{archive:<70} {f:>9,} {t:>9,} {rate:>6}  {status}")
        else:
            print(f"{archive:<70} {'?':>9} {'?':>9} {'?':>6}  {status}")

    print("-" * 105)
    print(f"\nNew failures found in this scan:    {total_failures:>9,}")
    print(f"Previously counted (single-file):    19,864")
    print(f"Grand total (lower bound):          {total_failures + 19864:>9,}")


if __name__ == "__main__":
    main()
