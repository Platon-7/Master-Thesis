#!/usr/bin/env python3
"""
generate_breakdown_md.py — read audit_report.json (+ pair report.json) and
emit ROBOMETER_BREAKDOWN.md answering every dataset-level question in one
authoritative document.

Inputs:
  --audit   /projects/prjs1958/robometer_full_dataset/audit_report.json
            (expected: produced by audit_all_archives.py --no-exclusions,
             covering every archive with no blind spots)
  --pairs   /projects/prjs1958/robometer_full_dataset/pairs/report.json
            (in-context pair counts per archive, tier histogram)

Output:
  --output  /gpfs/home3/pkarageorgis1/Master-Thesis/Real-World-Failures/Robometer/ROBOMETER_BREAKDOWN.md

Sections written:
  1. Totals: successes, failures, episodes
  2. Breakdown by robot type: humanoid / human-hand / standard-robot
  3. Orphan successes (archives with succ>0, fail=0)
  4. Orphan failures (archives with fail>0, succ=0)
  5. In-context learning pairs (from pair report)
  6. Per-archive table (every archive, sorted)
"""

import argparse
import datetime
import json
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────
# Robot-type categorization (authoritative)
#
# Humanoid embodiments and human-hand / egocentric datasets are enumerated
# explicitly. Everything else is a standard robot arm (including sim arms
# like MetaWorld and failsafe/playworld).
# ────────────────────────────────────────────────────────────────────────────

HUMANOID = {
    "abraranwar_agibotworld_alpha_rfm_agibotworld",
    "abraranwar_agibotworld_alpha_headcam_rfm_agibotworld",
    "jesbu1_galaxea_rfm_galaxea_part1_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part2_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part3_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part4_r1_lite",
    "jesbu1_galaxea_rfm_galaxea_part5_r1_lite",
    "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm",
}

HUMAN_HAND = {
    "jesbu1_egodex_rfm_egodex_part1", "jesbu1_egodex_rfm_egodex_part2",
    "jesbu1_egodex_rfm_egodex_part3", "jesbu1_egodex_rfm_egodex_part4",
    "jesbu1_egodex_rfm_egodex_part5", "jesbu1_egodex_rfm_egodex_test",
    "jesbu1_epic_rfm_epic",
    "anqil_rh20t_subset_rfm_rh20t_human",
    "jesbu1_h2r_rfm_h2r",
    "jesbu1_hand_paired_rfm_hand_paired_human",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human",
}


def categorize(archive_name: str) -> str:
    if archive_name in HUMANOID:
        return "humanoid"
    if archive_name in HUMAN_HAND:
        return "human_hand"
    return "standard"


# ────────────────────────────────────────────────────────────────────────────
# MD builders
# ────────────────────────────────────────────────────────────────────────────

def fmt(n):
    return f"{n:,}"


def build_md(audit: dict, pairs: dict) -> str:
    per = audit["per_archive"]
    # Bucket by group
    buckets = {"humanoid": [], "human_hand": [], "standard": []}
    for row in per:
        buckets[categorize(row["archive"])].append(row)

    def totals(rows):
        s = sum(r["successes"] for r in rows)
        f = sum(r["failures"] for r in rows)
        return s, f, len(rows)

    s_h, f_h, n_h = totals(buckets["humanoid"])
    s_m, f_m, n_m = totals(buckets["human_hand"])
    s_s, f_s, n_s = totals(buckets["standard"])

    total_s = s_h + s_m + s_s
    total_f = f_h + f_m + f_s
    total_n = n_h + n_m + n_s

    # Orphan successes: succ > 0 and fail == 0
    orphan_succ_rows = [r for r in per if r["successes"] > 0 and r["failures"] == 0]
    orphan_succ_total = sum(r["successes"] for r in orphan_succ_rows)
    orphan_succ_by_group = {g: sum(r["successes"] for r in buckets[g] if r in orphan_succ_rows) for g in buckets}

    # Orphan failures: fail > 0 and succ == 0
    orphan_fail_rows = [r for r in per if r["failures"] > 0 and r["successes"] == 0]
    orphan_fail_total = sum(r["failures"] for r in orphan_fail_rows)

    # In-context pairs
    pair_archives = pairs.get("per_archive", {})
    pair_tier_hist = {}
    pair_total_fail = 0
    pair_total_succ = 0
    pair_total_unpaired = 0
    pair_total_unused = 0
    for a in pair_archives.values():
        pair_total_fail += a.get("total_failures", 0)
        pair_total_succ += a.get("total_successes", 0)
        pair_total_unpaired += a.get("unpaired", 0)
        pair_total_unused += a.get("successes_unused", 0)
        for k, v in a.get("tier_hist", {}).items():
            pair_tier_hist[k] = pair_tier_hist.get(k, 0) + v
    pair_total_built = sum(pair_tier_hist.values())

    today = datetime.date.today().isoformat()
    lines = []

    lines.append("# Robometer Dataset — Full Breakdown")
    lines.append("")
    lines.append(f"**Single-source-of-truth document.** Generated automatically from `audit_report.json` (full no-exclusions scan) and `pairs/report.json`. Last generated: {today}.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Totals
    lines.append("## 1. Totals")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Archives scanned | **{fmt(total_n)}** |")
    lines.append(f"| Successes | **{fmt(total_s)}** |")
    lines.append(f"| Failures | **{fmt(total_f)}** |")
    lines.append(f"| Total episodes (succ + fail + partial) | **{fmt(audit['total_episodes'])}** |")
    lines.append("")
    lines.append(f"Source: `{audit.get('__source__', 'audit_report.json')}`. Scan covers every archive in `robometer_full_dataset/raw_archives/`.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. By robot type
    lines.append("## 2. Breakdown by robot type")
    lines.append("")
    lines.append("| Group | Archives | Successes | Failures |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Humanoid | {n_h} | {fmt(s_h)} | {fmt(f_h)} |")
    lines.append(f"| Human-only / human-hand | {n_m} | {fmt(s_m)} | {fmt(f_m)} |")
    lines.append(f"| Standard robot arms | {n_s} | {fmt(s_s)} | {fmt(f_s)} |")
    lines.append(f"| **Total** | **{total_n}** | **{fmt(total_s)}** | **{fmt(total_f)}** |")
    lines.append("")
    lines.append("Categorization is maintained in `generate_breakdown_md.py` (`HUMANOID` and `HUMAN_HAND` sets). Everything not in those sets is treated as a standard robot arm (includes sim arms like MetaWorld and failsafe/PlayWorld).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Orphan successes
    lines.append("## 3. Orphan successes (archives with successes but zero failures)")
    lines.append("")
    lines.append("| Group | Orphan successes | Archives |")
    lines.append("|---|---:|---:|")
    for g, label in [("humanoid", "Humanoid"), ("human_hand", "Human-only / human-hand"), ("standard", "Standard robot arms")]:
        g_orphan_rows = [r for r in buckets[g] if r in orphan_succ_rows]
        lines.append(f"| {label} | {fmt(sum(r['successes'] for r in g_orphan_rows))} | {len(g_orphan_rows)} |")
    lines.append(f"| **Total orphan successes** | **{fmt(orphan_succ_total)}** | **{len(orphan_succ_rows)}** |")
    non_orphan_s = total_s - orphan_succ_total
    lines.append(f"| Non-orphan successes (in archives that also have failures) | {fmt(non_orphan_s)} | — |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. Orphan failures
    lines.append("## 4. Orphan failures (archives with failures but zero successes)")
    lines.append("")
    if orphan_fail_rows:
        lines.append("| Archive | Failures |")
        lines.append("|---|---:|")
        for r in sorted(orphan_fail_rows, key=lambda x: -x["failures"]):
            lines.append(f"| `{r['archive']}` | {fmt(r['failures'])} |")
        lines.append(f"| **Total** | **{fmt(orphan_fail_total)}** |")
    else:
        lines.append("(none — every archive with failures also has at least one success internally)")
    lines.append("")
    lines.append("These are typically failure-only dumps whose failures get paired cross-archive using their robot family's success archive (tier-2 in `pair_robometer.py`).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. In-context pairs
    lines.append("## 5. In-context learning pairs")
    lines.append("")
    lines.append(f"Source: `{pairs.get('__source__', 'pairs/report.json')}` (produced by `pair_robometer.py`).")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Archives with failures (covered) | {len(pair_archives)} |")
    lines.append(f"| Total failures in those archives | {fmt(pair_total_fail)} |")
    lines.append(f"| Successes available for pairing | {fmt(pair_total_succ)} |")
    lines.append(f"| **Pairs built** | **{fmt(pair_total_built)}** |")
    lines.append(f"| Unpaired failures | {fmt(pair_total_unpaired)} |")
    lines.append(f"| Unused successes | {fmt(pair_total_unused)} |")
    lines.append("")
    lines.append("### Tier breakdown")
    lines.append("")
    lines.append("| Tier | Description | Pairs |")
    lines.append("|---|---|---:|")
    tier_descs = {
        "1_same_task_fresh": "same task, fresh success (never reused)",
        "2_same_task_family_fresh": "same task, success from other archive in same family, fresh",
        "3_same_task_reused": "same task, success reused",
        "4_same_family_other_task_fresh": "same family, other task, fresh",
        "5_same_family_other_task_reused": "same family, other task, reused",
        "6_cross_family_fallback": "cross-family fallback",
    }
    for k in sorted(tier_descs.keys()):
        lines.append(f"| {k[0]} | {tier_descs[k]} | {fmt(pair_tier_hist.get(k, 0))} |")
    lines.append(f"| **Total** | | **{fmt(pair_total_built)}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. Per-archive table
    lines.append("## 6. Per-archive table (every archive scanned)")
    lines.append("")
    lines.append("| Archive | Group | Successes | Failures | Orphan? |")
    lines.append("|---|---|---:|---:|:--:|")
    for r in sorted(per, key=lambda x: (categorize(x["archive"]), -x["successes"])):
        grp = categorize(r["archive"])
        orphan = "succ-only" if (r["successes"] > 0 and r["failures"] == 0) else \
                 ("fail-only" if (r["failures"] > 0 and r["successes"] == 0) else "")
        lines.append(f"| `{r['archive']}` | {grp} | {fmt(r['successes'])} | {fmt(r['failures'])} | {orphan} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 7. TL;DR
    lines.append("## 7. TL;DR — the six questions")
    lines.append("")
    lines.append(f"1. **How many successes?** {fmt(total_s)}")
    lines.append(f"2. **How many failures?** {fmt(total_f)}")
    lines.append(f"3. **Split by type:** humanoid = {fmt(s_h + f_h)} ep ({n_h} archives), human-hand = {fmt(s_m + f_m)} ep ({n_m} archives), standard-robot = {fmt(s_s + f_s)} ep ({n_s} archives).")
    lines.append(f"4. **Orphan successes:** {fmt(orphan_succ_total)} ({len(orphan_succ_rows)} archives).")
    lines.append(f"5. **Orphan failures:** {fmt(orphan_fail_total)} ({len(orphan_fail_rows)} archives).")
    lines.append(f"6. **In-context learning pairs:** {fmt(pair_total_built)}.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Regenerating this document")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 generate_breakdown_md.py \\")
    lines.append("    --audit  /projects/prjs1958/robometer_full_dataset/audit_report.json \\")
    lines.append("    --pairs  /projects/prjs1958/robometer_full_dataset/pairs/report.json \\")
    lines.append("    --output /gpfs/home3/pkarageorgis1/Master-Thesis/Real-World-Failures/Robometer/ROBOMETER_BREAKDOWN.md")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit",  required=True, help="Path to audit_report.json")
    parser.add_argument("--pairs",  required=True, help="Path to pairs/report.json")
    parser.add_argument("--output", required=True, help="Path to ROBOMETER_BREAKDOWN.md")
    args = parser.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    pairs = json.loads(Path(args.pairs).read_text())
    audit["__source__"] = args.audit
    pairs["__source__"] = args.pairs

    md = build_md(audit, pairs)
    Path(args.output).write_text(md)
    print(f"Wrote {args.output} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
