"""Generate dataset_success_cutoff.txt entries for all families that
pairs_unified.jsonl emits, mapping each to the closest paper-calibrated
value already in the cutoff file. Append the new entries in place.

Rules:
  1. If `<family>` already has an entry → skip
  2. Else find any cutoff key that contains `<family>` as a substring
     (paper uses long names like `oxe_fractal20220817_data`; family is the
     truncated `oxe_fractal` — substring match recovers the calibration)
  3. Special remappings for known short→long families
  4. Otherwise default to 0.95 (defensible robometer-family value)
"""
import json
from pathlib import Path

CUTOFF = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer/robometer/data/dataset_success_cutoff.txt")
PAIRS = Path("/projects/prjs1958/robometer_frame_dataset/pairs_unified.jsonl")

# Manual remappings where substring-match would pick the wrong (or no) entry.
# Each maps a bare family in pairs_unified → an existing key in the cutoff file.
EXPLICIT_REMAP = {
    "droid": "oxe_droid",
    "metaworld": "metaworld_train",
    "soar": "soar_rfm",
    "auto_eval": "auto_eval_rfm",
    "rh20t": "rh20t_robot",
    "molmoact": "molmoact_dataset_household",
    "motif": "motif_rfm",
    "fino_net": "fino_net_rfm",
    "robo_arena": "roboarena",
    "fractal": "oxe_fractal20220817_data",
    "oxe_fractal": "oxe_fractal20220817_data",
    "oxe_furniture_bench": "oxe_furniture_bench_dataset_converted_externally_to_rlds",
    "oxe_iamlab_cmu": "oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "oxe_stanford_hydra": "oxe_stanford_hydra_dataset_converted_externally_to_rlds",
    "oxe_berkeley_rpt": "oxe_berkeley_rpt_converted_externally_to_rlds",
    "oxe_berkeley_mvp": "oxe_berkeley_mvp_converted_externally_to_rlds",
    "oxe_berkeley_fanuc": "oxe_berkeley_fanuc_manipulation",
    "oxe_imperial_sawyer": "oxe_imperialcollege_sawyer_wrist_cam",
    "oxe_ucsd_kitchen": "oxe_ucsd_kitchen_dataset_converted_externally_to_rlds",
    "oxe_dlr_edan": "oxe_dlr_edan_shared_control_converted_externally_to_rlds",
    "oxe_austin_buds": "oxe_austin_buds_dataset_converted_externally_to_rlds",
    "oxe_tokyo_lsmo": "oxe_tokyo_u_lsmo_converted_externally_to_rlds",
    "oxe_nyu_rot": "oxe_nyu_rot_dataset_converted_externally_to_rlds",
    # Roboreward archive-level families (use paper-calibrated roboreward_* entries)
    "toto": "roboreward_toto",
    "bridge": "roboreward_bridge",
    "berkeley_autolab_ur5": "roboreward_berkeley_autolab_ur5",
    "ucsd_pick_and_place": "roboreward_ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "austin_sirius": "roboreward_austin_sirius_dataset_converted_externally_to_rlds",
    "roboturk": "roboreward_roboturk",
    "berkeley_mvp": "roboreward_berkeley_mvp_converted_externally_to_rlds",
    "iamlab_cmu": "roboreward_iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "cmu_play_fusion": "roboreward_cmu_play_fusion",
    "berkeley_fanuc": "roboreward_berkeley_fanuc_manipulation",
    "berkeley_rpt": "roboreward_berkeley_rpt_converted_externally_to_rlds",
    "stanford_hydra": "roboreward_stanford_hydra_dataset_converted_externally_to_rlds",
    "ucsd_kitchen": "roboreward_ucsd_kitchen_dataset_converted_externally_to_rlds",
    "utokyo_pr2_tabletop": "roboreward_utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "kaist_nonprehensile": "roboreward_kaist_nonprehensile_converted_externally_to_rlds",
    "utokyo_xarm_bimanual": "roboreward_utokyo_xarm_bimanual_converted_externally_to_rlds",
    "dlr_edan": "roboreward_dlr_edan_shared_control_converted_externally_to_rlds",
    "austin_buds": "roboreward_austin_buds_dataset_converted_externally_to_rlds",
    "tokyo_u_lsmo": "roboreward_tokyo_u_lsmo_converted_externally_to_rlds",
    "nyu_rot": "oxe_nyu_rot_dataset_converted_externally_to_rlds",
    # Sources without any upstream analog — defensible defaults
    "failsafe": None,        # user-added; tasks complete at last frame in sim → 1.0
    "racer": None,           # robometer-family — 0.95
    "libero": None,          # robometer-family — 0.95
    "usc_koch": None,
    "mit_franka": None,
    "utokyo_pr2_fridge": None,
    "utokyo_xarm_pick_place": None,
    "utd_so101": None,
    "viola": None,
    "usc_xarm": None,
    "usc_trossen": None,
    "usc_franka": None,
    "nyu_door_opening": None,
    "taco_play": None,
    "jaco_play": None,
    "oxe_jaco_play": None,
    "roboreward": None,      # user-added top-level family; 0.95 matches paper-calibrated archive avg
}

# Defaults for families with no upstream analog
DEFAULTS = {
    "failsafe": 1.0,        # sim env, success at last frame
    # everything else → 0.95
}


def main():
    existing = {}
    for line in open(CUTOFF):
        line = line.strip()
        if "," in line:
            k, v = line.split(",", 1)
            existing[k.strip()] = float(v.strip())

    # Collect families from pairs_unified
    families = set()
    for line in open(PAIRS):
        r = json.loads(line)
        f = r.get("family")
        if f:
            families.add(f)

    additions = []
    for f in sorted(families):
        if f in existing:
            continue
        if f in EXPLICIT_REMAP and EXPLICIT_REMAP[f] is not None:
            ref = EXPLICIT_REMAP[f]
            if ref in existing:
                additions.append((f, existing[ref], f"mirrors {ref}"))
                continue
            else:
                print(f"  WARN: explicit remap {f} → {ref} not in cutoff file")
        # No remap (or remap target missing) — use defaults
        val = DEFAULTS.get(f, 0.95)
        additions.append((f, val, "default"))

    # Append
    with open(CUTOFF, "a") as fh:
        fh.write("\n# --- bake-off addendum: bare-family names from pairs_unified.jsonl ---\n")
        for f, v, why in additions:
            fh.write(f"{f},{v}\n")

    print(f"appended {len(additions)} entries to {CUTOFF}")
    for f, v, why in additions:
        print(f"  {f:30s} = {v}  ({why})")


if __name__ == "__main__":
    main()
