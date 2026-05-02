# Robometer Dataset — Full Breakdown

**Single-source-of-truth document.** Generated automatically from `audit_report.json` (full no-exclusions scan) and `pairs/report.json`. Last generated: 2026-04-18.

---

## 1. Totals

| Metric | Count |
|---|---:|
| Archives scanned | **93** |
| Successes | **1,476,322** |
| Failures | **215,537** |
| Total episodes (succ + fail + partial) | **1,691,859** |

Source: `/projects/prjs1958/robometer_full_dataset/audit_report.json`. Scan covers every archive in `robometer_full_dataset/raw_archives/`.

---

## 2. Breakdown by robot type

| Group | Archives | Successes | Failures |
|---|---:|---:|---:|
| Humanoid | 8 | 551,147 | 0 |
| Human-only / human-hand | 11 | 366,699 | 0 |
| Standard robot arms | 74 | 558,476 | 215,537 |
| **Total** | **93** | **1,476,322** | **215,537** |

Categorization is maintained in `generate_breakdown_md.py` (`HUMANOID` and `HUMAN_HAND` sets). Everything not in those sets is treated as a standard robot arm (includes sim arms like MetaWorld and failsafe/PlayWorld).

---

## 3. Orphan successes (archives with successes but zero failures)

| Group | Orphan successes | Archives |
|---|---:|---:|
| Humanoid | 551,147 | 8 |
| Human-only / human-hand | 366,699 | 11 |
| Standard robot arms | 505,219 | 45 |
| **Total orphan successes** | **1,423,065** | **64** |
| Non-orphan successes (in archives that also have failures) | 53,257 | — |

---

## 4. Orphan failures (archives with failures but zero successes)

| Archive | Failures |
|---|---:|
| `ykorkmaz_libero_failure_rfm_libero_90_failure` | 4,312 |
| `ykorkmaz_libero_failure_rfm_libero_10_failure` | 498 |
| `ykorkmaz_libero_failure_rfm_libero_object_failure` | 489 |
| `ykorkmaz_libero_failure_rfm_libero_spatial_failure` | 486 |
| `ykorkmaz_libero_failure_rfm_libero_goal_failure` | 456 |
| **Total** | **6,241** |

These are typically failure-only dumps whose failures get paired cross-archive using their robot family's success archive (tier-2 in `pair_robometer.py`).

---

## 5. In-context learning pairs

Source: `/projects/prjs1958/robometer_full_dataset/pairs/report.json` (produced by `pair_robometer.py`).

| Metric | Count |
|---|---:|
| Archives with failures (covered) | 21 |
| Total failures in those archives | 68,933 |
| Successes available for pairing | 19,859 |
| **Pairs built** | **68,933** |
| Unpaired failures | 0 |
| Unused successes | 2,107 |

### Tier breakdown

| Tier | Description | Pairs |
|---|---|---:|
| 1 | same task, fresh success (never reused) | 14,076 |
| 2 | same task, success from other archive in same family, fresh | 6,932 |
| 3 | same task, success reused | 36,377 |
| 4 | same family, other task, fresh | 2,293 |
| 5 | same family, other task, reused | 9,255 |
| 6 | cross-family fallback | 0 |
| **Total** | | **68,933** |

---

## 6. Per-archive table (every archive scanned)

| Archive | Group | Successes | Failures | Orphan? |
|---|---|---:|---:|:--:|
| `jesbu1_egodex_rfm_egodex_part2` | human_hand | 94,488 | 0 | succ-only |
| `jesbu1_egodex_rfm_egodex_part5` | human_hand | 75,076 | 0 | succ-only |
| `jesbu1_egodex_rfm_egodex_part3` | human_hand | 51,899 | 0 | succ-only |
| `jesbu1_egodex_rfm_egodex_part1` | human_hand | 45,232 | 0 | succ-only |
| `jesbu1_egodex_rfm_egodex_part4` | human_hand | 43,199 | 0 | succ-only |
| `jesbu1_epic_rfm_epic` | human_hand | 37,030 | 0 | succ-only |
| `anqil_rh20t_subset_rfm_rh20t_human` | human_hand | 14,225 | 0 | succ-only |
| `jesbu1_egodex_rfm_egodex_test` | human_hand | 3,215 | 0 | succ-only |
| `jesbu1_h2r_rfm_h2r` | human_hand | 2,254 | 0 | succ-only |
| `jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human` | human_hand | 72 | 0 | succ-only |
| `jesbu1_hand_paired_rfm_hand_paired_human` | human_hand | 9 | 0 | succ-only |
| `abraranwar_agibotworld_alpha_headcam_rfm_agibotworld` | humanoid | 216,911 | 0 | succ-only |
| `abraranwar_agibotworld_alpha_rfm_agibotworld` | humanoid | 216,910 | 0 | succ-only |
| `jesbu1_galaxea_rfm_galaxea_part2_r1_lite` | humanoid | 24,888 | 0 | succ-only |
| `jesbu1_galaxea_rfm_galaxea_part3_r1_lite` | humanoid | 24,741 | 0 | succ-only |
| `jesbu1_galaxea_rfm_galaxea_part1_r1_lite` | humanoid | 22,110 | 0 | succ-only |
| `jesbu1_galaxea_rfm_galaxea_part4_r1_lite` | humanoid | 21,511 | 0 | succ-only |
| `jesbu1_galaxea_rfm_galaxea_part5_r1_lite` | humanoid | 14,868 | 0 | succ-only |
| `jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm` | humanoid | 9,208 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_droid` | standard | 149,804 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_fractal20220817_data` | standard | 87,204 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_bridge_v2` | standard | 72,930 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_language_table` | standard | 50,000 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_bc_z` | standard | 39,347 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_robo_set` | standard | 36,480 | 0 | succ-only |
| `anqil_rh20t_subset_rfm_rh20t_robot` | standard | 15,744 | 0 | succ-only |
| `jesbu1_failsafe_rfm_failsafe` | standard | 13,461 | 58,153 |  |
| `jesbu1_molmoact_rfm_molmoact_dataset_household` | standard | 11,872 | 0 | succ-only |
| `jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval` | standard | 10,094 | 0 | succ-only |
| `jesbu1_roboreward_rfm_high_res_roboreward_train` | standard | 8,425 | 36,647 |  |
| `jesbu1_roboreward_rfm_roboreward_train` | standard | 8,425 | 36,647 |  |
| `jesbu1_racer_rfm_racer_train` | standard | 5,724 | 23,391 |  |
| `jesbu1_oxe_rfm_oxe_furniture_bench_dataset_converted_externally_to_rlds` | standard | 5,100 | 0 | succ-only |
| `jesbu1_auto_eval_rfm_auto_eval_rfm` | standard | 4,956 | 3,721 |  |
| `jesbu1_soar_rfm_soar_rfm` | standard | 4,803 | 12,009 |  |
| `abraranwar_libero_rfm_libero256_90` | standard | 3,950 | 0 | succ-only |
| `jesbu1_oxe_rfm_eval_oxe_bc_z_eval` | standard | 3,914 | 0 | succ-only |
| `jesbu1_molmoact_rfm_molmoact_dataset_tabletop` | standard | 3,674 | 0 | succ-only |
| `jesbu1_ph2d_rfm_ph2d` | standard | 3,596 | 0 | succ-only |
| `jesbu1_roboarena_0825_rfm_roboarena` | standard | 1,626 | 10,753 |  |
| `jesbu1_oxe_rfm_oxe_utaustin_mutex` | standard | 1,500 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_berkeley_cable_routing` | standard | 1,482 | 0 | succ-only |
| `jesbu1_racer_rfm_racer_val` | standard | 1,407 | 5,820 |  |
| `jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist` | standard | 1,009 | 6,757 |  |
| `jesbu1_oxe_rfm_oxe_jaco_play` | standard | 976 | 0 | succ-only |
| `jesbu1_roboreward_rfm_high_res_roboreward_val` | standard | 974 | 5,258 |  |
| `jesbu1_roboreward_rfm_roboreward_val` | standard | 974 | 5,258 |  |
| `jesbu1_oxe_rfm_oxe_berkeley_rpt_converted_externally_to_rlds` | standard | 904 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_toto` | standard | 902 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds` | standard | 631 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_stanford_hydra_dataset_converted_externally_to_rlds` | standard | 570 | 0 | succ-only |
| `jesbu1_roboreward_rfm_high_res_roboreward_test` | standard | 527 | 2,304 |  |
| `jesbu1_roboreward_rfm_roboreward_test` | standard | 527 | 2,304 |  |
| `jesbu1_oxe_rfm_oxe_berkeley_mvp_converted_externally_to_rlds` | standard | 480 | 0 | succ-only |
| `abraranwar_libero_rfm_libero256_object` | standard | 456 | 0 | succ-only |
| `abraranwar_libero_rfm_libero256_spatial` | standard | 433 | 0 | succ-only |
| `abraranwar_libero_rfm_libero256_goal` | standard | 432 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_berkeley_fanuc_manipulation` | standard | 415 | 0 | succ-only |
| `abraranwar_usc_koch_rewind_rfm_usc_koch_rewind` | standard | 407 | 0 | succ-only |
| `abraranwar_libero_rfm_libero256_10` | standard | 388 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_aloha_mobile` | standard | 272 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_imperialcollege_sawyer_wrist_cam` | standard | 168 | 0 | succ-only |
| `jesbu1_oxe_rfm_eval_oxe_berkeley_cable_routing_eval` | standard | 165 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_ucsd_kitchen_dataset_converted_externally_to_rlds` | standard | 150 | 0 | succ-only |
| `jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm` | standard | 138 | 80 |  |
| `jesbu1_oxe_rfm_eval_oxe_jaco_play_eval` | standard | 109 | 0 | succ-only |
| `jesbu1_oxe_rfm_eval_oxe_toto_eval` | standard | 101 | 0 | succ-only |
| `aliangdw_metaworld_metaworld_train` | standard | 100 | 0 | succ-only |
| `jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot` | standard | 100 | 0 | succ-only |
| `aliangdw_metaworld_metaworld_eval` | standard | 85 | 33 |  |
| `jesbu1_motif_rfm_motif_rfm` | standard | 83 | 0 | succ-only |
| `jesbu1_fino_net_rfm_fino_net` | standard | 82 | 0 | succ-only |
| `jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist` | standard | 69 | 40 |  |
| `jesbu1_oxe_rfm_oxe_austin_buds_dataset_converted_externally_to_rlds` | standard | 50 | 0 | succ-only |
| `jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all` | standard | 50 | 50 |  |
| `jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds` | standard | 48 | 0 | succ-only |
| `jesbu1_oxe_rfm_oxe_tokyo_u_lsmo_converted_externally_to_rlds` | standard | 48 | 0 | succ-only |
| `aliangdw_utd_so101_human_utd_so101_human` | standard | 20 | 0 | succ-only |
| `aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking` | standard | 20 | 20 |  |
| `jesbu1_oxe_rfm_eval_oxe_viola_eval` | standard | 15 | 0 | succ-only |
| `ykorkmaz_usc_trossen_rfm_usc_trossen` | standard | 15 | 6 |  |
| `jesbu1_oxe_rfm_oxe_nyu_rot_dataset_converted_externally_to_rlds` | standard | 14 | 0 | succ-only |
| `aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking` | standard | 12 | 12 |  |
| `jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top` | standard | 10 | 10 |  |
| `jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist` | standard | 10 | 10 |  |
| `jesbu1_hand_paired_rfm_hand_paired_robot` | standard | 9 | 0 | succ-only |
| `aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking` | standard | 8 | 8 |  |
| `jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm` | standard | 2 | 5 |  |
| `ykorkmaz_libero_failure_rfm_libero_10_failure` | standard | 0 | 498 | fail-only |
| `ykorkmaz_libero_failure_rfm_libero_90_failure` | standard | 0 | 4,312 | fail-only |
| `ykorkmaz_libero_failure_rfm_libero_goal_failure` | standard | 0 | 456 | fail-only |
| `ykorkmaz_libero_failure_rfm_libero_object_failure` | standard | 0 | 489 | fail-only |
| `ykorkmaz_libero_failure_rfm_libero_spatial_failure` | standard | 0 | 486 | fail-only |

---

## 7. TL;DR — the six questions

1. **How many successes?** 1,476,322
2. **How many failures?** 215,537
3. **Split by type:** humanoid = 551,147 ep (8 archives), human-hand = 366,699 ep (11 archives), standard-robot = 774,013 ep (74 archives).
4. **Orphan successes:** 1,423,065 (64 archives).
5. **Orphan failures:** 6,241 (5 archives).
6. **In-context learning pairs:** 68,933.

---

## Regenerating this document

```bash
python3 generate_breakdown_md.py \
    --audit  /projects/prjs1958/robometer_full_dataset/audit_report.json \
    --pairs  /projects/prjs1958/robometer_full_dataset/pairs/report.json \
    --output /gpfs/home3/pkarageorgis1/Master-Thesis/Real-World-Failures/Robometer/ROBOMETER_BREAKDOWN.md
```
