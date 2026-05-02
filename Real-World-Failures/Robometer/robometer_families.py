"""
Unified robot-family registry for Robometer archives.

Single source of truth for `archive_name -> family` mapping. Used by:
  - extract_successes.py           (SUCCESS_ARCHIVES → FAMILY_REGISTRY)
  - extract_orphan_successes.py    (ORPHAN_ARCHIVES  → FAMILY_REGISTRY)
  - repack_to_match_format.py      (meta.json family field)
  - patch_meta_family.py           (post-repack fix-up)

Adding a new archive here — rather than in a per-script dict — makes the
family available everywhere automatically.

Membership semantics (which archives are orphan vs success-pool vs excluded)
still live in the respective extract scripts; this registry only resolves the
family name for any archive we've ever seen.
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Family registry — union of every archive we extract anywhere in the pipeline
# ────────────────────────────────────────────────────────────────────────────

FAMILY_REGISTRY: dict[str, str] = {
    # ── Success-pool archives (from extract_successes.py / SUCCESS_ARCHIVES) ──
    # Failure-archive families (Group A single-file)
    "jesbu1_racer_rfm_racer_train":                                                     "racer",
    "jesbu1_racer_rfm_racer_val":                                                       "racer",
    "jesbu1_auto_eval_rfm_auto_eval_rfm":                                               "auto_eval",
    "jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm":                             "mit_franka",
    "jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist":             "mit_franka",
    "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm":                               "mit_franka",
    "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all":                             "usc_koch",
    "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking":                       "utd_so101",
    "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top":     "utd_so101",
    "jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist": "utd_so101",
    "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking":                         "usc_xarm",
    "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking":                     "usc_franka",
    "ykorkmaz_usc_trossen_rfm_usc_trossen":                                             "usc_trossen",
    # Failure-archive families (Group B split)
    "jesbu1_soar_rfm_soar_rfm":                                                         "soar",
    "jesbu1_roboarena_0825_rfm_roboarena":                                              "roboarena",
    "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist":                 "roboarena",
    # Success-only archives (extract_successes.py success-pool extras)
    "abraranwar_libero_rfm_libero256_10":                                               "libero",
    "abraranwar_libero_rfm_libero256_90":                                               "libero",
    "abraranwar_libero_rfm_libero256_goal":                                             "libero",
    "abraranwar_libero_rfm_libero256_object":                                           "libero",
    "abraranwar_libero_rfm_libero256_spatial":                                          "libero",
    "abraranwar_usc_koch_rewind_rfm_usc_koch_rewind":                                   "usc_koch",
    "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot":             "usc_koch",
    "aliangdw_utd_so101_human_utd_so101_human":                                         "utd_so101",

    # ── Orphan-success archives (from extract_orphan_successes.py / ORPHAN_ARCHIVES) ──
    # OXE robot-arm subset
    "jesbu1_oxe_rfm_oxe_aloha_mobile":                                                  "oxe_aloha_mobile",
    "jesbu1_oxe_rfm_oxe_austin_buds_dataset_converted_externally_to_rlds":              "oxe_austin_buds",
    "jesbu1_oxe_rfm_oxe_bc_z":                                                          "oxe_bc_z",
    "jesbu1_oxe_rfm_oxe_berkeley_cable_routing":                                        "oxe_berkeley_cable_routing",
    "jesbu1_oxe_rfm_oxe_berkeley_fanuc_manipulation":                                   "oxe_berkeley_fanuc",
    "jesbu1_oxe_rfm_oxe_berkeley_mvp_converted_externally_to_rlds":                     "oxe_berkeley_mvp",
    "jesbu1_oxe_rfm_oxe_berkeley_rpt_converted_externally_to_rlds":                     "oxe_berkeley_rpt",
    "jesbu1_oxe_rfm_oxe_bridge_v2":                                                     "oxe_bridge_v2",
    "jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval":                                           "oxe_bridge_v2",
    "jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds":          "oxe_dlr_edan",
    "jesbu1_oxe_rfm_oxe_fractal20220817_data":                                          "oxe_fractal",
    "jesbu1_oxe_rfm_oxe_furniture_bench_dataset_converted_externally_to_rlds":          "oxe_furniture_bench",
    "jesbu1_oxe_rfm_oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds":         "oxe_iamlab_cmu",
    "jesbu1_oxe_rfm_oxe_imperialcollege_sawyer_wrist_cam":                              "oxe_imperial_sawyer",
    "jesbu1_oxe_rfm_oxe_jaco_play":                                                     "oxe_jaco_play",
    "jesbu1_oxe_rfm_oxe_language_table":                                                "oxe_language_table",
    "jesbu1_oxe_rfm_oxe_nyu_rot_dataset_converted_externally_to_rlds":                  "oxe_nyu_rot",
    "jesbu1_oxe_rfm_oxe_robo_set":                                                      "oxe_robo_set",
    "jesbu1_oxe_rfm_oxe_stanford_hydra_dataset_converted_externally_to_rlds":           "oxe_stanford_hydra",
    "jesbu1_oxe_rfm_oxe_tokyo_u_lsmo_converted_externally_to_rlds":                     "oxe_tokyo_lsmo",
    "jesbu1_oxe_rfm_oxe_toto":                                                          "oxe_toto",
    "jesbu1_oxe_rfm_oxe_ucsd_kitchen_dataset_converted_externally_to_rlds":             "oxe_ucsd_kitchen",
    "jesbu1_oxe_rfm_oxe_utaustin_mutex":                                                "oxe_utaustin_mutex",
    # Non-OXE orphan archives
    "anqil_rh20t_subset_rfm_rh20t_robot":                                               "rh20t",
    "jesbu1_molmoact_rfm_molmoact_dataset_household":                                   "molmoact",
    "jesbu1_molmoact_rfm_molmoact_dataset_tabletop":                                    "molmoact",
    "jesbu1_motif_rfm_motif_rfm":                                                       "motif",
    "jesbu1_fino_net_rfm_fino_net":                                                     "fino_net",

    # ── Additional orphan archives (closing the 505,219-vs-341,220 gap) ──
    # OXE eval splits — same family as their non-eval counterparts
    "jesbu1_oxe_rfm_eval_oxe_bc_z_eval":                                                "oxe_bc_z",
    "jesbu1_oxe_rfm_eval_oxe_berkeley_cable_routing_eval":                              "oxe_berkeley_cable_routing",
    "jesbu1_oxe_rfm_eval_oxe_jaco_play_eval":                                           "oxe_jaco_play",
    "jesbu1_oxe_rfm_eval_oxe_toto_eval":                                                "oxe_toto",
    "jesbu1_oxe_rfm_eval_oxe_viola_eval":                                               "oxe_viola",
    # Standalone small archives
    "jesbu1_ph2d_rfm_ph2d":                                                             "ph2d",
    "aliangdw_metaworld_metaworld_train":                                               "metaworld",
    "jesbu1_hand_paired_rfm_hand_paired_robot":                                         "hand_paired",

    # ── DROID — user's own droid/ subtree is separate (5,503 failures × 3 views
    # + matched successes). The Robometer `oxe_droid` archive holds 149,804
    # ORPHAN successes that belong under keyframes_orphan_success/, NOT droid/.
    "jesbu1_oxe_rfm_oxe_droid":                                                         "droid",
}


def family_of(archive: str) -> str:
    """Return family name for an archive; falls back to the archive name
    itself when the archive isn't in the registry (safer than raising — keeps
    the pipeline moving on partially-registered new archives)."""
    return FAMILY_REGISTRY.get(archive, archive)
