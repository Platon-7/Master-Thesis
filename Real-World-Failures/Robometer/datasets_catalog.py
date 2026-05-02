"""
Catalog of all datasets in robometer/processed_datasets on HuggingFace.

Each entry:
  "archive_name": name of the .tar file (without .tar) or split archive prefix
  "category": broad grouping
  "split_parts": True if the archive is split into .tar.part-aa, .tar.part-ab, ...
  "skip": True if already processed elsewhere (e.g. DROID failures done separately)
"""

DATASETS = [
    # ----------------------------------------------------------------
    # REAL ROBOT — OXE / DROID / Cross-embodiment
    # ----------------------------------------------------------------
    {
        "archive": "jesbu1_oxe_rfm_oxe_droid",
        "category": "real_robot",
        "description": "DROID dataset (OXE split)",
        "split_parts": False,
        "skip": False,  # different from our DROID failure subset
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_bridge_v2",
        "category": "real_robot",
        "description": "Bridge V2",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_bc_z",
        "category": "real_robot",
        "description": "BC-Z (Google robot)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_fractal20220817_data",
        "category": "real_robot",
        "description": "RT-1 / Fractal data",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_robo_set",
        "category": "real_robot",
        "description": "RoboSet",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_aloha_mobile",
        "category": "real_robot",
        "description": "ALOHA Mobile",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_toto",
        "category": "real_robot",
        "description": "TOTO",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_jaco_play",
        "category": "real_robot",
        "description": "Jaco Play",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_berkeley_cable_routing",
        "category": "real_robot",
        "description": "Berkeley Cable Routing",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_berkeley_fanuc_manipulation",
        "category": "real_robot",
        "description": "Berkeley Fanuc",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_berkeley_mvp_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "Berkeley MVP",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_berkeley_rpt_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "Berkeley RPT",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_austin_buds_dataset_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "Austin BUDS",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_stanford_hydra_dataset_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "Stanford Hydra",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "CMU IAM Lab Pickup-Insert",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_dlr_edan_shared_control_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "DLR EDAN",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_furniture_bench_dataset_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "Furniture Bench",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_imperialcollege_sawyer_wrist_cam",
        "category": "real_robot",
        "description": "Imperial College Sawyer",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_nyu_rot_dataset_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "NYU RoT",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_tokyo_u_lsmo_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "Tokyo-U LSMO",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_ucsd_kitchen_dataset_converted_externally_to_rlds",
        "category": "real_robot",
        "description": "UCSD Kitchen",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_utaustin_mutex",
        "category": "real_robot",
        "description": "UT Austin MUTEX",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_oxe_language_table",
        "category": "real_robot",
        "description": "Language Table",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_racer_rfm_racer_train",
        "category": "real_robot",
        "description": "RACER (train)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_racer_rfm_racer_val",
        "category": "real_robot",
        "description": "RACER (val)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_rfm_new_mit_franka_rfm_rfm_new_mit_franka_rfm",
        "category": "real_robot",
        "description": "MIT Franka (with wrist)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_rfm_new_mit_franka_rfm_nowrist_rfm_new_mit_franka_rfm_nowrist",
        "category": "real_robot",
        "description": "MIT Franka (no wrist)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "aliangdw_usc_franka_policy_ranking_usc_franka_policy_ranking",
        "category": "real_robot",
        "description": "USC Franka Policy Ranking",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "aliangdw_usc_xarm_policy_ranking_usc_xarm_policy_ranking",
        "category": "real_robot",
        "description": "USC xArm Policy Ranking",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "aliangdw_utd_so101_human_utd_so101_human",
        "category": "real_robot",
        "description": "UTD SO-101 (human demos)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "aliangdw_utd_so101_policy_ranking_utd_so101_policy_ranking",
        "category": "real_robot",
        "description": "UTD SO-101 Policy Ranking",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "ykorkmaz_usc_trossen_rfm_usc_trossen",
        "category": "real_robot",
        "description": "USC Trossen",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "abraranwar_usc_koch_rewind_rfm_usc_koch_rewind",
        "category": "real_robot",
        "description": "USC Koch Rewind",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_robot",
        "category": "real_robot",
        "description": "USC Koch Human-Robot Paired (robot)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_usc_koch_p_ranking_rfm_usc_koch_p_ranking_all",
        "category": "real_robot",
        "description": "USC Koch Policy Ranking",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_utd_so101_clean_policy_ranking_top_utd_so101_clean_policy_ranking_top",
        "category": "real_robot",
        "description": "UTD SO-101 Clean Policy Ranking (top camera)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_utd_so101_clean_policy_ranking_wrist_utd_so101_clean_policy_ranking_wrist",
        "category": "real_robot",
        "description": "UTD SO-101 Clean Policy Ranking (wrist camera)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_auto_eval_rfm_auto_eval_rfm",
        "category": "real_robot",
        "description": "Auto Eval RFM",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_h2r_rfm_h2r",
        "category": "real_robot",
        "description": "H2R (Human-to-Robot)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_fino_net_rfm_fino_net",
        "category": "real_robot",
        "description": "FINO-Net",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_ph2d_rfm_ph2d",
        "category": "real_robot",
        "description": "PH2D",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_soar_rfm_soar_rfm",
        "category": "real_robot",
        "description": "SOAR",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_roboarena_0825_rfm_roboarena",
        "category": "real_robot",
        "description": "RoboArena",
        "split_parts": False,
        "skip": False,
    },
    # ----------------------------------------------------------------
    # REAL ROBOT — RoboReward subset (own neural-annotation pipeline, integrated as
    # a first-class family in pairs_unified.jsonl alongside droid / robometer / etc.)
    # ----------------------------------------------------------------
    {
        "archive": "jesbu1_roboreward_rfm_roboreward_train",
        "category": "real_robot",
        "description": "RoboReward benchmark train",
        "split_parts": True,
        "skip": False,
    },
    {
        "archive": "jesbu1_roboreward_rfm_roboreward_val",
        "category": "real_robot",
        "description": "RoboReward benchmark val",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_roboreward_rfm_roboreward_test",
        "category": "real_robot",
        "description": "RoboReward benchmark test",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_roboreward_rfm_high_res_roboreward_train",
        "category": "real_robot",
        "description": "RoboReward high-res train",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_roboreward_rfm_high_res_roboreward_val",
        "category": "real_robot",
        "description": "RoboReward high-res val",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_roboreward_rfm_high_res_roboreward_test",
        "category": "real_robot",
        "description": "RoboReward high-res test",
        "split_parts": False,
        "skip": False,
    },
    # ----------------------------------------------------------------
    # REAL ROBOT — Large split archives (need part assembly)
    # ----------------------------------------------------------------
    {
        "archive": "abraranwar_agibotworld_alpha_rfm_agibotworld",
        "category": "real_robot",
        "description": "AgiBot World Alpha (~430 GB)",
        "split_parts": True,
        "skip": False,
    },
    {
        "archive": "abraranwar_agibotworld_alpha_headcam_rfm_agibotworld",
        "category": "real_robot",
        "description": "AgiBot World Alpha (headcam, ~546 GB)",
        "split_parts": True,
        "skip": False,
    },
    # ----------------------------------------------------------------
    # SIMULATED
    # ----------------------------------------------------------------
    {
        "archive": "aliangdw_metaworld_metaworld_train",
        "category": "simulated",
        "description": "MetaWorld (train) — SKIP: custom MetaWorld failures generated separately",
        "split_parts": False,
        "skip": True,  # user generated own MetaWorld failures
    },
    {
        "archive": "aliangdw_metaworld_metaworld_eval",
        "category": "simulated",
        "description": "MetaWorld (eval) — SKIP: custom MetaWorld failures generated separately",
        "split_parts": False,
        "skip": True,  # user generated own MetaWorld failures
    },
    {
        "archive": "abraranwar_libero_rfm_libero256_10",
        "category": "simulated",
        "description": "LIBERO-10 (256x256)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "abraranwar_libero_rfm_libero256_90",
        "category": "simulated",
        "description": "LIBERO-90 (256x256)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "abraranwar_libero_rfm_libero256_goal",
        "category": "simulated",
        "description": "LIBERO-Goal (256x256)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "abraranwar_libero_rfm_libero256_object",
        "category": "simulated",
        "description": "LIBERO-Object (256x256)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "abraranwar_libero_rfm_libero256_spatial",
        "category": "simulated",
        "description": "LIBERO-Spatial (256x256)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "ykorkmaz_libero_failure_rfm_libero_10_failure",
        "category": "simulated",
        "description": "LIBERO-10 explicit failures",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "ykorkmaz_libero_failure_rfm_libero_90_failure",
        "category": "simulated",
        "description": "LIBERO-90 explicit failures",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "ykorkmaz_libero_failure_rfm_libero_goal_failure",
        "category": "simulated",
        "description": "LIBERO-Goal explicit failures",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "ykorkmaz_libero_failure_rfm_libero_object_failure",
        "category": "simulated",
        "description": "LIBERO-Object explicit failures",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "ykorkmaz_libero_failure_rfm_libero_spatial_failure",
        "category": "simulated",
        "description": "LIBERO-Spatial explicit failures",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_motif_rfm_motif_rfm",
        "category": "simulated",
        "description": "MOTIF (Atari-based reward learning)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_molmoact_rfm_molmoact_dataset_tabletop",
        "category": "simulated",
        "description": "MolmoAct tabletop",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_molmoact_rfm_molmoact_dataset_household",
        "category": "simulated",
        "description": "MolmoAct household",
        "split_parts": False,
        "skip": False,
    },
    # ----------------------------------------------------------------
    # HUMAN / EGOCENTRIC
    # ----------------------------------------------------------------
    {
        "archive": "jesbu1_egodex_rfm_egodex_test",
        "category": "human_egocentric",
        "description": "EgoDex test (egocentric dexterous manipulation)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_epic_rfm_epic",
        "category": "human_egocentric",
        "description": "EPIC Kitchens",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "anqil_rh20t_subset_rfm_rh20t_human",
        "category": "human_egocentric",
        "description": "RH20T human demos",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "anqil_rh20t_subset_rfm_rh20t_robot",
        "category": "real_robot",
        "description": "RH20T robot trajectories",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_hand_paired_rfm_hand_paired_human",
        "category": "human_egocentric",
        "description": "Hand Paired (human demos)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_hand_paired_rfm_hand_paired_robot",
        "category": "real_robot",
        "description": "Hand Paired (robot trajectories)",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_humanoid_everyday_rfm_humanoid_everyday_rfm",
        "category": "human_egocentric",
        "description": "Humanoid Everyday Activities",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_usc_koch_human_robot_paired_usc_koch_human_robot_paired_human",
        "category": "human_egocentric",
        "description": "USC Koch Human-Robot Paired (human demos)",
        "split_parts": False,
        "skip": False,
    },
    # Large split archives - egocentric/humanoid
    {
        "archive": "jesbu1_egodex_rfm_egodex_part1",
        "category": "human_egocentric",
        "description": "EgoDex part 1 (large)",
        "split_parts": True,
        "skip": False,
    },
    {
        "archive": "jesbu1_egodex_rfm_egodex_part2",
        "category": "human_egocentric",
        "description": "EgoDex part 2 (large)",
        "split_parts": True,
        "skip": False,
    },
    {
        "archive": "jesbu1_egodex_rfm_egodex_part3",
        "category": "human_egocentric",
        "description": "EgoDex part 3 (large)",
        "split_parts": True,
        "skip": False,
    },
    {
        "archive": "jesbu1_egodex_rfm_egodex_part4",
        "category": "human_egocentric",
        "description": "EgoDex part 4 (large)",
        "split_parts": True,
        "skip": False,
    },
    {
        "archive": "jesbu1_egodex_rfm_egodex_part5",
        "category": "human_egocentric",
        "description": "EgoDex part 5 (large)",
        "split_parts": True,
        "skip": False,
    },
    {
        "archive": "jesbu1_galaxea_rfm_galaxea_part1_r1_lite",
        "category": "real_robot",
        "description": "Galaxea R1-Lite part 1 (large)",
        "split_parts": True,
        "skip": False,
    },
    # ----------------------------------------------------------------
    # EVAL / BENCHMARK
    # ----------------------------------------------------------------
    {
        "archive": "jesbu1_oxe_rfm_eval_oxe_bc_z_eval",
        "category": "eval_benchmark",
        "description": "OXE Eval: BC-Z",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_eval_oxe_bridge_v2_eval",
        "category": "eval_benchmark",
        "description": "OXE Eval: Bridge V2",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_eval_oxe_jaco_play_eval",
        "category": "eval_benchmark",
        "description": "OXE Eval: Jaco Play",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_eval_oxe_toto_eval",
        "category": "eval_benchmark",
        "description": "OXE Eval: TOTO",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_eval_oxe_viola_eval",
        "category": "eval_benchmark",
        "description": "OXE Eval: VIOLA",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_oxe_rfm_eval_oxe_berkeley_cable_routing_eval",
        "category": "eval_benchmark",
        "description": "OXE Eval: Berkeley Cable Routing",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_mit_franka_p-rank_rfm_mit_franka_p-rank_rfm",
        "category": "eval_benchmark",
        "description": "MIT Franka Policy Ranking",
        "split_parts": False,
        "skip": False,
    },
    {
        "archive": "jesbu1_roboarena_eval_debug_nowrist_roboarena_eval_debug_nowrist",
        "category": "eval_benchmark",
        "description": "RoboArena Eval (no wrist)",
        "split_parts": False,
        "skip": False,
    },
]


def list_datasets(category=None, skip_done=True, no_splits=False):
    """Return filtered list of datasets.

    Args:
        category: filter by category ('real_robot', 'simulated', 'human_egocentric', 'eval_benchmark')
        skip_done: skip datasets marked skip=True (already processed)
        no_splits: exclude split-part archives (require assembly)
    """
    result = DATASETS
    if skip_done:
        result = [d for d in result if not d.get("skip", False)]
    if category:
        result = [d for d in result if d["category"] == category]
    if no_splits:
        result = [d for d in result if not d.get("split_parts", False)]
    return result


if __name__ == "__main__":
    from collections import Counter
    cats = Counter(d["category"] for d in DATASETS if not d.get("skip"))
    print("Datasets by category (excluding already-processed):")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    splits = sum(1 for d in DATASETS if d.get("split_parts") and not d.get("skip"))
    single = sum(1 for d in DATASETS if not d.get("split_parts") and not d.get("skip"))
    print(f"\nSingle-file archives (easy): {single}")
    print(f"Split archives (need assembly): {splits}")
