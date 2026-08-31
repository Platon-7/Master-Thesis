"""ManiSkill3 task metadata for RoboRef downstream RL.

Task selection constraint
-------------------------
FailSafe (part of the RoboRef training corpus) is built on three ManiSkill
tasks -- ``PickCube``, ``PushCube`` and ``StackCube`` -- via its fault-injected
``FailPickCube`` / ``FailPushCube`` / ``FailStackCube`` variants.  Those three
task families are therefore IN-DISTRIBUTION for our reward models and must not
be used as a held-out downstream benchmark.  Every task registered below is
outside that set.  ``FORBIDDEN_TASK_STEMS`` is enforced at env-construction
time so the constraint cannot be violated by a config typo.

The language instructions are written in the same imperative style as the
MetaWorld instructions in ``metaworld_utils.TASK_TO_LANG`` (and as the task
strings in the RoboRef training corpus), since the reward model conditions on
them directly.
"""

from __future__ import annotations

from typing import Dict, NamedTuple

# ---------------------------------------------------------------------------
# Hard constraint: task families that leaked into training through FailSafe.
# Matched case-insensitively as substrings against the ManiSkill env id.
# ---------------------------------------------------------------------------
FORBIDDEN_TASK_STEMS = ("pickcube", "pushcube", "stackcube")


class ManiSkillTaskSpec(NamedTuple):
    """Per-task defaults.

    Attributes:
        instruction: Natural-language task description handed to the reward model.
        max_episode_steps: Horizon. ManiSkill's own registered defaults are used
            by the RL baselines; we restate them so a config can override.
        control_mode: Action space. ``pd_ee_delta_pos`` (3-DoF translation +
            gripper) is the easiest to learn from scratch for the top-down
            tabletop tasks and is what the ManiSkill RL baselines use for the
            cube-manipulation family. ``pd_ee_delta_pose`` adds rotation.
    """

    instruction: str
    max_episode_steps: int
    control_mode: str


# ---------------------------------------------------------------------------
# Curated shortlist. All are "easy tier" tabletop push/pull/pick-and-place
# tasks, chosen to be learnable by SAC *from scratch* (no BC warm start) while
# staying clear of the FailSafe three.
# ---------------------------------------------------------------------------
MANISKILL_TASKS: Dict[str, ManiSkillTaskSpec] = {
    # --- pull family -------------------------------------------------------
    # The direct analogue of PushCube (which is forbidden): the arm reaches
    # past the cube and drags it back toward a goal region. Easiest task here.
    "PullCube-v1": ManiSkillTaskSpec(
        instruction="Pull the cube to the goal region",
        max_episode_steps=50,
        control_mode="pd_ee_delta_pos",
    ),
    # Tool use: an L-shaped tool must be grasped and used to drag a cube that
    # is out of direct reach. Noticeably harder; keep as a stretch task.
    "PullCubeTool-v1": ManiSkillTaskSpec(
        instruction="Pull the cube closer using the tool",
        max_episode_steps=100,
        control_mode="pd_ee_delta_pose",
    ),
    # --- push family -------------------------------------------------------
    # Poke a cube toward a goal with a held peg.
    "PokeCube-v1": ManiSkillTaskSpec(
        instruction="Poke the cube to the goal region with the peg",
        max_episode_steps=50,
        control_mode="pd_ee_delta_pos",
    ),
    # Roll a ball across the table into a goal region.
    "RollBall-v1": ManiSkillTaskSpec(
        instruction="Roll the ball to the goal region",
        max_episode_steps=80,
        control_mode="pd_ee_delta_pos",
    ),
    # --- pick-and-place family --------------------------------------------
    # Pick up a YCB object and move it to a goal position. This is the
    # pick-and-place task with real object diversity, which is the most
    # informative setting for a vision-language reward model.
    "PickSingleYCB-v1": ManiSkillTaskSpec(
        instruction="Pick up the object and move it to the goal position",
        max_episode_steps=100,
        control_mode="pd_ee_delta_pose",
    ),
    # Grasp a peg lying flat and stand it upright. Simple, single-object.
    "LiftPegUpright-v1": ManiSkillTaskSpec(
        instruction="Lift the peg and stand it upright",
        max_episode_steps=50,  # ManiSkill registers 50, not 100,
        control_mode="pd_ee_delta_pose",
    ),
    # Push a T-shaped block onto a matching outline. Success is 90% AREA OVERLAP
    # between block and target, which makes it the most visually-legible success
    # criterion of the shortlist -- and its camera sits closest to the scene
    # (0.64 m vs PullCube's 0.96 m). Uses the panda_stick agent (no gripper), so
    # the action space is 3-D.
    "PushT-v1": ManiSkillTaskSpec(
        instruction="Push the T-shaped block onto the target outline",
        max_episode_steps=100,
        control_mode="pd_ee_delta_pos",
    ),
    # Place a sphere into a shallow bin.
    "PlaceSphere-v1": ManiSkillTaskSpec(
        instruction="Place the sphere into the bin",
        max_episode_steps=50,  # ManiSkill registers 50, not 80,
        control_mode="pd_ee_delta_pose",
    ),
}

# Mirrors ``metaworld_utils.TASK_TO_LANG`` so the language wrappers can be fed
# from one flat mapping.
TASK_TO_LANG: Dict[str, str] = {name: spec.instruction for name, spec in MANISKILL_TASKS.items()}

# Suggested preliminary sweep, easiest first. These three cover one pull, one
# push and one pick-and-place, which is the spread the results chapter needs.
RECOMMENDED_TASKS = ("PullCube-v1", "PokeCube-v1", "PickSingleYCB-v1")


def assert_task_allowed(env_id: str) -> None:
    """Reject ManiSkill tasks that are in-distribution via FailSafe.

    Raises:
        ValueError: if ``env_id`` belongs to the PickCube/PushCube/StackCube
            families that FailSafe contributed to the RoboRef training corpus.
    """
    stem = env_id.split("-")[0].replace("_", "").lower()
    for forbidden in FORBIDDEN_TASK_STEMS:
        if forbidden in stem:
            raise ValueError(
                f"ManiSkill task '{env_id}' is off-limits as a downstream benchmark: the "
                f"'{forbidden}' family entered the RoboRef training corpus through FailSafe "
                f"(FailPickCube / FailPushCube / FailStackCube). Pick one of: "
                f"{', '.join(sorted(MANISKILL_TASKS))}."
            )


def get_task_spec(env_id: str) -> ManiSkillTaskSpec:
    """Return defaults for ``env_id``, falling back to generic values.

    Unknown-but-allowed tasks are permitted so the shortlist is not a cage;
    they get a generic instruction that should be overridden in the config.
    """
    assert_task_allowed(env_id)
    if env_id in MANISKILL_TASKS:
        return MANISKILL_TASKS[env_id]
    return ManiSkillTaskSpec(
        instruction=f"Complete the {env_id.split('-')[0]} task",
        max_episode_steps=100,
        control_mode="pd_ee_delta_pose",
    )
