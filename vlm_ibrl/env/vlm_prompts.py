"""Shared prompts and per-task descriptions for the VLM reward critics.

The two ``VLMRobosuite`` (env/robosuite_vlm_env.py) and ``VLMCritic_PixelMetaWorld``
(env/vlm_envs.py) classes both consume these. Keep textual prompts and per-task
strings here so we have a single source of truth.
"""


# ---------------------------------------------------------------------------
# System prompts (shared by RoboMimic and MetaWorld envs)
# ---------------------------------------------------------------------------

vlmcritic_prompt = """You are an expert roboticist tasked to decide whether a simulated robot has completed a given task using a short video and a task instruction.

Output format:
Return a single character with no extra text:
"1" if the task is completed in this frame;
"0" otherwise.
Do not explain your answer.
"""

roboreward_prompt = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.
"""


# ---------------------------------------------------------------------------
# MetaWorld
# ---------------------------------------------------------------------------

METAWORLD_TASK_DESCRIPTIONS = {
    "Assembly": "pick up a nut and place it onto a peg",
    "BoxClose": "grasp the cover and close the box with it",
    "CoffeePush": "push a mug under a coffee machine",
    "StickPull": "grasp a stick and pull a box with the stick",
}

# Demo2Reward-optimized task instructions for MetaWorld.
METAWORLD_DEMO2REWARD_REPLIES = {
    "Assembly": "In the final frame, determine if the robot has successfully completed the task by visually verifying that the nut (a gray circular object) is fully seated on top of the peg (a red cylindrical object), with no visible separation between them. Output \"1\" if the nut is completely resting on the peg; otherwise, output \"0\". Ignore intermediate actions or partial placements. Do not consider motion or rotation as failure — success is determined by final positional alignment.",
    "BoxClose": "In the final frame, determine if the robot has successfully completed the task \"grasp the cover and close the box with it\" by checking whether the cover is visibly grasped by the robot's end-effector and placed on top of the box such that it overlaps with the box's top surface; ignore incomplete alignment or small gaps. If this condition is met, output \"1\"; otherwise, output \"0\".",
    "CoffeePush": "The robot's task is to push a white mug under a red coffee machine. Evaluate the final frame: if the white mug is fully under the coffee machine (no part of it is visible on the table surface) and there is a small green dot visible near the coffee machine's spout (indicating correct placement), output \"1\". If the mug is still visible on the table, not positioned under the machine, or the green dot is absent, output \"0\". Ignore the robot's arm position or motion. Do not consider partial positioning or intermediate contact as success. Success is defined solely by the mug's final position and the presence of the green dot.",
    "StickPull": "Output \"1\" only if, in the final frame, the red stick is visibly held by the robot's gripper and is in direct contact with the multicolored box, and the box is positioned close to the green ball (within touching distance or adjacent). Ignore motion or intermediate frames. If the box is not near the green ball, or the stick is not clearly contacting the box, or the stick is not gripped, output \"0\". The gripper must be visibly closed around the stick, and the stick must appear to be touching the side or top of the multicolored box.",
}


# ---------------------------------------------------------------------------
# RoboMimic
# ---------------------------------------------------------------------------

ROBOMIMIC_TASK_DESCRIPTIONS = {
    "NutAssemblySquare": "Two colored pegs (one square and one round) are mounted on the tabletop, and a suare nut is placed on the table in front of a single robot arm. The robot must fit the square nut onto the square peg.",
    "PickPlaceCan": "A can is placed in a bin in front of a single robot arm. There are four containers next to the bin. The robot must place the can into its corresponding container.",
}

# Demo2Reward-optimized task instructions for RoboMimic (3 demos, 8B + 32B meta-critic).
ROBOMIMIC_DEMO2REWARD_REPLIES = {
    "PickPlaceCan": 'In the final frame, output "1" only if the red can is visually fully and stably seated inside a compartment of the container AND the robot’s arm and gripper are stationary, disengaged from the can, and not in motion — otherwise output "0".',
    "NutAssemblySquare": 'The robot must fit the brown square nut onto the brown square peg. Success is achieved in the final frame if the brown square nut is fully seated on top of the brown square peg with no visible gaps, and the gripper is not in contact with or holding the nut. The nut and peg must be in stable physical contact with no motion. If the nut is floating above, partially inserted, or the gripper is touching the nut, output "0". Ignore slight rotation or minor misalignment if the nut is visibly seated and stable. Output "1" only if the nut is clearly seated and not being held.',
}
