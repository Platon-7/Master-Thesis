# FailSafe Failure Generation Rubric

## Tasks
- FailPickCube-v1: Pick up the red cube and lift it to the goal position
- FailPushCube-v1: Push the red cube to the green goal position
- FailStackCube-v1: Pick up the red cube and stack it on top of the green cube

## RoboMeter Scoring (1-5)
- 1: No progress
- 2: Approached the object
- 3: Grasped/contacted the object (stack only; pick and push skip score 3)
- 4: Near completion (closest failure to success)
- 5: Success (task completed)

## Score 1 (6 total: 2 per task)
For all three tasks:
(a) Arm moves AWAY from the object (away-biased random actions)
(b) Arm totally freezes and does nothing (zero actions)

## Score 2 (6 total: 2 per task)
For all three tasks:
(a) Arm approaches then FREEZES near the object
(b) Arm approaches then moves away again (labels: 1->2->1)

## Score 3 (3 total: stack only)
Pick and push do NOT have score 3 cases.
Stack only:
(a) Grasp then freeze (arm holds cube, frozen in place)
(b) Grasp and carry away WITHOUT dropping (cube carried to wrong spot, placed gently -> 3->2)
(c) Grasp and DROP on the way (cube falls mid-air during carry -> 3->2)

## Score 4 (7 total: 2 pick + 2 push + 3 stack)
Pick (grasped = score 4, since grasping IS the hard part for pick):
(a) Grasp then freeze
(b) Grasp then ungrasp/drop (4->2)

Push (meaningful push toward goal = score 4):
(a) Push partway toward goal then freeze
(b) Push toward goal then push wrong direction (4->3->2, gradual regression as cube moves away)

Stack (cube near/above goal = score 4):
(a) Drop near goal (wrist tilt causes cube to slip near cubeB)
(b) Freeze near goal (arm stops above cubeB while holding cube)
(c) Move away from goal while near it

## Bonus: Success Then Mess Up (2 total)
- Pick: Pick up, lift to goal (5), then drop it (5->2)
- Push: Push to bullseye (5), keep pushing past it (5->4->3 as cube moves away)
- Stack: SKIP (too hard, would need to grip and smash the stack)

## Success (3 total: 1 per task)
One clean successful trajectory per task showing full 1->2->...->5 progression.
These MUST show observed score = 5 (not 4).

## Total: 27 unique episodes
- Score 1: 6
- Score 2: 6
- Score 3: 3
- Score 4: 7
- Bonus: 2
- Success: 3

## Labeling Rules

### Pick labeler (no score 3)
- 5: grasped AND cube near goal position (obj_to_goal < 0.05)
- 4: grasped (but not at goal)
- 2: end-effector near cube (ee_to_obj < 0.08)
- 1: no progress

### Push labeler (no score 3 slots, but score 3 appears in labels during regression)
- 5: cube at goal (obj_to_goal < 0.03)
- 4: cube pushed >50% toward goal OR obj_to_goal < 0.06
- 3: cube pushed >20% toward goal (meaningful progress)
- 2: end-effector near cube
- 1: no progress

### Stack labeler
- 5: cube stacked on cubeB (not grasped, xy aligned < 0.03, z at stack height)
- 4: cube grasped AND near cubeB (horiz < 0.08, z > cubeB_z)
- 3: cube grasped OR cube elevated (in the air)
- 2: end-effector near cubeA
- 1: no progress

## Diversity Requirement
Each generated case within the same score must show DISTINCT visual outcomes.
The HTML verification page should show visually different failure modes.

## Solver Stage References

### Push: stages [0, 1, 2]
- Stage 0: close_gripper
- Stage 1: reach behind cube
- Stage 2: push to goal

### Stack: stages [0, 1, 2, 3, 4, 5, 6]
- Stage 0: find grasp pose (dry run)
- Stage 1: reach above cube
- Stage 2: move to grasp pose
- Stage 3: close gripper
- Stage 4: lift cube
- Stage 5: align above cubeB (stack)
- Stage 6: open gripper (release)

### Pick: stages [0, 1, 2, 3]
- Stage 0: reach near cube
- Stage 1: move to grasp pose
- Stage 2: close gripper
- Stage 3: lift to goal position
