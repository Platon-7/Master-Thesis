import random
import re
import time


gvl_system_prompt = """You are an expert roboticist tasked with predicting task completion percentages.

The task completion percentage must be between 0 and 100, where:
- 0% corresponds to the initial state
- 100% corresponds to full task completion

Frames may be presented in random order. Do not assume any temporal ordering.
Estimate completion based only on the visual state of each frame.

For each frame, output strictly:
"Frame {i}: Task Completion Percentages:{}%"
Do not include any additional text.
"""

gvl_prompt_intro = """You are an expert roboticist tasked with predicting task completion percentages for a robot performing the following task: "{task_description}". The task completion percentages are between 0 and 100, where 100 corresponds to full task completion.

We provide several examples of the robot performing the task at various stages and their corresponding task completion percentages. Note that these frames are in random order, so please pay attention to the individual frames when reasoning about task completion percentage.

Example frames (with known completion):
"""

gvl_prompt_new_episode = """
Now consider a new episode.

Initial frame of this episode:
"""

gvl_prompt_query = """
In the initial robot scene, the task completion percentage is 0.

Now, estimate task completion for the task: "{task_description}". Output the task completion percentage for the following frames that are presented in random order. For each frame, format your response as follow: Frame {{i}}: Task Completion Percentages:{{}}%
"""


def GVL_extract_perc(response_str: str, frame_index: int) -> int:
    """Extract the task-completion percentage for a given frame index from a VLM response."""
    pattern = rf"Frame\s+{frame_index}:\s.*?Task Completion Percentages:\s*(\d+)%"
    match = re.search(pattern, response_str, re.DOTALL)
    if not match:
        raise ValueError(f"Frame {frame_index} not found in response.")
    perc = int(match.group(1))
    assert 0 <= perc <= 100, perc
    return perc


def GVL_prompt_eval(
    model,
    processor,
    prompt_vlm,
    task_description,
    example_frames,
    example_percs,
    test_init_frame,
    test_frames,
    debug,
):
    start = time.perf_counter()
    content = []

    content.append({"type": "text", "text": gvl_prompt_intro.format(task_description=task_description)})

    examples = list(zip(example_frames, example_percs))
    random.shuffle(examples)

    for i, (ex_frame, ex_perc) in enumerate(examples):
        content.append({"type": "text", "text": f"\nFrame {i}:"})
        content.append({"type": "image", "image": ex_frame})
        content.append({"type": "text", "text": f"\nTask completion: {ex_perc}%"})

    content.append({"type": "text", "text": gvl_prompt_new_episode})
    content.append({"type": "image", "image": test_init_frame})
    content.append({"type": "text", "text": gvl_prompt_query.format(task_description=task_description)})

    indexed_frames = list(enumerate(test_frames))
    random.shuffle(indexed_frames)

    orig_to_shuffled = {}
    last_index = len(indexed_frames) - 1
    for new_idx, (orig_idx, frame) in enumerate(indexed_frames):
        orig_to_shuffled[orig_idx] = new_idx
        content.append({"type": "text", "text": f"\nFrame {new_idx}:"})
        content.append({"type": "image", "image": frame})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": gvl_system_prompt}]},
        {"role": "user", "content": content},
    ]
    precise_prompt = dict(max_new_tokens=500, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw_output = prompt_vlm(
        model=model, processor=processor, messages=messages, debug=debug, prompt_kwargs=precise_prompt
    )
    if debug:
        elapsed_seconds = time.perf_counter() - start
        print(f".......... Elapsed time: {elapsed_seconds:.3f} .................")
        for i in range(last_index + 1):
            print(i, "->", orig_to_shuffled[i])

    return raw_output, GVL_extract_perc(raw_output, orig_to_shuffled[last_index])


def reward_from_GVL(
    model, processor, prompt_vlm, task_description, example_frames, example_percs, test_init_frame, test_frames
):
    _, perc = GVL_prompt_eval(
        model,
        processor,
        prompt_vlm,
        task_description,
        example_frames,
        example_percs,
        test_init_frame,
        test_frames,
        debug=False,
    )
    return perc / 100.0  # normalize to [0, 1]
