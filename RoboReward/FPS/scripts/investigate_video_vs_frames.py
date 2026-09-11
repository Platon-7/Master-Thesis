#!/usr/bin/env python3
"""
Investigate the difference between feeding Qwen3-VL a video (mp4 path)
vs a list of pre-extracted frames.

This script does NOT run the model — it only traces what happens inside
qwen_vl_utils.process_vision_info and the Qwen3VL processor

Usage:
    conda activate roboreward
    python scripts/investigate_video_vs_frames.py
"""

import os, sys, json, torch, cv2, numpy as np
from PIL import Image
from pprint import pprint

# ── Make qwen_vl_utils importable ──────────────────────────────────────────
from qwen_vl_utils.vision_process import (
    fetch_video,
    smart_nframes,
    process_vision_info,
    FPS,           # the global default = 2.0
    FRAME_FACTOR,  # = 2
    FPS_MIN_FRAMES,
    FPS_MAX_FRAMES,
)

# ── Pick a sample video ───────────────────────────────────────────────────
SAMPLE_VIDEO = (
    "/var/scratch/pkarageo/my_success_case/"
    "Sun_Jul__9_18:59:44_2023/recordings/MP4/22246076.mp4"
)

TASK = "rearrange pillows on sofa"

SEPARATOR = "=" * 80


def get_video_info(path):
    """Get basic video info with OpenCV."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    cap.release()
    return fps, total, duration


def extract_frames_manually(path, target_fps=2.0):
    """Extract frames from video at target_fps, mimicking what fetch_video does."""
    cap = cv2.VideoCapture(path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Same logic as smart_nframes
    nframes = total_frames / video_fps * target_fps
    nframes = min(min(max(nframes, FPS_MIN_FRAMES), min(FPS_MAX_FRAMES, total_frames)), total_frames)
    nframes = int(nframes // FRAME_FACTOR * FRAME_FACTOR)

    indices = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()

    return frames, indices, video_fps, total_frames


# ============================================================================
# EXPERIMENT 1: Trace what fetch_video returns for mp4 vs frame list
# ============================================================================

def experiment_1_fetch_video():
    print(f"\n{SEPARATOR}")
    print("EXPERIMENT 1: fetch_video internals — MP4 path vs List of frames")
    print(SEPARATOR)

    video_fps, total_frames, duration = get_video_info(SAMPLE_VIDEO)
    print(f"\nVideo: {SAMPLE_VIDEO}")
    print(f"  Original FPS: {video_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f}s")

    # ── Path A: MP4 file path ──
    print(f"\n{'─'*60}")
    print("PATH A: Feeding MP4 file path")
    print(f"{'─'*60}")

    ele_mp4 = {"type": "video", "video": SAMPLE_VIDEO}
    (video_a, metadata_a), sample_fps_a = fetch_video(
        ele_mp4, return_video_sample_fps=True, return_video_metadata=True
    )

    print(f"  Output tensor shape: {video_a.shape}  (T, C, H, W)")
    print(f"  sample_fps (returned): {sample_fps_a:.4f}")
    print(f"  video_metadata:")
    print(f"    fps (= raw video fps):   {metadata_a['fps']}")
    print(f"    frames_indices:          {metadata_a['frames_indices'].tolist()}")
    print(f"    total_num_frames:        {metadata_a['total_num_frames']}")

    # ── Path B: List of frames ──
    print(f"\n{'─'*60}")
    print("PATH B: Feeding list of PIL frames (extracted at same FPS)")
    print(f"{'─'*60}")

    frames, manual_indices, _, _ = extract_frames_manually(SAMPLE_VIDEO, target_fps=2.0)
    print(f"  Extracted {len(frames)} frames at indices: {manual_indices}")

    ele_frames = {"type": "video", "video": frames}
    (video_b, metadata_b), sample_fps_b = fetch_video(
        ele_frames, return_video_sample_fps=True, return_video_metadata=True
    )

    print(f"  Output tensor shape: {video_b.shape}  (T, C, H, W)")
    print(f"  sample_fps (returned): {sample_fps_b:.4f}")
    print(f"  video_metadata:")
    print(f"    fps (= raw_fps, FAKE):   {metadata_b['fps']}")
    print(f"    frames_indices (FAKE):   {metadata_b['frames_indices']}")
    print(f"    total_num_frames (FAKE): {metadata_b['total_num_frames']}")

    # ── Compare ──
    print(f"\n{'─'*60}")
    print("COMPARISON")
    print(f"{'─'*60}")
    print(f"  Tensor shapes match:    {video_a.shape == video_b.shape}")
    print(f"  Tensor values match:    {torch.allclose(video_a, video_b, atol=1.0)}")
    if video_a.shape == video_b.shape:
        diff = (video_a.float() - video_b.float()).abs()
        print(f"  Max pixel difference:   {diff.max().item():.2f}")
        print(f"  Mean pixel difference:  {diff.mean().item():.4f}")

    print(f"\n  metadata['fps']:")
    print(f"    MP4 path:   {metadata_a['fps']}  (real video fps)")
    print(f"    Frame list: {metadata_b['fps']}  (defaults to sample_fps=2.0)")

    print(f"\n  metadata['frames_indices']:")
    print(f"    MP4 path:   {metadata_a['frames_indices'].tolist()}  (real sampled positions)")
    print(f"    Frame list: {metadata_b['frames_indices']}  (just 0,1,2,...)")

    return metadata_a, metadata_b


# ============================================================================
# EXPERIMENT 2: Show how metadata affects timestamp tokens
# ============================================================================

def experiment_2_timestamps(metadata_a, metadata_b):
    print(f"\n\n{SEPARATOR}")
    print("EXPERIMENT 2: How metadata becomes timestamp tokens")
    print(SEPARATOR)

    print("""
    The Qwen3VL processor computes timestamps using _calculate_timestamps():

        timestamps = [idx / video_fps for idx in frames_indices]
        # Then merge pairs (temporal_patch_size=2):
        timestamps = [(t[i] + t[i+1]) / 2 for i in range(0, len(t), 2)]

    These become the "<X.X seconds>" tokens in the prompt.
    """)

    merge_size = 2

    def calculate_timestamps(indices, video_fps):
        """Replicates Qwen3VLProcessor._calculate_timestamps"""
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend([indices[-1]] * (merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    # Path A: real metadata
    ts_a = calculate_timestamps(metadata_a['frames_indices'], metadata_a['fps'])
    # Path B: fake metadata
    ts_b = calculate_timestamps(metadata_b['frames_indices'], metadata_b['fps'])

    print(f"PATH A (MP4 path) — real metadata:")
    print(f"  fps used for division: {metadata_a['fps']} (real video fps)")
    print(f"  frames_indices: {metadata_a['frames_indices'].tolist()}")
    print(f"  Computed timestamps: {[f'{t:.1f}' for t in ts_a]}")
    print(f"  Prompt tokens: {' '.join(f'<{t:.1f} seconds>' for t in ts_a)}")

    print(f"\nPATH B (frame list) — fake metadata:")
    print(f"  fps used for division: {metadata_b['fps']} (fake, defaults to 2.0)")
    print(f"  frames_indices: {metadata_b['frames_indices']}")
    print(f"  Computed timestamps: {[f'{t:.1f}' for t in ts_b]}")
    print(f"  Prompt tokens: {' '.join(f'<{t:.1f} seconds>' for t in ts_b)}")

    print(f"\n{'─'*60}")
    print("KEY INSIGHT:")
    print(f"{'─'*60}")
    print(f"  Path A: timestamps reflect REAL positions in the original video")
    print(f"          e.g., frame 150 in a 30fps video = <5.0 seconds>")
    print(f"  Path B: timestamps are SYNTHETIC")
    print(f"          indices [0,1,2,...] / fake_fps(2.0) = [0.0, 0.5, 1.0, ...]")
    print(f"          These may not match the actual video timing at all!")


# ============================================================================
# EXPERIMENT 3: The FPS terminology explained
# ============================================================================

def experiment_3_fps_terminology():
    print(f"\n\n{SEPARATOR}")
    print("EXPERIMENT 3: FPS terminology — what each variable means")
    print(SEPARATOR)

    video_fps, total_frames, duration = get_video_info(SAMPLE_VIDEO)

    # Simulate what happens with different "fps" settings
    for requested_fps in [2.0, 10.0, 30.0]:
        nframes = total_frames / video_fps * requested_fps
        nframes = min(min(max(nframes, FPS_MIN_FRAMES), min(FPS_MAX_FRAMES, total_frames)), total_frames)
        nframes = int(nframes // FRAME_FACTOR * FRAME_FACTOR)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        actual_sample_fps = nframes / max(total_frames, 1e-6) * video_fps

        print(f"\n  Requested fps={requested_fps}:")
        print(f"    nframes extracted:  {nframes}")
        print(f"    sample_fps:         {actual_sample_fps:.2f}")
        print(f"    (sample_fps = nframes / total_frames * video_fps)")

    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    FPS TERMINOLOGY GUIDE                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  "fps" (in config dict)                                         │
    │     = The DESIRED sampling rate you request.                    │
    │     = Default: 2.0 (hardcoded in qwen_vl_utils as FPS=2.0)     │
    │     = Used by smart_nframes to compute how many frames to grab. │
    │     = Formula: nframes = total_frames / video_fps * fps         │
    │                                                                 │
    │  "video_fps" / metadata["fps"] / "raw_fps"                      │
    │     = The ORIGINAL fps of the video file (e.g., 30fps).         │
    │     = Read from the video decoder (torchvision/decord).         │
    │     = For frame lists: defaults to sample_fps (FAKE!)           │
    │     = Used to compute timestamps: timestamp = frame_idx / fps   │
    │                                                                 │
    │  "sample_fps" (returned by fetch_video)                         │
    │     = The ACTUAL effective fps after sampling.                   │
    │     = Formula: nframes / total_frames * video_fps               │
    │     = Close to requested "fps" but not exact due to rounding.   │
    │     = Passed to processor via video_kwargs for bookkeeping.     │
    │                                                                 │
    │  Example: 30fps video, 10s long (300 frames), requested fps=2:  │
    │     nframes = 300 / 30 * 2 = 20 frames                         │
    │     sample_fps = 20 / 300 * 30 = 2.0                           │
    │     raw_fps (metadata) = 30.0                                   │
    │     timestamps = frame_indices / 30.0 (real video time)         │
    │                                                                 │
    │  Video file: {SAMPLE_VIDEO.split('/')[-1]:>25}                  │
    │     video_fps (raw): {video_fps:>10.1f}                         │
    │     total_frames:    {total_frames:>10d}                        │
    │     duration:        {duration:>10.2f}s                         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# EXPERIMENT 4: Full process_vision_info comparison
# ============================================================================

def experiment_4_process_vision_info():
    print(f"\n{SEPARATOR}")
    print("EXPERIMENT 4: Full process_vision_info output comparison")
    print(SEPARATOR)

    try:
        # Extract frames first
        frames, indices, video_fps, total_frames = extract_frames_manually(SAMPLE_VIDEO, target_fps=2.0)

        # ── Path A: mp4 ──
        print("\nProcessing Path A (MP4 path)...")
        messages_mp4 = [{
            "role": "user",
            "content": [
                {"type": "video", "video": SAMPLE_VIDEO},
                {"type": "text", "text": f"Task: {TASK}"}
            ]
        }]

        img_a, vid_a, kwargs_a = process_vision_info(messages_mp4, return_video_kwargs=True)

        print(f"\nPATH A (MP4 path):")
        print(f"  image_inputs:  {len(img_a) if img_a else 0} images")
        print(f"  video_inputs:  {len(vid_a) if vid_a else 0} videos")
        print(f"  video tensor:  {vid_a[0].shape}")
        print(f"  video_kwargs:  {kwargs_a}")

        # ── Path B: frame list ──
        print("\nProcessing Path B (frame list)...")
        messages_frames = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": f"Task: {TASK}"}
            ]
        }]

        img_b, vid_b, kwargs_b = process_vision_info(messages_frames, return_video_kwargs=True)

        print(f"\nPATH B (frame list):")
        print(f"  image_inputs:  {len(img_b) if img_b else 0} images")
        print(f"  video_inputs:  {len(vid_b) if vid_b else 0} videos")
        print(f"  video tensor:  {vid_b[0].shape}")
        print(f"  video_kwargs:  {kwargs_b}")

        # ── Compare tensors ──
        print(f"\nTensor comparison:")
        print(f"  Shapes match: {vid_a[0].shape == vid_b[0].shape}")
        if vid_a[0].shape == vid_b[0].shape:
            diff = (vid_a[0] - vid_b[0]).abs()
            print(f"  Max diff:  {diff.max().item():.4f}")
            print(f"  Mean diff: {diff.mean().item():.6f}")
            print(f"  Are identical: {torch.equal(vid_a[0], vid_b[0])}")
            if not torch.equal(vid_a[0], vid_b[0]):
                print(f"  (Differences due to double-resize in frame list path)")

    except Exception as e:
        print(f"\n  ERROR in experiment 4: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not os.path.exists(SAMPLE_VIDEO):
        print(f"ERROR: Video not found: {SAMPLE_VIDEO}")
        sys.exit(1)

    print(SEPARATOR)
    print("  Qwen3-VL Video Input Investigation")
    print("  MP4 file path vs List of pre-extracted frames")
    print(SEPARATOR)

    meta_a, meta_b = experiment_1_fetch_video()
    experiment_2_timestamps(meta_a, meta_b)
    experiment_3_fps_terminology()
    experiment_4_process_vision_info()

    print(f"\n{SEPARATOR}")
    print("DONE — All experiments complete!")
    print(SEPARATOR)
