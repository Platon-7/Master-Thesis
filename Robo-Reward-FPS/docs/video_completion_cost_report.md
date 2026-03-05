# Video Completion Cost Report

## Overview

This report estimates costs for completing temporally clipped videos from the RoboReward dataset using commercial video generation APIs. Temporal clips are truncated versions of successful robot manipulation episodes where the success ending has been removed.

**Important distinction:** The `_attempt_N_score_N` filename suffix in RoboReward is used for two different augmentation types:
- **Temporal clipping** (11,116 videos): Same task, truncated video — the episode is cut short at different progress levels
- **Counterfactual relabeling** (3,228 videos): Same video, different task instruction — the task is swapped to one that doesn't match the outcome

Only temporal clips are relevant for video completion. Counterfactual entries reuse the original untruncated video with a swapped task, so there is nothing to "complete."

## Dataset Numbers

| Metric | Value |
|--------|-------|
| Total RoboReward videos | 45,072 |
| Total `_attempt_` entries | 14,479 |
| True temporal clips (same task) | **11,116** |
| Counterfactual relabeling (different task) | 3,228 |
| Unmatched (no original found) | 135 |
| Source datasets | 28 |
| Estimated download size (clips only) | ~1.3 GB |

## Video Duration Estimates

| Metric | Value |
|--------|-------|
| Average full video duration | ~31s |
| Average clip duration (truncated) | ~9.5s |
| Average seconds to generate (completion) | ~21.5s |
| Total seconds to generate (completion) | 238,994s (66.4 hrs) |
| Total seconds from scratch | 344,596s (95.7 hrs) |
| **Savings from completion vs scratch** | **31%** |

## API Cost Comparison

### Per-Second Billing Models

| Model | $/sec | Completion Cost | From Scratch Cost | Savings |
|-------|-------|----------------|-------------------|---------|
| Kling V2.5 Turbo Std | $0.042 | **$10,038** | $14,473 | $4,435 |
| Runway Gen-4 Turbo | $0.050 | **$11,950** | $17,230 | $5,280 |
| Sora 2 Standard (720p) | $0.100 | **$23,899** | $34,460 | $10,560 |
| Sora 2 Pro (720p) | $0.300 | **$71,698** | $103,379 | $31,681 |

### Per-Video Billing: NVIDIA Cosmos Predict 2.5

Cosmos Predict 2.5 uses **per-video flat-rate pricing** instead of per-second billing. Each API call generates a fixed **5.8-second clip** (93 frames @ 16fps) at 1280x704 resolution. It supports a **video-to-video mode** that conditions generation on an input video, making it suitable for video completion/extension.

**API calls needed per clip:**
- Completion (~21.5s): ceil(21.5 / 5.8) = **4 calls per clip**
- From scratch (~31s): ceil(31 / 5.8) = **6 calls per clip**

| Provider | $/video | Completion Cost | From Scratch Cost | Savings |
|----------|---------|----------------|-------------------|---------|
| fal.ai | $0.20 | **$8,893** (44,464 calls) | **$13,339** (66,696 calls) | $4,446 |
| WaveSpeedAI | $0.25 | **$11,116** (44,464 calls) | **$16,674** (66,696 calls) | $5,558 |

**Cosmos-specific notes:**
- Output resolution (1280x704) is much higher than RoboReward input (320x192) — will need downsampling
- Each call takes ~7 minutes to process, so 44,464 calls would take ~216 days sequentially; parallelism is essential
- Video-to-video mode uses the input video as a structural guide with text prompting, not pure autoregressive frame prediction
- The 2B parameter model is open-source (Apache 2.0) and can also be self-hosted on GPU — this eliminates API costs entirely but requires compute

## Caveats

1. **Video extension support**: Not all models support extending arbitrary uploaded videos. Veo 3.1, for example, does NOT support this. Kling, Runway, Sora 2, and Cosmos Predict 2.5 (video-to-video) do support video-conditioned generation in various forms.

2. **Resolution mismatch**: RoboReward videos are 320x192 @ 10fps — much lower resolution and framerate than typical video generation model output (720p+ @ 24fps). This means:
   - Generated completions will need to be downsampled to match
   - We may be paying for higher-quality output than we need
   - Some models may not accept such low-resolution inputs

3. **Billing granularity**: These are ballpark estimates. Actual costs depend on:
   - Whether the API charges per-second or per-video with fixed duration tiers
   - Minimum video length requirements
   - Whether there are per-request overhead costs
   - For Cosmos: the fixed 5.8s output means we may generate slightly more video than needed per call

4. **Quality considerations**: Robot manipulation videos have specific visual characteristics (fixed cameras, precise object interactions) that may challenge general-purpose video generation models. Quality assessment will be needed after generation.

5. **Self-hosting option**: Cosmos Predict 2.5 (2B) is open-source and can be run locally. On a single A100 GPU, generation takes ~7 minutes per 5.8s clip. With access to HPC GPU nodes, this could eliminate API costs entirely — see the self-hosting estimate below.

## Self-Hosting Estimate (Cosmos Predict 2.5)

| Resource | Value |
|----------|-------|
| Model size | 2B parameters |
| GPU requirement | 1x A100 (40GB+) |
| Time per 5.8s clip | ~7 minutes |
| Total API calls needed (completion) | 44,464 |
| Total GPU-hours (1 GPU) | ~5,187 hrs (~216 days) |
| Total GPU-hours (8 GPUs parallel) | ~648 hrs (~27 days) |
| Cloud cost (A100 @ ~$2/hr) | ~$1,297 (8 GPUs) |

Self-hosting is dramatically cheaper than any API option if GPU compute is available.

## Recommendation

For a pilot study, start with a small subset (~100 clips) using either:
- **fal.ai Cosmos Predict 2.5** ($0.20/video, ~$80 for 100 clips with 4 calls each) — cheapest API option with video-to-video support
- **Kling V2.5 Turbo** (~$90 for 100 clips) — per-second billing, simpler single-call workflow

For full-scale generation, **self-hosting Cosmos Predict 2.5** on HPC GPUs is the most cost-effective approach if compute is available.
