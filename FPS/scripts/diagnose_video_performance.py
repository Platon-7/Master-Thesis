#!/usr/bin/env python3
"""
Video Performance Diagnostic Tool
Measures actual video reading and processing times to understand speed differences
"""

import os
import sys
import time
import glob
import json
import cv2
import torch
from typing import Dict, List, Tuple
from torchvision import io

def get_video_properties_cv2(video_path: str) -> Dict:
    """Get video properties using OpenCV"""
    cap = cv2.VideoCapture(video_path)

    props = {
        'path': video_path,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'bitrate': cap.get(cv2.CAP_PROP_BITRATE) if hasattr(cv2, 'CAP_PROP_BITRATE') else None,
    }

    # Decode fourcc
    fourcc = props['fourcc']
    props['codec'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    props['duration_sec'] = props['frame_count'] / props['fps'] if props['fps'] > 0 else 0
    props['resolution'] = f"{props['width']}x{props['height']}"
    props['total_pixels'] = props['width'] * props['height'] * props['frame_count']

    cap.release()
    return props


def measure_torchvision_read_time(video_path: str, num_trials: int = 3) -> Dict:
    """Measure how long torchvision takes to read a video"""
    times = []

    for _ in range(num_trials):
        start = time.time()
        try:
            video, audio, info = io.read_video(
                video_path,
                pts_unit="sec",
                output_format="TCHW",
            )
            elapsed = time.time() - start
            times.append(elapsed)

            result = {
                'success': True,
                'mean_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'frames_loaded': video.size(0),
                'video_fps': info.get('video_fps', 0),
            }
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'mean_time': None,
            }
            break

    return result


def analyze_dataset(video_paths: List[str], label: str, max_videos: int = 10):
    """Analyze a set of videos"""
    print(f"\n{'='*80}")
    print(f"📊 Analyzing {label} Dataset")
    print(f"{'='*80}")

    results = []

    for i, video_path in enumerate(video_paths[:max_videos], 1):
        print(f"\n[{i}/{min(len(video_paths), max_videos)}] {os.path.basename(video_path)}")

        # Get properties
        props = get_video_properties_cv2(video_path)
        print(f"  Properties: {props['resolution']}, {props['fps']:.1f} FPS, "
              f"{props['frame_count']} frames, {props['duration_sec']:.1f}s, codec: {props['codec']}")

        # Measure read time
        print(f"  Measuring torchvision read time...", end=' ', flush=True)
        timing = measure_torchvision_read_time(video_path, num_trials=3)

        if timing['success']:
            print(f"{timing['mean_time']:.3f}s (min: {timing['min_time']:.3f}s, max: {timing['max_time']:.3f}s)")
            print(f"  Frames loaded: {timing['frames_loaded']}, "
                  f"Speed: {props['frame_count'] / timing['mean_time']:.1f} frames/sec")
        else:
            print(f"FAILED: {timing['error']}")

        results.append({
            'filename': os.path.basename(video_path),
            'properties': props,
            'timing': timing,
        })

    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute aggregate statistics"""
    successful = [r for r in results if r['timing']['success']]

    if not successful:
        return {'error': 'No successful reads'}

    return {
        'num_videos': len(successful),
        'avg_read_time': sum(r['timing']['mean_time'] for r in successful) / len(successful),
        'avg_frames': sum(r['properties']['frame_count'] for r in successful) / len(successful),
        'avg_resolution': sum(r['properties']['width'] * r['properties']['height'] for r in successful) / len(successful),
        'avg_fps': sum(r['properties']['fps'] for r in successful) / len(successful),
        'avg_duration': sum(r['properties']['duration_sec'] for r in successful) / len(successful),
        'codecs': list(set(r['properties']['codec'] for r in successful)),
        'resolutions': list(set(r['properties']['resolution'] for r in successful)),
        'total_pixels_per_sec': sum(r['properties']['total_pixels'] / r['timing']['mean_time']
                                     for r in successful) / len(successful),
    }


def main():
    print("="*80)
    print("🔬 VIDEO PERFORMANCE DIAGNOSTIC")
    print("="*80)

    # Find droid_general videos
    droid_success = glob.glob('/var/scratch/pkarageo/roboreward_tasks/droid_general/success/*.mp4')
    droid_failure = glob.glob('/var/scratch/pkarageo/roboreward_tasks/droid_general/failure/*.mp4')
    droid_videos = droid_success + droid_failure

    # Find pillows videos
    pillows_success = glob.glob('/var/scratch/pkarageo/my_success_case/**/recordings/MP4/*.mp4', recursive=True)
    pillows_failure = glob.glob('/var/scratch/pkarageo/my_failure_case/**/recordings/MP4/*.mp4', recursive=True)
    pillows_videos = pillows_success + pillows_failure

    print(f"\nFound {len(droid_videos)} droid videos")
    print(f"Found {len(pillows_videos)} pillows videos")

    # Analyze both datasets
    droid_results = analyze_dataset(droid_videos, "DROID", max_videos=10)
    pillows_results = analyze_dataset(pillows_videos, "PILLOWS", max_videos=10)

    # Compare statistics
    print(f"\n{'='*80}")
    print("📈 COMPARATIVE STATISTICS")
    print(f"{'='*80}")

    droid_stats = compute_statistics(droid_results)
    pillows_stats = compute_statistics(pillows_results)

    print(f"\n{'Metric':<30} {'DROID':>15} {'PILLOWS':>15} {'Ratio':>10}")
    print("-"*80)

    metrics = [
        ('Avg read time (sec)', 'avg_read_time'),
        ('Avg frames', 'avg_frames'),
        ('Avg FPS', 'avg_fps'),
        ('Avg duration (sec)', 'avg_duration'),
        ('Avg resolution (pixels)', 'avg_resolution'),
        ('Processing speed (pixels/sec)', 'total_pixels_per_sec'),
    ]

    for label, key in metrics:
        if key in droid_stats and key in pillows_stats:
            d_val = droid_stats[key]
            p_val = pillows_stats[key]
            ratio = d_val / p_val if p_val > 0 else 0
            print(f"{label:<30} {d_val:>15.2f} {p_val:>15.2f} {ratio:>10.2f}x")

    print(f"\n{'Codecs':<30} {', '.join(droid_stats['codecs']):>15} {', '.join(pillows_stats['codecs']):>15}")
    print(f"{'Unique resolutions':<30} {len(droid_stats['resolutions']):>15} {len(pillows_stats['resolutions']):>15}")

    # Save detailed results
    output = {
        'droid': {
            'results': droid_results,
            'statistics': droid_stats,
        },
        'pillows': {
            'results': pillows_results,
            'statistics': pillows_stats,
        },
    }

    output_path = '/var/scratch/pkarageo/roboreward_results/video_performance_diagnostic.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_path}")

    print(f"\n{'='*80}")
    print("🔍 KEY FINDINGS")
    print(f"{'='*80}")

    if droid_stats['avg_read_time'] < pillows_stats['avg_read_time']:
        speedup = pillows_stats['avg_read_time'] / droid_stats['avg_read_time']
        print(f"✓ DROID videos read {speedup:.2f}x FASTER than pillows videos")
        print(f"  DROID: {droid_stats['avg_read_time']:.3f}s avg")
        print(f"  Pillows: {pillows_stats['avg_read_time']:.3f}s avg")
    else:
        speedup = droid_stats['avg_read_time'] / pillows_stats['avg_read_time']
        print(f"✓ Pillows videos read {speedup:.2f}x FASTER than DROID videos")

    if droid_stats['codecs'] != pillows_stats['codecs']:
        print(f"\n✓ Different codecs detected:")
        print(f"  DROID: {droid_stats['codecs']}")
        print(f"  Pillows: {pillows_stats['codecs']}")

    if abs(droid_stats['avg_resolution'] - pillows_stats['avg_resolution']) > 10000:
        print(f"\n✓ Significant resolution difference:")
        print(f"  DROID: {droid_stats['avg_resolution']:.0f} pixels")
        print(f"  Pillows: {pillows_stats['avg_resolution']:.0f} pixels")


if __name__ == "__main__":
    main()
