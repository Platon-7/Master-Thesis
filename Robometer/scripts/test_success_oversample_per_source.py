#!/usr/bin/env python3
"""End-to-end test of success_oversample_per_source.

Verifies the full chain Snellius runs will exercise:
  1. yaml override survives the job-script sed pipeline and emerges as a valid
     Hydra ++ override
  2. DataConfig accepts success_oversample_per_source as Optional[Dict[str, float]]
  3. The trainer's WeightedRandomSampler computes weights that match the
     advertised semantic (failure_oversample × failure-weight, per-source
     multiplier on success-weight)
  4. Empirically sampling from the constructed sampler reproduces the
     theoretical batch-fraction distribution to within ~1 percentage point

Run:
    /home/pkarageorgis1/.conda/envs/robometer_qwen35_gpu/bin/python3 \
        /gpfs/home3/pkarageorgis1/Master-Thesis/Robometer/scripts/test_success_oversample_per_source.py
"""
import sys
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, "/gpfs/home3/pkarageorgis1/Master-Thesis/Robometer")


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ===========================================================================
section("TEST 1 — sed pipeline emits a valid Hydra ++ override")
# ===========================================================================
YAML_LINE = "data.success_oversample_per_source: {metaworld: 13.0}"
pipeline = (
    f"echo '{YAML_LINE}' | sed -e 's/[[:space:]]*#.*//' "
    "| grep -v '^[[:space:]]*$' "
    "| sed -e 's/: */=/' -e 's/^/++/'"
)
out = subprocess.check_output(pipeline, shell=True, executable="/bin/bash").decode().strip()
print(f"  yaml line:   {YAML_LINE!r}")
print(f"  sed output:  {out!r}")
expected = "++data.success_oversample_per_source={metaworld: 13.0}"
assert out == expected, f"sed mismatch: got {out!r}, want {expected!r}"
print("  ✓ sed pipeline emits the expected ++ override")

from omegaconf import OmegaConf
parsed = OmegaConf.from_dotlist([out.lstrip("+")])
got = dict(parsed.data.success_oversample_per_source)
assert got == {"metaworld": 13.0}, f"OmegaConf parse mismatch: got {got!r}"
print(f"  ✓ OmegaConf parses to dict {got}")


# ===========================================================================
section("TEST 2 — DataConfig accepts the new field")
# ===========================================================================
from robometer.configs.experiment_configs import DataConfig

c_default = DataConfig()
assert hasattr(c_default, "success_oversample_per_source")
assert c_default.success_oversample_per_source is None
print(f"  default: success_oversample_per_source = {c_default.success_oversample_per_source}")

c_with = DataConfig(success_oversample_per_source={"metaworld": 13.0, "failsafe": 2.0})
assert c_with.success_oversample_per_source == {"metaworld": 13.0, "failsafe": 2.0}
print(f"  with override: {c_with.success_oversample_per_source}")
print("  ✓ DataConfig accepts the field")


# ===========================================================================
section("TEST 3 — sampler weight computation matches semantic")
# ===========================================================================
# Reproduce the exact weight-construction logic from rbm_heads_trainer.py
# (extracted because we don't want to instantiate the full Trainer in a test).

def build_weights(quality_labels, data_sources, failure_oversample, succ_per_src):
    """Mirror rbm_heads_trainer.py weights construction."""
    succ_per_src = {str(k): float(v) for k, v in (succ_per_src or {}).items()}
    weights = []
    for lbl, src in zip(quality_labels, data_sources):
        if lbl != "successful":
            weights.append(failure_oversample)
        else:
            w = 1.0
            if succ_per_src and src in succ_per_src:
                w *= succ_per_src[src]
            weights.append(w)
    return torch.tensor(weights, dtype=torch.double)


# Synthetic dataset that mirrors the real train_raw composition (scaled down by 10×):
# (1,946 mw_succ + 27,010 mw_fail) / 10 ≈ (195 + 2,701)
# Plus a comparable amount of non-metaworld successes and failures to make the
# proportions realistic.
N_MW_SUCC = 195
N_MW_FAIL = 2701
N_OTHER_SUCC = 74_300
N_OTHER_FAIL = 10_306

quality = (
    ["successful"] * N_MW_SUCC
    + ["failure"] * N_MW_FAIL
    + ["successful"] * N_OTHER_SUCC
    + ["failure"] * N_OTHER_FAIL
)
sources = (
    ["metaworld"] * N_MW_SUCC
    + ["metaworld"] * N_MW_FAIL
    + ["droid"] * N_OTHER_SUCC
    + ["droid"] * N_OTHER_FAIL
)
N_TOTAL = len(quality)
assert N_TOTAL == N_MW_SUCC + N_MW_FAIL + N_OTHER_SUCC + N_OTHER_FAIL

# Case A: no boost (current production behavior on cancelled run4)
W_A = build_weights(quality, sources, failure_oversample=2.61, succ_per_src=None)
# Each mw_succ: weight 1.0; each mw_fail: weight 2.61
assert W_A[0].item() == 1.0
assert W_A[N_MW_SUCC].item() == 2.61
# Total weight breakdown
mw_succ_w_A = N_MW_SUCC * 1.0
mw_fail_w_A = N_MW_FAIL * 2.61
total_w_A = float(W_A.sum().item())
print(f"  Case A (no boost): total weight = {total_w_A:,.1f}")
print(f"    metaworld success weight share: {mw_succ_w_A/total_w_A*100:.3f}%   (expected per-batch fraction)")
print(f"    metaworld failure weight share: {mw_fail_w_A/total_w_A*100:.3f}%")

# Case B: with the proposed boost
W_B = build_weights(quality, sources, failure_oversample=2.61, succ_per_src={"metaworld": 13.0})
assert W_B[0].item() == 13.0, f"mw success weight should be 13.0, got {W_B[0].item()}"
assert W_B[N_MW_SUCC].item() == 2.61, "mw failure weight unchanged"
assert W_B[-1].item() == 2.61, "non-mw failure weight unchanged"
assert W_B[N_MW_SUCC + N_MW_FAIL].item() == 1.0, "non-mw success weight unchanged"
mw_succ_w_B = N_MW_SUCC * 13.0
mw_fail_w_B = N_MW_FAIL * 2.61
total_w_B = float(W_B.sum().item())
print(f"  Case B (mw=13×): total weight = {total_w_B:,.1f}")
print(f"    metaworld success weight share: {mw_succ_w_B/total_w_B*100:.3f}%   ← boosted")
print(f"    metaworld failure weight share: {mw_fail_w_B/total_w_B*100:.3f}%   ← unchanged")
print("  ✓ weight construction matches semantic (successes boosted by 13×, failures untouched)")


# ===========================================================================
section("TEST 4 — empirical draws reproduce theoretical fractions (< 1pp error)")
# ===========================================================================
# Draw a large number of samples and compare observed mw_succ rate to theory.

N_DRAWS = 200_000  # roughly comparable to a 3000-step run on the 192k cancelled run
gen = torch.Generator().manual_seed(42)
sampler_B = WeightedRandomSampler(W_B, num_samples=N_DRAWS, replacement=True, generator=gen)
drawn = list(sampler_B)
counts = Counter()
for idx in drawn:
    counts[(quality[idx], sources[idx])] += 1

# Theoretical: per-draw probability = weight / total_weight
def theory_count(weight_share):
    return N_DRAWS * weight_share

theory_mw_succ_share = mw_succ_w_B / total_w_B
theory_mw_fail_share = mw_fail_w_B / total_w_B

obs_mw_succ = counts.get(("successful", "metaworld"), 0)
obs_mw_fail = counts.get(("failure", "metaworld"), 0)
obs_mw_succ_share = obs_mw_succ / N_DRAWS
obs_mw_fail_share = obs_mw_fail / N_DRAWS

# Also estimate unique mw_succ visits (coupon-collector)
unique_mw_succ_seen = len({idx for idx in drawn if sources[idx] == "metaworld" and quality[idx] == "successful"})

print(f"  {'metric':<35s} {'theory':>10s} {'observed':>10s} {'abs error':>10s}")
print(f"  " + "-" * 70)
print(f"  {'mw succ batch share':<35s} {theory_mw_succ_share*100:>9.3f}% {obs_mw_succ_share*100:>9.3f}% {abs(theory_mw_succ_share-obs_mw_succ_share)*100:>9.3f} pp")
print(f"  {'mw fail batch share':<35s} {theory_mw_fail_share*100:>9.3f}% {obs_mw_fail_share*100:>9.3f}% {abs(theory_mw_fail_share-obs_mw_fail_share)*100:>9.3f} pp")
print(f"  {'mw succ total draws':<35s} {theory_count(theory_mw_succ_share):>10.0f} {obs_mw_succ:>10d}")
print(f"  {'mw succ unique seen (of {})':<35s}".format(N_MW_SUCC), f" — — — — — {unique_mw_succ_seen:>10d} ({unique_mw_succ_seen/N_MW_SUCC*100:.1f}%)")

ERR_TOL = 0.01  # 1pp error tolerance on the 200k-draw sample
assert abs(obs_mw_succ_share - theory_mw_succ_share) < ERR_TOL
assert abs(obs_mw_fail_share - theory_mw_fail_share) < ERR_TOL
print("  ✓ observed shares within 1pp of theory")


# ===========================================================================
section("TEST 5 — Case A vs Case B side-by-side, showing the improvement")
# ===========================================================================
gen_A = torch.Generator().manual_seed(42)
sampler_A = WeightedRandomSampler(W_A, num_samples=N_DRAWS, replacement=True, generator=gen_A)
drawn_A = list(sampler_A)
obs_succ_A = sum(1 for idx in drawn_A if sources[idx] == "metaworld" and quality[idx] == "successful")
unique_succ_A = len({idx for idx in drawn_A if sources[idx] == "metaworld" and quality[idx] == "successful"})

print(f"  {'recipe':<25s} {'mw_succ draws':>15s} {'unique seen':>12s} {'% of 195':>10s} {'avg visits':>11s}")
print(f"  " + "-" * 80)
print(f"  {'A: default (no boost)':<25s} {obs_succ_A:>15d} {unique_succ_A:>12d} {unique_succ_A/N_MW_SUCC*100:>9.1f}% {obs_succ_A/N_MW_SUCC:>10.2f}")
print(f"  {'B: metaworld=13.0':<25s} {obs_mw_succ:>15d} {unique_mw_succ_seen:>12d} {unique_mw_succ_seen/N_MW_SUCC*100:>9.1f}% {obs_mw_succ/N_MW_SUCC:>10.2f}")
boost = obs_mw_succ / max(obs_succ_A, 1)
print(f"  → effective boost: {boost:.1f}× more metaworld success exposures with the new knob")
assert boost > 10, f"expected ~13× boost, got {boost:.1f}×"
print(f"  ✓ boost in observed draws ≈ configured 13× multiplier")


section("ALL TESTS PASSED")
