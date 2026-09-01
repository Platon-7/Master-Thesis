#!/usr/bin/env python3
"""Clip demo actions away from the tanh saturation boundary, in place on a copy.

A tanh-squashed Gaussian actor parameterises a = tanh(u), so a demo action of
|a| = 1-1e-6 demands a pre-squash mean of atanh(0.999999) = 7.25. Measured on our
converted ManiSkill demos, 23-25% of all action components sit at that boundary,
which makes Gaussian-NLL cloning (the codebase's BC and IQL/AWR actor losses)
structurally unable to fit them and inflates log_prob to ~50. Clipping to 0.99
puts the target at atanh(0.99) = 2.65, which the actor can actually reach.
"""
import argparse, h5py, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--clip", type=float, default=0.99)
    a = ap.parse_args()
    with h5py.File(a.h5, "r+") as f:
        d, n_sat, n_tot = f["data"], 0, 0
        for k in d.keys():
            act = np.array(d[k]["actions"])
            n_sat += int((np.abs(act) >= 0.999).sum()); n_tot += act.size
            d[k]["actions"][...] = np.clip(act, -a.clip, a.clip)
        f.attrs["actions_clipped_to"] = a.clip
        print(f"{a.h5}: clipped to +-{a.clip}  ({n_sat}/{n_tot} = {n_sat/n_tot:.1%} were saturated)")

if __name__ == "__main__":
    main()
