#!/usr/bin/env python3
"""Copy the first N episodes of a converted demo h5 into a new file.

Cheaper than re-running convert_maniskill_demos.py, which would replay every
episode through the simulator again just to keep a prefix of them.
"""
import argparse, h5py

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--num", type=int, required=True)
    a = ap.parse_args()
    with h5py.File(a.src, "r") as fi, h5py.File(a.dst, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        di, do = fi["data"], fo.create_group("data")
        keys = sorted(di.keys(), key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))[: a.num]
        for k in keys:
            di.copy(k, do)
        print(f"{a.dst}: {len(keys)} episodes (from {len(di.keys())})")

if __name__ == "__main__":
    main()
