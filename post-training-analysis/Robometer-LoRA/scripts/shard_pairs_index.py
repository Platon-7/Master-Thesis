"""Split a pairs_index_<split>.jsonl into N round-robin shards for parallel
HF cache builds. Round-robin (line_num % N) keeps each shard's source-mix
representative of the whole, so per-shard cache build times are roughly equal.

Output:
  <splits_dir>/shards/pairs_index_<split>_part00.jsonl
  ...
  <splits_dir>/shards/pairs_index_<split>_part<N-1>.jsonl
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", type=Path, required=True,
                    help="dir containing pairs_index_<split>.jsonl")
    ap.add_argument("--split", required=True, help="split name (e.g. train, eval_droid)")
    ap.add_argument("--n-shards", type=int, default=16)
    args = ap.parse_args()

    src = args.splits_dir / f"pairs_index_{args.split}.jsonl"
    if not src.is_file():
        raise FileNotFoundError(src)

    out_dir = args.splits_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = [
        (out_dir / f"pairs_index_{args.split}_part{i:02d}.jsonl").open("w")
        for i in range(args.n_shards)
    ]
    counts = [0] * args.n_shards
    with src.open() as f:
        for i, line in enumerate(f):
            handles[i % args.n_shards].write(line)
            counts[i % args.n_shards] += 1
    for h in handles:
        h.close()

    print(f"sharded {src.name} → {out_dir}/")
    for i, n in enumerate(counts):
        print(f"  part{i:02d}: {n:,} rows")
    print(f"  total: {sum(counts):,} rows")


if __name__ == "__main__":
    main()
