"""Convert the MetaWorld figure's error bands from standard deviation to standard error.

The bands are filled polygons of mean +/- population sd over 5 seeds. Verified
against the raw per-seed data on W&B: base4b_coffeepush at step 5000 gives
mean 0.050 and sd(ddof=0) 0.100, and the figure's band top is 0.150.

With n = 5, s.e. = sd / sqrt(5), so each band is rescaled about its own mean.
Polygon layout is 8 upper points with ascending x, then 8 lower points with
descending x. The lower edge was clipped at zero in places, so sd is recovered
from the upper edge, which is not clipped, and the new lower edge is re-clipped.
"""
import re, sys, math
import numpy as np

N_SEEDS = 5
SCALE = 1.0 / math.sqrt(N_SEEDS)
COORD = re.compile(r"\(([-\d.eE+]+),([-\d.eE+]+)\)")


def coords(line):
    return [(float(a), float(b)) for a, b in COORD.findall(line)]


def fmt(pts):
    out = []
    for x, y in pts:
        xs = f"{x:g}"
        ys = f"{y:.4f}".rstrip("0").rstrip(".")
        out.append(f"({xs},{ys if ys else '0'})")
    return " ".join(out)


def main(src, dst):
    lines = open(src).read().split("\n")
    means = {}                                  # (panel, color) -> {x: mean}
    panel = -1
    for ln in lines:
        if ln.startswith("\\nextgroupplot"):
            panel += 1
        m = re.search(r"\\addplot\[color=(c\w+)", ln)
        if m:
            means[(panel, m.group(1))] = dict(coords(ln))

    panel = -1
    out, n = [], 0
    for ln in lines:
        if ln.startswith("\\nextgroupplot"):
            panel += 1
        m = re.search(r"\\addplot\[draw=none, fill=(c\w+)", ln)
        if not m:
            out.append(ln); continue
        col = m.group(1)
        mu = means.get((panel, col))
        pts = coords(ln)
        if mu is None or len(pts) % 2:
            out.append(ln); continue
        h = len(pts) // 2
        hi, lo = pts[:h], pts[h:][::-1]         # lo comes back reversed
        assert [x for x, _ in hi] == [x for x, _ in lo], "x grids differ"
        new_hi, new_lo = [], []
        for (x, yh), (_, yl) in zip(hi, lo):
            m0 = mu.get(x)
            if m0 is None:
                new_hi.append((x, yh)); new_lo.append((x, yl)); continue
            sd = max(yh - m0, 0.0)              # upper edge is unclipped
            se = sd * SCALE
            new_hi.append((x, m0 + se))
            new_lo.append((x, max(0.0, m0 - se)))
        out.append(re.sub(r"coordinates \{[^}]*\}",
                          "coordinates {" + fmt(new_hi + new_lo[::-1]) + "}", ln))
        n += 1
    open(dst, "w").write("\n".join(out))
    print(f"rescaled {n} bands by 1/sqrt({N_SEEDS}) = {SCALE:.4f} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
