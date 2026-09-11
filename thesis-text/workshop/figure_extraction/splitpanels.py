import json, sys
CFG={
 "metaworld_rl.pdf": dict(legy=(8,42), y0=95.463, sy=212.37,
   panels=[("Coffee-Push", 49.06, 413.0, 58.834, 0.0097746, 5000),
           ("Box-Close",  413.0, 800.0, 436.414, 0.0097746, 5000)]),
 "robomimic_rl.pdf": dict(legy=(8,45), y0=94.421, sy=278.235,
   panels=[("PickPlace-Can", 40.0, 800.0, 54.091, 0.00094017, 0)]),
}
def legend(items, tx, lo, hi):
    labs=[(x,y,t) for x,y,t in tx if lo<=y<=hi]
    m={}
    for p in items:
        pts=p['pts']
        if len(pts) not in (2,3): continue
        ys=[q[1] for q in pts]; xs=[q[0] for q in pts]
        if abs(max(ys)-min(ys))>1e-6 or not (lo<=ys[0]<=hi+8): continue
        if not (10 < max(xs)-min(xs) < 40): continue
        c=[(abs(y-ys[0]), x-max(xs), t) for x,y,t in labs if 0 < x-max(xs) < 45 and abs(y-ys[0])<12]
        if c: m[tuple(p['stroke'])]=min(c)[2]
    return m
pdf=sys.argv[1]; cfg=CFG[pdf]
D=json.load(open(pdf+".json")); items=D['items']; tx=D['texts']
lm=legend(items,tx,*cfg['legy'])
print(f"=== {pdf} colour -> model ===")
for k,v in lm.items(): print(f"   {k} -> {v}")
out={}
for p in items:
    if p['op']!='S' or len(p['pts'])<5: continue
    xs=[q[0] for q in p['pts']]
    name=lm.get(tuple(p['stroke']))
    if not name: continue
    for pn, a, b, x0, sx, t0 in cfg['panels']:
        if a <= min(xs) <= b:
            d=[[round((qx-x0)/sx+t0,1), round((qy-cfg['y0'])/cfg['sy'],4)] for qx,qy in p['pts']]
            out[f"{pn}|{name}"]=d
            print(f"   {pn:14s} {name:18s} {len(d):3d} pts  x {d[0][0]:.0f}-{d[-1][0]:.0f}  final y={d[-1][1]:.3f}  max={max(v for _,v in d):.3f}")
json.dump(out, open(pdf+".curves.json","w"), indent=1)
