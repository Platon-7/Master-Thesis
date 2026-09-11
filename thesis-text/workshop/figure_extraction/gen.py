import json
MODELS=[("RoboRef-Asym","cAsym","square*"),("RoboRef-Std","cStd","triangle*"),
        ("Robometer-4B-Base","cBase","diamond*"),("RoboDopamine","cDopa","triangle*,every mark/.append style={rotate=180}"),
        ("RoboReward","cRew","+"),("LRM-TRI","cLRM","x")]
COLORS={"cAsym":(0.8824,0.1137,0.1843),"cStd":(0.2902,0.2275,0.6549),
        "cBase":(0.0,0.5137,0.0),"cDopa":(0.8314,0.6275,0.0902),
        "cRew":(0.5451,0.2706,0.0745),"cLRM":(0.15,0.15,0.15)}
def defs():
    return "\n".join(f"\\definecolor{{{k}}}{{rgb}}{{{r},{g},{b}}}" for k,(r,g,b) in COLORS.items())
def coords(xs,ys,sx=1.0):
    return " ".join(f"({x/sx:g},{y:g})" for x,y in zip(xs,ys))

cur=json.load(open("metaworld_rl.pdf.curves.json"))
band=json.load(open("metaworld_bands.json"))
L=[]
L.append("% Auto-generated from the original matplotlib PDF's vector coordinates.")
L.append("% Values are the exact plotted data, not a visual trace. RoboRef-ICL omitted.")
L.append(r"\begin{figure*}[t]")
L.append(r"\centering")
L.append(defs())
L.append(r"\begin{tikzpicture}")
L.append(r"""\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.15cm},
  width=0.47\textwidth, height=4.6cm,
  xmin=4000, xmax=41000, ymin=-0.02, ymax=0.95,
  scaled x ticks=false, xtick={5000,15000,25000,35000},
  xticklabels={5k,15k,25k,35k}, ytick={0,0.2,0.4,0.6,0.8},
  grid=major, grid style={gray!22, line width=0.3pt},
  tick label style={font=\scriptsize}, label style={font=\footnotesize},
  title style={font=\small}, xlabel={Training steps},
  every axis plot/.append style={line width=0.9pt},
  legend style={font=\scriptsize, draw=none, fill=none, legend columns=6,
                /tikz/every even column/.append style={column sep=6pt}},
]""")
for i,(panel,title) in enumerate([("Coffee-Push","Coffee-Push"),("Box-Close","Box-Close")]):
    extra = r"ylabel={Task success rate (ground truth)}," if i==0 else ""
    leg   = "legend to name=mwlegend," if i==0 else ""
    L.append(f"\\nextgroupplot[title={{{title}}},{extra}{leg}]")
    for name,c,mk in MODELS:
        b=band[f"{panel}|{name}"]
        poly=coords(b["x"],b["hi"])+" "+coords(list(reversed(b["x"])),list(reversed(b["lo"])))
        L.append(f"\\addplot[draw=none, fill={c}, fill opacity=0.13, forget plot] coordinates {{{poly}}} --cycle;")
    for name,c,mk in MODELS:
        d=cur[f"{panel}|{name}"]
        L.append(f"\\addplot[color={c}, mark={mk}, mark size=1.5pt] coordinates {{{coords([p[0] for p in d],[p[1] for p in d])}}};")
        if i==0: L.append(f"\\addlegendentry{{{name}}}")
L.append(r"\end{groupplot}")
L.append(r"\node at ($(group c1r1.south west)!0.5!(group c2r1.south east) + (0,-1.05cm)$) {\pgfplotslegendfromname{mwlegend}};")
L.append(r"\end{tikzpicture}")
L.append(r"""\caption{MetaWorld downstream RL on two held-out tasks absent from every reward
model's training data, five seeds per reward, mean ground-truth success with a
one standard deviation band. Only the asymmetric fine-tune trains a competent
policy; every other reward ends far behind.}
\label{fig:mw-rl}
\end{figure*}""")
open("metaworld_tikz.tex","w").write("\n".join(L)+"\n")
print("metaworld_tikz.tex", len("\n".join(L)), "chars")

# ---- Robomimic ----
rc=json.load(open("robomimic_rl.pdf.curves.json"))
R=[]
R.append("% Auto-generated from the original matplotlib PDF's vector coordinates.")
R.append(r"\begin{figure}[t]")
R.append(r"\centering")
R.append(defs())
R.append(r"\begin{tikzpicture}")
R.append(r"""\begin{axis}[
  width=0.92\columnwidth, height=5.2cm,
  xmin=0, xmax=760000, ymin=-0.02, ymax=0.92,
  scaled x ticks=false, xtick={0,200000,400000,600000},
  xticklabels={0,200k,400k,600k}, ytick={0,0.2,0.4,0.6,0.8},
  grid=major, grid style={gray!22, line width=0.3pt},
  tick label style={font=\scriptsize}, label style={font=\footnotesize},
  xlabel={Training steps}, ylabel={Task success rate (ground truth)},
  every axis plot/.append style={line width=0.9pt},
  legend style={font=\scriptsize, draw=none, fill=none, at={(0.5,-0.30)},
                anchor=north, legend columns=3,
                /tikz/every even column/.append style={column sep=6pt}},
]""")
R.append(r"\addplot[gray!70, dashed, line width=0.8pt, forget plot] coordinates {(0,0.56) (760000,0.56)};")
R.append(r"\node[font=\scriptsize, text=gray!45!black, anchor=south, fill=white, inner sep=1pt] at (axis cs:350000,0.60) {behavior cloning};")
for name,c,mk in MODELS:
    k=f"PickPlace-Can|{name}" if name!="LRM-TRI" else "PickPlace-Can|LRM"
    d=rc.get(k)
    if d is None: continue
    R.append(f"\\addplot[color={c}, mark={mk}, mark size=1.3pt, mark repeat=11] coordinates {{{coords([p[0] for p in d],[p[1] for p in d])}}};")
    R.append(f"\\addlegendentry{{{'LRM' if name=='LRM-TRI' else name}}}")
R.append(r"\end{axis}")
R.append(r"\end{tikzpicture}")
R.append(r"""\caption{Robomimic PickPlace-Can, a domain absent from every reward model's
training data, one seed per reward, smoothed ground-truth success over $750$k
steps. The dashed line marks the behavior-cloning policy IBRL starts from, at
$0.56$. RoboRef-Asym is the only reward that ends above it.}
\label{fig:robomimic-rl}
\end{figure}""")
open("robomimic_tikz.tex","w").write("\n".join(R)+"\n")
print("robomimic_tikz.tex", len("\n".join(R)), "chars")
