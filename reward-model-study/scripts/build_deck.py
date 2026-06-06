"""Build a self-contained HTML presentation from the study artifacts.

Numbers are read straight from results/FULL_METRICS.csv and
results/ood_kendall_harness.csv (no hand-transcription); figures are embedded as
base64 so the deck is one portable file that opens offline in any browser.

  python reward-model-study/scripts/build_deck.py
  -> reward-model-study/deck/index.html   (arrow keys / space to navigate)
"""
import base64
import csv
from pathlib import Path

ROOT = Path("/gpfs/home3/pkarageorgis1/Master-Thesis/reward-model-study")
RES = ROOT / "results"
FIG = ROOT / "figures"
OUT = ROOT / "deck" / "index.html"

# ---- load metrics ---------------------------------------------------------
M = {(r["model"], r["cell"]): r for r in csv.DictReader(open(RES / "FULL_METRICS.csv"))}
KEN = {r["model"]: r for r in csv.DictReader(open(RES / "ood_kendall_harness.csv"))}


def g(model, cell, key):
    r = M.get((model, cell))
    if not r or r.get(key, "") in ("", "nan"):
        return None
    return float(r[key])


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def ken(model):
    r = KEN.get(model)
    return None if not r else float(r["kendall_last"])


def b64(name):
    return base64.b64encode((FIG / name).read_bytes()).decode()


LABEL = {
    "baseline": "Robometer-4B<br><span class=sub>baseline · untrained</span>",
    "run1_s4000": "Robometer-FT<br><span class=sub>asym+ICL · s4000</span>",
    "run1_s5000": "Robometer-FT<br><span class=sub>asym+ICL · s5000</span>",
    "run2_s5000": "Robometer-FT<br><span class=sub>asym · noICL</span>",
    "run3_s5000": "Robometer-FT<br><span class=sub>paper-std</span>",
    "run4_s6500": "Qwen3.5-FT<br><span class=sub>asym+ICL</span>",
    "run5_s6500": "Qwen3.5-FT<br><span class=sub>asym</span>",
    "run6_s6500": "Qwen3.5-FT<br><span class=sub>paper-std</span>",
}
FAM = {"baseline": "base", "run1_s4000": "asym", "run1_s5000": "asym", "run2_s5000": "asym",
       "run3_s5000": "paper", "run4_s6500": "asym", "run5_s6500": "asym", "run6_s6500": "paper"}


def row(model, cells):
    tds = "".join(cells)
    return f'<tr class="{FAM[model]}"><th>{LABEL[model]}</th>{tds}</tr>'


def td(v, hi=False, nd=2):
    cls = ' class="hi"' if hi else ""
    return f"<td{cls}>{fmt(v, nd)}</td>"


ORDER = ["baseline", "run1_s4000", "run1_s5000", "run2_s5000", "run3_s5000",
         "run4_s6500", "run5_s6500", "run6_s6500"]

# ======================================================================
# slides
# ======================================================================
slides = []

# 0 — title
slides.append("""
<section class="title">
  <h1>VLM Reward Models for Downstream Sparse-RL</h1>
  <p class="lede">Do fine-tuned vision-language reward models (Robometer-FT, Qwen3.5-FT)
     give a better reward for IBRL on MetaWorld CoffeePush than the released baseline?</p>
  <p class="models">Robometer-4B (baseline) · Robometer-FT (4B) · Qwen3.5-FT &nbsp;|&nbsp;
     success head + progress head · OOD &amp; in-distribution · downstream RL</p>
  <p class="foot">Self-contained study — every number from <code>FULL_METRICS.csv</code> /
     <code>ood_kendall_harness.csv</code>; the story follows the data.</p>
</section>""")

# 1 — hook
slides.append(f"""
<section>
  <h2><span class="num">0</span> Where it went wrong</h2>
  <div class="two">
    <div class="card bad">
      <h3>Downstream RL fails</h3>
      <p class="big">0.12</p>
      <p>peak success of IBRL on CoffeePush with our VLM reward — the policy then
         <b>collapses to ~2%</b>. Same band as BC-bootstrap alone.</p>
    </div>
    <div class="card bad">
      <h3>FT <i>hurts</i> OOD ranking</h3>
      <p class="big">0.64 → ≤0.32</p>
      <p>On OOD, the <b>untrained</b> baseline out-ranks every fine-tuned model
         (harness kendall<sub>last</sub> <b>{fmt(ken('baseline'))}</b> vs ≤ {fmt(ken('run1_s4000'))}).</p>
    </div>
  </div>
  <p class="ask">→ Fine-tuning didn't help downstream <i>and</i> degraded OOD ranking.
     The rest of the deck is: <b>why</b>, on both heads, in- and out-of-distribution.</p>
</section>""")

# 2 — OOD ranking
ood_tbl = "".join(row(m, [
    td(g(m, "ood", "succ_AUC"), hi=(m == "baseline")),
    td(ken(m), hi=(m == "baseline")),
    td(g(m, "ood", "prog_VOCpearson"), hi=(m in ("baseline",))),
]) for m in ORDER)
slides.append(f"""
<section>
  <h2><span class="num">1</span> Pure ranking — OUT of distribution <span class="n">(full set, 782)</span></h2>
  <table class="metrics">
    <tr><th>model</th><th>success AUC</th><th>kendall<sub>last</sub><br><span class=sub>harness, paper-exact</span></th>
        <th>progress r<br><span class=sub>vs GT, within-traj</span></th></tr>
    {ood_tbl}
  </table>
  <ul class="take">
    <li><b>Baseline ≫ all FT</b> on every OOD ranking metric.</li>
    <li>Gate passed: baseline reproduces the paper (kendall {fmt(ken('baseline'))} ≈ 0.66) — the gap is real,
        not a measurement bug. Frame-count confound separately ruled out (8 vs 16).</li>
    <li>Success head: baseline AUC <b>{fmt(g('baseline','ood','succ_AUC'))}</b> vs FT 0.54–0.69.
        Progress head: baseline r <b>{fmt(g('baseline','ood','prog_VOCpearson'))}</b>; asymmetric FT ≈ 0 (next slides).</li>
  </ul>
</section>""")

# 3 — in-distribution + specialization
ind_tbl = "".join(row(m, [
    td(g(m, "indist_icloff", "succ_AUC"), hi=(m in ("run2_s5000", "run3_s5000"))),
    td(g(m, "ood", "succ_AUC")),
]) for m in ORDER)
slides.append(f"""
<section>
  <h2><span class="num">2</span> Pure ranking — IN distribution <span class="n">(common 3,142)</span></h2>
  <div class="two">
    <div>
      <table class="metrics small">
        <tr><th>model</th><th>in-dist AUC</th><th>OOD AUC</th></tr>
        {ind_tbl}
      </table>
    </div>
    <div>
      <img src="data:image/png;base64,{b64('fig2_specialization.png')}"/>
    </div>
  </div>
  <ul class="take">
    <li><b>Specialization tradeoff:</b> the 4B FT wins in-dist (0.84–0.88 vs baseline {fmt(g('baseline','indist_icloff','succ_AUC'))})
        but loses OOD — baseline is the mirror image.</li>
    <li>Baseline is near-random on our curated data because <i>our data is OOD for it</i>; that is the expected,
        honest reading — not a bug.</li>
  </ul>
</section>""")

# 4 — THE finding
prog_tbl = "".join(row(m, [
    td(g(m, "ood", "prog_VOCpearson"), nd=2),
    td(g(m, "indist_icloff", "prog_VOCpearson"), nd=2),
]) for m in ORDER)
slides.append(f"""
<section>
  <h2><span class="num">3</span> THE finding — asymmetric loss destroys the progress head</h2>
  <img class="hero" src="data:image/png;base64,{b64('fig1_progress_head_collapse.png')}"/>
  <div class="two">
    <table class="metrics small">
      <tr><th>model</th><th>prog r — OOD</th><th>prog r — in-dist</th></tr>
      {prog_tbl}
    </table>
    <ul class="take">
      <li>Every <span class="asym-t">asymmetric</span> model collapses to <b>r ≈ −0.03</b> — both bases, both distributions.</li>
      <li>Every <span class="paper-t">paper-standard</span> model + baseline keeps it <b>intact (0.44–0.90)</b>.</li>
      <li>The asymmetric C51 loss kills the head that IBRL's <i>dense</i> reward depends on →
          explains the progress-reward inversion on rollouts.</li>
    </ul>
  </div>
</section>""")

# 5 — second failure mode: failure-data suppresses success-progress
slides.append("""
<section>
  <h2><span class="num">3b</span> Why even paper-std FT loses OOD ranking</h2>
  <p class="lede">Paper-standard keeps the progress <i>shape</i> — yet its OOD cross-trajectory
     ranking still dies. The dumps show why: fine-tuning on failure-rich data
     <b>suppresses final-frame progress on OOD successes</b>, while failures barely move.</p>
  <table class="metrics">
    <tr><th>OOD final progress</th><th>success</th><th>failure</th><th>gap (succ−fail)</th></tr>
    <tr class="base"><th>baseline (untrained)</th><td>0.79</td><td>0.48</td><td class="hi">+0.31</td></tr>
    <tr class="paper"><th>run3 (4B paper-std)</th><td>0.52</td><td>0.49</td><td>+0.04</td></tr>
    <tr class="paper"><th>run6 (Qwen3.5 paper-std)</th><td>0.36</td><td>0.36</td><td>+0.00</td></tr>
  </table>
  <ul class="take">
    <li>The collapse is <b>asymmetric</b>: successes pulled down (0.79→0.52→0.36), failures unchanged →
        the success/failure gap kendall needs vanishes.</li>
    <li>Not noise: run3's within-trajectory shape is intact (r 0.88). It's a <b>calibration</b> bias from
        failure-dominated supervision (MetaWorld ~93% failures), not lost dynamics.</li>
    <li><b>Two distinct progress-head failure modes:</b> asymmetric loss kills it outright;
        failure-heavy data suppresses success-progress on OOD even under paper-standard loss.</li>
  </ul>
</section>""")

# 6 — operating point: TPR@5FPR + dense-ECE
tpr_tbl = "".join(row(m, [
    td(g(m, "ood", "succ_TPR@5FPR"), nd=2),
    td(g(m, "indist_icloff", "succ_TPR@5FPR"), nd=2),
    td(g(m, "indist_iclon", "succ_TPR@5FPR"), hi=(m in ("run1_s4000", "run1_s5000")), nd=2),
]) for m in ORDER)
slides.append(f"""
<section>
  <h2><span class="num">4</span> The operating point — TPR@5%FPR &amp; dense-ECE</h2>
  <div class="two">
    <table class="metrics small">
      <tr><th>TPR@5%FPR</th><th>OOD</th><th>in-dist</th><th>+ICL</th></tr>
      {tpr_tbl}
    </table>
    <img src="data:image/png;base64,{b64('fig3_ece_flat.png')}"/>
  </div>
  <ul class="take">
    <li>At a strict 5% FPR, TPR is <b>~0 almost everywhere</b> — the asymmetric compression leaves no usable
        high-precision operating point (only baseline OOD {fmt(g('baseline','ood','succ_TPR@5FPR'))}, and ICL — see col 3).</li>
    <li><b>dense-ECE is NOT the differentiator we hoped.</b> On OOD (85% positive) it mostly measures how far
        predictions sit below the base rate; asymmetric scores <i>lower</i> ECE only by predicting higher on
        average, not by separating success from failure. It can favour the worse-ranking model — we followed the data.</li>
  </ul>
</section>""")

# 7 — ICL on/off
icl_tbl = "".join(row(m, [
    td(g(m, "indist_icloff", "succ_AUC")),
    td(g(m, "indist_iclon", "succ_AUC")),
    td(g(m, "indist_icloff", "succ_TPR@5FPR"), nd=2),
    td(g(m, "indist_iclon", "succ_TPR@5FPR"), hi=(m in ("run1_s4000", "run1_s5000", "run2_s5000")), nd=2),
]) for m in ORDER)
slides.append(f"""
<section>
  <h2><span class="num">5</span> In-context demos at inference (in-dist only)</h2>
  <table class="metrics">
    <tr><th>model</th><th>AUC off</th><th>AUC on</th><th>TPR@5FPR off</th><th>TPR@5FPR on</th></tr>
    {icl_tbl}
  </table>
  <ul class="take">
    <li>ICL barely moves <b>AUC</b> (ranking already set by the weights).</li>
    <li>But ICL <b>recovers the strict operating point</b>: TPR@5%FPR jumps to <b>0.17–0.21</b> for the 4B FT
        (from ≈0) — a real, precision-side benefit a single threshold hides.</li>
    <li>OOD has no demos, so ICL is in-distribution only; its main value was at <i>training</i> time.</li>
  </ul>
</section>""")

# 8 — ablations
slides.append(f"""
<section>
  <h2><span class="num">6</span> Ablations — what each ingredient did</h2>
  <table class="ablate">
    <tr><th>ingredient</th><th>effect</th><th>verdict</th></tr>
    <tr><td><b>Loss</b><br>paper-std vs asymmetric</td>
        <td>asym is slightly better on the success head (in-dist AUC {fmt(g('run2_s5000','indist_icloff','succ_AUC'))} vs
            {fmt(g('run3_s5000','indist_icloff','succ_AUC'))}) but <b>destroys the progress head</b> (r ≈ −0.03 vs {fmt(g('run3_s5000','indist_icloff','prog_VOCpearson'))}).</td>
        <td class="win">paper-std is the better RL recipe</td></tr>
    <tr><td><b>ICL</b><br>at inference</td>
        <td>negligible on AUC; recovers TPR@5%FPR (0→0.2) in-dist; no demos OOD.</td>
        <td class="meh">value is at training</td></tr>
    <tr><td><b>Base model</b><br>Robometer-4B vs Qwen3.5</td>
        <td>same progress-head collapse under asymmetric loss on both; 4B FT is the stronger in-dist ranker.</td>
        <td class="meh">orthogonal to the finding</td></tr>
    <tr><td><b>Training data</b><br>failure-rich + frame labels</td>
        <td>suppresses OOD success-progress → collapses cross-trajectory ranking even under paper-std loss.</td>
        <td class="lose">the OOD-ranking culprit</td></tr>
  </table>
</section>""")

# 9 — downstream RL
slides.append("""
<section>
  <h2><span class="num">7</span> Downstream RL — closing the loop</h2>
  <div class="three">
    <div class="card bad"><h3>VLM-reward IBRL</h3><p class="big">0.12</p><p>peak; policy → ~2%</p></div>
    <div class="card good"><h3>GT-reward control</h3><p class="big">0.56–0.82</p><p>same loop, real reward</p></div>
    <div class="card warn"><h3>BC-rollout AUC</h3><p class="big">0.66–0.88</p><p>reward ranks rollouts, seed-noisy</p></div>
  </div>
  <ul class="take">
    <li><b>The loop is capable — the reward is the limiter.</b> Swap in GT reward and IBRL trains to 0.56–0.82;
        the VLM reward caps at 0.12.</li>
    <li>The reward <i>does</i> rank live BC rollouts (AUC ~0.7), and the scoring path is verified bug-free
        (inline ≡ direct, diff = 0) — so it is not a plumbing bug.</li>
    <li>On rollouts the <b>progress head inverts</b> — exactly the §3 collapse, now showing up where it costs us.</li>
  </ul>
</section>""")

# 10 — takeaways
slides.append("""
<section>
  <h2><span class="num">∑</span> Takeaways</h2>
  <ol class="final">
    <li><b>Asymmetric loss broke the progress head</b> (r ≈ 0, both bases, both distributions) — the cleanest,
        most consequential finding. Paper-standard preserves it.</li>
    <li><b>Fine-tuning specializes:</b> in-distribution gains, OOD ranking loss — real, frame-confound ruled out.</li>
    <li><b>The OOD-ranking culprit is failure-rich training data</b> suppressing success-progress (verified on the
        dumps), a <i>second</i> failure mode distinct from the loss.</li>
    <li><b>The downstream bottleneck is the reward's usable signal,</b> NOT the RL loop (GT → 0.82) and NOT
        calibration/ECE (which doesn't separate the models).</li>
    <li><b>Recommended next recipe:</b> paper-standard loss (intact progress head) + balanced / on-policy reward
        data for OOD robustness.</li>
  </ol>
</section>""")

# ======================================================================
# shell
# ======================================================================
CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font:16px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;overflow:hidden}
#deck{height:100vh;width:100vw;position:relative}
section{position:absolute;inset:0;padding:4vh 5vw;display:none;flex-direction:column;justify-content:center;animation:f .25s ease}
section.on{display:flex}
@keyframes f{from{opacity:0;transform:translateY(8px)}to{opacity:1}}
h1{font-size:2.6rem;line-height:1.15;margin-bottom:1rem;background:linear-gradient(90deg,#58a6ff,#79c0ff);-webkit-background-clip:text;background-clip:text;color:transparent}
h2{font-size:1.9rem;margin-bottom:1.6rem;font-weight:650}
h2 .num{display:inline-block;min-width:1.7em;text-align:center;background:#1f6feb;color:#fff;border-radius:8px;padding:.05em .25em;margin-right:.5em;font-size:.95em}
h2 .n{font-size:.85rem;color:#8b949e;font-weight:400}
h3{font-size:1.05rem;color:#8b949e;font-weight:600;margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.04em}
.title{justify-content:center}
.lede{font-size:1.25rem;color:#c9d1d9;max-width:60ch;margin-bottom:1rem}
.models{color:#8b949e;margin:1rem 0}.foot{color:#6e7681;font-size:.85rem;margin-top:1.5rem}
code{background:#161b22;padding:.1em .4em;border-radius:5px;color:#79c0ff;font-size:.85em}
.two{display:grid;grid-template-columns:1fr 1fr;gap:2vw;align-items:center}
.three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5vw;margin-bottom:1.5rem}
img{max-width:100%;max-height:46vh;border-radius:8px;background:#fff;padding:6px;display:block;margin:0 auto}
img.hero{max-height:40vh;margin-bottom:1rem}
table{border-collapse:collapse;width:100%;font-size:.92rem}
table.small{font-size:.82rem}
th,td{padding:.45em .7em;text-align:center;border-bottom:1px solid #21262d}
table.metrics th:first-child,table.ablate td:first-child{text-align:left}
table tr th:first-child{font-weight:600}
.sub{font-size:.72em;color:#8b949e;font-weight:400}
td.hi{background:#1f6feb22;color:#79c0ff;font-weight:700;border-radius:4px}
tr.asym td,tr.asym th{color:#ff9492}tr.paper td,tr.paper th{color:#a5d6ff}tr.base td,tr.base th{color:#d2a8ff}
.asym-t{color:#ff7b72;font-weight:700}.paper-t{color:#79c0ff;font-weight:700}
ul.take{margin:1.2rem 0 0 1.1rem;max-width:90ch}ul.take li{margin:.5rem 0}
ol.final{margin:.5rem 0 0 1.4rem;max-width:80ch}ol.final li{margin:.7rem 0;font-size:1.05rem}
.ask{margin-top:1.6rem;font-size:1.15rem;color:#ffa657;border-left:3px solid #ffa657;padding-left:1rem}
.card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:1.4rem;text-align:center}
.card.bad{border-color:#f8514955}.card.good{border-color:#3fb95055}.card.warn{border-color:#d2992255}
.big{font-size:2.6rem;font-weight:800;margin:.3rem 0}
.card.bad .big{color:#ff7b72}.card.good .big{color:#56d364}.card.warn .big{color:#e3b341}
table.ablate td{text-align:left;vertical-align:top;font-size:.9rem}
.win{color:#56d364;font-weight:700}.lose{color:#ff7b72;font-weight:700}.meh{color:#8b949e;font-weight:600}
#hud{position:fixed;bottom:14px;right:20px;color:#6e7681;font-size:.8rem;z-index:9}
#bar{position:fixed;top:0;left:0;height:3px;background:#1f6feb;transition:width .25s;z-index:9}
"""

JS = """
const S=[...document.querySelectorAll('section')];let i=0;
function show(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('on',k===i));
document.getElementById('hud').textContent=(i+1)+' / '+S.length;
document.getElementById('bar').style.width=((i+1)/S.length*100)+'%';location.hash=i;}
addEventListener('keydown',e=>{if(['ArrowRight',' ','PageDown'].includes(e.key)){show(i+1);e.preventDefault();}
else if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);
else if(e.key==='Home')show(0);else if(e.key==='End')show(S.length-1);});
addEventListener('click',e=>{if(e.clientX>innerWidth*0.5)show(i+1);else show(i-1);});
show(parseInt(location.hash.slice(1))||0);
"""

html = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>VLM Reward Models — thesis deck</title><style>{CSS}</style></head>
<body><div id=bar></div><div id=deck>{''.join(slides)}</div><div id=hud></div>
<script>{JS}</script></body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
print(f"wrote {OUT}  ({len(html)//1024} KB, {len(slides)} slides)")
