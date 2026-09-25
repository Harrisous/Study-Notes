"""Figure 11.7: a toy run of bias-based (auxiliary-loss-free) load balancing. AUTHOR'S SIMULATION, not paper data.

Setup (same as research notes sec. 5.2, fig. 11.6): 16 experts, top-2, d = 32, centroids e_i ~ N(0, I/d),
fixed "hotness" offsets c_i linearly spaced in [-1, 1], 4096 random tokens u ~ N(0, I) per step,
affinity s = sigmoid(u . e_i + c_i), selection by Top-2 of s + b, sign rule
b_i <- b_i - gamma * sign(load_i - mean), seed 0, 3000 steps; gamma in {1e-4, 1e-3, 1e-2}.
Nothing is learned except b, so the run isolates the controller.
Panels: (a) relative load of every expert, gamma = 1e-3; (b) MaxVio_batch for three gammas plus the
sampling-noise floor (uniform random top-2); (c) final bias vs hotness for gamma = 1e-3 and 1e-2.
Writes deepseek/figures/l11-sim-balance.svg
"""

import math

import numpy as np

from l11_svg import SVG, polyline

N, K, D, B, STEPS = 16, 2, 32, 4096, 3000
MEAN = B * K / N


def run(gamma, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1 / math.sqrt(D), size=(N, D))
    c = np.linspace(-1, 1, N)
    b = np.zeros(N)
    mv, loads = [], []
    for _ in range(STEPS):
        U = rng.normal(size=(B, D))
        s = 1 / (1 + np.exp(-(U @ E.T + c)))
        idx = np.argpartition(-(s + b), K, axis=1)[:, :K]
        load = np.bincount(idx.ravel(), minlength=N)
        mv.append((load.max() - MEAN) / MEAN)
        loads.append(load / MEAN)
        b = b - gamma * np.sign(load - MEAN)
    return np.array(mv), np.array(loads), b, c


runs = {g: run(g) for g in (1e-4, 1e-3, 1e-2)}
pts = [1, 10, 30, 100, 300, 1000, 3000]
table = {g: [round(float(runs[g][0][p - 1]), 3) for p in pts] for g in runs}
assert table[1e-3] == [1.727, 1.557, 1.453, 0.67, 0.129, 0.098, 0.055], table[1e-3]
for g in runs:
    print("gamma=%g  MaxVio at %s: %s  mean(last 100)=%.3f  final b in [%.3f, %.3f] spread %.3f"
          % (g, pts, table[g], runs[g][0][-100:].mean(), runs[g][2].min(), runs[g][2].max(),
             runs[g][2].max() - runs[g][2].min()))

# sampling-noise floor: every token picks 2 distinct experts uniformly at random
rng = np.random.default_rng(1)
floor = []
for _ in range(2000):
    r = rng.random((B, N))
    idx = np.argpartition(r, K, axis=1)[:, :K]
    load = np.bincount(idx.ravel(), minlength=N)
    floor.append((load.max() - MEAN) / MEAN)
floor = float(np.mean(floor))
print("noise floor (uniform random top-2): %.3f" % floor)

W, H = 720, 640
svg = SVG(W, H, "玩具模拟：偏置如何把 16 个专家的负载拉平（笔者模拟）")

# ---------------------------------------------------------------- shared log-x axis helper
LX0, LX1 = 0.0, math.log10(STEPS)


def xaxis(x0, x1, y1, label=True):
    for t in (1, 10, 100, 1000, 3000):
        x = x0 + (math.log10(t) - LX0) / (LX1 - LX0) * (x1 - x0)
        svg.line(x, y1, x, y1 + 4, color="ink", width=1)
        svg.text(x, y1 + 17, "{:,}".format(t), size=11, anchor="middle", color="muted")
    if label:
        svg.text((x0 + x1) / 2, y1 + 33, "训练步（对数坐标）", size=12, anchor="middle")


def sx(step, x0, x1):
    return x0 + (math.log10(step) - LX0) / (LX1 - LX0) * (x1 - x0)


steps_plot = np.unique(np.round(np.logspace(0, LX1, 140)).astype(int))

# ---------------------------------------------------------------- (a) per-expert loads
AX0, AX1, AY0, AY1 = 70, 600, 44, 250
svg.text(16, 22, "Ⓐ γ = 10^{−3} 时每个专家的相对负载（负载 / 均值）", size=13, weight="600")
YMAX = 2.8


def ay(v):
    return AY1 - v / YMAX * (AY1 - AY0)


for v in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
    svg.line(AX0, ay(v), AX1, ay(v), color="grid", width=1)
    svg.text(AX0 - 8, ay(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(AX0, AY1, AX1, AY1, color="ink", width=1.2)
xaxis(AX0, AX1, AY1, label=False)
loads = runs[1e-3][1]
for i in range(N):
    if i in (0, N - 1):
        continue
    polyline(svg, [(sx(t, AX0, AX1), ay(loads[t - 1, i])) for t in steps_plot], color="muted", width=0.9, opacity=0.55)
polyline(svg, [(sx(t, AX0, AX1), ay(loads[t - 1, N - 1])) for t in steps_plot], color="red", width=1.8)
polyline(svg, [(sx(t, AX0, AX1), ay(loads[t - 1, 0])) for t in steps_plot], color="blue", width=1.8)
svg.line(AX0, ay(1), AX1, ay(1), color="green", width=1.6, dash="6 4")
svg.text(AX1 + 6, ay(loads[0, N - 1]) + 4, "最热门专家", size=12, color="red")
svg.text(AX1 + 6, ay(loads[0, N - 1]) + 20, "（c = +1）", size=11, color="red")
svg.text(AX1 + 6, ay(1) + 4, "均值 = 1", size=12, color="green")
svg.text(AX1 + 6, ay(0.42) + 4, "最冷门专家", size=12, color="blue")
svg.text(AX1 + 6, ay(0.42) + 20, "（c = −1）", size=11, color="blue")
svg.text(AX1 - 6, ay(0.45), "灰线：其余 14 个专家", size=11, color="muted", anchor="end")

# ---------------------------------------------------------------- (b) MaxVio for three gammas
BX0, BX1, BY0, BY1 = 70, 400, 318, 560
svg.text(16, 296, "Ⓑ 每步的 MaxVio_{batch}，三种更新速度 γ", size=13, weight="600")
VMAX = 1.9


def by(v):
    return BY1 - v / VMAX * (BY1 - BY0)


for v in (0, 0.5, 1.0, 1.5):
    svg.line(BX0, by(v), BX1, by(v), color="grid", width=1)
    svg.text(BX0 - 8, by(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(BX0, BY1, BX1, BY1, color="ink", width=1.2)
xaxis(BX0, BX1, BY1)
svg.rect(BX0, by(floor + 0.02), BX1 - BX0, by(max(0, floor - 0.02)) - by(floor + 0.02), fill="muted",
         fill_opacity=0.18)
svg.text(BX0 + 4, by(floor) - 8, "采样噪声地板 ≈ %.2f" % floor, size=11, color="muted")
cols = {1e-4: "violet", 1e-3: "blue", 1e-2: "orange"}
labs = {1e-4: "γ = 10^{−4}", 1e-3: "γ = 10^{−3}", 1e-2: "γ = 10^{−2}"}
ly = {1e-4: 0.62, 1e-3: 1.30, 1e-2: 0.30}
for g in (1e-4, 1e-3, 1e-2):
    mv = runs[g][0]
    polyline(svg, [(sx(t, BX0, BX1), by(mv[t - 1])) for t in steps_plot], color=cols[g], width=1.6)
# label each curve just left of where it first drops below a chosen height, so no label sits on a line
lab_v = {1e-4: 0.80, 1e-3: 0.80, 1e-2: 0.62}
for g, v in lab_v.items():
    mv = runs[g][0]
    t_cross = next(t for t in range(1, STEPS + 1) if mv[t - 1] <= v)
    svg.text(sx(t_cross, BX0, BX1) - 10, by(v) + 4, labs[g], size=12, color=cols[g], weight="600", anchor="end")
# the shared log axis of panel (a) gets its own label between the 100 and 1,000 ticks
svg.text((sx(100, AX0, AX1) + sx(1000, AX0, AX1)) / 2, AY1 + 17, "训练步（对数坐标）", size=11, anchor="middle",
         color="muted")

# ---------------------------------------------------------------- (c) final bias vs hotness
CX0, CX1, CY0, CY1 = 486, 700, 318, 560
svg.text(452, 296, "Ⓒ 3000 步后的偏置 b_{i}", size=13, weight="600")
b3, c = runs[1e-3][2], runs[1e-3][3]
b2 = runs[1e-2][2]
BMIN, BMAXV = -0.25, 0.7


def cy(v):
    return CY1 - (v - BMIN) / (BMAXV - BMIN) * (CY1 - CY0)


def cx(v):
    return CX0 + (v + 1) / 2 * (CX1 - CX0)


for v in (-0.2, 0.0, 0.2, 0.4, 0.6):
    svg.line(CX0, cy(v), CX1, cy(v), color="grid", width=1)
    svg.text(CX0 - 8, cy(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(CX0, CY1, CX1, CY1, color="ink", width=1.2)
for t in (-1, 0, 1):
    svg.text(cx(t), CY1 + 17, "%+d" % t if t else "0", size=11, anchor="middle", color="muted")
svg.text((CX0 + CX1) / 2, CY1 + 33, "专家的热度 c_{i}", size=12, anchor="middle")
for bb, col in ((b3, "blue"), (b2, "orange")):
    polyline(svg, [(cx(c[i]), cy(bb[i])) for i in range(N)], color=col, width=1.2)
    for i in range(N):
        svg.circle(cx(c[i]), cy(bb[i]), 2.8, fill=col, fill_opacity=0.9)
svg.text(cx(0.05), cy(0.45), "γ = 10^{−2}", size=12, color="orange", weight="600")
svg.text(cx(0.05), cy(0.07), "γ = 10^{−3}", size=12, color="blue", weight="600")
# the two final bias vectors differ by an almost constant shift (std of b2 - b3 is about 0.006)
shift = b2 - b3
assert shift.std() < 0.01, shift.std()
svg.text(CX0 + 4, CY0 + 2, "形状几乎相同，约差一个常数 %.2f" % shift.mean(), size=11, color="muted")

svg.text(16, 618, "设定：16 个专家、每 token 选 2 个、每步 4096 个随机 token，专家有固定的“热度”偏移；只有偏置在变，", size=12,
         color="muted")
svg.text(16, 636, "所以这张图只检验偏置这个控制器本身。数值全部由脚本算出（seed 0）。", size=12, color="muted")

svg.save("l11-sim-balance.svg")
