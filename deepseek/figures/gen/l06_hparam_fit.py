"""Figure 6.12: the hyper-parameter scaling laws of DeepSeek LLM (dsllm eq. 1), computed.

  eta_opt = 0.3118 * C^-0.1250      B_opt = 0.2920 * C^0.3271   (B in tokens, C in FLOPs = M * D)
Stars: predictions for 7B (C = 8.49e22) and 67B (C = 8.65e23) with M from the real structure (notes R6/R7).
Open circles: values actually used (Table 2): B = 2304 / 4608 sequences x 4096 tokens, eta = 4.2e-4 / 3.2e-4.
Bars at C = 1e17: the near-optimal ranges read off the Fig. 2(a) grid (B 2^16.5..2^18, eta 2^-8.75..2^-8).
Shaded band: the fitting range stated in the text, 1e17..2e19.
Writes deepseek/figures/l06-hparam-fit.svg
"""

import math

import numpy as np

from l06_svg import SVG

C7, C67 = 4.2467e10 * 2e12, 4.3273e11 * 2e12


def eta(C):
    return 0.3118 * C ** -0.1250


def bsz(C):
    return 0.2920 * C ** 0.3271


W, H = 760, 400
svg = SVG(W, H, "DeepSeek LLM 的超参数 scaling law：最优 batch size 与学习率随算力的变化")
PY0, PY1 = 40, 320
panels = [
    dict(x0=74, x1=350, title="最优 batch size B（token，log₂ 刻度）", f=bsz, ylo=14, yhi=27, ylog=2,
         ticks=[14, 16, 18, 20, 22, 24, 26], col="blue",
         used=[(C7, 2304 * 4096), (C67, 4608 * 4096)], near=(2 ** 16.5, 2 ** 18)),
    dict(x0=450, x1=726, title="最优学习率 η（log₁₀ 刻度）", f=eta, ylo=math.log10(1e-4), yhi=math.log10(1e-2),
         ylog=10, ticks=[-4, -3, -2], col="violet",
         used=[(C7, 4.2e-4), (C67, 3.2e-4)], near=(2 ** -8.75, 2 ** -8)),
]
LC0, LC1 = 16, 25

for P in panels:
    x0, x1 = P["x0"], P["x1"]

    def px(C, x0=x0, x1=x1):
        return x0 + (math.log10(C) - LC0) / (LC1 - LC0) * (x1 - x0)

    def py(v, P=P):
        lv = math.log(v, P["ylog"])
        return PY1 - (lv - P["ylo"]) / (P["yhi"] - P["ylo"]) * (PY1 - PY0)

    # fitting range band
    svg.rect(px(1e17), PY0, px(2e19) - px(1e17), PY1 - PY0, fill="grid", fill_opacity=0.9)
    for e in range(LC0, LC1 + 1):
        x = px(10 ** e)
        svg.line(x, PY0, x, PY1, color="grid", width=1)
        if e % 2 == 0:
            svg.text(x, PY1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
    for t in P["ticks"]:
        v = P["ylog"] ** t
        y = py(v)
        svg.line(x0, y, x1, y, color="grid", width=1)
        lab = ("2^{%d}" % t) if P["ylog"] == 2 else ("10^{%d}" % t)
        svg.text(x0 - 6, y + 4, lab, size=11, anchor="end", color="muted")
    svg.line(x0, PY1, x1, PY1, color="muted", width=1.2)
    svg.line(x0, PY0, x0, PY1, color="muted", width=1.2)
    svg.text((x0 + x1) / 2, PY0 - 14, P["title"], size=12, anchor="middle", color="ink", weight="600")
    svg.text((x0 + x1) / 2, PY1 + 36, "训练算力 C（FLOPs）", size=12, anchor="middle", color="muted")
    svg.text(px(3e17), PY0 + 16, "正文称的", size=11, color="muted")
    svg.text(px(3e17), PY0 + 30, "拟合范围", size=11, color="muted")

    Cs = np.logspace(LC0, LC1, 120)
    svg.path("M" + " L".join("%.1f %.1f" % (px(C), py(P["f"](C))) for C in Cs), stroke=P["col"], width=2.2)

    # near-optimal range at 1e17 from the grid
    lo_v, hi_v = P["near"]
    svg.line(px(1e17), py(lo_v), px(1e17), py(hi_v), color="green", width=5, cap="round")

    # stars (predictions) and circles (used)
    for C in (C7, C67):
        x, y = px(C), py(P["f"](C))
        pts = []
        for k in range(10):
            r = 8 if k % 2 == 0 else 3.5
            a = math.pi / 2 + k * math.pi / 5
            pts.append((x + r * math.cos(a), y - r * math.sin(a)))
        svg.polygon(pts, fill="green", stroke="green", width=1)
    for C, v in P["used"]:
        svg.circle(px(C), py(v), 6, stroke="ink", width=1.5)

# annotations (numbers checked in the lesson text)
bp = panels[0]


def pxb(C):
    return bp["x0"] + (math.log10(C) - LC0) / (LC1 - LC0) * (bp["x1"] - bp["x0"])


def pyb(v):
    return PY1 - (math.log2(v) - bp["ylo"]) / (bp["yhi"] - bp["ylo"]) * (PY1 - PY0)


svg.text(pxb(C7) - 10, pyb(bsz(C7)) + 4, "7B：9.24M", size=12, anchor="end", color="green")
svg.text(pxb(C67) - 10, pyb(bsz(C67)) - 12, "67B：19.7M", size=12, anchor="end", color="green")
ep = panels[1]


def pxe(C):
    return ep["x0"] + (math.log10(C) - LC0) / (LC1 - LC0) * (ep["x1"] - ep["x0"])


def pye(v):
    return PY1 - (math.log10(v) - ep["ylo"]) / (ep["yhi"] - ep["ylo"]) * (PY1 - PY0)


svg.text(pxe(C7) - 12, pye(eta(C7)) + 20, "7B：4.24×10^{-4}", size=12, anchor="end", color="green")
svg.text(pxe(C67) - 10, pye(eta(C67)) + 22, "67B：3.17×10^{-4}", size=12, anchor="end", color="green")
svg.text(pxe(1e17) + 8, pye(2 ** -8.4) - 2, "10^{17} 网格的近优范围", size=11, color="green")

# legend (bottom)
ly = 382
svg.polygon([(80 + 7 * math.cos(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else 0.45),
              ly - 4 - 7 * math.sin(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else 0.45)) for k in range(10)],
            fill="green", stroke="green", width=1)
svg.text(92, ly, "公式对 7B、67B 的预测", size=12)
svg.circle(262, ly - 4, 6, stroke="ink", width=1.5)
svg.text(274, ly, "实际采用的值（表 2）", size=12)
svg.line(430, ly - 4, 446, ly - 4, color="green", width=5, cap="round")
svg.text(454, ly, "10^{17} 网格的近优范围（图 6.11）", size=12)

svg.save("l06-hparam-fit.svg")
