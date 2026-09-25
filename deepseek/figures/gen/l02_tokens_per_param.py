"""Figure 2.14: tokens per parameter. Left: the large models listed in Chinchilla table 1 (D/N):
  LaMDA 137B/168B 1.23; GPT-3 175B/300B 1.71; Jurassic-1 178B/300B 1.69; Gopher 280B/300B 1.07;
  MT-NLG 530B/270B 0.51; Chinchilla 70B/1.4T 20.0
Right: D_opt/N_opt from Chinchilla's method-3 formula (eq. 4) versus compute, for three parameter sets
(lesson-02-doc.md section 3.6 / 4.3): rounded paper values (E, A, B, alpha, beta) = (1.69, 406.4, 410.7, 0.34, 0.28);
unrounded TeX-source values; Besiroglu et al. (2024) refit (1.8172, 482.01, 2085.43, 0.3478, 0.3658).
At C = 1e21 / 1e25: rounded 50.1 / 122.1, TeX 33.9 / 75.5, Besiroglu 21.6 / 17.1. Reference line: 20.
Writes deepseek/figures/l02-tokens-per-param.svg
"""

import math

import numpy as np

from l02_svg import SVG, LogAxis

sets = {
    "rounded": (1.69, 406.4, 410.7, 0.34, 0.28),
    "tex": (math.exp(0.5267228), math.exp(6.0073404), math.exp(6.0179186), 0.33917084, 0.2849083),
    "bes": (1.8172, 482.01, 2085.43, 0.3478, 0.3658),
}


def ratio(C, E, A, B, al, be):
    G = (al * A / (be * B)) ** (1 / (al + be))
    a, b = be / (al + be), al / (al + be)
    N = G * (C / 6) ** a
    D = (C / 6) ** b / G
    return D / N


assert abs(ratio(1e21, *sets["rounded"]) - 50.08) < 0.05
assert abs(ratio(1e25, *sets["bes"]) - 17.11) < 0.05

W, H = 720, 360
svg = SVG(W, H, "每个参数训练多少 token：Chinchilla 之前的大模型，以及方法 3 的三种参数给出的答案")

# ---------------- left: bars
models = [("LaMDA", 137, 168), ("GPT-3", 175, 300), ("Jurassic-1", 178, 300), ("Gopher", 280, 300),
          ("MT-NLG", 530, 270), ("Chinchilla", 70, 1400)]
X0, X1, Y0, Y1 = 50, 372, 44, 290
ay = LogAxis(0.3, 100, Y1, Y0)
svg.text(X0 - 36, 22, "(a) 训练 token ÷ 参数（Chinchilla 表 1）", size=13, color="ink", weight="600")
for v in (0.3, 1, 3, 10, 30, 100):
    y = ay(v)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 6, y + 4, "%g" % v, size=11, anchor="end", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
bw = (X1 - X0) / len(models)
for i, (name, n, d) in enumerate(models):
    r = d / n
    x = X0 + i * bw + 8
    col = "green" if name == "Chinchilla" else "red"
    svg.rect(x, ay(r), bw - 16, Y1 - ay(r), fill=col, fill_opacity=0.7)
    svg.text(x + (bw - 16) / 2, ay(r) - 6, "%.2f" % r if r < 10 else "%.0f" % r, size=11, anchor="middle",
             color="ink", weight="600")
    svg.text(x + (bw - 16) / 2, Y1 + 15, name, size=11, anchor="middle", color="ink")
    svg.text(x + (bw - 16) / 2, Y1 + 30, "%gB" % n, size=11, anchor="middle", color="muted")
svg.line(X0, ay(20), X1, ay(20), color="green", width=1.4, dash="6 4")
svg.text(X0 + 4, ay(20) - 6, "20：经验法则（论文里没有这句话）", size=11, color="green")

# ---------------- right: D/N vs compute for three fits
RX0, RX1, RY0, RY1 = 448, 704, 44, 290
rx = LogAxis(1e20, 1e26, RX0, RX1)
ry = LogAxis(10, 200, RY1, RY0)
svg.text(RX0 - 40, 22, "(b) 方法 3 的公式给出的最优 D/N", size=13, color="ink", weight="600")
for v in (10, 20, 50, 100, 200):
    y = ry(v)
    svg.line(RX0, y, RX1, y, color="grid", width=1)
    svg.text(RX0 - 6, y + 4, "%g" % v, size=11, anchor="end", color="muted")
for e in range(20, 27):
    x = rx(10.0 ** e)
    svg.line(x, RY0, x, RY1, color="grid", width=1)
    if e % 2 == 1:
        svg.text(x, RY1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
svg.line(RX0, RY1, RX1, RY1, color="muted", width=1.2)
svg.line(RX0, RY0, RX0, RY1, color="muted", width=1.2)
svg.text((RX0 + RX1) / 2, RY1 + 36, "训练算力 C（FLOPs）", size=12, anchor="middle", color="muted")
Cs = np.logspace(20, 26, 120)
styles = {"rounded": ("red", None, "正文四舍五入参数"), "tex": ("orange", "6 3", "TeX 源码未四舍五入"),
          "bes": ("blue", None, "Besiroglu 2024 重拟合")}
for i, (k, (col, dash, lab)) in enumerate(styles.items()):
    r = np.array([ratio(c, *sets[k]) for c in Cs])
    svg.path("M" + " L".join("%.1f %.1f" % (rx(c), ry(v)) for c, v in zip(Cs, r)), stroke=col, width=2.4, dash=dash)
    ly = RY0 + 16 + i * 17
    svg.line(RX0 + 8, ly - 4, RX0 + 30, ly - 4, color=col, width=2.4, dash=dash)
    svg.text(RX0 + 36, ly, lab, size=11, color=col, weight="600")
svg.line(RX0, ry(20), RX1, ry(20), color="green", width=1.4, dash="6 4")
svg.text(RX0 + 4, ry(20) + 14, "20", size=11, color="green")
xg = rx(5.76e23)
svg.line(xg, RY0, xg, RY1, color="muted", width=1, dash="2 3")
svg.text(xg + 3, RY1 - 6, "Gopher 的算力", size=11, color="muted")

svg.text(16, H - 10, "左：Chinchilla 之前的大模型每个参数只训练 0.5–1.7 个 token。右：同一个公式，参数写法不同，在 Gopher 的算力上结论差 5 倍。",
         size=12, color="ink")

svg.save("l02-tokens-per-param.svg")
