"""Figure 2.13: compute-optimal model size versus training compute - Kaplan vs Chinchilla, log-log.

Kaplan et al. (2020) table 6: N_opt = 1.3e9 * C_min^0.73 with C_min in PF-days (8.64e19 FLOPs);
  N is non-embedding and C_min is Kaplan's minimum compute, so this line is only an order-of-magnitude
  comparison with Chinchilla's (N includes embeddings, C = 6ND).
Chinchilla (Hoffmann et al. 2022):
  method 1 projections, table 3 (FLOPs, parameters): 400M 1.92e19; 1B 1.21e20; 10B 1.23e22; 67B 5.76e23;
    175B 3.85e24; 280B 9.90e24; 520B 3.43e25; 1T 1.27e26
  method 3 frontier, eq. (4) with the unrounded TeX-source fit: N_opt = G (C/6)^a, G = 1.300, a = 0.4565
  rule of thumb D = 20 N  =>  N = sqrt(C / 120)
Models (N, C = 6ND except Gopher, which uses the 5.76e23 quoted by Chinchilla):
  GPT-3 175B / 3.14e23; Gopher 280B / 5.76e23; Chinchilla 70B / 5.88e23; MT-NLG 530B / 8.59e23 (6ND arithmetic)
Writes deepseek/figures/l02-optimal-n.svg
"""

import math

import numpy as np

from l02_svg import SVG, LogAxis

E, A, B = math.exp(0.5267228), math.exp(6.0073404), math.exp(6.0179186)
al, be = 0.33917084, 0.2849083
G = (al * A / (be * B)) ** (1 / (al + be))
a = be / (al + be)
assert abs(G - 1.300) < 0.002 and abs(a - 0.4565) < 1e-3

W, H = 720, 450
X0, X1, Y0, Y1 = 78, 560, 30, 380
ax = LogAxis(1e18, 1e27, X0, X1)
ay = LogAxis(1e8, 1e13, Y1, Y0)
svg = SVG(W, H, "算力最优的模型规模：Kaplan 与 Chinchilla")

for e in range(8, 14):
    y = ay(10.0 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    lab = {8: "100M", 9: "1B", 10: "10B", 11: "100B", 12: "1T", 13: "10T"}[e]
    svg.text(X0 - 8, y + 4, lab, size=11, anchor="end", color="muted")
for e in range(18, 28):
    x = ax(10.0 ** e)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    if e % 2 == 0 and e > 18:
        svg.text(x, Y1 + 16, "10^{%d}" % e, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "训练算力 C（FLOPs，对数）", size=12, anchor="middle", color="muted")
svg.text(22, (Y0 + Y1) / 2, "算力最优的参数量 N（对数）", size=12, anchor="middle", color="muted", rotate=-90)

Cs = np.logspace(18, 27, 200)


def curve(ys, **kw):
    m = (ys >= 1e8) & (ys <= 1e13)
    svg.path("M" + " L".join("%.1f %.1f" % (ax(c), ay(y)) for c, y in zip(Cs[m], ys[m])), **kw)


curve(1.3e9 * (Cs / 8.64e19) ** 0.73, stroke="violet", width=2.6)
curve(np.sqrt(Cs / 120), stroke="green", width=1.8, dash="6 4")
curve(G * (Cs / 6) ** a, stroke="blue", width=2.2)

t3 = [(1.92e19, 400e6), (1.21e20, 1e9), (1.23e22, 10e9), (5.76e23, 67e9), (3.85e24, 175e9), (9.90e24, 280e9),
      (3.43e25, 520e9), (1.27e26, 1e12)]
for c, n in t3:
    svg.circle(ax(c), ay(n), 4, fill="green")

models = [("GPT-3", 3.14e23, 175e9, -8, 16, "end"), ("Gopher", 5.76e23, 280e9, 9, 5, "start"),
          ("MT-NLG", 6 * 530e9 * 270e9, 530e9, 8, -6, "start"), ("Chinchilla", 5.88e23, 70e9, 8, 16, "start")]
for name, c, n, dx, dy, anc in models:
    x, y = ax(c), ay(n)
    svg.rect(x - 4.5, y - 4.5, 9, 9, fill="red" if name != "Chinchilla" else "green", fill_opacity=0.9)
    svg.text(x + dx, y + dy, name, size=12, anchor=anc, color="ink", weight="600")

# legend on the right
LX = 572
leg = [("violet", None, "Kaplan 表 6", "N = 1.3×10^{9} · C^{0.73}", "C 为 C_{min}（PF-days）", None),
       ("green", "dot", "Chinchilla 方法 1", "表 3 的投影点", None, None),
       ("green", "6 4", "每参数 20 token", "N = √(C/120)", None, None),
       ("blue", None, "Chinchilla 方法 3", "本课式 (2.27)", "未四舍五入参数", "N ∝ C^{0.4565}")]
yy = 46
for col, dash, t1, t2, t3_, t4 in leg:
    if dash == "dot":
        svg.circle(LX + 12, yy - 4, 4, fill=col)
    else:
        svg.line(LX, yy - 4, LX + 24, yy - 4, color=col, width=2.4, dash=dash if dash else None)
    svg.text(LX + 30, yy, t1, size=12, color=col, weight="600")
    extra = [t for t in (t2, t3_, t4) if t]
    for k, t in enumerate(extra):
        svg.text(LX + 30, yy + 21 * (k + 1), t, size=11, color="muted")   # 21: room for superscripts
    yy += 28 + 21 * len(extra)
svg.rect(LX + 7, yy - 4, 9, 9, fill="red", fill_opacity=0.9)
svg.text(LX + 30, yy + 5, "实际训练的模型", size=12, color="ink", weight="600")

svg.text(X0, H - 8, "Kaplan 的 N 不含嵌入、C 是 C_{min}，两家的常数口径不同，这张图只作量级比较。", size=12, color="ink")

svg.save("l02-optimal-n.svg")
