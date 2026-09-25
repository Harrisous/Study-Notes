"""Figure M4.8: softmax saturation. Mean of the largest softmax weight over 512 scores ~ N(0, s^2).

For each scale s the same 2000 x 512 standard-normal draws (numpy default_rng(1)) are
multiplied by s, so the curve is smooth. s = 1 is what 1/sqrt(d_k) scaling aims for;
s = 8 is d_k = 64 without scaling (section 3).
Writes math/figures/m04-saturation.svg
"""

import math

import numpy as np

from m04_svg import SVG, Axes

NREP, NKEY = 2000, 512
rng = np.random.default_rng(1)
g = rng.standard_normal((NREP, NKEY))


def mean_max_weight(s):
    a = g * s
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return float((e.max(1) / e.sum(1)).mean())


ss = np.logspace(math.log10(0.25), math.log10(16), 120)
ys = [mean_max_weight(s) for s in ss]
marks = {s: mean_max_weight(s) for s in (1, 2, 4, 8, 11.31, 16)}
for s, v in marks.items():
    print("s=%g mean max weight %.4f" % (s, v))

W, H = 720, 330
svg = SVG(W, H, "分数的标准差越大，softmax 越接近独热")
A = Axes(svg, 70, 40, 440, 220, (0.25, 16), (0, 1), logx=True)
A.frame(xticks=[0.25, 0.5, 1, 2, 4, 8, 16], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        xfmt=lambda v: "%g" % v, xlabel="分数的标准差 s（对数轴）", ylabel="最大权重的平均值",
        ylabel_dx=-46)
A.curve(ss, ys, stroke="blue", width=2.4)
svg.line(A.x0, A.Y(1 / NKEY), A.x0 + A.w, A.Y(1 / NKEY), color="green", width=1.3, dash="5 4")
svg.text(A.X(2.6), A.Y(1 / NKEY) - 6, "均匀分布：每个权重 1/512 ≈ 0.002", size=12, color="green")

v1, v8 = marks[1], marks[8]
svg.circle(*A.P(1, v1), 5, fill="green")
svg.text(A.X(1) - 8, A.Y(v1) - 12, "s = 1：%.3f" % v1, size=12, color="green", anchor="end")
svg.circle(*A.P(8, v8), 5, fill="red")
svg.text(A.X(8) - 8, A.Y(v8) - 10, "s = 8：%.2f" % v8, size=12, color="red", anchor="end")

# notes on the right
nx = 540
svg.text(nx, 60, "512 个分数 ~ N(0, s²)，", size=12)
svg.text(nx, 78, "重复 2000 次取平均。", size=12)
# lines that carry a subscript get 24 units of leading, so the lowered "k" clears the next line
svg.text(nx, 110, "s = 1：除以 √d_{k} 之后", size=12, color="green")
svg.text(nx, 134, "的尺度，权重分得很散。", size=12, color="green")
svg.text(nx, 166, "s = 8：d_{k} = 64 而不缩放，", size=12, color="red")
svg.text(nx, 190, "一个键平均拿走约 3/4。", size=12, color="red")
svg.text(nx, 222, "等价的说法：乘以 s", size=12, color="muted")
svg.text(nx, 240, "= 温度 τ = 1/s。", size=12, color="muted")

svg.save("m04-saturation.svg")
