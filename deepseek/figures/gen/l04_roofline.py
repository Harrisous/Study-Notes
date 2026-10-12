"""Figure 4.8: roofline of A100 80GB SXM and H100 SXM with the decode / prefill operating points.

Roof: attainable = min(F, BW * I). A100: F = 312 TFLOP/s dense BF16, BW = 2.039 TB/s -> ridge 153 FLOP/B.
H100 SXM: F = 989 TFLOP/s (dense, FlashAttention-3), BW = 3.35 TB/s -> ridge 295 FLOP/B.
Points (bf16, placed by the intensity formulas of the lesson, not measured kernel profiles):
  decode attention part  I = n_h / n_g : MHA 1, Llama-3-8B 4, GQA-8 with 64 heads 8, MQA with 64 heads 64
  decode weight part     I = B          : B = 1, 16, 64, 256
  prefill of 2048 tokens, Llama-2-7B    : I = FLOPs / (weights + KV written) ~ 1971
Numbers from lesson-04-doc §3.2, §4.5. Writes deepseek/figures/l04-roofline.svg
"""

import math

from l04_data import HW, MODELS, kv_bytes, n_mm_llama
from l04_svg import SVG

A, Hh = HW["A100"], HW["H100"]
ridgeA, ridgeH = A["F"] / A["BW"], Hh["F"] / Hh["BW"]
assert round(ridgeA) == 153 and round(ridgeH) == 295

m = MODELS["Llama-2-7B"]
NMM = n_mm_llama(m)
P = 2048
pre_flops = 2 * NMM * P + 2 * m["l"] * m["nh"] * m["dh"] * P * (P + 1)
pre_bytes = 2 * NMM + kv_bytes(m) * P
I_pre = pre_flops / pre_bytes
assert abs(I_pre - 1971) < 2


def att(hw, I):
    return min(hw["F"], hw["BW"] * I) / 1e12


assert abs(att(A, 1) - 2.039) < 1e-3 and abs(att(A, 8) - 16.3) < 0.05 and abs(att(A, 64) - 130.5) < 0.05

W, H = 720, 470
X0, X1, Y0, Y1 = 84, 640, 40, 380
LX0, LX1 = -1, 12          # log2 I: 0.5 .. 4096
LY0, LY1 = -0.5, 3.5       # log10 TFLOP/s: 0.32 .. 3162


def px(I):
    return X0 + (math.log2(I) - LX0) / (LX1 - LX0) * (X1 - X0)


def py(T):
    return Y1 - (math.log10(T) - LY0) / (LY1 - LY0) * (Y1 - Y0)


svg = SVG(W, H, "屋顶线模型：A100 与 H100，以及 decode、prefill 各部分的算术强度")
for e, lab in ((0, "1"), (1, "10"), (2, "100"), (3, "1000")):
    y = py(10 ** e)
    svg.line(X0, y, X1, y, color="grid", width=1)
    svg.text(X0 - 8, y + 4, lab, size=11, anchor="end", color="muted")
for k in range(0, 13, 2):
    x = px(2 ** k)
    svg.line(x, Y0, x, Y1, color="grid", width=1)
    svg.text(x, Y1 + 17, str(2 ** k), size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 40, "算术强度 I（FLOP/字节，对数刻度）", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "可达算力（TFLOP/s，对数刻度）", size=12, anchor="middle", color="muted", rotate=-90)


def roof(hw, col, width):
    ridge = hw["F"] / hw["BW"]
    Ilo = 2 ** LX0
    d = "M%.1f %.1f L%.1f %.1f L%.1f %.1f" % (px(Ilo), py(att(hw, Ilo)), px(ridge), py(hw["F"] / 1e12),
                                               px(2 ** LX1), py(hw["F"] / 1e12))
    svg.path(d, stroke=col, width=width, join="round")
    svg.line(px(ridge), py(hw["F"] / 1e12), px(ridge), Y1, color=col, width=1, dash="3 3")
    return ridge


roof(Hh, "muted", 2.0)
roof(A, "ink", 2.6)
svg.text(px(2 ** LX1) - 4, py(Hh["F"] / 1e12) - 8, "H100 SXM：989 TFLOP/s，3.35 TB/s", size=12, anchor="end", color="muted")
svg.text(px(330), py(A["F"] / 1e12) + 20, "A100 80GB SXM", size=12, weight="600")
svg.text(px(330), py(A["F"] / 1e12) + 36, "312 TFLOP/s，2.039 TB/s", size=12)
svg.text(px(ridgeA) - 4, Y1 - 8, "脊点 153", size=12, anchor="end")
svg.text(px(ridgeH) + 4, Y1 - 8, "脊点 295", size=12, color="muted")
# memory-bound note: three short lines that end left of the A100 ridge line, so no dashed line crosses them
xr = px(ridgeA) - 10
svg.text(xr, 286, "受带宽限制", size=13, color="muted", weight="600", anchor="end")
svg.text(xr, 302, "（memory-bound）", size=12, color="muted", anchor="end")
svg.text(xr, 318, "斜线：可达算力 = 带宽 × I", size=12, color="muted", anchor="end")
svg.text(px(620), py(45), "受算力限制", size=13, color="muted", weight="600")
svg.text(px(620), py(30), "（compute-bound）", size=12, color="muted")

# decode weight part: I = B (squares, labelled above the line)
for B in (1, 16, 64, 256):
    x, y = px(B), py(att(A, B))
    svg.rect(x - 4.5, y - 4.5, 9, 9, fill="ink")
    lab = "B = %d" % B
    svg.text(x - 8, y - 9, lab, size=12, anchor="end", weight="600")
svg.text(px(1.1), py(att(A, 16)) - 34, "■ decode 的权重部分：I = B", size=12, weight="600")

# decode attention part: I = n_h / n_g (amber circles, labelled below the line)
for I, lab in ((1, "MHA 1"), (4, "Llama-3-8B 4"), (8, "GQA-8（64 头）8"), (64, "MQA（64 头）64")):
    x, y = px(I), py(att(A, I))
    svg.circle(x, y, 5.5, fill="amber", stroke="surface", width=1.2)
    if I == 64:
        # to the right the two dashed ridge lines would cut through the label: put it under the slope
        svg.line(x + 2, y + 7, x + 14, y + 50, color="amber", width=1)
        svg.text(x + 26, y + 64, lab, size=12, color="amber", weight="600", anchor="end")
    else:
        svg.text(x + 9, y + 16, lab, size=12, color="amber", weight="600")
svg.text(px(2.2), py(0.62), "● decode 的注意力部分：I = n_{h}/n_{g}，与 B 无关", size=12, color="amber", weight="600")

# prefill
x, y = px(I_pre), py(att(A, I_pre))
svg.polygon([(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)], fill="blue")
svg.text(x + 2, y - 30, "prefill 2048 token", size=12, anchor="middle", color="blue", weight="600")
svg.text(x + 2, y - 14, "Llama-2-7B，I ≈ %d" % round(I_pre), size=12, anchor="middle", color="blue")

svg.text(16, H - 14, "点的位置按正文的强度公式算出（bf16），不是实测的 kernel 剖析；H100 的 989 是稠密值，NVIDIA 官网只印稀疏值。",
         size=12, color="muted")
svg.save("l04-roofline.svg")
