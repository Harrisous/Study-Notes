"""Figure 6.6: depth versus width, drawn to scale (height = number of layers, width = d).

LLaMA-1 6.7B: 32 layers x 4096; LLaMA-1 65.2B: 80 x 8192 (llama1 Table 2).
DeepSeek LLM 7B: 30 x 4096; 67B: 95 x 8192 (dsllm Table 2).
Per-layer parameter counts (d_ff = 22016, d_h = 128) are computed here:
  MHA layer 4d^2 + 3 d d_ff + 2d, GQA-8 layer 2d^2 + 2 d (8*128) + 3 d d_ff + 2d.
Writes deepseek/figures/l06-depth-width.svg
"""

from l06_svg import SVG

d67, dff = 8192, 22016
mha = 4 * d67 * d67 + 3 * d67 * dff + 2 * d67
gqa = 2 * d67 * d67 + 2 * d67 * 8 * 128 + 3 * d67 * dff + 2 * d67
eq_layers = 80 * mha / gqa          # GQA layers with the parameters of 80 MHA layers
assert abs(eq_layers - 93.575) < 0.01

W, H = 760, 470
svg = SVG(W, H, "按比例绘制：层数（高）× 隐藏维度（宽）")
BASE = 380          # bottom of all rectangles
PX_L = 3.0          # px per layer
PX_D = 150 / 8192   # px per unit of d

models = [
    ("LLaMA-1 6.7B", 32, 4096, "MHA", "muted", 40),
    ("DeepSeek 7B", 30, 4096, "MHA", "blue", 150),
    ("LLaMA-1 65.2B", 80, 8192, "MHA", "muted", 285),
    ("DeepSeek 67B", 95, 8192, "GQA-8", "blue", 470),
]
for name, L, d, att, col, x in models:
    w, h = d * PX_D, L * PX_L
    y = BASE - h
    svg.rect(x, y, w, h, fill=col, fill_opacity=0.12, stroke=col, width=1.5)
    # faint layer lines every 10 layers
    for k in range(10, L, 10):
        yy = BASE - k * PX_L
        svg.line(x, yy, x + w, yy, color=col, width=0.6, opacity=0.35)
    svg.text(x + w / 2, BASE + 22, name, size=13, anchor="middle", weight="600", color="ink")
    svg.text(x + w / 2, BASE + 40, "%d 层 × %d" % (L, d), size=12, anchor="middle", color="muted")
    svg.text(x + w / 2, BASE + 58, att, size=12, anchor="middle", color=col)
    svg.text(x + w / 2, y - 8, "%d" % L, size=12, anchor="middle", color=col)

# DeepSeek 67B: mark the parameter-equivalent of 80 MHA layers
x67, w67 = 470, 8192 * PX_D
y_eq = BASE - eq_layers * PX_L
svg.line(x67 - 6, y_eq, x67 + w67 + 6, y_eq, color="ink", width=1.4, dash="5 3")
svg.text(x67 + w67 + 12, y_eq - 6, "93.6 层 GQA-8", size=12, color="ink")
svg.text(x67 + w67 + 12, y_eq + 10, "= 80 层 MHA 的参数", size=12, color="ink")
y80 = BASE - 80 * PX_L
svg.rect(x67, BASE - 95 * PX_L, w67, 15 * PX_L, fill="blue", fill_opacity=0.18)
svg.line(x67 - 6, y80, x67 + w67 + 6, y80, color="muted", width=1.0, dash="2 3")
svg.text(x67 + w67 + 12, y80 + 16, "80 层", size=12, color="muted")
svg.text(x67 + w67 + 12, y80 + 50, "80 层以上的 15 层", size=12, color="blue")
svg.text(x67 + w67 + 12, y80 + 66, "（深色）", size=12, color="blue")

# legend-like note at top-left
svg.text(24, 30, "高 = 层数（每层 3 px），宽 = 隐藏维度 d（8192 = 150 px）", size=12, color="muted")
svg.text(24, 50, "每层参数（d = 8192，FFN 22016）：MHA %.2f 亿，GQA-8 %.2f 亿" % (mha / 1e8, gqa / 1e8), size=12,
         color="muted")
svg.text(24, 70, "GQA 每层省 %.2f 亿；95 层共省 %.1f 亿" % ((mha - gqa) / 1e8, 95 * (mha - gqa) / 1e8), size=12,
         color="muted")

svg.save("l06-depth-width.svg")
