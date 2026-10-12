"""Figure M7.12: one Muon step as used by DeepSeek-V4 (algorithm 1 of the V4 report), with a real shape.

Weight W ∈ R^{n×m}; the worked shape is [7168, 18432], the dense-FFN shape of DeepSeek-V3 (hidden 7168,
intermediate_size 18432 in its config.json). DeepSeek-V4 has MoE layers in every block and no dense FFN
(V4 §4.2.1); the V3 shape is borrowed only as a representative large matrix (numbers are computed below):
  RMS(U V^T) = 1/√max(n, m) = 0.007366;  scale 0.18·√18432 = 24.44;  RMS after scaling = 0.18.
  NS cost per iteration 2(2nm² + m³) with m = min = 7168, n = max = 18432.
Bottom: which parameters DeepSeek-V4 gives to AdamW and which to Muon (V4 §2.4).
Writes math/figures/m07-muon-pipeline.svg
"""

import math

from m07_svg import SVG

n, m = 18432, 7168
rms_o = 1 / math.sqrt(max(n, m))
scale = 0.18 * math.sqrt(max(n, m))
assert abs(rms_o * scale - 0.18) < 1e-12

W, H = 740, 350
svg = SVG(W, H, "DeepSeek-V4 的一步 Muon 更新，以及哪些参数用 Muon、哪些用 AdamW")
boxes = [
    (4, "blue", "梯度 *G*_{t}", ["形状 [n, m]", "本步反传得到"]),
    (127, "amber", "动量缓冲 *M*_{t}", ["μ*M*_{t−1} + *G*_{t}", "μ = 0.95（状态）"]),
    (250, "blue", "Nesterov", ["μ*M*_{t} + *G*_{t}", "再除以 ‖·‖_{F}"]),
    (373, "orange", "牛顿–舒尔茨 × 10", ["前 8 步 (3.4445, …)", "后 2 步 (2, −1.5, 0.5)"]),
    (496, "green", "缩放", ["× 0.18·√max(n, m)", "RMS 对齐 AdamW"]),
    (619, "ink", "更新 W", ["W(1 − ηλ) − η*O*_{t}", "λ = 0.1（解耦衰减）"]),
]
bw, bh, by = 112, 92, 44
for x, col, title, lines in boxes:
    svg.rect(x, by, bw, bh, fill=col, fill_opacity=0.10, stroke=col, width=1.6, rx=8)
    svg.text(x + bw / 2, by + 24, title, size=13, anchor="middle", weight="600", color=col)
    for i, t in enumerate(lines):
        svg.text(x + bw / 2, by + 50 + 18 * i, t, size=11, anchor="middle")
for i in range(len(boxes) - 1):
    x0 = boxes[i][0] + bw
    x1 = boxes[i + 1][0]
    svg.arrow(x0 + 1, by + bh / 2, x1 - 1, by + bh / 2, color="muted", width=1.4)
svg.text(20, 22, "一步 Muon（DeepSeek-V4 算法 1）：只对二维权重矩阵做；动量先加再正交化", size=13, weight="600")

# worked numbers under the pipeline
yy = by + bh + 30
svg.text(12, yy, "以 [7168, 18432] 的矩阵（DeepSeek-V3 稠密 FFN 的形状）为例：", size=12, weight="600")
svg.text(12, yy + 20, "正交化后 UV^{⊤} 有 7168 个奇异值为 1，RMS = 1/√18432 = %.5f；" % rms_o, size=12)
svg.text(12, yy + 40, "乘 0.18·√18432 = %.2f 后 RMS = 0.18，与 AdamW 更新的典型 RMS 同量级（Moonlight 用 0.2）。" % scale,
         size=12)

# bottom: parameter split
yb = 240
svg.rect(12, yb, 350, 96, fill="violet", fill_opacity=0.08, stroke="violet", width=1.2, rx=8)
svg.text(24, yb + 22, "AdamW（β_{1} = 0.9，β_{2} = 0.95，ε = 10^{−20}，衰减 0.1）", size=12.5, weight="600", color="violet")
for i, t in enumerate(["词嵌入、预测头（输出层）", "mHC 的静态偏置与门控因子", "全部 RMSNorm 权重（向量）"]):
    svg.text(30, yb + 46 + 18 * i, "· " + t, size=12)
svg.rect(378, yb, 350, 96, fill="green", fill_opacity=0.08, stroke="green", width=1.2, rx=8)
svg.text(390, yb + 22, "Muon（动量 0.95，衰减 0.1，RMS 0.18）", size=12.5, weight="600", color="green")
for i, t in enumerate(["其余全部模块", "每个二维权重矩阵各自正交化", "同形状的矩阵可以合成一批做 NS"]):
    svg.text(396, yb + 46 + 18 * i, "· " + t, size=12)
svg.save("m07-muon-pipeline.svg")
print("rms", rms_o, "scale", scale)
