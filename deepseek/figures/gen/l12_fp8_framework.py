"""Figure 12.8: the FP8 mixed-precision framework for one Linear operator
(redrawn after DeepSeek-V3 report Figure 6, re-laid-out in the course colours).

Labels follow the paper's figure: Input (BF16) -To FP8-> Fprop (sum in FP32) -To BF16-> Output;
Output Gradient (BF16) -To FP8-> Dgrad -To BF16-> Input Gradient; Wgrad takes both FP8 inputs,
accumulates in FP32 and emits the Weight Gradient in FP32 -To BF16-> Optimizer States
-To FP32-> Master Weight -To FP8-> Weight. Schematic.
Writes deepseek/figures/l12-fp8-framework.svg
"""

from l12_svg import SVG

W, H = 760, 478
svg = SVG(W, H, "一个 Linear 算子的 FP8 混合精度数据流：三个 GEMM 的输入都是 FP8，累加在 FP32")


def node(cx, cy, title, dtype, col, w=150):
    svg.rect(cx - w / 2, cy - 22, w, 44, fill=col, fill_opacity=0.12, stroke=col, width=1.3, rx=6)
    svg.text(cx, cy - 4, title, size=13, anchor="middle", weight="600")
    svg.text(cx, cy + 14, dtype, size=12, anchor="middle", color=col if col != "muted" else "ink")


def gemm(cx, cy, name, sub):
    svg.rect(cx - 90, cy - 32, 180, 64, fill="blue", fill_opacity=0.08, stroke="blue", width=1.8, rx=8)
    svg.text(cx, cy - 10, name + "（" + sub + "）", size=14, anchor="middle", weight="700", color="blue")
    svg.text(cx, cy + 9, "⊗ FP8 × FP8", size=12, anchor="middle", color="orange")
    svg.text(cx, cy + 25, "Σ 累加在 FP32", size=12, anchor="middle", color="green")


def conv(x1, y1, x2, y2, label, col, lx=None, ly=None, anchor="start"):
    svg.arrow(x1, y1, x2, y2, color=col, width=1.5)
    if label:
        mx = (x1 + x2) / 2 + 6 if lx is None else lx
        my = (y1 + y2) / 2 + 4 if ly is None else ly
        svg.text(mx, my, label, size=12, color=col, anchor=anchor, weight="600")


LX, CXc, RX = 120, 380, 630
# weight chain (top centre)
node(CXc, 34, "主权重 Master Weight", "FP32", "amber", w=196)
conv(CXc, 54, CXc, 104, "To FP8", "orange")
node(CXc, 124, "权重 Weight", "FP8（128×128 块缩放）", "orange", w=196)
# inputs
node(LX, 124, "输入 Input", "BF16", "muted", w=160)
node(RX, 124, "输出梯度 Output Grad", "BF16", "muted", w=190)
# GEMMs
gemm(LX, 228, "Fprop", "前向")
gemm(RX, 228, "Dgrad", "求输入梯度")
gemm(CXc, 318, "Wgrad", "求权重梯度")
conv(LX, 144, LX, 196, "To FP8", "orange")
conv(RX, 144, RX, 196, "To FP8", "orange")
conv(CXc - 50, 144, LX + 70, 198, "", "orange")
conv(CXc + 50, 144, RX - 70, 198, "", "orange")
# FP8 copies into Wgrad
svg.circle(LX + 20 - 20, 172, 3.2, fill="orange")
svg.circle(RX, 172, 3.2, fill="orange")
conv(LX, 172, CXc - 60, 288, "FP8 输入（缓存到反向）", "orange", lx=278, ly=296, anchor="end")
conv(RX, 172, CXc + 60, 288, "FP8 输出梯度", "orange", lx=482, ly=296, anchor="start")
# outputs
conv(LX, 258, LX, 360, "To BF16", "green", lx=LX + 6, ly=344)
node(LX, 380, "输出 Output", "BF16", "green", w=160)
conv(RX, 258, RX, 360, "To BF16", "green", lx=RX + 6, ly=344)
node(RX, 380, "输入梯度 Input Grad", "BF16", "green", w=190)
# weight-gradient chain back to the master weight
conv(CXc, 348, CXc, 382, "", "green")
node(CXc, 402, "权重梯度 Weight Grad", "FP32（跨 micro-batch 累加）", "amber", w=224)
conv(CXc + 112, 402, 556, 450, "", "amber")
svg.text(CXc + 118, 438, "To BF16", size=12, color="amber", weight="600")
node(636, 450, "优化器状态", "BF16（AdamW 两个矩）", "amber", w=168)
svg.path("M720 450 L744 450 L744 34 L%d 34" % (CXc + 90), stroke="amber", width=1.5, arrow=True)
svg.text(738, 300, "To FP32", size=12, color="amber", weight="600", anchor="end")
svg.save("l12-fp8-framework.svg")
