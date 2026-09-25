"""Figure 12.9: which parts of DeepSeek-V3 run in which precision
(after Insights into DeepSeek-V3, Figure 1, plus V3 report 3.3.1 and 3.3.3).

Sources for each tag (research notes 3.3):
  embedding, output head: BF16 (Insights Fig. 1; V3 3.3.1 keeps them in "original precision")
  RMSNorm, router, RoPE, cross-entropy: FP32 (Insights Fig. 1)
  linear layers and experts: FP8 GEMM inputs (V3 3.3.1), core attention: BF16 (Insights Fig. 1)
  input of the linear after attention cached in E5M6; SwiGLU input cached in FP8 (V3 3.3.3)
  dispatch FP8, combine BF16 (V3 3.3.3); MTP linear projection BF16 (Insights Fig. 1)
  master weights and weight gradients FP32, AdamW moments BF16 (V3 3.3.3)
Schematic.
Writes deepseek/figures/l12-precision-map.svg
"""

from l12_svg import SVG

W, H = 760, 572
svg = SVG(W, H, "DeepSeek-V3 各组件的数值精度：只有线性层的乘法用 FP8")

TAGCOL = {"FP8": "orange", "BF16": "blue", "FP32": "green", "E5M6": "orange"}


def tag(x, y, t, sub=None):
    col = TAGCOL[t.split()[0]]
    w = 12 + 8 * len(t)
    svg.rect(x, y - 13, w, 20, fill=col, fill_opacity=0.18, stroke=col, width=1.1, rx=10)
    svg.text(x + w / 2, y + 1, t, size=12, anchor="middle", color=col, weight="700")
    if sub:
        svg.text(x + w + 6, y + 1, sub, size=11, color="muted")


def box(x, y, w, h, title, col="ink", fill=None):
    svg.rect(x, y, w, h, fill=fill, fill_opacity=0.06 if fill else None, stroke=col, width=1.2, rx=6)
    svg.text(x + 10, y + 19, title, size=13, weight="600")


# legend
lx = 30
for i, (t, lab) in enumerate((("FP8", "低精度乘法 / 存储 / 通信"), ("BF16", "半精度"), ("FP32", "单精度"))):
    tag(lx + i * 200, 22, t, lab)

L, R = 30, 330         # component box span
TX = 344               # tag column
y = 50
box(L, y, R - L, 30, "嵌入层 Embedding")
tag(TX, y + 20, "BF16")
y += 44
# transformer block container
svg.rect(L - 12, y - 8, 712, 322, stroke="muted", width=1.2, dash="6 4", rx=10)
svg.text(L - 4, y + 8, "Transformer 块 × 61", size=12, color="muted", weight="600")
y += 16
rows = [
    ("RMSNorm", ["FP32"], None),
    ("MLA 的线性投影（下投影、上投影）", ["FP8"], "GEMM 输入"),
    ("RoPE", ["FP32"], None),
    ("核心注意力 softmax(QKᵀ)V", ["BF16"], None),
    ("注意力输出投影 Wᴼ", ["FP8"], "GEMM；输入按 E5M6 缓存给反向"),
    ("RMSNorm", ["FP32"], None),
    ("路由器（门控打分）", ["FP32"], None),
    ("dispatch：把 token 发给专家", ["FP8"], "通信"),
    ("专家 FFN（路由 + 共享）", ["FP8"], "GEMM；SwiGLU 输入按 FP8 缓存，反向重算"),
    ("combine：把专家输出收回来", ["BF16"], "通信"),
]
for name, tags, sub in rows:
    box(L, y, R - L, 26, name)
    tag(TX, y + 18, tags[0], sub)
    y += 30
y += 20
box(L, y, R - L, 30, "输出头 Output Head")
tag(TX, y + 20, "BF16")
y += 38
box(L, y, R - L, 30, "交叉熵损失")
tag(TX, y + 20, "FP32")
y += 38
box(L, y, R - L, 30, "MTP 模块的线性投影")
tag(TX, y + 20, "BF16")

# optimizer column
OX, OY = 480, 436
svg.rect(OX - 12, OY - 22, 244, 134, fill="amber", fill_opacity=0.06, stroke="amber", width=1.2, rx=8)
svg.text(OX, OY - 4, "优化器与梯度（不在前向路径上）", size=12, weight="600", color="amber")
for i, (t, lab) in enumerate((("FP32", "主权重"), ("FP32", "权重梯度"), ("BF16", "AdamW 一阶矩"), ("BF16", "AdamW 二阶矩"))):
    tag(OX, OY + 22 + i * 25, t, lab)
svg.text(L, H - 10, "据 Insights 论文图 1 与 V3 技术报告 §3.3.1、§3.3.3 整理；各组件之间传递的激活都是 BF16。", size=11, color="muted")
svg.save("l12-precision-map.svg")
