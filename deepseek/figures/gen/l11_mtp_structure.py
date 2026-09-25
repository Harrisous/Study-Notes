"""Figure 11.11: the MTP module of DeepSeek-V3 (redrawn after V3 report fig. 3, with D_MTP = 1 as actually used).

V3 eqs. 21-25 (arXiv:2412.19437 sec. 2.2): h~_i = W_k [RMSNorm(h_i^(k-1)); RMSNorm(Emb(x_{i+k}))],
h^(k)_{1:T-k} = TRM_k(h~_{1:T-k}), p^(k)_{i+k+1} = OutHead(h^(k)_i). Embedding and output head are shared
with the main model. Sizes from the HF config: d = 7168, W_1 in R^{7168 x 14336} (eh_proj), vocab 129280.
Example length T = 6 (research notes sec. 5.2 fig. 11.9). The paper's figure draws D = 2; V3 uses D = 1.
Writes deepseek/figures/l11-mtp-structure.svg
"""

from l11_svg import SVG, box

W, H = 720, 570
svg = SVG(W, H, "MTP 模块的结构与 token 对齐（D_MTP = 1）")

TW, TG = 34, 6


def tokens(x0, y, labels, color, fo=0.14):
    for k, lab in enumerate(labels):
        x = x0 + k * (TW + TG)
        svg.rect(x, y, TW, 26, fill=color, fill_opacity=fo, stroke=color, width=1.1, rx=4)
        svg.text(x + TW / 2, y + 17, lab, size=12, anchor="middle", color="ink")
    return x0 + len(labels) * (TW + TG) - TG


# ---------------------------------------------------------------- main model
MX, MW = 40, 234
svg.text(MX, 24, "主模型（61 层）", size=14, weight="700")
tokens(MX, 492, ["x_{%d}" % i for i in range(1, 7)], "ink", 0.06)
svg.text(MX, 546, "输入 x_{1}…x_{6}（位置 i = 1…6）", size=12, color="muted")
box(svg, MX, 432, MW, 36, "嵌入层 Emb（共享）", color="violet", fill_opacity=0.12, size=12)
box(svg, MX, 300, MW, 96, "Transformer 块 × 61", color="blue", fill_opacity=0.08, size=13, weight="600",
    sub="MLA + DeepSeekMoE", sub_size=12)
box(svg, MX, 150, MW, 36, "输出头 OutHead（共享）", color="violet", fill_opacity=0.12, size=12)
tokens(MX, 88, ["x_{%d}" % i for i in range(2, 8)], "green", 0.14)
svg.text(MX, 76, "目标 x_{2}…x_{7} → 交叉熵 ℒ_{Main}", size=12, color="green", weight="600")
for y1, y2 in ((490, 469), (430, 397), (298, 187), (148, 115)):
    svg.arrow(MX + MW / 2, y1, MX + MW / 2, y2, color="muted", width=1.4)
# hidden states h^(0)
svg.circle(MX + MW / 2, 262, 3, fill="blue")

# ---------------------------------------------------------------- MTP module
X0, XW = 400, 300
svg.text(X0, 24, "MTP 模块 1（深度 k = 1）", size=14, weight="700", color="green")
box(svg, X0 - 12, 40, XW + 24, 486, "", color="green", fill_opacity=0.03, dash="5 4")
tokens(X0 + 40, 492, ["x_{%d}" % i for i in range(2, 7)], "ink", 0.06)
svg.text(X0 + 40, 546, "输入 x_{2}…x_{6}（位置 i = 1…5 各取 x_{i+1}）", size=12, color="muted")
box(svg, X0 + 130, 440, 170, 30, "嵌入层 Emb（共享）", color="violet", fill_opacity=0.12, size=12)
box(svg, X0 + 150, 396, 130, 28, "RMSNorm", color="ink", fill_opacity=0.05, size=12)
box(svg, X0, 396, 130, 28, "RMSNorm", color="ink", fill_opacity=0.05, size=12)
box(svg, X0 + 40, 344, 220, 32, "拼接 [ · ; · ] → 14336 维", color="ink", fill_opacity=0.05, size=12)
box(svg, X0 + 20, 290, 260, 36, "线性投影 W₁^{MTP}：7168 × 14336", color="orange", fill_opacity=0.12, size=12)
box(svg, X0 + 20, 214, 260, 54, "Transformer 块 TRM₁", color="blue", fill_opacity=0.08, size=13, weight="600",
    sub="1 层：MLA + MoE（256 + 1 个专家）", sub_size=11.5)
box(svg, X0 + 40, 150, 220, 36, "输出头 OutHead（共享）", color="violet", fill_opacity=0.12, size=12)
tokens(X0 + 40, 88, ["x_{%d}" % i for i in range(3, 8)], "green", 0.14)
svg.text(X0 + 40, 76, "目标 x_{3}…x_{7} → 交叉熵 ℒ_{MTP}^{(1)}", size=12, color="green", weight="600")
# arrows inside MTP
svg.arrow(X0 + 215, 490, X0 + 215, 471, color="muted", width=1.3)
svg.arrow(X0 + 215, 440, X0 + 215, 425, color="muted", width=1.3)
svg.arrow(X0 + 215, 396, X0 + 200, 377, color="muted", width=1.3)
svg.arrow(X0 + 65, 396, X0 + 100, 377, color="muted", width=1.3)
svg.arrow(X0 + 150, 344, X0 + 150, 327, color="muted", width=1.3)
svg.arrow(X0 + 150, 290, X0 + 150, 269, color="muted", width=1.3)
svg.arrow(X0 + 150, 214, X0 + 150, 187, color="muted", width=1.3)
svg.arrow(X0 + 150, 148, X0 + 150, 115, color="muted", width=1.3)
# h^(0) from main model into the MTP RMSNorm
svg.path("M %s %s H %s V %s H %s" % (MX + MW / 2, 262, X0 - 2, 410, X0 - 1), stroke="blue", width=1.8, arrow=True)
svg.text(MX + MW + 8, 256, "*h*^{(0)}_{i}（i = 1…5）", size=12, color="blue")
# sharing links
svg.path("M %s %s H %s" % (MX + MW, 450, X0 + 129), stroke="violet", width=1.2, dash="4 3")
svg.path("M %s %s H %s" % (MX + MW, 168, X0 + 39), stroke="violet", width=1.2, dash="4 3")
svg.text((MX + MW + X0 - 12) / 2, 162, "同一份参数", size=11, anchor="middle", color="violet")
svg.text((MX + MW + X0 - 12) / 2, 444, "同一份参数", size=11, anchor="middle", color="violet")

svg.save("l11-mtp-structure.svg")
