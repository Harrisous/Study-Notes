"""Figure 11.1: DeepSeek-V3 at a glance (redrawn after V3 report fig. 2, with the real sizes).

Left: the 61-layer stack (3 dense + 58 MoE layers) plus the MTP module used in training.
Right: one MoE layer zoomed in: MLA with its cache, the router (sigmoid + bias, node-limited
Top-8) and the 1 shared + 256 routed experts drawn as 8 nodes x 32 experts.
All sizes from arXiv:2412.19437 sec. 4.2 and the HF config.json (research notes sec. 3.1).
Writes deepseek/figures/l11-v3-architecture.svg
"""

from l11_svg import SVG, box

W, H = 720, 572
svg = SVG(W, H, "DeepSeek-V3 总体结构：61 层、MLA + DeepSeekMoE、训练时附加 MTP 模块")

# ------------------------------------------------------------------ left: the stack
LX, LW = 18, 214
svg.text(LX, 22, "整体（自下而上）", size=13, weight="600")


def stack_box(y, h, label, sub, color, fo=0.10, dash=None):
    box(svg, LX, y, LW, h, label, color=color, fill_opacity=fo, size=13, sub=sub, dash=dash)


stack_box(40, 50, "MTP 模块（层号 61）", "训练时预测 x_{i+2}；推理可丢", "green", 0.08, dash="5 4")
stack_box(104, 42, "RMSNorm + 输出头", "129280 × 7168 → 下一个 token", "ink", 0.05)
stack_box(160, 210, "", "", "blue", 0.08)
svg.text(LX + LW / 2, 196, "第 3–60 层 × 58", size=14, anchor="middle", color="blue", weight="700")
svg.text(LX + LW / 2, 222, "MLA 注意力", size=13, anchor="middle")
svg.text(LX + LW / 2, 244, "+ DeepSeekMoE", size=13, anchor="middle")
svg.text(LX + LW / 2, 272, "1 个共享专家", size=12, anchor="middle", color="violet")
svg.text(LX + LW / 2, 292, "256 个路由专家，激活 8 个", size=12, anchor="middle", color="blue")
svg.text(LX + LW / 2, 312, "专家中间维 2048", size=12, anchor="middle", color="muted")
svg.text(LX + LW / 2, 340, "放大见右图 →", size=12, anchor="middle", color="muted")
stack_box(384, 56, "第 0–2 层 × 3", "MLA + 稠密 FFN（中间维 18432）", "ink", 0.05)
stack_box(454, 42, "嵌入层 Emb", "129280 × 7168", "ink", 0.05)
svg.text(LX + LW / 2, 520, "输入 token x_{1}, …, x_{T}", size=12, anchor="middle", color="muted")
for y1, y2 in ((508, 498), (452, 442), (382, 372), (158, 148), (102, 92)):
    svg.arrow(LX + LW / 2, y1, LX + LW / 2, y2, color="muted", width=1.4)
svg.text(LX, 544, "d = 7168，共 61 层", size=12, color="ink")
svg.text(LX, 562, "总参 671B，每 token 激活 37B", size=12, color="ink")

# ------------------------------------------------------------------ right: one MoE layer
RX = 262
svg.line(RX - 14, 30, RX - 14, 540, color="rule", width=1)
svg.text(RX, 22, "一个 MoE 层（放大）", size=13, weight="600")

# MLA block
box(svg, RX, 38, 440, 104, "", color="blue", fill_opacity=0.05)
svg.text(RX + 12, 60, "MLA 注意力（第 9 课）", size=13, color="blue", weight="600")
svg.text(RX + 12, 82, "n_{h} = 128 个头，d_{h} = 128，查询压缩 d′_{c} = 1536", size=12)
svg.text(RX + 12, 102, "每层只缓存两样东西：", size=12)
svg.rect(RX + 12, 109, 128, 21, fill="orange", fill_opacity=0.25, stroke="orange", width=1.2)
svg.text(RX + 76, 125.5, "*c*^{KV}：512", size=12, anchor="middle", color="orange", weight="600")
svg.rect(RX + 144, 109, 24, 21, fill="red", fill_opacity=0.18, stroke="red", width=1.2)
svg.text(RX + 176, 125.5, "*k*^{R}：64（RoPE 分支）", size=12, color="red")
svg.rect(RX + 8, 105, 164, 29, fill=None, stroke="amber", width=1.4, dash="4 3")
svg.text(RX + 432, 125.5, "缓存 576 个数/层", size=12, anchor="end", color="amber", weight="600")

svg.arrow(RX + 220, 144, RX + 220, 160, color="ink", width=1.4)
svg.text(RX + 228, 157, "*u*_{t}（7168 维）", size=12, color="ink")

# Router
box(svg, RX, 164, 440, 96, "", color="violet", fill_opacity=0.05)
svg.text(RX + 12, 184, "路由器（§2–§5）", size=13, color="violet", weight="600")
steps = [("亲和度", "s_{i} = σ(*u*·*e*_{i})"), ("只用于选择", "s_{i} + b_{i}"),
         ("节点受限", "≤ 4 个节点"), ("Top-8", "选 8 个专家"), ("门控值", "g_{i} ∝ s_{i}")]
bx, bw, gap = RX + 10, 78, 8
for k, (a, b) in enumerate(steps):
    x = bx + k * (bw + gap)
    col = "violet"
    box(svg, x, 196, bw, 50, a, color=col, fill_opacity=0.12, size=12, sub=b, sub_size=11)
    if k:
        svg.arrow(x - gap, 221, x - 1, 221, color="violet", width=1.3)

# Experts: 8 nodes x 32 routed experts
svg.text(RX, 284, "专家（每个节点 8 张 GPU、32 个路由专家）", size=13, weight="600")
active = {(0, 5), (0, 21), (2, 9), (3, 2), (3, 17), (3, 28), (5, 12), (5, 30)}
act_nodes = {0, 2, 3, 5}
GX, GY, CELL = RX + 142, 296, 9.0
box(svg, RX, 304, 90, 86, "共享专家 ×1", color="violet", fill_opacity=0.16, size=12, sub="每个 token 都用", sub_size=11)
for n in range(8):
    y = GY + n * (CELL + 4.2)
    lab_col = "blue" if n in act_nodes else "muted"
    svg.text(GX - 6, y + CELL - 1.5, "节点 %d" % (n + 1), size=11, anchor="end", color=lab_col)
    for j in range(32):
        x = GX + j * CELL
        if (n, j) in active:
            svg.rect(x + 0.8, y + 0.8, CELL - 1.6, CELL - 1.6, fill="blue", fill_opacity=0.85, rx=1.5)
        else:
            svg.rect(x + 0.8, y + 0.8, CELL - 1.6, CELL - 1.6, fill="muted", fill_opacity=0.16, rx=1.5)
svg.text(RX, GY + 8 * (CELL + 4.2) + 16, "蓝色 = 这个 token 激活的 8 个路由专家，分布在节点 1、3、4、6", size=12,
         color="blue")

# Output line
svg.text(RX, 446, "输出：*h*′_{t} = *u*_{t} + FFN^{(s)}(*u*_{t}) + Σ_{i∈Top-8} g_{i}·FFN_{i}(*u*_{t})", size=13)
svg.text(RX, 468, "单个专家：3 × 7168 × 2048 ≈ 44.0M 参数（SwiGLU）", size=12, color="muted")
svg.text(RX, 488, "每层 257 个专家 ≈ 11.3B；58 层的路由专家占总参 97.4%", size=12, color="muted")
svg.text(RX, 514, "与 V2 相比：sigmoid 亲和度、偏置只管选、节点受限 M = 4、", size=12, color="ink")
svg.text(RX, 532, "不丢 token，外加 MTP 训练目标", size=12, color="ink")

svg.save("l11-v3-architecture.svg")
