"""Figure 12.16: prefill / decode disaggregated deployment units of DeepSeek-V3 (V3 report 3.4,
December 2024), with the later configuration from the Open-Source-Week Day 6 post (committed 2025-03-01) as a footnote.

Prefill unit: 4 nodes, 32 GPUs; attention TP4 + SP with DP8; MoE EP32; each GPU hosts its 8
routed experts plus 1 redundant expert (32 redundant experts in total).
Decode unit: 40 nodes, 320 GPUs; attention TP4 + SP with DP80; MoE EP320; each GPU hosts one
expert; 64 GPUs host the redundant and shared experts (their exact placement is not given; drawn
as the last 8 nodes for illustration).
Writes deepseek/figures/l12-pd-deploy.svg
"""

from l12_svg import SVG

assert (256 + 32) // 32 == 9 and 256 + 64 == 320 and 4 * 8 == 32 and 4 * 80 == 320

W, H = 760, 456
svg = SVG(W, H, "PD 分离：prefill 用 4 节点 32 GPU 为一个单元，decode 用 40 节点 320 GPU 为一个单元")

# request flow
svg.text(20, 26, "请求", size=13, weight="600")
svg.arrow(56, 21, 96, 21, color="ink", width=1.5)
svg.text(104, 26, "prefill：一次并行处理整段提示（算力受限）", size=12, color="blue")
svg.arrow(372, 21, 412, 21, color="amber", width=1.8)
svg.text(392, 14, "KV cache", size=11, anchor="middle", color="amber")
svg.text(420, 26, "decode：每步只生成 1 个 token（访存受限）", size=12, color="green")

# ---- prefill unit
PX, PY = 20, 58
svg.rect(PX, PY, 300, 262, fill="blue", fill_opacity=0.04, stroke="blue", width=1.3, rx=10)
svg.text(PX + 12, PY + 22, "prefill 单元：4 节点 × 8 GPU = 32 GPU", size=13, weight="600", color="blue")
cell = 26
for n in range(4):
    y = PY + 40 + n * 38
    svg.text(PX + 12, y + 18, "节点 %d" % n, size=11, color="muted")
    for g in range(8):
        x = PX + 62 + g * (cell + 2)
        svg.rect(x, y, cell, cell, fill="blue", fill_opacity=0.14, stroke="blue", width=0.8)
        svg.text(x + cell / 2, y + 17, "8+1", size=11, anchor="middle")
    for k in range(2):     # two TP4 groups per node
        x = PX + 62 + k * 4 * (cell + 2) - 2
        svg.rect(x, y - 3, 4 * (cell + 2), cell + 6, stroke="violet", width=1.4, rx=4, dash="4 2")
ty = PY + 40 + 4 * 38 + 14
svg.text(PX + 12, ty, "紫虚框 = 一个注意力 TP4 组（+SP），共 DP8 组", size=12, color="violet")
svg.text(PX + 12, ty + 20, "格内 8+1 = 8 个路由专家 + 1 个冗余专家（EP32）", size=12)
svg.text(PX + 12, ty + 40, "浅层稠密 MLP 用 TP1；两个 micro-batch 交替重叠", size=12, color="muted")

# ---- decode unit
DX, DY = 340, 58
svg.rect(DX, DY, 400, 262, fill="green", fill_opacity=0.04, stroke="green", width=1.3, rx=10)
svg.text(DX + 12, DY + 22, "decode 单元：40 节点 × 8 GPU = 320 GPU", size=13, weight="600", color="green")
c2 = 8
for n in range(40):
    r, cidx = divmod(n, 8)
    x0 = DX + 16 + cidx * 47
    y0 = DY + 40 + r * 24
    for g in range(8):
        red = n >= 32
        svg.rect(x0 + g * (c2 - 3) + (g // 4) * 0, y0, c2 - 4, 16,
                 fill="violet" if red else "blue", fill_opacity=0.45 if red else 0.28)
    svg.rect(x0 - 2, y0 - 2, 8 * (c2 - 3) + 3, 20, stroke="muted", width=0.6, rx=2)
ty = DY + 40 + 5 * 24 + 16
svg.text(DX + 12, ty, "蓝：256 块 GPU 各放 1 个路由专家（EP320）", size=12, color="blue")
svg.text(DX + 12, ty + 20, "紫：64 块 GPU 放冗余专家与共享专家（位置为示意）", size=12, color="violet")
svg.text(DX + 12, ty + 40, "共享专家当作必选的路由专家：每 token 选 9 个", size=12)
svg.text(DX + 12, ty + 60, "注意力 TP4 + SP，× DP80；all-to-all 走 IB 点对点 + IBGDA", size=12, color="muted")

# footnote: the Day 6 (2025-03-01) numbers
fy = 350
svg.text(20, fy, "据 DeepSeek-V3 技术报告 §3.4（2024-12）。开源周 Day 6 文章（2025-03-01）公布的线上配置已不同：", size=12, color="muted")
svg.text(20, fy + 22, "· prefill：EP32、注意力 DP32，每单元 4 节点，每 GPU 9 个路由专家 + 1 个共享专家", size=12)
svg.text(20, fy + 44, "· decode：EP144、注意力 DP144，每单元 18 节点，每 GPU 2 个路由专家 + 1 个共享专家；5 级流水重叠", size=12)
svg.text(20, fy + 66, "· 精度：矩阵乘法与 dispatch 用 FP8（与训练一致），核心 MLA 计算与 combine 用 BF16", size=12)
svg.text(20, fy + 94, "两种来源不要混用；profile-data 仓库又是 decode EP128、TP1。", size=11, color="muted")
svg.save("l12-pd-deploy.svg")
