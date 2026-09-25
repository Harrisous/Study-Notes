"""Figure 20.1: the six threads x eight generations of the course, one short entry per change.

A schematic (no data-driven geometry): rows are generations in release order (dates from
lesson-00-doc section 3.2), columns are the six threads of the course map, and every entry is a
change taught in the lesson named at the bottom of the figure. Empty cells mean "no change on this
thread in this generation". Thread colours follow lesson-00-doc section 5.2 (A blue, B violet,
C orange, D green, E amber, F ink).
Writes deepseek/figures/l20-matrix.svg
"""

from l20_svg import SVG

THREADS = [("A", "注意力 · KV", "blue"), ("B", "MoE", "violet"), ("C", "数值 · 系统", "orange"),
           ("D", "目标 · 后训练", "green"), ("E", "残差 · 记忆", "amber"), ("F", "规模 · 优化器", "ink")]

ROWS = [
    ("DeepSeek LLM", "2023-11", [
        ["GQA-8 · 95 层", "389 KB/token"], ["稠密 FFN"], ["BF16 混合精度"], ["SFT + DPO"],
        ["Pre-RMSNorm", "残差"], ["超参数规律", "C = MD"]]),
    ("MoE / Math", "2024-01～02", [
        None, ["细粒度 + 共享", "2 + 64 选 6"], None, ["GRPO", "去掉价值网络"], None, None]),
    ("V2", "2024-05", [
        ["MLA", "576 个数/层"], ["设备受限 M=3", "三种均衡损失"], ["部署 FP8 权重", "6 bit KV"],
        ["SFT + 两阶段", "GRPO"], None, None]),
    ("V3", "2024-12", [
        None, ["偏置均衡", "节点受限 M=4"], ["FP8 训练", "DualPipe"], ["MTP（D = 1）"],
        None, ["学习率 / γ / λ", "三条日程"]]),
    ("R1", "2025-01", [
        None, None, None, ["规则奖励纯 RL", "四阶段 · 蒸馏"], None, None]),
    ("V3.1 / V3.2", "2025-08～12", [
        ["DSA", "top-2048"], None, ["UE8M0 缩放", "FP8 KV 656 B"], ["规模化 RL", "智能体"],
        ["OCR 光学压缩", "mHC 论文"], None]),
    ("V4", "2026-04", [
        ["CSA / HCA", "3,514 B/token"], ["取消节点限制", "前 3 层哈希"], ["FP4 专家与", "索引器"],
        ["专家 → 多教师", "OPD"], ["mHC（n = 4）"], ["Muon"]]),
    ("V4.1-Flash", "2026-09", [
        ["CSA2 · CED", "890 B/token"], ["1 + 384 选 6", "按模态偏置"], ["FP4 主 KV", "SWA 有界回放"],
        ["算法不变", "MTP → DSpark"], ["Engram 196B", "单趟 mHC"], ["45T 多模态", "无稠密预热"]]),
]

W, H = 760, 540
svg = SVG(W, H, "六条主线 × 八代模型：每一格是这一代在这条主线上的改动")
X0, CW = 128, 105
Y0, RH = 64, 52

for j, (letter, name, col) in enumerate(THREADS):
    x = X0 + j * CW
    svg.rect(x + 2, 8, CW - 4, 44, fill=col, fill_opacity=0.14, stroke=col, width=1.2, rx=5)
    svg.text(x + CW / 2, 26, "主线 " + letter, size=13, anchor="middle", color=col, weight="700")
    svg.text(x + CW / 2, 44, name, size=12, anchor="middle")

for i, (gen, date, cells) in enumerate(ROWS):
    y = Y0 + i * RH
    if i % 2 == 0:
        svg.rect(4, y, W - 8, RH, fill="grid", fill_opacity=0.5)
    svg.text(10, y + 22, gen, size=13, weight="700")
    svg.text(10, y + 40, date, size=12, color="muted")
    for j, cell in enumerate(cells):
        x = X0 + j * CW
        col = THREADS[j][2]
        if cell is None:
            svg.text(x + CW / 2, y + RH / 2 + 4, "·", size=14, anchor="middle", color="rule")
            continue
        svg.rect(x + 3, y + 4, CW - 6, RH - 8, fill=col, fill_opacity=0.10, stroke=col, width=0.9, rx=4)
        if len(cell) == 1:
            svg.text(x + CW / 2, y + RH / 2 + 4, cell[0], size=12, anchor="middle")
        else:
            svg.text(x + CW / 2, y + RH / 2 - 4, cell[0], size=12, anchor="middle")
            svg.text(x + CW / 2, y + RH / 2 + 13, cell[1], size=12, anchor="middle", color="muted")

FY = Y0 + len(ROWS) * RH + 22
svg.text(10, FY, "出处：A 第 4、6、9、14、18、19 课；B 第 7、10、11、18、19 课；C 第 2、10、12、14、15、18、19 课；", size=12,
         color="muted")
svg.text(10, FY + 18, "D 第 3、6、8、10、11、13、15、18、19 课；E 第 5、16、17、19 课；F 第 6、11、18、19 课。", size=12,
         color="muted")

svg.save("l20-matrix.svg")
