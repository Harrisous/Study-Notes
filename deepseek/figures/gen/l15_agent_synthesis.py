"""Figure 15.12: where V3.2's 85,267 agentic RL tasks come from, and how the general-agent tasks are synthesised.

Top: V3.2 paper Table 1 (number of tasks, environment real / synthesized, prompt extracted / synthesized).
Bottom: the general-agent environment-synthesis workflow of V3.2 section 3.2.3 (task category + sandbox with
bash and search -> data into the sandbox database -> task-specific tool functions -> simple task with Python
solution and verification functions, fix until verified -> iteratively harder, augment tools if needed ->
RL with DeepSeek-V3.2, keep instances with non-zero pass@100 -> 1,827 environments, 4,417 tasks).
Side note: Table 5 pass@1 on 50 sampled synthetic tasks.
Writes deepseek/figures/l15-agent-synthesis.svg
"""

from l15_svg import SVG

TASKS = [  # label, count, env, prompt
    ("搜索智能体", 50275, "real", "合成"),
    ("代码智能体", 24667, "real", "取自网上"),
    ("代码解释器", 5908, "real", "取自网上"),
    ("通用智能体", 4417, "synth", "合成"),
]
TOTAL = sum(t[1] for t in TASKS)
assert TOTAL == 85267
assert abs(4417 / 1827 - 2.4176) < 1e-3

W, H = 760, 500
svg = SVG(W, H, "V3.2 的智能体 RL 任务：四类来源，以及通用智能体环境的合成流程")

# ================================================================ top: bars
svg.text(16, 22, "85,267 个智能体 RL 任务（V3.2 论文表 1）", size=13, weight="700")
BX0, BMAX = 132, 330
for i, (lab, n, env, prm) in enumerate(TASKS):
    y = 38 + i * 30
    w = n / 50275 * BMAX
    real = env == "real"
    svg.text(BX0 - 10, y + 15, lab, size=12, anchor="end")
    svg.rect(BX0, y + 2, w, 18, fill="blue", fill_opacity=0.40 if real else 0.12, stroke="blue", width=1.2,
             dash=None if real else "4 3")
    svg.text(BX0 + w + 8, y + 15, "{:,}（{:.1f}%）".format(n, n / TOTAL * 100), size=12, weight="700")
    svg.text(BX0 + BMAX + 112, y + 15, "环境%s · 提示%s" % ("真实" if real else "合成", prm), size=12,
             color="muted")
svg.text(BX0, 172, "实心 = 真实环境（真实搜索 API、代码仓库、Jupyter）；虚线 = 合成环境。", size=11, color="muted")

# ================================================================ bottom: general-agent synthesis loop
svg.line(12, 186, 748, 186, color="rule", width=1)
svg.text(16, 208, "通用智能体：自动合成环境与任务（“难解、易验”）", size=13, weight="700")

bw, bh = 168, 58
row_y = 224
xs = [16, 200, 384, 568]
steps = [
    ["输入", "任务类别 + 沙箱", "（bash 与搜索工具）"],
    ["① 取数据", "上网生成或检索数据，", "存进沙箱数据库"],
    ["② 合成工具", "一组任务专用的工具，", "每个是一个函数"],
    ["③ 出题", "简单任务 + Python 解法", "+ Python 验证函数"],
]
for i, (x, lines) in enumerate(zip(xs, steps)):
    svg.box(x, row_y, bw, bh, [(lines[0], {"weight": "700", "size": 12}), (lines[1], {"size": 12}),
                                (lines[2], {"size": 12})], color="ink" if i == 0 else "blue",
            fill_opacity=0.05 if i == 0 else 0.10, line_gap=16)
    if i < 3:
        svg.arrow(x + bw + 2, row_y + bh / 2, xs[i + 1] - 4, row_y + bh / 2, color="ink", width=1.4)

# verification loop under step 3
lx = xs[3]
svg.box(lx, 320, bw, 44, [("解法没通过验证？", {"size": 12, "weight": "700"}), ("改解法或验证函数，直到通过", {"size": 12})],
        color="red", fill_opacity=0.06, line_gap=16)
svg.path("M%d %d v%d" % (lx + 60, row_y + bh, 320 - row_y - bh - 3), stroke="red", width=1.2, arrow=True)
svg.path("M%d %d v-%d" % (lx + 110, 320, 320 - row_y - bh - 3), stroke="red", width=1.2, arrow=True)

# harder loop
svg.box(xs[2], 320, bw, 44, [("④ 逐步加难", {"size": 12, "weight": "700"}), ("同步更新解法与验证函数", {"size": 12})],
        color="blue", fill_opacity=0.10, line_gap=16)
svg.arrow(lx - 2, 342, xs[2] + bw + 4, 342, color="ink", width=1.4)
svg.text(xs[2] + bw / 2 - 8, 382, "工具不够用时扩充工具集", size=11, anchor="end", color="muted")

# filter and result
svg.box(xs[1], 398, bw, 58, [("⑤ 用 V3.2 做 RL 试解", {"size": 12, "weight": "700"}), ("只保留 pass@100 > 0", {"size": 12}),
                              ("（100 次里至少解出 1 次）", {"size": 12})], color="green", fill_opacity=0.10, line_gap=16)
svg.path("M%d %d V%d H%d" % (xs[2] + bw / 2, 364, 427, xs[1] + bw + 4), stroke="ink", width=1.4, arrow=True)
svg.box(xs[0], 398, bw, 58, [("结果：1,827 个环境", {"size": 12, "weight": "700"}), ("4,417 个任务", {"size": 12}),
                              ("平均每个环境约 2.4 个", {"size": 12})], color="green", fill_opacity=0.20,
        line_gap=16)
svg.arrow(xs[1] - 2, 427, xs[0] + bw + 4, 427, color="ink", width=1.4)
svg.text(xs[0], 320, "中间产物：", size=12, color="muted")
svg.text(xs[0], 338, "数千个 ⟨环境, 工具, 任务, 验证器⟩", size=12, color="muted")
svg.text(xs[0], 356, "四元组", size=12, color="muted")

# table 5
svg.rect(560, 398, 188, 82, fill="grid", fill_opacity=0.6, rx=6)
svg.text(570, 416, "抽 50 个合成任务：pass@1", size=11, color="muted")
for j, (m, v) in enumerate([("V3.2-Exp", "12%"), ("Sonnet-4.5", "34%"), ("Gemini-3.0 Pro", "51%"),
                            ("GPT-5-Thinking", "62%")]):
    svg.text(570, 433 + j * 14, m, size=11)
    svg.text(738, 433 + j * 14, v, size=11, anchor="end", weight="700")
svg.text(16, 492, "据 DeepSeek-V3.2 论文表 1、表 5 与 §3.2.3 绘制。", size=11, color="muted")

svg.save("l15-agent-synthesis.svg")
