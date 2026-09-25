"""Figures 13.13-13.14: distillation of DeepSeek-R1 into six dense models.

13.13 l13-distill.svg          schematic: the 804,745-sample SFT set -> six students (SFT only), and the
                              control experiment: large-scale RL directly on Qwen2.5-32B-Base.
13.14 l13-distill-results.svg (a) AIME 2024 pass@1 versus parameter count (log axis) for the six students,
                              with reference lines; (b) Qwen-32B: QwQ-32B-Preview vs RL-from-base vs
                              distillation on five metrics.
Numbers: R1 paper v1 Table 5/6 = v2 Table 15/16, v2 Table 6 (bases, learning rates), v2 section B.4.1/B.4.3,
as recorded in reference-docs/lesson-13-doc.md sections 3.5 and 3.15.
"""

from l13_svg import SVG, Axes

STUDENTS = [  # name, base, lr, params (B), family, AIME pass@1
    ("R1-Distill-Qwen-1.5B", "Qwen2.5-Math-1.5B", "1×10^{-4}", 1.5, "qwen", 28.9),
    ("R1-Distill-Qwen-7B", "Qwen2.5-Math-7B", "8×10^{-5}", 7, "qwen", 55.5),
    ("R1-Distill-Llama-8B", "Llama-3.1-8B", "5×10^{-5}", 8, "llama", 50.4),
    ("R1-Distill-Qwen-14B", "Qwen2.5-14B", "7×10^{-5}", 14, "qwen", 69.7),
    ("R1-Distill-Qwen-32B", "Qwen2.5-32B", "6×10^{-5}", 32, "qwen", 72.6),
    ("R1-Distill-Llama-70B", "Llama-3.3-70B-Instruct", "2×10^{-5}", 70, "llama", 70.0),
]

# ------------------------------------------------------------------ schematic
svg = SVG(760, 392, "把 R1 蒸馏进 6 个稠密小模型，以及在 Qwen2.5-32B 上直接做 RL 的对照实验")
svg.box(16, 40, 196, 62, color="green", lines=["DeepSeek-R1 流水线", "老师：671B MoE，37B 激活"], size=13.5, sub_size=12)
svg.arrow(114, 104, 114, 128, width=1.5)
svg.rect(16, 130, 196, 78, fill="amber", fill_opacity=0.12, stroke="amber", width=1.3, rx=8)
svg.text(114, 153, "804,745 条 SFT 数据", size=13.5, anchor="middle", weight="700")
svg.text(114, 173, "推理 626,933 + 一般 177,812", size=12, anchor="middle")
svg.text(114, 192, "即 §5 拒绝采样得到的那份", size=12, anchor="middle", color="muted")
# fan out to students
svg.text(16, 234, "只做 SFT，不做 RL", size=13, color="violet", weight="700")
svg.text(16, 254, "2–3 轮 · batch 64 · 上下文 32,768", size=12, color="muted")
svg.text(386, 28, "底座", size=12, color="muted", weight="600")
svg.text(560, 28, "学生（蒸馏后）", size=12, color="muted", weight="600")
svg.text(756, 28, "初始 lr", size=12, anchor="end", color="muted", weight="600")
y0 = 44
for k, (name, base, lr, _, fam, _) in enumerate(STUDENTS):
    y = y0 + k * 34
    col = "blue" if fam == "qwen" else "violet"
    svg.path("M214 170 C 300 170, 300 %d, 380 %d" % (y + 14, y + 14), stroke="violet", width=1.1, arrow=True)
    svg.rect(382, y, 150, 28, fill="surface", stroke=col, width=1.2, rx=6)
    svg.text(457, y + 19, base, size=12, anchor="middle")
    svg.arrow(534, y + 14, 552, y + 14, width=1.2, color=col)
    svg.rect(554, y, 150, 28, fill=col, fill_opacity=0.12, stroke=col, width=1.2, rx=6)
    svg.text(629, y + 19, name.replace("R1-Distill-", "Distill-"), size=12, anchor="middle", weight="600")
    svg.text(756, y + 19, lr, size=12, anchor="end", color="muted")
# control experiment
cy = 300
svg.line(16, cy - 26, 744, cy - 26, color="rule", width=1, dash="4 4")
svg.text(16, cy - 6, "对照实验：同一个 32B 底座，不蒸馏，直接做 R1-Zero 式的大规模 RL", size=13, weight="600")
svg.rect(16, cy + 8, 170, 40, fill="surface", stroke="blue", width=1.2, rx=6)
svg.text(101, cy + 33, "Qwen2.5-32B-Base", size=12.5, anchor="middle")
svg.arrow(188, cy + 28, 226, cy + 28, width=1.4, color="blue")
svg.rect(228, cy + 8, 300, 40, fill="surface", stroke="ink", width=1.2, rx=6)
svg.text(378, cy + 25, "RL 超过 1 万步：数学、代码、STEM", size=12.5, anchor="middle", weight="600")
svg.text(378, cy + 42, "lr 2×10^{-6}，KL 0.001，G = 16，最大 32,768", size=12, anchor="middle", color="muted")
svg.arrow(530, cy + 28, 552, cy + 28, width=1.4, color="blue")
svg.rect(554, cy + 8, 190, 40, fill="blue", fill_opacity=0.12, stroke="blue", width=1.2, rx=6)
svg.text(649, cy + 33, "Qwen2.5-32B-Zero", size=12.5, anchor="middle", weight="600")
svg.text(16, cy + 76, "v1 称之为 DeepSeek-R1-Zero-Qwen-32B；结果对比见图 13.14(b)", size=12, color="muted")
svg.save("l13-distill.svg")

# ------------------------------------------------------------------ results
svg = SVG(760, 640, "蒸馏模型的 AIME 成绩随参数量变化，以及 Qwen-32B 上蒸馏与直接 RL 的对比")
ax = Axes(70, 560, 40, 290, (1, 100), (0, 90), xlog=True)
svg.text(16, 20, "(a) 蒸馏模型的 AIME 2024 pass@1 与参数量（横轴对数）", size=13, weight="600")
for v in range(0, 91, 20):
    y = ax.py(v)
    svg.line(ax.X0, y, ax.X1, y, color="grid", width=1)
    svg.text(ax.X0 - 6, y + 4, "%d%%" % v, size=11, anchor="end", color="muted")
for p in (1, 2, 5, 10, 20, 50, 100):
    x = ax.px(p)
    svg.line(x, ax.Y0, x, ax.Y1, color="grid", width=1)
    svg.text(x, ax.Y1 + 16, "%gB" % p, size=11, anchor="middle", color="muted")
svg.text((ax.X0 + ax.X1) / 2, ax.Y1 + 36, "学生模型参数量（十亿，对数刻度）", size=12, anchor="middle", color="muted")
refs = [("DeepSeek-R1（老师）79.8", 79.8, "green"), ("o1-mini 63.6（仅 v1 表）", 63.6, "muted"),
        ("QwQ-32B-Preview 50.0", 50.0, "muted"), ("Claude-3.5-Sonnet 16.0", 16.0, "muted"),
        ("GPT-4o-0513 9.3", 9.3, "muted")]
for name, v, col in refs:
    y = ax.py(v)
    svg.line(ax.X0, y, ax.X1, y, color=col, width=1.2, dash="5 4")
    svg.text(ax.X1 + 6, y + 4, name, size=12, color=col if col != "muted" else "ink")
q = [(p, a) for _, _, _, p, f, a in STUDENTS if f == "qwen"]
svg.polyline([(ax.px(p), ax.py(a)) for p, a in q], stroke="blue", width=1.8)
for name, base, lr, p, fam, a in STUDENTS:
    x, y = ax.px(p), ax.py(a)
    if fam == "qwen":
        svg.circle(x, y, 5, fill="blue")
    else:
        svg.rect(x - 5, y - 5, 10, 10, fill="violet")
    short = name.replace("R1-Distill-", "")
    dx, anc, dy = 8, "start", 4
    if short in ("Qwen-14B",):
        dx, anc, dy = -8, "end", -8
    if short == "Llama-8B":
        dy = 18
    if short == "Qwen-7B":
        dx, anc, dy = -8, "end", -6
    if short == "Llama-70B":
        dx, anc, dy = 8, "start", 4
    if short == "Qwen-32B":
        dy = -8
    svg.text(x + dx, y + dy, "%s %.1f" % (short, a), size=12, anchor=anc,
             color="blue" if fam == "qwen" else "violet", weight="600")

# (b) grouped bars
b0 = 360
svg.text(16, b0, "(b) 同一个 Qwen-32B 底座：蒸馏 vs 直接做大规模 RL", size=13, weight="600")
metrics = [("AIME pass@1", [50.0, 47.0, 72.6]), ("AIME cons@64", [60.0, 60.0, 83.3]),
           ("MATH-500", [90.6, 91.6, 94.3]), ("GPQA Diamond", [54.5, 55.0, 62.1]),
           ("LiveCodeBench", [41.9, 40.2, 57.2])]
models = [("QwQ-32B-Preview", "muted"), ("Qwen2.5-32B-Zero（RL）", "blue"), ("Distill-Qwen-32B（蒸馏）", "green")]
bx = Axes(70, 744, b0 + 40, b0 + 220, (0, 5), (0, 100))
for v in (0, 25, 50, 75, 100):
    y = bx.py(v)
    svg.line(bx.X0, y, bx.X1, y, color="grid", width=1)
    svg.text(bx.X0 - 6, y + 4, "%d%%" % v, size=11, anchor="end", color="muted")
gw = (bx.X1 - bx.X0) / len(metrics)
bw = 34
for i, (mname, vals) in enumerate(metrics):
    cx = bx.X0 + gw * (i + 0.5)
    for j, v in enumerate(vals):
        x = cx + (j - 1) * (bw + 4) - bw / 2
        col = models[j][1]
        svg.rect(x, bx.py(v), bw, bx.Y1 - bx.py(v), fill=col, fill_opacity=0.8 if col != "muted" else 0.45)
        svg.text(x + bw / 2, bx.py(v) - 5, "%.1f" % v, size=11, anchor="middle",
                 color=col if col != "muted" else "ink")
    svg.text(cx, bx.Y1 + 18, mname, size=12, anchor="middle")
svg.line(bx.X0, bx.Y1, bx.X1, bx.Y1, color="muted", width=1.1)
lx = 70
for name, col in models:
    svg.rect(lx, 624, 16, 12, fill=col, fill_opacity=0.8 if col != "muted" else 0.45)
    svg.text(lx + 22, 635, name, size=12)
    lx += 36 + len(name) * 10.5
svg.save("l13-distill-results.svg")
