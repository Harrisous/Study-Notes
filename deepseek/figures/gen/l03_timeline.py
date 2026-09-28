"""Figure 3.18: lineage of the methods in lesson 3, placed by arXiv first-version date.

Dates (lesson-03 research notes, sec. 0): GAE 2015-06-08; Christiano 2017-06-12; PPO 2017-07-20;
Ziegler 2019-09-18; Stiennon 2020-09-02; InstructGPT 2022-03-04; HH-RLHF 2022-04-12; DPO 2023-05-29;
DeepSeek LLM 2024-01-05; DeepSeekMath 2024-02-05. Writes deepseek/figures/l03-timeline.svg
"""
import datetime as dt
from l03_svg import SVG, Axis

W, H = 740, 322
svg = SVG(W, H, "第 3 课方法谱系：按 arXiv 首版日期")
t0, t1 = 2015.0, 2024.6
ax = Axis(t0, t1, 36, 716)
Y = 150


def year(s):
    d = dt.date.fromisoformat(s)
    start = dt.date(d.year, 1, 1)
    return d.year + (d - start).days / 365.25


svg.line(ax(t0), Y, ax(t1), Y, color="ink", width=1.6)
for y in range(2015, 2025):
    svg.line(ax(y), Y - 5, ax(y), Y + 5, color="ink", width=1.2)
    svg.text(ax(y) + 2, Y + 18, str(y), size=11, color="muted")
LV = {-1: 104, -2: 58, 1: 210, 2: 256, 3: 302}
items = [
    ("2015-06-08", -1, "start", "blue", "GAE", "优势估计"),
    ("2017-07-20", -1, "start", "blue", "PPO", "裁剪目标"),
    ("2023-05-29", -1, "end", "blue", "DPO", "去掉奖励模型与 RL 循环"),
    ("2024-02-05", -2, "end", "green", "DeepSeekMath · GRPO", "去掉价值网络（第 8 课）"),
    ("2017-06-12", 1, "end", "ink", "Christiano 等", "从比较中学奖励（Atari）"),
    ("2019-09-18", 1, "middle", "ink", "Ziegler 等", "GPT-2 + PPO + KL 惩罚"),
    ("2020-09-02", 2, "end", "ink", "Stiennon 等", "摘要；四个同尺寸网络"),
    ("2022-03-04", 1, "middle", "ink", "InstructGPT", "三步法，到 175B"),
    ("2022-04-12", 2, "middle", "ink", "HH-RLHF", "KL 系数 0.001"),
    ("2024-01-05", 3, "end", "green", "DeepSeek LLM", "SFT + DPO（第 6 课）"),
]
for date, lv, anc, col, name, note in items:
    x = ax(year(date))
    y = LV[lv]
    ytop = y + (8 if lv < 0 else -30)
    svg.line(x, Y, x, ytop, color=col if col != "ink" else "muted", width=1.1, dash="2 2")
    svg.circle(x, Y, 4.6, fill=col)
    dx = {"start": -2, "middle": 0, "end": 2}[anc]
    svg.text(x + dx, y - 14, name, size=12.5, anchor=anc, color=col, weight="700")
    svg.text(x + dx, y + 2, "%s · %s" % (date[:7], note), size=11.5, anchor=anc, color="muted")
svg.text(ax(t0), 22, "上方：优化方法；下方：用它们训练的系统。蓝色是方法，绿色是 DeepSeek。", size=12, color="muted")
svg.save("l03-timeline.svg")
