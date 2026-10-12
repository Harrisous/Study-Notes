"""Figure 8.13: the unified paradigm (DeepSeekMath v3 Table 10) with a worked example.

Rows: SFT, RFT, Online RFT, DPO, PPO, GRPO. Columns: data source (offline = frozen SFT model,
online = current policy), reward function (Table 10: '-', Rule, Model), gradient coefficient
(derived in the lesson), and the coefficients the method gives to one group of four sampled answers
with rewards (1, 0, 0, 1) (research notes X8). DPO uses the original sequence-level loss with
beta = 0.1 at initialisation (implicit rewards equal): weight sigma(0) = 0.5, so +0.05 / -0.05.
GRPO uses population std and ignores the small KL term.
Writes deepseek/figures/l08-unified.svg
"""

import numpy as np

from l08_svg import SVG

r = np.array([1, 0, 0, 1], dtype=float)
grpo = (r - r.mean()) / r.std()
beta_dpo = 0.1
dpo_w = beta_dpo * 1 / (1 + np.exp(-0.0))
print("GRPO GC", grpo, "DPO weight", dpo_w)

rows = [
    ("SFT", ("人工标注数据", "offline"), "—", "1", None, "只用人工写的解答"),
    ("RFT", ("SFT 模型采样", "offline"), "规则", "𝕀(o)：对 1、错 0", [1, 0, 0, 1], None),
    ("Online RFT", ("当前策略采样", "online"), "规则", "𝕀(o)", [1, 0, 0, 1], None),
    ("DPO", ("SFT 模型采样（成对）", "offline"), "规则", "±β·σ(r̂_{负} − r̂_{正})", [dpo_w, -dpo_w, None, None],
     None),
    ("PPO", ("当前策略采样", "online"), "模型", "A_{t}（GAE + 价值网络）", None, "每个 token 各不相同"),
    ("GRPO", ("当前策略采样（每题 G 个）", "online"), "模型", "Â_{i,t} + β(w − 1)", list(grpo), None),
]

W, H = 740, 440
svg = SVG(W, H, "统一范式：六种方法的数据源、奖励函数与梯度系数")
cols = [16, 118, 290, 346, 530]      # method, data, reward, GC, example
hy = 30
for x, h in zip(cols, ("方法", "数据源", "奖励", "梯度系数 GC", "例：四个回答，奖励 (1, 0, 0, 1)")):
    svg.text(x, hy, h, size=12.5, weight="700", color="ink")
svg.line(16, hy + 10, W - 16, hy + 10, color="ink", width=1.2)
RH = 52
for k, (name, (src, kind), rew, gc, ex, note) in enumerate(rows):
    y = hy + 22 + k * RH
    if k % 2 == 0:
        svg.rect(16, y - 6, W - 32, RH, fill="grid", fill_opacity=0.5)
    svg.text(cols[0], y + 22, name, size=14, weight="700", color="blue" if name == "GRPO" else "ink")
    col = "amber" if kind == "offline" else "blue"
    svg.rect(cols[1], y + 1, 160, 36, fill=col, fill_opacity=0.12, stroke=col, width=1, rx=5)
    svg.text(cols[1] + 80, y + 16, src, size=11.5, anchor="middle", color="ink")
    svg.text(cols[1] + 80, y + 31, "离线" if kind == "offline" else "在线", size=11, anchor="middle", color=col)
    svg.text(cols[2], y + 22, rew, size=12.5, color="ink")
    svg.text(cols[3], y + 22, gc, size=12.5, color="ink")
    if ex is not None:
        for j, v in enumerate(ex):
            x = cols[4] + j * 48
            if v is None:
                svg.rect(x, y + 6, 42, 26, stroke="rule", width=1, rx=4, dash="3 3")
                svg.text(x + 21, y + 24, "—", size=12, anchor="middle", color="muted")
                continue
            c = "green" if v > 0 else ("red" if v < 0 else "muted")
            svg.rect(x, y + 6, 42, 26, fill=c, fill_opacity=0.18 if v else 0.08, stroke=c, width=1, rx=4)
            lab = ("%+.2f" % v if abs(v) < 0.5 else "%+.0f" % v) if v else "0"
            svg.text(x + 21, y + 24, lab.replace("-", "−"), size=12, anchor="middle", color="ink", weight="600")
    else:
        svg.text(cols[4], y + 22, note, size=12, color="muted")

yb = hy + 22 + len(rows) * RH + 14
svg.text(16, yb, "DPO：一对（第 1 个正确 vs 第 2 个错误），β = 0.1，训练开始时隐式奖励相等，σ(0) = 0.5。", size=12, color="muted")
svg.text(16, yb + 18, "GRPO：总体标准差，忽略很小的 KL 项。RFT / Online RFT 不惩罚错误回答（系数为 0），GRPO 会（系数为负）。",
         size=12, color="muted")
svg.text(16, yb + 36, "黄框 = 离线数据（来自冻结的 SFT 模型），蓝框 = 在线数据（来自正在训练的策略）。", size=12, color="muted")
svg.save("l08-unified.svg")
