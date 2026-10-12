"""Figure 13.6: PPO versus GRPO (redrawn after R1 paper arXiv v2 Fig. 3 / Nature Extended Data Fig. 2).

PPO: policy -> output -> reference (KL), reward (r), value (v) -> r - beta*KL per token -> GAE -> A_t.
GRPO: policy -> G outputs -> reference and reward models -> r_1..r_G -> group computation -> A_1..A_G,
the KL term goes straight into the loss. Trained models blue, frozen models muted.
Writes deepseek/figures/l13-ppo-vs-grpo.svg
"""

from l13_svg import SVG

W, H = 760, 432
svg = SVG(W, H, "PPO 与 GRPO 的结构对比：GRPO 去掉价值模型，用一组回答的奖励统计量代替它")


def model(x, y, w, h, name, trained):
    col = "blue" if trained else "muted"
    svg.rect(x, y, w, h, fill=col, fill_opacity=0.16 if trained else 0.10, stroke=col, width=1.4, rx=8)
    svg.text(x + w / 2, y + h / 2 + 5, name, size=13, anchor="middle", color="ink", weight="600")


def small(x, y, w, h, label, col="ink"):
    svg.rect(x, y, w, h, fill="surface", stroke=col, width=1.2, rx=5)
    svg.text(x + w / 2, y + h / 2 + 4.5, label, size=12.5, anchor="middle", color=col)


# ------------------------------------------------------------------ PPO
svg.text(16, 24, "PPO", size=15, weight="700")
svg.text(56, 24, "（第 3 课）四个模型，其中两个要训练", size=12, color="muted")
cy = 110
small(16, cy - 16, 34, 32, "q")
svg.arrow(52, cy, 70, cy, width=1.4)
model(72, cy - 24, 100, 48, "策略模型", True)
svg.arrow(174, cy, 190, cy, width=1.4)
small(192, cy - 16, 30, 32, "o")
ys = [44, 94, 144]
names = [("参考模型", False, "KL", "red"), ("奖励模型", False, "r", "green"), ("价值模型", True, "v", "blue")]
for (nm, tr, out, oc), y in zip(names, ys):
    svg.arrow(224, cy, 250, y + 16, width=1.2)
    model(252, y, 118, 32, nm, tr)
    svg.arrow(372, y + 16, 392, y + 16, width=1.2)
    svg.text(400, y + 21, out, size=13, color=oc, weight="700")
svg.text(311, 196, "与策略同规模，要训练", size=12, anchor="middle", color="blue")
# combine r and KL
svg.circle(452, 85, 13, fill="surface", stroke="ink", width=1.4)
svg.text(452, 90, "⊕", size=14, anchor="middle")
svg.arrow(418, 60, 440, 76, width=1.2, color="red")
svg.arrow(414, 110, 440, 94, width=1.2, color="green")
svg.text(472, 64, "r − β·KL：KL 逐 token 进奖励", size=12, color="ink")
svg.arrow(466, 85, 494, 106, width=1.2)
svg.path("M414 160 H527 V128", stroke="blue", width=1.2, arrow=True)
small(496, 94, 62, 32, "GAE")
svg.arrow(560, 110, 580, 110, width=1.4)
small(582, 94, 42, 32, "A_{t}", col="ink")
svg.text(636, 104, "每个 token", size=12, color="ink")
svg.text(636, 121, "各有一个优势", size=12, color="ink")
svg.text(636, 138, "（依赖 λ）", size=12, color="muted")

svg.line(16, 214, 744, 214, color="rule", width=1, dash="4 4")

# ------------------------------------------------------------------ GRPO
svg.text(16, 240, "GRPO", size=15, weight="700")
svg.text(66, 240, "（第 8 课）去掉价值模型，只有策略要训练", size=12, color="muted")
cy = 318
small(16, cy - 16, 34, 32, "q")
svg.arrow(52, cy, 70, cy, width=1.4)
model(72, cy - 24, 100, 48, "策略模型", True)
outs = [cy - 52, cy - 16, cy + 36]
labels = ["o_{1}", "o_{2}", "o_{G}"]
for y, lab in zip(outs, labels):
    svg.arrow(174, cy, 190, y + 14, width=1.1)
    small(192, y, 36, 28, lab)
svg.text(210, cy + 28, "⋮", size=13, anchor="middle")
model(262, cy - 58, 108, 32, "参考模型", False)
model(262, cy + 8, 108, 32, "奖励模型", False)
svg.arrow(230, cy - 2, 260, cy - 42, width=1.1)
svg.arrow(230, cy + 2, 260, cy + 24, width=1.1)
# KL goes to the loss
svg.path("M372 %d H420" % (cy - 42), stroke="red", width=1.2, dash="4 3", arrow=True)
svg.text(426, cy - 38, "KL 估计直接加进损失（不进奖励）", size=12, color="red")
# rewards
svg.arrow(372, cy + 24, 392, cy + 24, width=1.2)
for k, lab in enumerate(["r_{1}", "r_{2}", "⋮", "r_{G}"]):
    y = cy - 26 + k * 25
    if lab == "⋮":
        svg.text(412, y + 12, lab, size=13, anchor="middle", color="green")
    else:
        small(396, y - 4, 34, 20, lab, col="green")
svg.arrow(432, cy + 18, 452, cy + 18, width=1.2)
svg.rect(454, cy - 8, 108, 52, fill="amber", fill_opacity=0.10, stroke="amber", width=1.3, rx=8)
svg.text(508, cy + 14, "组内计算", size=13, anchor="middle", weight="600")
svg.text(508, cy + 32, "(r − 均值)/标准差", size=12, anchor="middle")
svg.arrow(564, cy + 18, 584, cy + 18, width=1.2)
for k, lab in enumerate(["Â_{1}", "Â_{2}", "⋮", "Â_{G}"]):
    y = cy - 26 + k * 25
    if lab == "⋮":
        svg.text(604, y + 12, lab, size=13, anchor="middle")
    else:
        small(588, y - 4, 34, 20, lab)
svg.text(636, cy + 4, "同一回答的", size=12)
svg.text(636, cy + 21, "所有 token", size=12)
svg.text(636, cy + 38, "共用一个 Â_{i}", size=12)

# legend
svg.rect(16, 408, 18, 14, fill="blue", fill_opacity=0.16, stroke="blue", width=1.2, rx=3)
svg.text(40, 420, "训练中的模型", size=12)
svg.rect(136, 408, 18, 14, fill="muted", fill_opacity=0.10, stroke="muted", width=1.2, rx=3)
svg.text(160, 420, "冻结的模型（R1-Zero 的“奖励模型”就是规则检查器）", size=12)
svg.save("l13-ppo-vs-grpo.svg")
