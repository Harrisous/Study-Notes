"""Figure 8.6: PPO versus GRPO, redrawn after DeepSeekMath v3 Figure 4, plus a memory strip.

Top: PPO (policy and value model trained; reference and reward model frozen; KL added to the
per-token reward; GAE turns r and v into A). Bottom: GRPO (no value model; G outputs per
question; group computation turns r_1..r_G into A_1..A_G; KL added directly to the loss).
Memory strip: model-state bytes with 16 B per trained parameter and 2 B per frozen parameter
(ZeRO rule), N = 6,910,365,696 (DeepSeekMath 7B config) - the author's estimate, not a paper number.
Writes deepseek/figures/l08-ppo-vs-grpo.svg
"""

from l08_svg import SVG, box

N = 6_910_365_696
GB = 1e9
trained, frozen = 16 * N / GB, 2 * N / GB
ppo_total, grpo_total = 2 * trained + 2 * frozen, trained + 2 * frozen
print("16N = %.1f GB, 2N = %.2f GB, PPO %.1f GB, GRPO %.1f GB, saving %.1f%%"
      % (trained, frozen, ppo_total, grpo_total, 100 * (1 - grpo_total / ppo_total)))

W, H = 740, 600
svg = SVG(W, H, "PPO 与 GRPO 的结构对比")


def model(x, y, w, h, name, trained_flag, sub=None):
    col = "blue" if trained_flag else "muted"
    lines = [name] + ([sub] if sub else [])
    box(svg, x, y, w, h, lines, color=col, fill_opacity=0.14 if trained_flag else 0.10, size=13,
        width=1.8 if trained_flag else 1.2)


def token(x, y, s, col="ink", size=14):
    svg.text(x, y + 5, s, size=size, anchor="middle", color=col, weight="600")


# ---------------- PPO
svg.text(16, 26, "PPO", size=15, color="ink", weight="700")
svg.text(60, 26, "四个模型，其中两个要训练", size=12, color="muted")
cy = 118
token(30, cy, "q")
model(52, cy - 26, 104, 52, "策略模型", True)
svg.arrow(38, cy, 52, cy, shorten=1)
token(186, cy, "o")
svg.arrow(156, cy, 176, cy, shorten=1)
model(222, 44, 124, 42, "参考模型", False)
model(222, 97, 124, 42, "奖励模型", False)
model(222, 150, 124, 42, "价值模型", True)
for yy in (65, 118, 171):
    svg.path("M196 %d C206 %d 208 %d 222 %d" % (cy, cy, yy, yy), stroke="ink", width=1.4, arrow=True)
# KL circle-plus
kx, ky = 392, 92
svg.circle(kx, ky, 11, fill="surface", stroke="violet", width=1.6)
svg.line(kx - 6, ky, kx + 6, ky, color="violet", width=1.6)
svg.line(kx, ky - 6, kx, ky + 6, color="violet", width=1.6)
svg.arrow(346, 65, kx - 10, ky - 6, color="violet", width=1.4)
svg.text(356, 60, "KL", size=12, color="violet", weight="600")
svg.arrow(346, 118, kx - 10, ky + 6, color="green", width=1.4)
token(430, ky, "r", "green")
svg.arrow(kx + 11, ky, 422, ky, color="green", width=1.4)
token(430, 171, "v", "blue")
svg.arrow(346, 171, 422, 171, color="blue", width=1.4)
box(svg, 462, 104, 84, 54, ["GAE", "(8.9)"], color="ink", fill_opacity=0.05, size=13)
svg.arrow(438, ky + 2, 462, 122, width=1.4)
svg.arrow(438, 169, 462, 142, width=1.4)
token(576, 131, "A_{t}")
svg.arrow(546, 131, 562, 131, shorten=1)
svg.text(600, 104, "每个 token 一个优势，", size=12, color="muted")
svg.text(600, 122, "靠价值模型逐步估出", size=12, color="muted")
svg.text(600, 150, "KL 作为每 token 的", size=12, color="violet")
svg.text(600, 168, "惩罚加进奖励 r", size=12, color="violet")

svg.line(16, 212, W - 16, 212, color="rule", width=1)

# ---------------- GRPO
svg.text(16, 240, "GRPO", size=15, color="ink", weight="700")
svg.text(70, 240, "去掉价值模型；每道题采 G 个回答", size=12, color="muted")
cy = 338
token(30, cy, "q")
model(52, cy - 26, 104, 52, "策略模型", True)
svg.arrow(38, cy, 52, cy, shorten=1)
outs = [("o₁", 290), ("o₂", 318), ("⋮", 350), ("o_{G}", 384)]
for s, yy in outs:
    token(186, yy, s)
svg.path("M156 %d L170 %d" % (cy, cy), stroke="ink", width=1.4)
for yy in (290, 318, 384):
    svg.path("M170 %d C172 %d 172 %d 176 %d" % (cy, cy, yy, yy), stroke="ink", width=1.2)
model(222, 262, 124, 42, "参考模型", False)
model(222, 317, 124, 42, "奖励模型", False)
svg.path("M200 282 L206 282 L206 392 L200 392", stroke="ink", width=1.3)
svg.arrow(206, 283, 222, 283, width=1.3, shorten=1)
svg.arrow(206, 338, 222, 338, width=1.3, shorten=1)
rs = [("r₁", 300), ("r₂", 326), ("⋮", 350), ("r_{G}", 378)]
for s, yy in rs:
    token(376, yy, s, "green")
svg.arrow(346, 338, 362, 338, color="green", width=1.4)
box(svg, 402, 306, 130, 64, ["组计算", "(r − 均值) / 标准差"], color="ink", fill_opacity=0.05, size=13)
svg.arrow(390, 338, 402, 338, width=1.4)
As = [("Â₁", 300), ("Â₂", 326), ("⋮", 350), ("Â_{G}", 378)]
for s, yy in As:
    token(556, yy, s)
svg.arrow(532, 338, 543, 338, width=1.4)
svg.text(582, 318, "每个回答一个优势，", size=12, color="muted")
svg.text(582, 336, "所有 token 共用", size=12, color="muted")
# KL straight into the loss
box(svg, 402, 250, 104, 40, ["目标 J_{GRPO}"], color="violet", fill_opacity=0.08, size=13)
svg.arrow(346, 276, 402, 270, color="violet", width=1.4)
svg.text(356, 262, "KL", size=12, color="violet", weight="600")
svg.text(520, 266, "KL 直接作为损失里的一项，", size=12, color="violet")
svg.text(520, 284, "不进优势的计算", size=12, color="violet")

# legend
ly = 420
svg.rect(16, ly - 11, 26, 16, fill="blue", fill_opacity=0.14, stroke="blue", width=1.8, rx=4)
svg.text(48, ly + 2, "训练中的模型", size=12)
svg.rect(150, ly - 11, 26, 16, fill="muted", fill_opacity=0.10, stroke="muted", width=1.2, rx=4)
svg.text(182, ly + 2, "冻结的模型", size=12)

svg.line(16, 440, W - 16, 440, color="rule", width=1)

# ---------------- memory strip
svg.text(16, 466, "模型状态显存（笔者估算：训练 16 字节/参数，冻结 2 字节/参数，N = 6.91B，不含激活与 KV cache）",
         size=12, color="ink", weight="600")
X0, X1 = 70, 600
scale = (X1 - X0) / ppo_total
segs_ppo = [("策略", trained, "blue", 0.75), ("价值", trained, "blue", 0.35),
            ("参考", frozen, "muted", 0.6), ("奖励", frozen, "muted", 0.35)]
segs_grpo = [("策略", trained, "blue", 0.75), ("参考", frozen, "muted", 0.6), ("奖励", frozen, "muted", 0.35)]
for row, (lab, segs, tot) in enumerate((("PPO", segs_ppo, ppo_total), ("GRPO", segs_grpo, grpo_total))):
    y = 486 + row * 46
    svg.text(16, y + 19, lab, size=13, weight="700")
    x = X0
    for name, v, col, op in segs:
        w = v * scale
        svg.rect(x, y, w, 28, fill=col, fill_opacity=op, stroke="surface", width=1)
        if w > 60:
            svg.text(x + w / 2, y + 19, "%s %.1f" % (name, v), size=12, anchor="middle",
                     color="surface" if op >= 0.6 else "ink", weight="600")
        x += w
    svg.text(x + 8, y + 19, "%.1f GB" % tot, size=13, weight="700")
svg.text(X0, 588, "参考、奖励模型各 13.8 GB（条太窄，未标字）。去掉价值模型：248.8 → 138.2 GB，少 44.4%。",
         size=12, color="muted")

svg.save("l08-ppo-vs-grpo.svg")
