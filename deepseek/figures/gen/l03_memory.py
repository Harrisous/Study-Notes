"""Figure 3.12: model-state memory of SFT, DPO, GRPO-style RL and PPO-RLHF (bookkeeping estimate).

Assumptions (stated in the lesson): a trained network costs 16 bytes/parameter (mixed-precision Adam:
fp16 weights 2 + fp16 grads 2 + fp32 master weights, momentum, variance 12; ZeRO paper sec. 3.1);
a frozen network stores 16-bit weights only, 2 bytes/parameter. Activations, KV cache of the rollouts
and temporary buffers are NOT counted. GB = 1e9 bytes.
Left : every network 7B.   Right: InstructGPT's 175B policy with a 6B RM and a 6B value function.
Writes deepseek/figures/l03-memory.svg
"""
from l03_svg import SVG, Axis

TR, FR = 16, 2          # bytes per parameter: trained / frozen
N = 7                   # billions of parameters
cols = [
    ("SFT", [("policy", TR * N)]),
    ("DPO", [("policy", TR * N), ("ref", FR * N)]),
    ("GRPO 式", [("policy", TR * N), ("ref", FR * N), ("rm", FR * N)]),
    ("PPO", [("policy", TR * N), ("value", TR * N), ("ref", FR * N), ("rm", FR * N)]),
]
tot = [sum(v for _, v in parts) for _, parts in cols]
assert tot == [112, 126, 140, 252]
ig = [("policy", TR * 175), ("ref", FR * 175), ("value", TR * 6), ("rm", FR * 6)]
assert sum(v for _, v in ig) == 3258
COL = {"policy": ("blue", "策略（训练）"), "value": ("violet", "价值网络（训练）"),
       "ref": ("muted", "参考模型（冻结）"), "rm": ("green", "奖励模型（冻结）")}

W, H = 740, 390
svg = SVG(W, H, "四种后训练方法的模型状态显存（7B）与 InstructGPT 175B 配置")
# ---------------- left: stacked bars, 7B
ax_y = Axis(0, 280, 300, 50)
svg.text(16, 24, "每个网络都是 7B（单位 GB）", size=13.5, weight="600")
for v in (0, 50, 100, 150, 200, 250):
    svg.line(60, ax_y(v), 420, ax_y(v), color="grid", width=1)
    svg.text(54, ax_y(v) + 4, str(v), size=11, anchor="end", color="muted")
svg.line(60, ax_y(0), 420, ax_y(0), color="muted", width=1.1)
bw = 58
for i, (name, parts) in enumerate(cols):
    x = 84 + i * 86
    y = ax_y(0)
    for key, v in parts:
        col, _ = COL[key]
        h = ax_y(0) - ax_y(v)
        svg.rect(x, y - h, bw, h, fill=col, fill_opacity=0.30 if key in ("policy", "value") else 0.18,
                 stroke=col, width=1.1)
        if h > 16:
            svg.text(x + bw / 2, y - h / 2 + 4, str(v), size=11.5, anchor="middle")
        y -= h
    svg.text(x + bw / 2, y - 6, "%d" % tot[i], size=12.5, anchor="middle", weight="700")
    svg.text(x + bw / 2, ax_y(0) + 17, name, size=12.5, anchor="middle", weight="600")
    ratio = ("%.3f" % (tot[i] / tot[0])).rstrip("0").rstrip(".")   # 1, 1.125, 1.25, 2.25 (exact)
    svg.text(x + bw / 2, ax_y(0) + 33, ratio + "×", size=11, anchor="middle", color="muted")
# legend
for k, key in enumerate(("policy", "value", "ref", "rm")):
    col, lab = COL[key]
    lx = 16 + k * 122
    svg.rect(lx, 352, 14, 12, fill=col, fill_opacity=0.3, stroke=col, width=1)
    svg.text(lx + 19, 363, lab, size=11.5)
svg.text(16, 384, "训练的网络按 16 字节/参数，冻结的按 2 字节/参数；不含激活值和采样用的 KV cache。", size=11.5, color="muted")

# ---------------- right: InstructGPT 175B
x0, x1 = 470, 724
svg.text(460, 24, "InstructGPT，175B 策略（单位 GB）", size=13.5, weight="600")
total = 3258.0
y = 70
cx = x0
for key, v in ig:
    col, lab = COL[key]
    w = (x1 - x0) * v / total
    svg.rect(cx, y, w, 34, fill=col, fill_opacity=0.30 if key in ("policy", "value") else 0.18, stroke=col, width=1.1)
    cx += w
rows = [("策略 175B × 16", 2800, "blue"), ("参考 175B × 2", 350, "muted"), ("价值 6B × 16", 96, "violet"), ("奖励 6B × 2", 12, "green")]
for k, (lab, v, col) in enumerate(rows):
    yy = 140 + k * 24
    svg.rect(x0, yy - 10, 12, 12, fill=col, fill_opacity=0.3, stroke=col, width=1)
    svg.text(x0 + 18, yy, lab, size=12)
    svg.text(x1, yy, "%s（%.1f%%）" % (format(v, ","), 100 * v / total), size=12, anchor="end")
svg.line(x0, 236, x1, 236, color="rule", width=1)
svg.text(x0, 256, "合计 3,258 GB", size=12.5, weight="700")
svg.text(x0, 282, "RM 与价值网络固定为 6B，", size=12, color="muted")
svg.text(x0, 300, "对 175B 策略几乎不占地方；", size=12, color="muted")
svg.text(x0, 318, "四个网络同尺寸时，价值网络", size=12, color="muted")
svg.text(x0, 336, "和策略一样贵（左图 PPO）。", size=12, color="muted")
svg.save("l03-memory.svg")
