"""Figure M10.12: one DeepSeek-V3 training step (2048 H800) against the time the same FLOPs need at peak.

Measured stage times (Insights into DeepSeek-V3, ISCA 2025, table 4, MPFT column; seconds):
  1F 1.13, bubble 2.06, 1B 1.99, 1W 0.48, 1F1B 13.95, opt 0.29; time/step 19.926.
  The paper defines 1F = forward time, 1B = input backward time, 1W = weight backward time;
  "bubble", "1F1B" and "opt" are not defined further.
Ideal time: 15360 sequences x 4096 tokens per step, 250.45 GFLOPs per token (lesson-12 notes),
spread over 2048 GPUs at 990 TFLOP/s (FlashMLA blog, H800 SXM5).
Writes math/figures/m10-v3-step.svg
"""

from m10_svg import SVG

stages = [("1F", 1.13, "blue", 0.6), ("bubble", 2.06, "red", 0.55), ("1B", 1.99, "blue", 0.6),
          ("1W", 0.48, "blue", 0.4), ("1F1B", 13.95, "violet", 0.55), ("opt", 0.29, "muted", 0.6)]
STEP = 19.926
tokens = 15360 * 4096
ideal = tokens * 250.45e9 / 2048 / 990e12
mfu = ideal / STEP

W, H = 740, 276
svg = SVG(W, H, "DeepSeek-V3 的一个训练步：实测各阶段耗时与同样运算量的峰值耗时")
X0, X1 = 40, 700
scale = (X1 - X0) / STEP
svg.text(X0, 26, "V3 在 2048 块 H800 上的一个训练步（{:,} 个 token）".format(tokens), size=14, weight="600")

y = 58
svg.text(X0, y - 8, "实测：每步 %.3f s（Insights 表 4）" % STEP, size=12, color="muted")
x = X0
for name, t, col, op in stages:
    w = t * scale
    svg.rect(x, y, w, 40, fill=col, fill_opacity=op, stroke="surface", width=1)
    if w > 36:
        svg.text(x + w / 2, y + 18, name, size=12, anchor="middle", weight="600")
        svg.text(x + w / 2, y + 33, "%.2f s" % t, size=11, anchor="middle")
    x += w
rest = STEP - sum(t for _, t, _, _ in stages)
# labels for the two thin segments under the bar
x1w = X0 + (1.13 + 2.06 + 1.99) * scale
svg.line(x1w + 0.48 * scale / 2, y + 40, x1w + 0.48 * scale / 2, y + 52, color="muted", width=1)
svg.text(x1w + 0.48 * scale / 2, y + 64, "1W 0.48", size=11, anchor="middle", color="muted")
xo = X0 + (STEP - rest - 0.29) * scale
svg.line(xo + 0.29 * scale / 2, y + 40, xo + 0.29 * scale / 2, y + 52, color="muted", width=1)
svg.text(xo + 0.29 * scale / 2, y + 64, "opt 0.29", size=11, anchor="end", color="muted")

y2 = 160
svg.text(X0, y2 - 8, "同样的运算量，每块卡都以 990 TFLOP/s 满速运行：%.2f s" % ideal, size=12, color="muted")
svg.rect(X0, y2, ideal * scale, 40, fill="green", fill_opacity=0.6)
svg.text(X0 + ideal * scale / 2, y2 + 25, "%.2f s" % ideal, size=13, anchor="middle", weight="600")
svg.text(X0 + ideal * scale + 10, y2 + 25, "÷ %.2f s = %.1f%%（MFU）" % (STEP, 100 * mfu), size=13, weight="600")

svg.text(X0, 238, "红色：流水线气泡 2.06 s，占一步的 %.1f%%。1F、1B、1W 按论文定义为前向、对输入的反向、对权重的反向；"
         % (100 * 2.06 / STEP), size=11, color="muted")
svg.text(X0, 256, "bubble、1F1B、opt 三行论文没有另行定义，名字照抄。各段之和 %.2f s，与每步 %.3f s 相差 %.3f s。"
         % (STEP - rest, STEP, rest), size=11, color="muted")

svg.save("m10-v3-step.svg")
print("ideal %.3f s, MFU %.4f, bubble share %.4f, sum %.2f" % (ideal, mfu, 2.06 / STEP, STEP - rest))
