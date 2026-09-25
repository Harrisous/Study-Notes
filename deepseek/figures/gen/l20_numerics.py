"""Figure 20.4 (summary of thread C): (a) bits per stored number, scale factors amortised, for the
formats the course met; (b) H800 GPU hours per 10^12 training tokens and MFU for 67B, V2, V3.

(a) from l20_num.BITS (lesson 12 p3, lesson 14 p7, lesson 18 p7, lesson 19 p5, lesson 9 p8);
(b) from l20_num.GPUH_PER_T (lesson 10 p5: 300.6K and 172.8K, MFU ~38% / ~20% are the lesson's own
6N estimates; lesson 12 p1: 180K and 38.94% from the Insights paper, BF16 peak).
Writes deepseek/figures/l20-numerics.svg
"""

from l20_svg import SVG
from l20_num import BITS, GPUH_PER_T

W, H = 760, 566
svg = SVG(W, H, "主线 C 的总账：每个数用几位存，以及每万亿 token 的 GPU 小时")
X0, X1 = 330, 670


def xb(v):
    return X0 + v / 16 * (X1 - X0)


svg.text(10, 22, "(a) 每个数的有效位数（含摊到每个数上的缩放因子）", size=13, weight="700")
TOP, RH = 56, 30
for v in (0, 4, 8, 12, 16):
    svg.line(xb(v), TOP - 8, xb(v), TOP + len(BITS) * RH - 4, color="grid", width=1)
    svg.text(xb(v), TOP - 14, "%d 位" % v, size=11, anchor="middle", color="muted")
for i, (name, bits, where) in enumerate(BITS):
    y = TOP + i * RH
    col = "muted" if bits >= 16 else ("orange" if bits <= 6.01 else "amber")
    svg.text(X0 - 10, y + 15, name, size=12, anchor="end")
    svg.rect(X0, y + 4, xb(bits) - X0, 16, fill=col, fill_opacity=0.55, stroke=col, width=0.8)
    svg.text(xb(bits) + 5, y + 16, ("%.3f" % bits).rstrip("0").rstrip(".") + " · " + where, size=11, color="muted")

# (b)
Y2 = TOP + len(BITS) * RH + 30
svg.text(10, Y2, "(b) 每训练 10^{12} 个 token 的 H800 GPU 小时，与模型算力利用率 MFU", size=13, weight="700")
XB0, XB1 = 250, 540


def xg(v):
    return XB0 + v / 320 * (XB1 - XB0)


T2 = Y2 + 32
for v in (0, 100, 200, 300):
    svg.line(xg(v), T2 - 8, xg(v), T2 + 3 * 40 - 8, color="grid", width=1)
    svg.text(xg(v), T2 - 12, "%dK" % v, size=11, anchor="middle", color="muted")
svg.text(680, T2 - 12, "MFU", size=12, anchor="middle", weight="700")
LABEL = {"DeepSeek LLM 67B": "67B（稠密，激活 67.4B）", "DeepSeek-V2": "V2（MoE，激活 21B）",
         "DeepSeek-V3": "V3（MoE + FP8，激活 37B）"}
NOTE = {"DeepSeek LLM 67B": "约 38%（估算）", "DeepSeek-V2": "约 20%（估算）", "DeepSeek-V3": "38.9%（Insights）"}
for i, (name, h, act, mfu) in enumerate(GPUH_PER_T):
    y = T2 + i * 40
    col = "blue" if name.endswith("V3") else "muted"
    svg.text(XB0 - 10, y + 15, LABEL[name], size=12, anchor="end")
    svg.rect(XB0, y + 3, xg(h) - XB0, 18, fill=col, fill_opacity=0.5, stroke=col, width=0.8)
    svg.text(xg(h) + 6, y + 16, "%sK" % ("%.1f" % h).rstrip("0").rstrip("."), size=12, weight="700")
    svg.text(680, y + 16, NOTE[name], size=12, anchor="middle", color="blue" if col == "blue" else "muted")

FY = T2 + 3 * 40 + 14
svg.text(10, FY, "(a) 橙色：6 位及以下；黄色：约 8–9 位；灰色：BF16。缓存行里 RoPE 维若用 BF16 存，会把整行的平均位数拉高。", size=12,
         color="muted")
svg.text(10, FY + 18, "(b) 三行不是同一种模型：67B 稠密、V2 与 V3 是 MoE；MFU 都按 BF16 峰值计，前两个是课里按 6N 的估算。", size=12,
         color="muted")

svg.save("l20-numerics.svg")
