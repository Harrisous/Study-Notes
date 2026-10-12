"""Figure 11.3: softmax (V2) vs sigmoid + normalisation (V3) gating on the same six logits.

Worked example of lesson 11 sec. 2 (constructed numbers, K = 2):
  z = [2.0, 1.5, 0.5, 0.0, -0.5, -1.0]
  V2:  s = softmax(z), g = s on the top-2 (no normalisation)      -> g = [0.4769, 0.2893], sum 0.7662
  V3:  s = sigmoid(z), g = s / sum(selected s)                    -> g = [0.5186, 0.4814], sum 1
Then raise the logit of the UNSELECTED expert 3 from 0.5 to 1.4 and recompute.
Writes deepseek/figures/l11-softmax-sigmoid.svg
"""

import numpy as np

from l11_svg import SVG

z = np.array([2.0, 1.5, 0.5, 0.0, -0.5, -1.0])
z2 = z.copy()
z2[2] = 1.4


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


s_sm, s_sg = softmax(z), sigmoid(z)
sel = [0, 1]
g_v2 = s_sm[sel]
g_v3 = s_sg[sel] / s_sg[sel].sum()
g_v2b = softmax(z2)[sel]
g_v3b = sigmoid(z2)[sel] / sigmoid(z2)[sel].sum()
assert np.allclose(np.round(g_v2, 4), [0.4769, 0.2893]) and np.allclose(np.round(g_v3, 4), [0.5186, 0.4814])
assert np.allclose(np.round(g_v2b, 4), [0.4128, 0.2504]) and np.allclose(g_v3b, g_v3)

W, H = 720, 470
svg = SVG(W, H, "同一组 logit：softmax 门控（V2）与 sigmoid + 归一化门控（V3）")

# ---------------------------------------------------------------- panel A
svg.text(16, 22, "Ⓐ 亲和度 s_{i}：6 个专家，logit z = [2.0, 1.5, 0.5, 0.0, −0.5, −1.0]", size=13, weight="600")
AX0, AY0, AY1 = 70, 44, 214          # plot box (y grows down)
AW = 620


def ay(v):
    return AY1 - v * (AY1 - AY0)


for k in range(0, 6):
    v = k * 0.2
    svg.line(AX0, ay(v), AX0 + AW, ay(v), color="grid", width=1)
    svg.text(AX0 - 8, ay(v) + 4, "%.1f" % v, size=11, anchor="end", color="muted")
svg.line(AX0, AY1, AX0 + AW, AY1, color="ink", width=1.2)
gw = AW / 6
bw = 30
for i in range(6):
    cx = AX0 + gw * (i + 0.5)
    for j, (val, col, fo) in enumerate(((s_sm[i], "muted", 0.55), (s_sg[i], "violet", 0.7))):
        x = cx - bw - 2 + j * (bw + 4)
        svg.rect(x, ay(val), bw, AY1 - ay(val), fill=col, fill_opacity=fo, rx=2)
        svg.text(x + bw / 2, ay(val) - 5, "%.3f" % val, size=11, anchor="middle", color=col if col != "muted" else "ink")
    lab = "专家 %d" % (i + 1) + ("（被选）" if i in sel else "")
    svg.text(cx, AY1 + 17, lab, size=12, anchor="middle", color="blue" if i in sel else "ink",
             weight="600" if i in sel else None)
lx = AX0 + AW - 250
svg.rect(lx, AY0 - 2, 12, 12, fill="muted", fill_opacity=0.55, rx=2)
svg.text(lx + 18, AY0 + 8, "softmax：6 个加起来 = 1", size=12)
svg.rect(lx, AY0 + 18, 12, 12, fill="violet", fill_opacity=0.7, rx=2)
svg.text(lx + 18, AY0 + 28, "sigmoid：各自独立，和 = %.2f" % s_sg.sum(), size=12)

# ---------------------------------------------------------------- panels B and C
def gate_panel(x0, y0, title, rows, note):
    svg.text(x0, y0, title, size=13, weight="600")
    BX = x0 + 100
    BW = 170
    for r, (lab, vals, col) in enumerate(rows):
        y = y0 + 20 + r * 58
        svg.text(x0, y + 15, lab, size=12, color=col if col != "muted" else "ink")
        acc = 0.0
        for k, v in enumerate(vals):
            svg.rect(BX + acc * BW, y + 2, v * BW, 20, fill=col, fill_opacity=0.75 - 0.3 * k,
                     stroke=col, width=0.8)
            svg.text(BX + (acc + v / 2) * BW, y + 16.5, "%.3f" % v, size=11, anchor="middle", color="ink")
            acc += v
        svg.text(BX + max(acc, 1.0) * BW + 8, y + 16.5, "和 %.3f" % acc, size=11, color="muted")
        svg.line(BX + BW, y - 2, BX + BW, y + 26, color="rule", width=1, dash="3 3")
    svg.text(x0, y0 + 20 + 2 * 58 + 4, note, size=12, color="muted")


gate_panel(16, 270, "Ⓑ 选 Top-2 后的门控值 g", [
    ("V2：g = s", list(g_v2), "muted"),
    ("V3：s 归一化", list(g_v3), "violet"),
], "虚线处 = 1。V2 的两个门控加起来只有 0.766")
gate_panel(376, 270, "Ⓒ 把没被选的专家 3 的 logit 调到 1.4", [
    ("V2：g = s", list(g_v2b), "muted"),
    ("V3：s 归一化", list(g_v3b), "violet"),
], "V2 的门控随之变小；V3 纹丝不动")
svg.line(360, 258, 360, 440, color="rule", width=1)
svg.text(16, 458, "图中每个数都由 z 直接算出；softmax 的分母含全部 6 个专家，sigmoid 只看自己的 logit。", size=12,
         color="muted")

svg.save("l11-softmax-sigmoid.svg")
