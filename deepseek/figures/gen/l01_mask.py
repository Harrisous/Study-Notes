"""Figure 1.10: the causal mask turns one forward pass into T honest predictions.

Scores: the illustrative 4x4 matrix of reference-docs/lesson-01-doc.md sec. 4.4 (not from a real
model), rows/columns labelled with the GPT-2 tokens of "The unbelievable tokenization".
Masked row-wise softmax is computed here and asserted against the recorded rows.
Writes deepseek/figures/l01-mask.svg
"""

import numpy as np

from l01_svg import SVG

S = np.array([[1, .5, -.3, 2], [.2, 1.1, .4, -1], [.3, 0, .9, .5], [1.2, -.4, .1, .8]])
M = np.triu(np.full((4, 4), -np.inf), 1)
Z = S + M
A = np.exp(Z - Z.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)
ref = [[1, 0, 0, 0], [0.2891, 0.7109, 0, 0], [0.2807, 0.2079, 0.5114, 0], [0.4535, 0.0916, 0.1510, 0.3040]]
assert np.allclose(A, ref, atol=6e-5)
row1 = np.exp(S[0]) / np.exp(S[0]).sum()          # what row 1 would do without the mask
print("unmasked row 1", np.round(row1, 3), "future share", round(row1[1:].sum(), 3))

toks = ["The", "Ġunbelievable", "Ġtoken", "ization"]
target = ["Ġunbelievable", "Ġtoken", "ization", "（下一个）"]

W, H = 760, 332
s = SVG(W, H, "因果掩码：一次前向传播给出 T 个只看过去的预测")
CW, CH = 52, 34
SX, AX, Y0 = 124, 404, 78


def fmt(v):
    return ("%.1f" % v).replace("-", "−")


s.text(10, 22, "打分表 S（示意数值）加上掩码 M，再逐行 softmax", size=13, weight="700")
for x0, title in ((SX, "S + M"), (AX, "A = softmax(S + M)")):
    s.text(x0 + 2 * CW, Y0 - 30, title, size=12.5, anchor="middle", weight="600")
    for j in range(4):
        s.text(x0 + j * CW + CW / 2, Y0 - 8, "j=%d" % (j + 1), size=11, anchor="middle", color="muted")
for i in range(4):
    s.text(SX - 10, Y0 + i * CH + 22, "t=%d  %s" % (i + 1, toks[i]), size=12, anchor="end",
           color="ink")
    for j in range(4):
        x, y = SX + j * CW, Y0 + i * CH
        if j > i:
            s.rect(x + 1, y + 1, CW - 2, CH - 2, fill="red", fill_opacity=0.16, stroke="red", width=0.8, rx=3)
            s.text(x + CW / 2, y + 22, "−∞", size=12, anchor="middle", color="red")
        else:
            s.rect(x + 1, y + 1, CW - 2, CH - 2, fill="violet", fill_opacity=0.10, rx=3)
            s.text(x + CW / 2, y + 22, fmt(S[i, j]), size=12, anchor="middle", color="ink")
        xa = AX + j * CW
        if j > i:
            s.rect(xa + 1, y + 1, CW - 2, CH - 2, fill="red", fill_opacity=0.06, rx=3)
            s.text(xa + CW / 2, y + 22, "0", size=12, anchor="middle", color="red")
        else:
            s.rect(xa + 1, y + 1, CW - 2, CH - 2, fill="blue", fill_opacity=0.08 + 0.6 * A[i, j], rx=3)
            s.text(xa + CW / 2, y + 22, "%.4f" % A[i, j] if A[i, j] < 1 else "1", size=11.5, anchor="middle",
                   color="ink")
    s.text(AX + 4 * CW + 14, Y0 + i * CH + 22, target[i], size=12, color="green")
s.arrow(SX + 4 * CW + 10, Y0 + 2 * CH, AX - 10, Y0 + 2 * CH, width=1.5)
s.text(AX + 4 * CW + 14, Y0 - 8, "该行要预测", size=11, color="muted")
# the gap between the tables is only 72 units wide, so the label is stacked on two lines
s.text((SX + 4 * CW + AX) / 2, Y0 + 2 * CH - 26, "逐行", size=12, anchor="middle", color="muted")
s.text((SX + 4 * CW + AX) / 2, Y0 + 2 * CH - 10, "softmax", size=12, anchor="middle", color="muted")

YB = Y0 + 4 * CH + 34
s.line(10, YB - 14, 750, YB - 14, color="rule", width=1)
lines = [
    ("一次前向传播，4 行给出 4 个预测；第 t 行只用到位置 1…t 的键和值，每行的权重和都是 1。", "ink"),
    ("第 1 行的目标 Ġunbelievable 就是位置 2 的输入。如果不加掩码，第 1 行的权重会是", "muted"),
    ("(%.3f, %.3f, %.3f, %.3f)：%.0f%% 的信息来自“未来”，其中 %.3f 直接来自答案本身。"
     % (row1[0], row1[1], row1[2], row1[3], 100 * row1[1:].sum(), row1[1]), "red"),
    ("掩码之后，这些格子的权重恰好为 0，训练时得到的 4 个预测与逐个生成时完全一样。", "muted"),
]
for k, (t, col) in enumerate(lines):
    s.text(10, YB + 8 + k * 22, t, size=12.5, color=col)

s.save("l01-mask.svg")
