"""Figure 7.11: the expert-level and device-level balance losses computed on a 4-token batch.

N_r = 8 routed experts, K_r = 2, T = 4 tokens (research notes R13). Router logits per token:
  (2, 1.5, 0.5, 0, 0, 0, 0, -0.5), (2, 0.2, 1.8, 0, 0, 0, -0.3, 0),
  (1.6, 1.4, 0, 0.2, 0, 0, 0, 0),  (0, 0, 0.1, 1.2, 1.1, 0.3, 0, 0).
softmax -> s_{i,t}; top-2 -> counts; f_i = N_r/(K_r T) * count; P_i = mean_t s_{i,t};
L_ExpBal / alpha_1 = sum f_i P_i = 1.5736. Devices: experts 1-4 and 5-8:
f' = (1.75, 0.25), P' = (0.7121, 0.2879), L_DevBal / alpha_2 = 1.3181. All computed here.
Writes deepseek/figures/l07-balance-ledger.svg
"""

import math

from l07_svg import SVG, fmt

Z = [[2, 1.5, 0.5, 0, 0, 0, 0, -0.5],
     [2, 0.2, 1.8, 0, 0, 0, -0.3, 0],
     [1.6, 1.4, 0, 0.2, 0, 0, 0, 0],
     [0, 0, 0.1, 1.2, 1.1, 0.3, 0, 0]]
Nr, Kr, T = 8, 2, 4
S = []
for row in Z:
    e = [math.exp(v) for v in row]
    tot = sum(e)
    S.append([v / tot for v in e])
top = [sorted(range(Nr), key=lambda i: -s[i])[:Kr] for s in S]
cnt = [sum(1 for t in range(T) if i in top[t]) for i in range(Nr)]
f = [Nr / (Kr * T) * c for c in cnt]
P = [sum(S[t][i] for t in range(T)) / T for i in range(Nr)]
L = sum(a * b for a, b in zip(f, P))
fd = [sum(f[:4]) / 4, sum(f[4:]) / 4]
Pd = [sum(P[:4]), sum(P[4:])]
Ld = fd[0] * Pd[0] + fd[1] * Pd[1]
assert abs(L - 1.5736) < 1e-4 and abs(Ld - 1.3181) < 1e-4 and sum(f) == Nr

W, H = 740, 500
svg = SVG(W, H, "专家级与设备级均衡损失的逐项计算（4 个 token，8 个路由专家，每个 token 选 2 个）")

CX0, CW, CH = 118, 58, 30          # heatmap cell geometry
Y0 = 50
svg.text(16, 22, "① 亲和度 s_{i,t}（每行是一个 token 对 8 个路由专家的 softmax；粗框 = 该 token 的 top-2）", size=13,
         weight="600")
for i in range(Nr):
    svg.text(CX0 + i * CW + CW / 2, Y0 - 8, "专家 %d" % (i + 1), size=11, anchor="middle", color="muted")
for t in range(T):
    y = Y0 + t * CH
    svg.text(CX0 - 10, y + 20, "token %d" % (t + 1), size=12, anchor="end")
    for i in range(Nr):
        x = CX0 + i * CW
        on = i in top[t]
        svg.rect(x + 1, y + 1, CW - 2, CH - 2, fill="blue", fill_opacity=min(0.85, 0.08 + 1.6 * S[t][i]),
                 stroke="ink" if on else None, width=2 if on else 1, rx=3)
        svg.text(x + CW / 2, y + 20, "%.3f" % S[t][i], size=12, anchor="middle",
                 color="surface" if S[t][i] > 0.28 else "ink", weight="600" if on else None)
    gs = sum(S[t][i] for i in top[t])
    svg.text(CX0 + Nr * CW + 10, y + 20, "门控和 %.2f" % gs, size=12, color="muted")

# ledger rows
ly = Y0 + T * CH + 34
svg.text(16, ly - 8, "② 逐列记账", size=13, weight="600")
rows = [
    ("被选次数", [str(c) for c in cnt], "ink"),
    ("f_{i}", ["%g" % v for v in f], "ink"),
    ("P_{i}", ["%.4f" % v for v in P], "violet"),
    ("f_{i}·P_{i}", ["%.4f" % (a * b) if a else "0" for a, b in zip(f, P)], "ink"),
]
for r, (name, vals, col) in enumerate(rows):
    y = ly + 8 + r * 24
    svg.text(CX0 - 10, y + 12, name, size=12, anchor="end", color=col, weight="600")
    for i, v in enumerate(vals):
        svg.text(CX0 + i * CW + CW / 2, y + 12, v, size=12, anchor="middle", color=col)
svg.line(CX0, ly + 6, CX0 + Nr * CW, ly + 6, color="rule", width=1)
svg.text(CX0 + Nr * CW + 10, ly + 20, "f_{i} = 8/(2×4)×次数", size=11, color="muted")
svg.text(CX0 + Nr * CW + 10, ly + 44, "Σf_{i} = %g = N_{r}" % sum(f), size=11, color="muted")
svg.text(CX0 + Nr * CW + 10, ly + 68, "P_{i} = 各列平均", size=11, color="muted")
res_y = ly + 8 + 4 * 24 + 16
svg.text(CX0, res_y, "L_{ExpBal} / α_{1} = Σ f_{i}P_{i} = %.4f" % L, size=14, color="ink", weight="600")
svg.text(CX0 + 300, res_y, "（完全均匀时 = 1，全部坍缩时 = N_{r}/K_{r} = 4）", size=12, color="muted")

# device-level
dy = res_y + 36
svg.text(16, dy, "③ 设备级：专家 1–4 放在设备 A，5–8 放在设备 B", size=13, weight="600")
by = dy + 12
for k, (lab, x0) in enumerate((("设备 A", CX0), ("设备 B", CX0 + 4 * CW))):
    svg.path("M%.1f %.1f L%.1f %.1f L%.1f %.1f L%.1f %.1f" % (x0 + 4, by, x0 + 4, by + 8, x0 + 4 * CW - 4, by + 8,
                                                          x0 + 4 * CW - 4, by), stroke="ink", width=1.3)
    xc = x0 + 2 * CW
    svg.text(xc, by + 28, lab, size=13, anchor="middle", weight="600")
    svg.text(xc, by + 48, "f′ = 组内 f 的平均 = %g" % fd[k], size=12, anchor="middle", color="ink")
    svg.text(xc, by + 68, "P′ = 组内 P 之和 = %.4f" % Pd[k], size=12, anchor="middle", color="violet")
svg.text(CX0, by + 98, "L_{DevBal} / α_{2} = %.2f×%.4f + %.2f×%.4f = %.4f" % (fd[0], Pd[0], fd[1], Pd[1], Ld), size=14,
         weight="600")
svg.text(CX0, by + 124, "（两台设备完全均衡时 = 1）", size=12, color="muted")

svg.save("l07-balance-ledger.svg")
print("f", f, "P", [round(v, 4) for v in P], "L", round(L, 4), "fd", fd, "Pd", [round(v, 4) for v in Pd], round(Ld, 4))
