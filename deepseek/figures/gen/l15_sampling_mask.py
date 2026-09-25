"""Figure 15.10: Keep Sampling Mask on a five-token example (lesson-15-doc.md N26).

pi_old = [0.5, 0.3, 0.15, 0.04, 0.01] over tokens a..e, top-p = 0.95: nucleus {a, b, c} (cumulative 0.95),
truncated and renormalised sampling distribution [0.5263, 0.3158, 0.1579, 0, 0].
pi_theta = [0.45, 0.33, 0.12, 0.07, 0.03]; its own top-p nucleus is {a, b, c, d} (0.90 < 0.95 <= 0.97).
Ratio for token c: full pi_theta / truncated pi_old = 0.760; with the same mask and renormalisation 0.844.
Whether V3.2 renormalises pi_theta on the kept set is not stated in the paper.
Writes deepseek/figures/l15-sampling-mask.svg
"""

import numpy as np

from l15_svg import SVG

TOK = ["a", "b", "c", "d", "e"]
P_OLD = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
P_TH = np.array([0.45, 0.33, 0.12, 0.07, 0.03])
TOP_P = 0.95


def nucleus(p, top_p):
    order = np.argsort(-p)
    keep, cum = [], 0.0
    for i in order:
        keep.append(int(i))
        cum += p[i]
        if cum >= top_p - 1e-12:
            break
    return sorted(keep)


N_OLD, N_TH = nucleus(P_OLD, TOP_P), nucleus(P_TH, TOP_P)
assert N_OLD == [0, 1, 2] and N_TH == [0, 1, 2, 3]
mask = np.zeros(5)
mask[N_OLD] = 1
old_tr = P_OLD * mask / (P_OLD * mask).sum()
th_tr = P_TH * mask / (P_TH * mask).sum()
r_full = P_TH[2] / old_tr[2]
r_mask = th_tr[2] / old_tr[2]
assert abs(r_full - 0.760) < 1e-3 and abs(r_mask - 0.8444) < 1e-3
outside = 1 - P_TH[N_OLD].sum()
assert abs(outside - 0.10) < 1e-9

W, H = 760, 392
svg = SVG(W, H, "Keep Sampling Mask：训练时对 π_θ 施加采样时的截断掩码")

PY0, PY1 = 60, 250
VMAX = 0.6


def panel(x0, title, sub, vals, keep, col, labels, own=None, warn=()):
    w = 200
    svg.text(x0, 22, title, size=13, weight="700")
    svg.text(x0, 46, sub, size=12, color="muted")
    svg.line(x0, PY1, x0 + w, PY1, color="rule", width=1)
    cw = w / 5
    for i, v in enumerate(vals):
        x = x0 + i * cw + 6
        h = v / VMAX * (PY1 - PY0)
        on = i in keep
        svg.rect(x, PY1 - h, cw - 12, max(h, 0.8), fill=col if on else "red", fill_opacity=0.40 if on else 0.10,
                 stroke="red" if (i in warn or not on) else col, width=2 if i in warn else 1.2,
                 dash=None if on else "3 2")
        svg.text(x + (cw - 12) / 2, PY1 - h - 6, labels[i], size=11, anchor="middle", color="ink" if on else "red")
        svg.text(x + (cw - 12) / 2, PY1 + 16, TOK[i], size=12, anchor="middle", weight="700")
    if own is not None:
        # bracket for the policy's own nucleus
        xa, xb = x0 + 2, x0 + (max(own) + 1) * cw - 2
        svg.path("M%.1f %.1f v6 H%.1f v-6" % (xa, PY1 + 26, xb), stroke="muted", width=1.1)
        svg.text((xa + xb) / 2, PY1 + 42, "它自己的 top-p 核：{a, b, c, d}", size=11, anchor="middle", color="muted")


def f3(v):
    return "%.3f" % v if v else "0"


panel(16, "① 采样：π_{old}，top-p = 0.95", "核 = {a, b, c}（累计 0.95）",
      P_OLD, N_OLD, "blue", [f3(v) for v in P_OLD])
panel(270, "② 不保留掩码：π_{θ} 用全词表", "d、e 在采样分布里概率为 0",
      P_TH, [0, 1, 2, 3, 4], "blue", [f3(v) for v in P_TH], own=N_TH, warn=(3, 4))
panel(524, "③ Keep Sampling Mask", "对 π_{θ} 施加同一个核 {a, b, c}",
      P_TH * mask, N_OLD, "green", [f3(v) for v in P_TH * mask])

# ratios for token c
svg.rect(16, 308, 728, 78, fill="grid", fill_opacity=0.6, rx=6)
svg.text(28, 330, "token c 的重要性比，分母是截断后的 π_{old}(c) = 0.15 / 0.95 = %.4f：" % old_tr[2], size=12)
svg.text(28, 352, "② 分子用全词表的 π_{θ}(c)：0.12 / %.4f = %.3f" % (old_tr[2], r_full), size=12)
svg.text(28, 374, "③ 分子用同一掩码、在核内重新归一化的 π_{θ}(c)：(0.12 / 0.90) / %.4f = %.3f" % (old_tr[2], r_mask),
         size=12)
svg.text(560, 352, "π_{θ} 落在旧核之外的概率：0.10", size=12, color="red")

svg.save("l15-sampling-mask.svg")
