"""Figure 15.8: off-policy sequence masking on a worked example.

Four responses to one prompt (G = 4, rewards [1, 0, 0, 1], advantages r - mean = [+0.5, -0.5, -0.5, +0.5]).
Every response has six tokens; the sampling probabilities pi_old are the same for all four, the current
policy pi_theta differs (lesson-15-doc.md N24, N25; o3's 4th token changed from 0.40 to 0.42 so that no ratio
sits exactly on the clipping kink rho = 1 - eps, where the surrogate has no derivative). Threshold delta is an assumption (the paper does not
give it): two values 0.05 and 0.2 are shown.
(a) The (D_i, A_i) plane with the masked region A < 0 and D > delta.
(b) Response o2: per-token importance ratios and the clipped surrogate with epsilon = 0.2.
Writes deepseek/figures/l15-offpolicy-mask.svg
"""

import numpy as np

from l15_svg import SVG, fmt

OLD = np.array([0.9, 0.6, 0.8, 0.5, 0.7, 0.95])
THETA = {
    "o1": np.array([0.91, 0.62, 0.80, 0.52, 0.70, 0.95]),
    "o2": np.array([0.80, 0.30, 0.70, 0.20, 0.60, 0.90]),
    "o3": np.array([0.85, 0.50, 0.75, 0.42, 0.65, 0.93]),
}
THETA["o4"] = THETA["o2"]
REW = {"o1": 1, "o2": 0, "o3": 0, "o4": 1}
mean_r = np.mean(list(REW.values()))
ADV = {k: v - mean_r for k, v in REW.items()}
D = {k: float(np.mean(np.log(OLD / t))) for k, t in THETA.items()}
assert abs(D["o1"] + 0.0138) < 1e-4 and abs(D["o2"] - 0.3448) < 1e-4 and abs(D["o3"] - 0.0956) < 1e-4


def mask(k, delta):
    return 0 if (ADV[k] < 0 and D[k] > delta) else 1


assert [mask(k, 0.05) for k in ("o1", "o2", "o3", "o4")] == [1, 0, 0, 1]
assert [mask(k, 0.2) for k in ("o1", "o2", "o3", "o4")] == [1, 0, 1, 1]

EPS = 0.2
rho2 = THETA["o2"] / OLD
surr2 = np.minimum(rho2 * ADV["o2"], np.clip(rho2, 1 - EPS, 1 + EPS) * ADV["o2"])
assert np.allclose(np.round(surr2, 4), [-0.4444, -0.4, -0.4375, -0.4, -0.4286, -0.4737])

W, H = 760, 432
svg = SVG(W, H, "off-policy 序列掩码：只掩掉离策略太远的负优势序列")

# ================================================================ (a)
PX0, PX1, PY0, PY1 = 70, 360, 48, 318
DX0, DX1 = -0.1, 0.4
AY0_, AY1_ = -0.75, 0.75


def px(d):
    return PX0 + (d - DX0) / (DX1 - DX0) * (PX1 - PX0)


def py(a):
    return PY1 - (a - AY0_) / (AY1_ - AY0_) * (PY1 - PY0)


svg.text(16, 22, "(a) 每条回答一个点：横轴 D̄_{i}，纵轴优势 Â_{i}", size=13, weight="700")
# masked regions (delta = 0.05 lighter, 0.2 darker overlap)
svg.rect(px(0.05), py(0), px(DX1) - px(0.05), py(AY0_) - py(0), fill="red", fill_opacity=0.08)
svg.rect(px(0.2), py(0), px(DX1) - px(0.2), py(AY0_) - py(0), fill="red", fill_opacity=0.10)
for d in (-0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
    svg.line(px(d), PY0, px(d), PY1, color="grid", width=1)
    svg.text(px(d), PY1 + 16, fmt(d).replace("-", "−"), size=11, anchor="middle", color="muted")
for a in (-0.5, 0.0, 0.5):
    svg.line(PX0, py(a), PX1, py(a), color="rule" if a == 0 else "grid", width=1.2 if a == 0 else 1)
    svg.text(PX0 - 6, py(a) + 4, ("%+.1f" % a).replace("-", "−") if a else "0", size=11, anchor="end", color="muted")
svg.text((PX0 + PX1) / 2, PY1 + 36, "D̄_{i} = 序列内 log(π_{old}/π_{θ}) 的平均", size=12, anchor="middle", color="muted")
for dl, lab in ((0.05, "δ = 0.05"), (0.2, "δ = 0.2")):
    svg.line(px(dl), PY0 - 4, px(dl), PY1, color="red", width=1.4, dash="5 4")
    svg.text(px(dl) + 4, PY0 + 8, lab, size=12, color="red")
svg.text(px(0.3), py(-0.2), "掩掉（m_{i} = 0）", size=12, anchor="middle", color="red", weight="700")
svg.text(px(0.3), py(0.22), "Â > 0：永远保留", size=12, anchor="middle", color="green")


def est_w(s, size):
    """Rough advance width: CJK and full-width punctuation 1 em, everything else 0.6 em."""
    return sum(size if ord(ch) > 0x2E80 else size * 0.6 for ch in s)


def boxed(x, y, s, size=12, anchor="start", color="ink"):
    """Text on a surface-coloured plate, so the dashed threshold lines pass behind it."""
    w = est_w(s, size) + 6
    x0 = {"start": x - 3, "middle": x - w / 2, "end": x - w + 3}[anchor]
    svg.rect(x0, y - size + 1, w, size + 5, fill="surface", fill_opacity=0.9, rx=3)
    svg.text(x, y, s, size=size, anchor=anchor, color=color)


LABEL = {"o2": (-10, -9, "end"), "o3": (-10, -9, "end"), "o4": (-10, -9, "end")}
for k in ("o1", "o2", "o3", "o4"):
    x, y = px(D[k]), py(ADV[k])
    col = "green" if mask(k, 0.05) == 1 else "red"
    svg.circle(x, y, 6, fill=col, stroke="surface", width=1.5)
    val = ("%.4f" % D[k]).replace("-", "−")
    if k == "o1":
        # o1 sits just left of the delta = 0.05 line: stack name and value above the point, centred,
        # so neither line of text straddles the dashed threshold (value plate ends ~4 units short of it).
        cx = x - 5
        assert cx + est_w("(%s)" % val, 12) / 2 + 3 < px(0.05) - 2
        boxed(cx, y - 26, k, size=12, anchor="middle")
        boxed(cx, y - 11, "(%s)" % val, size=12, anchor="middle")
        continue
    dx, dy, anc = LABEL[k]
    boxed(x + dx, y + dy, "%s（%s）" % (k, val), size=12, anchor=anc)
boxed(px(D["o3"]), py(-0.5) + 22, "δ = 0.05 时掩，δ = 0.2 时留", size=11, anchor="middle", color="muted")

# ================================================================ (b)
BX0, BX1, BY0, BY1 = 450, 740, 48, 318
svg.text(410, 22, "(b) o2 的 6 个 token：重要性比 ρ 与裁剪（ε = 0.2）", size=13, weight="700")
R0, R1 = 0.0, 1.3


def ry(r):
    return BY1 - (r - R0) / (R1 - R0) * (BY1 - BY0)


svg.rect(BX0, ry(1 + EPS), BX1 - BX0, ry(1 - EPS) - ry(1 + EPS), fill="green", fill_opacity=0.08)
for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
    svg.line(BX0, ry(r), BX1, ry(r), color="grid", width=1)
    svg.text(BX0 - 6, ry(r) + 4, fmt(r), size=11, anchor="end", color="muted")
svg.line(BX0, ry(1 - EPS), BX1, ry(1 - EPS), color="green", width=1.2, dash="4 3")
svg.text(BX1, ry(1 + EPS) - 6, "裁剪区间 [0.8, 1.2]", size=11, anchor="end", color="green")
cw = (BX1 - BX0) / 6
for j, r in enumerate(rho2):
    x = BX0 + j * cw + cw * 0.2
    live = r > 1 - EPS
    col = "red" if live else "muted"
    svg.rect(x, ry(r), cw * 0.6, ry(0) - ry(r), fill=col, fill_opacity=0.35 if live else 0.18, stroke=col, width=1.2)
    svg.text(x + cw * 0.3, ry(r) - 6, "%.3f" % r, size=11, anchor="middle", color="ink")
    svg.text(x + cw * 0.3, BY1 + 16, "t=%d" % (j + 1), size=11, anchor="middle", color="muted")
    svg.text(x + cw * 0.3, BY1 + 32, ("%.3f" % surr2[j]).replace("-", "−"), size=11, anchor="middle",
             color="red" if live else "muted")
svg.text(BX0 - 8, BY1 + 32, "代理项", size=11, anchor="end", color="muted")
svg.text(BX0, BY1 + 54, "红色：ρ > 0.8，代理项 = ρÂ，仍有梯度（4 个）", size=12, color="red")
svg.text(BX0, BY1 + 72, "灰色：ρ < 0.8，代理项 = 0.8·Â = −0.4，无梯度（2 个）", size=12, color="muted")

svg.text(16, 424, "奖励 [1, 0, 0, 1] → 优势 [+0.5, −0.5, −0.5, +0.5]；o4 的 π_{θ} 与 o2 相同。δ 取值是本例的假设，论文没有给出。",
         size=12, color="muted")

svg.save("l15-offpolicy-mask.svg")
