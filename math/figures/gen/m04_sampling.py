"""Figure M4.9: sampling from a categorical distribution by cutting [0, 1) into segments.

Toy logits z = (2, 1, 0.5, 0, -1) at tau = 1 (the same as lesson 4). Row 1: the original
distribution and three uniform draws U. Rows 2-3: top-k (k = 2) and top-p (p = 0.9) keep a
prefix of the sorted tokens and renormalise it (Holtzman et al. 2020, eqs. 2-3).
Writes math/figures/m04-sampling.svg
"""

import numpy as np

from m04_svg import SVG

z = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
p = np.exp(z - z.max())
p /= p.sum()
F = np.cumsum(p)


def pick(u, probs):
    return int(np.searchsorted(np.cumsum(probs), u, side="right"))


def top_k(probs, k):
    keep = np.argsort(-probs)[:k]
    q = np.zeros_like(probs)
    q[keep] = probs[keep]
    return q / q.sum(), float(probs[keep].sum())


def top_p(probs, thr):
    order = np.argsort(-probs)
    c = np.cumsum(probs[order])
    n = int(np.searchsorted(c, thr) + 1)       # smallest prefix whose mass >= thr
    keep = order[:n]
    q = np.zeros_like(probs)
    q[keep] = probs[keep]
    return q / q.sum(), float(probs[keep].sum())


pk, mk = top_k(p, 2)
pp, mp = top_p(p, 0.9)
US = (0.30, 0.70, 0.95)
print("p", np.round(p, 4), "F", np.round(F, 4))
print("picks", [pick(u, p) for u in US])
print("top-2", np.round(pk, 4), mk, "top-p .9", np.round(pp, 4), mp)

W, H = 720, 384
svg = SVG(W, H, "从分布里采样：把 [0,1) 按概率切段，再撒一个均匀的点")
X0, XW = 110, 530
BH = 34
OPA = [0.62, 0.46, 0.32, 0.20, 0.10]


def strip(y, probs, title, sub, note=None, removed=()):
    svg.text(16, y + 14, title, size=13, weight="600")
    svg.text(16, y + 31, sub, size=12, color="muted")
    x = X0
    for i, v in enumerate(probs):
        w = XW * v
        if w > 0:
            svg.rect(x, y, w, BH, fill="blue", fill_opacity=OPA[i], stroke="blue", width=1.1)
            if w > 26:
                svg.text(x + w / 2, y + 15, "#%d" % i, size=12, anchor="middle", color="ink")
                svg.text(x + w / 2, y + 29, ("%.3f" % v), size=11, anchor="middle", color="ink")
            x += w
    if removed:
        svg.text(X0 + XW + 4, y + 22, "", size=11)
    if note:
        svg.text(X0, y + BH + 16, note, size=12, color="muted")


# row 1: original with cumulative ticks and U arrows
y1 = 84
svg.text(X0, 22, "U 在 [0, 1) 上均匀；它落在哪一段，就选哪个 token", size=12, color="ink")
strip(y1, p, "原分布", "τ = 1")
# tiny tail segment label
svg.text(X0 + XW * (F[3] + p[4] / 2), y1 - 6, "#4", size=11, anchor="middle", color="ink")
for i, f in enumerate(np.concatenate([[0.0], F])):
    x = X0 + XW * f
    svg.line(x, y1 + BH, x, y1 + BH + 6, color="muted", width=1)
    lab = "0" if i == 0 else ("1" if i == 5 else "%.3f" % f)
    if i in (0, 1, 2, 3, 5):
        svg.text(x, y1 + BH + 18, lab, size=11, anchor="middle", color="muted")
svg.text(X0 + XW * F[3] - 4, y1 + BH + 30, "F₄ = 0.972", size=11, anchor="end", color="muted")
for u in US:
    x = X0 + XW * u
    svg.arrow(x, y1 - 24, x, y1 - 2, color="red", width=1.6)
    svg.text(x + 4, y1 - 26, "U=%.2f→#%d" % (u, pick(u, p)), size=11, color="red")

# row 2: top-k
y2 = 200
strip(y2, pk, "top-k", "k = 2",
      note="只留最大的 2 个（原质量 %.4f），各除以 %.4f → (%.4f, %.4f)" % (mk, mk, pk[0], pk[1]))
svg.rect(X0 + XW + 6, y2 + 8, 20, BH - 16, fill="red", fill_opacity=0.12, stroke="red", width=1, dash="3 2")
svg.text(X0 + XW + 30, y2 + 22, "#2–#4", size=11, color="red")
svg.text(X0 + XW + 30, y2 + 36, "删去", size=11, color="red")

# row 3: top-p
y3 = 302
strip(y3, pp, "top-p", "阈值 0.9",
      note="累计概率首次 ≥ 0.9 的最小集合 {#0…#3}（质量 %.4f），再重新归一化" % mp)
svg.rect(X0 + XW + 6, y3 + 8, 20, BH - 16, fill="red", fill_opacity=0.12, stroke="red", width=1, dash="3 2")
svg.text(X0 + XW + 30, y3 + 22, "#4", size=11, color="red")
svg.text(X0 + XW + 30, y3 + 36, "删去", size=11, color="red")

svg.save("m04-sampling.svg")
