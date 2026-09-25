"""Figure 4.12: from logits to the next token: temperature, top-k and top-p on one toy example.

Logits z = [2.0, 1.0, 0.5, 0.0, -1.0] (lesson-04-doc §4.9); the candidate words are illustrative labels.
  p_tau(x) = exp(z_x / tau) / sum exp(z_x' / tau), tau = 0.5, 1, 2   (entropy in bits)
  top-k = 2 and top-p = 0.9 are applied to the tau = 1 distribution, then renormalised.
Writes deepseek/figures/l04-sampling.svg
"""

import math

from l04_svg import SVG

WORDS = ["山", "庙", "桥", "城", "车"]
U = [2.0, 1.0, 0.5, 0.0, -1.0]


def softmax(u, tau):
    e = [math.exp(x / tau) for x in u]
    s = sum(e)
    return [x / s for x in e]


def entropy(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


P = {tau: softmax(U, tau) for tau in (0.5, 1.0, 2.0)}
p1 = P[1.0]
topk = [p1[0], p1[1]]
topk = [x / sum(topk) for x in topk]
cum, keep = 0.0, 0
for x in p1:            # already sorted in decreasing order
    cum += x
    keep += 1
    if cum >= 0.9:
        break
topp_mass = sum(p1[:keep])
topp = [x / topp_mass for x in p1[:keep]]
# checks against lesson-04-doc §4.9
assert [round(x, 4) for x in P[0.5]] == [0.8292, 0.1122, 0.0413, 0.0152, 0.0021]
assert [round(x, 4) for x in p1] == [0.5630, 0.2071, 0.1256, 0.0762, 0.0280]
assert [round(x, 4) for x in P[2.0]] == [0.3745, 0.2272, 0.1769, 0.1378, 0.0836]
assert [round(entropy(P[t]), 3) for t in (0.5, 1.0, 2.0)] == [0.878, 1.741, 2.152]
assert [round(x, 4) for x in topk] == [0.7311, 0.2689]
assert keep == 4 and round(topp_mass, 4) == 0.9720 and [round(x, 4) for x in topp] == [0.5793, 0.2131, 0.1293, 0.0784]

W, H = 720, 630
svg = SVG(W, H, "从 logits 到下一个 token：温度、top-k 与 top-p（玩具例子）")


def bars(x0, y_base, probs, bw=26, gap=12, hmax=120, col="blue", kept=None, label_vals=True, words=WORDS):
    """Vertical probability bars (0..1 mapped to hmax px)."""
    for i, p in enumerate(probs):
        x = x0 + i * (bw + gap)
        on = kept is None or i in kept
        h = p * hmax
        if on:
            svg.rect(x, y_base - h, bw, h, fill=col, fill_opacity=0.6, rx=2)
        else:
            svg.rect(x, y_base - h, bw, h, fill=None, stroke="muted", width=1, dash="3 2", rx=2)
        if label_vals:
            svg.text(x + bw / 2, y_base - h - 5, "%.3f" % p, size=11,
                     anchor="middle", color="ink" if on else "muted")
        svg.text(x + bw / 2, y_base + 16, words[i], size=12, anchor="middle", color="ink" if on else "muted")
    svg.line(x0 - 4, y_base, x0 + len(probs) * (bw + gap) - gap + 4, y_base, color="muted", width=1)


# ---- row 0: logits
svg.text(16, 24, "① 模型给出的 logits z（“从前有座”之后，示意的 5 个候选词）", size=14, weight="600")
x0, yb = 150, 110
for i, u in enumerate(U):
    x = x0 + i * 60
    h = u * 30
    if h >= 0:
        svg.rect(x, yb - h, 30, h, fill="ink", fill_opacity=0.35, rx=2)
    else:
        svg.rect(x, yb, 30, -h, fill="ink", fill_opacity=0.35, rx=2)
    svg.text(x + 15, yb - max(h, 0) - 6, "%.1f" % u, size=12, anchor="middle")
    svg.text(x + 15, yb + (-h if h < 0 else 0) + 16, WORDS[i], size=12, anchor="middle")
svg.line(x0 - 8, yb, x0 + 5 * 60 - 22, yb, color="muted", width=1)
svg.text(480, 64, "除以温度 τ，再做 softmax：", size=12, color="muted")
svg.text(480, 82, "p_{τ}(x) = exp(z_{x}/τ) / Σ exp(z_{x'}/τ)", size=13, color="blue")

# ---- row 1: temperatures
svg.text(16, 176, "② 温度：同一组 logits，三种 τ", size=14, weight="600")
yb1 = 330
for j, tau in enumerate((0.5, 1.0, 2.0)):
    xs = 40 + j * 232
    bars(xs, yb1, P[tau], bw=24, gap=14, hmax=150)
    svg.text(xs + 88, yb1 + 38, "τ = %g，熵 %.3f bit" % (tau, entropy(P[tau])), size=12, anchor="middle",
             weight="600")
    svg.text(xs + 88, yb1 + 55, ("更尖，接近贪心", "原分布", "更平，尾部被抬高")[j], size=12, anchor="middle",
             color="muted")

# ---- row 2: truncation
svg.text(16, 420, "③ 截断再重新归一化（都从 τ = 1 的分布出发）", size=14, weight="600")
yb2 = 576
bars(40, yb2, p1, bw=24, gap=14, hmax=120, kept={0, 1}, col="green", label_vals=False)
for i, p in enumerate(topk):
    x = 40 + i * 38
    svg.rect(x + 6, yb2 - p * 120, 12, p * 120, fill="green", fill_opacity=0.9, rx=1)
    svg.text(x + 12, yb2 - p * 120 - 5, "%.3f" % p, size=11, anchor="middle", color="green", weight="600")
svg.text(128, yb2 + 38, "top-k，k = 2：只留 2 个", size=12, anchor="middle", weight="600")

bars(330, yb2, p1, bw=24, gap=14, hmax=120, kept=set(range(keep)), col="green", label_vals=False)
for i, p in enumerate(topp):
    x = 330 + i * 38
    svg.rect(x + 6, yb2 - p * 120, 12, p * 120, fill="green", fill_opacity=0.9, rx=1)
    svg.text(x + 12, yb2 - p * 120 - 5, "%.3f" % p, size=11, anchor="middle", color="green", weight="600")
svg.text(420, yb2 + 38, "top-p，p_{⋆} = 0.9：累计 %.3f 时停" % topp_mass, size=12, anchor="middle", weight="600")
cumv = 0
for i in range(5):
    cumv += p1[i]
    # x = 528 keeps the longest line ("... ← 首次 ≥ 0.9") inside the 720-wide viewBox
    svg.text(528, 456 + i * 18, "前 %d 个累计 %.3f%s" % (i + 1, cumv, " ← 首次 ≥ 0.9" if i == keep - 1 else ""),
             size=12, color="green" if i < keep else "muted")
svg.text(528, 456 + 5 * 18 + 8, "浅色宽柱：截断前；深色窄柱：", size=12, color="muted")
svg.text(528, 456 + 6 * 18 + 8, "重新归一化后的概率", size=12, color="muted")

svg.save("l04-sampling.svg")
