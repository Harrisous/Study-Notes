"""Figure 3.11: rewards, values, TD residuals and GAE advantages on a 4-token answer.

Toy numbers (lesson-03 research notes, sec. 4.4): beta = 0.05, RM score 1.0 on the last token,
per-token log-ratios log(pi/pi_ref) = [0.10, -0.05, 0.20, 0.00], values V = [0.5, 0.6, 0.7, 0.9],
V = 0 after the end, gamma = 1, lambda = 0.95.  All table entries are computed here, in exact rational
arithmetic, and rounded half-up (A_2 = 0.37325 exactly -> 0.3733, so that 0.095 + 0.95 x 0.3733 = 0.4496
checks out on a calculator; binary floats would round it down to 0.3732).
Writes deepseek/figures/l03-gae-tokens.svg
"""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from l03_svg import SVG, box

beta, RM, gamma = Fr(5, 100), Fr(1), Fr(1)
logr = [Fr(10, 100), Fr(-5, 100), Fr(20, 100), Fr(0)]
V = [Fr(50, 100), Fr(60, 100), Fr(70, 100), Fr(90, 100)]
T = len(V)
r = [-beta * x for x in logr]
r[-1] += RM
Vn = V[1:] + [Fr(0)]
delta = [r[t] + gamma * Vn[t] - V[t] for t in range(T)]


def gae(lam):
    A, acc = [Fr(0)] * T, Fr(0)
    for t in reversed(range(T)):
        acc = delta[t] + gamma * lam * acc
        A[t] = acc
    return A


A95, A0, A1 = gae(Fr(95, 100)), gae(Fr(0)), gae(Fr(1))
assert A95 == [Fr(4495875, 10 ** 7), Fr(37325, 10 ** 5), Fr(285, 1000), Fr(1, 10)]
assert sum(r) == Fr(9875, 10000)
assert A1 == [Fr(4875, 10000), Fr(3925, 10000), Fr(29, 100), Fr(1, 10)]


def f(v, n=4):
    """Exact value rounded half-up to n decimals, trailing zeros stripped, Unicode minus."""
    d = (Decimal(v.numerator) / Decimal(v.denominator)).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP)
    s = format(d, "f")
    s = s.rstrip("0").rstrip(".") if "." in s else s
    if s in ("-0", ""):
        s = "0"
    return s.replace("-", "−")


assert [f(a) for a in A95] == ["0.4496", "0.3733", "0.285", "0.1"]
assert [f(a + v) for a, v in zip(A95, V)] == ["0.9496", "0.9733", "0.985", "1"]


W, H = 740, 428
svg = SVG(W, H, "一段 4 个 token 的回答：逐 token 奖励、价值、TD 残差与 GAE 优势")
x0, cw = 214, 126
rows = [
    ("log π_{θ}/π_{ref}", [f(v, 2) for v in logr], "ink"),
    ("r_{t}（KL 奖励 + RM 分）", [f(v) for v in r], "ink"),
    ("V_{t}（价值网络）", [f(v, 2) for v in V], "violet"),
    ("δ_{t} = r_{t} + V_{t+1} − V_{t}", [f(v) for v in delta], "ink"),
    ("Â_{t}，λ = 0.95", [f(v) for v in A95], "blue"),
    ("Â_{t}，λ = 0（= δ_{t}）", [f(v) for v in A0], "muted"),
    ("Â_{t}，λ = 1", [f(v) for v in A1], "muted"),
    ("价值目标 Â_{t} + V_{t}", [f(a + v) for a, v in zip(A95, V)], "violet"),
]
# token header
for t in range(T):
    cx = x0 + t * cw
    last = t == T - 1
    box(svg, cx + 8, 16, cw - 16, 34, "o_{%d}%s" % (t + 1, "（结束）" if last else ""),
        stroke="green" if last else "blue", fill="green" if last else "blue", fill_opacity=0.10,
        color="green" if last else "blue", weight="700", size=13.5)
svg.text(x0 + 3 * cw + cw / 2, 66, "RM 分 1.0 只加在这里", size=12, anchor="middle", color="green", weight="600")
y = 92
rowy = {}
for k, (lab, vals, col) in enumerate(rows):
    if k in (4,):
        svg.rect(12, y - 20, W - 24, 30, fill="blue", fill_opacity=0.07, rx=6)
    svg.text(x0 - 12, y, lab, size=12.5, anchor="end", color=col if col != "muted" else "muted", weight="600" if k == 4 else None)
    for t, s in enumerate(vals):
        c = col
        if k == 1:
            c = "green" if t == T - 1 else "red"
        svg.text(x0 + t * cw + cw / 2, y, s, size=13, anchor="middle", color=c, weight="700" if k == 4 else None)
    rowy[k] = y
    y += 36 if k != 4 else 40
# recursion arrows on the lambda = 0.95 row
ya = rowy[4] + 14
for t in range(T - 1, 0, -1):
    xa = x0 + t * cw + cw / 2 - 20
    xb = x0 + (t - 1) * cw + cw / 2 + 26
    svg.path("M%d %d Q %d %d %d %d" % (xa, ya - 4, (xa + xb) / 2, ya + 12, xb, ya - 4), stroke="blue", width=1.2, arrow=True)
    svg.text((xa + xb) / 2, ya + 16, "×0.95，加 δ", size=11, anchor="middle", color="blue")
svg.text(16, H - 30, "从最后一个 token 往前递推：Â_{4} = δ_{4}，Â_{t} = δ_{t} + γλ·Â_{t+1}（γ = 1）。结束之后 V = 0。", size=12, color="muted")
svg.text(16, H - 10, "β = 0.05；红字是每个 token 的 KL 奖励 −β·log(π_{θ}/π_{ref})；数字为笔者自拟，由脚本计算。", size=12, color="muted")
svg.save("l03-gae-tokens.svg")
