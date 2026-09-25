"""Figure 3.5: reward-KL frontier of the toy problem (pi_ref = [0.6,0.3,0.1], r = [0,1,2]).

The optimal policies pi*_beta trace a curve in the (KL, E[r]) plane; no policy lies above it.
At a given beta the optimum is where a line of slope beta (E[r] - beta*KL = const) touches the curve.
One hand-picked policy is shown below the curve. Writes deepseek/figures/l03-frontier.svg
"""
import math
from l03_svg import SVG, Axis, polyline_d, fmt

pref = [0.6, 0.3, 0.1]
r = [0.0, 1.0, 2.0]


def stats(pi):
    Er = sum(p * x for p, x in zip(pi, r))
    KL = sum(p * math.log(p / q) for p, q in zip(pi, pref) if p > 0)
    return KL, Er


def opt(beta):
    w = [p * math.exp(x / beta) for p, x in zip(pref, r)]
    Z = sum(w)
    return [v / Z for v in w]


def frontier_at(kl):
    """E[r] of the optimal policy whose KL equals kl (bisection on log beta)."""
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if stats(opt(math.exp(mid)))[0] > kl:
            lo = mid
        else:
            hi = mid
    return stats(opt(math.exp(lo)))[1]


W, H = 720, 360
svg = SVG(W, H, "奖励–KL 前沿：最优策略族与一个非最优策略")
ax = Axis(0, 2.4, 70, 690)
ay = Axis(0.3, 2.1, 310, 40)
for v in (0, 0.5, 1.0, 1.5, 2.0):
    svg.line(ax(v), ay(0.3), ax(v), ay(2.1), color="grid", width=1)
    svg.text(ax(v), ay(0.3) + 17, fmt(v), size=11, anchor="middle", color="muted")
for v in (0.5, 1.0, 1.5, 2.0):
    svg.line(ax(0), ay(v), ax(2.4), ay(v), color="grid", width=1)
    svg.text(ax(0) - 7, ay(v) + 4, fmt(v), size=11, anchor="end", color="muted")
svg.line(ax(0), ay(0.3), ax(2.4), ay(0.3), color="muted", width=1.1)
svg.line(ax(0), ay(0.3), ax(0), ay(2.1), color="muted", width=1.1)
svg.text(ax(1.2), ay(0.3) + 38, "KL(π ‖ π_{ref})（单位：nat）", size=12.5, anchor="middle")
svg.text(ax(0) - 50, ay(2.1) - 14, "期望奖励 E[r]", size=12.5)
# frontier
betas = [10 ** (2 - 3.3 * i / 400) for i in range(401)]
pts = [stats(opt(b)) for b in betas]
svg.path(polyline_d([(ax(k), ay(e)) for k, e in pts]), stroke="green", width=2.6)
svg.text(ax(0.25), ay(1.92), "前沿左上方：任何策略都到不了", size=12, color="muted")
# tangent line at beta = 1
b = 1
k1, e1 = stats(opt(b))
c = e1 - b * k1
svg.line(ax(0), ay(c), ax(1.1), ay(c + b * 1.1), color="blue", width=1.4, dash="6 4")
svg.text(ax(1.12), ay(c + b * 1.1) - 4, "β = 1 的等目标线（斜率 1）", size=12, color="blue")
svg.line(ax(0.004), ay(c), ax(0.12), ay(0.625), color="blue", width=0.9)
svg.text(ax(0.13), ay(0.625) + 10, "截距 = β log Z = 0.768（β = 1 时的最优目标值）", size=11.5, color="blue")
for bb, dx, dy, anc in [(2, 10, 5, "start"), (1, 10, 16, "start"), (0.5, 10, 16, "start"), (0.25, 0, 24, "middle")]:
    k, e = stats(opt(bb))
    svg.circle(ax(k), ay(e), 4.2, fill="green")
    svg.text(ax(k) + dx, ay(e) + dy, "β = %s" % bb, size=12, anchor=anc, color="green", weight="600")
k, e = stats(pref)
svg.circle(ax(k), ay(e), 4.2, fill="ink")
svg.text(ax(k) + 10, ay(e) + 22, "π_{ref}（β → ∞）；β = 10 的点几乎与它重合", size=12)
k, e = stats(opt(10))
svg.circle(ax(k), ay(e), 3.4, fill="green")
# greedy: all mass on answer 3 (the beta -> 0 end point of the frontier, not a point below it)
kg, eg = math.log(1 / 0.1), 2.0
svg.circle(ax(kg), ay(eg), 4.2, fill="red")
svg.text(ax(kg) + 4, ay(eg) - 12, "全押回答 3：KL = ln 10 = 2.303", size=12, anchor="end", color="red")
# one sub-optimal policy
for pi, lab in [([0.1, 0.8, 0.1], "偏爱回答 2 的策略 [0.1, 0.8, 0.1]：比前沿低 0.32")]:
    k, e = stats(pi)
    fe = frontier_at(k)
    assert fe > e
    svg.line(ax(k), ay(e), ax(k), ay(fe), color="muted", width=1, dash="2 3")
    svg.circle(ax(k), ay(e), 4.2, fill="surface", stroke="ink", width=1.6)
    svg.text(ax(k) + 8, ay(e) + 20, lab, size=12)
    print("policy %s: KL=%.4f E[r]=%.4f frontier E[r]=%.4f" % (pi, k, e, fe))
svg.save("l03-frontier.svg")
