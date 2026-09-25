"""Figure 8.4: how PPO turns one end-of-answer reward into per-token advantages (toy numbers).

A five-token answer. Per-token reward r_t = -beta * k1_t (KL penalty, DeepSeekMath eq. 2) plus
the reward-model score on the last token. The value model's predictions V(s_t) are made up.
TD residual delta_t = r_t + gamma V(s_{t+1}) - V(s_t) and GAE A_t = sum_l (gamma lambda)^l delta_{t+l}
(Schulman et al. 2015, eqs. 16) are computed here with gamma = 1, lambda = 0.95 and lambda = 1.
All numbers are the author's construction for teaching; nothing here is from a paper.
Writes deepseek/figures/l08-gae-credit.svg and prints the table used in the lesson text.
"""

from l08_svg import SVG

beta, gamma = 0.04, 1.0
k1 = [0.20, -0.10, 0.50, 0.00, 0.30]            # log(pi_theta / pi_ref) of each sampled token
rm = 1.0                                         # reward-model score of the whole answer
V = [0.40, 0.45, 0.30, 0.55, 0.70]               # value model's guesses
T = len(V)
r = [-beta * k for k in k1]
r[-1] += rm
Vn = V[1:] + [0.0]
delta = [r[t] + gamma * Vn[t] - V[t] for t in range(T)]


def gae(lam):
    A, acc = [0.0] * T, 0.0
    for t in reversed(range(T)):
        acc = delta[t] + gamma * lam * acc
        A[t] = acc
    return A


A95, A1 = gae(0.95), gae(1.0)
for name, row in (("r", r), ("V", V), ("delta", delta), ("A(0.95)", A95), ("A(1.0)", A1)):
    print("%-8s" % name, " ".join("%+.4f" % v for v in row))

W, H = 740, 384
svg = SVG(W, H, "PPO 怎样把回答末尾的一个奖励分摊到每个 token（玩具数字）")
LX, CX, CW, RH = 16, 196, 100, 44
Y0 = 58
toks = ["o₁", "o₂", "o₃", "o₄", "o₅ (结束)"]
for t, s in enumerate(toks):
    x = CX + t * (CW + 8)
    svg.rect(x, Y0 - 34, CW, 26, fill="blue", fill_opacity=0.10, stroke="blue", width=1.1, rx=5)
    svg.text(x + CW / 2, Y0 - 16, s, size=13, anchor="middle", color="blue", weight="600")

rows = [
    ("每 token 奖励 r_{t}", r, "−β·k1，末尾加 RM 分 1.0"),
    ("价值预测 V(s_{t})", V, "笔者设定的猜测"),
    ("TD 残差 δ_{t}", delta, "r_{t} + V(s_{t+1}) − V(s_{t})"),
    ("GAE 优势，λ = 0.95", A95, "λ < 1：方差小，但有偏"),
    ("GAE 优势，λ = 1", A1, "之后奖励之和 − V(s_{t})"),
    ("GRPO 优势（对照）", [1.0] * T, "整条回答共用，例如 +1"),
]
for k, (lab, vals, note) in enumerate(rows):
    y = Y0 + k * RH
    svg.text(LX, y + 17, lab, size=12.5, color="ink", weight="600")
    svg.text(LX, y + 34, note, size=11, color="muted")
    for t, v in enumerate(vals):
        x = CX + t * (CW + 8)
        if k == 1:
            col, op = "blue", 0.08 + 0.25 * v
        else:
            col = "green" if v >= 0 else "red"
            op = min(0.55, 0.08 + 0.5 * abs(v))
        if k == 5:
            col, op = "violet", 0.16
        svg.rect(x, y + 4, CW, RH - 10, fill=col, fill_opacity=op, rx=4)
        v = 0.0 if abs(v) < 5e-4 else v
        lab_v = ("%+.3f" % v if v else "0.000") if k != 1 else "%.2f" % v
        svg.text(x + CW / 2, y + RH / 2 + 4, lab_v.replace("-", "−"), size=12.5,
                 anchor="middle", color="ink")
    if k == 4:
        svg.line(LX, y + RH + 1, W - 16, y + RH + 1, color="rule", width=1)

# notes under the grid
yb = Y0 + len(rows) * RH + 16
svg.text(LX, yb, "γ = 1。GAE 从最后一个 token 往前递推：Â_{t} = δ_{t} + γλ·Â_{t+1}。δ_{2} < 0：价值模型认为 o₂ 让前景变差了。",
         size=12, color="muted")
svg.text(LX, yb + 20, "PPO 的每个 token 各有一个优势，靠价值模型“猜”出来；GRPO 不要价值模型，整条回答共用一个优势。",
         size=12, color="muted")
svg.save("l08-gae-credit.svg")
