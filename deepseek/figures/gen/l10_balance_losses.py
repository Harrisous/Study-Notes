"""Figure 10.6: what each of V2's three balance losses counts, on one toy routing table.

Toy (author's construction): N_r = 8 routed experts on D = 4 devices (A: e1 e2, B: e3 e4, C: e5 e6,
D: e7 e8), K_r = 2, M = 2, T = 8 tokens. Average affinities P_i are given toy numbers (sum 1).
  f_i   = N_r / (K_r T) * #(tokens selecting expert i)          (V2 eq. 24)
  f'_i  = mean of f_j over the device's experts                  (eq. 27)
  f''_i = D / (M T) * #(tokens sent to device i)                 (eq. 30)
  P'_i  = P''_i = sum of P_j over the device's experts           (eqs. 28, 31)
Expert level is unbalanced (e1 hot, e2 idle), device compute is balanced, device A receives the most tokens.
Writes deepseek/figures/l10-balance-losses.svg
"""

from l10_svg import SVG, fmt

routes = [(1, 3), (1, 5), (1, 7), (1, 4), (3, 4), (5, 6), (6, 8), (7, 8)]
dev_of = {1: "A", 2: "A", 3: "B", 4: "B", 5: "C", 6: "C", 7: "D", 8: "D"}
devs = ["A", "B", "C", "D"]
Nr, Kr, T, D, M = 8, 2, 8, 4, 2
P = {1: 0.22, 2: 0.03, 3: 0.125, 4: 0.125, 5: 0.125, 6: 0.125, 7: 0.125, 8: 0.125}
assert abs(sum(P.values()) - 1) < 1e-12
assert all(len({dev_of[e] for e in r}) <= M for r in routes)
cnt = {i: sum(i in r for r in routes) for i in range(1, 9)}
f = {i: Nr / (Kr * T) * cnt[i] for i in cnt}
members = {dv: [i for i in range(1, 9) if dev_of[i] == dv] for dv in devs}
f1 = {dv: sum(f[i] for i in members[dv]) / len(members[dv]) for dv in devs}
P1 = {dv: sum(P[i] for i in members[dv]) for dv in devs}
recv = {dv: sum(any(dev_of[e] == dv for e in r) for r in routes) for dv in devs}
f2 = {dv: D / (M * T) * recv[dv] for dv in devs}
L1 = sum(f[i] * P[i] for i in f)
L2 = sum(f1[dv] * P1[dv] for dv in devs)
L3 = sum(f2[dv] * P1[dv] for dv in devs)
assert abs(L1 - 1.19) < 1e-12 and abs(L2 - 1.0) < 1e-12 and abs(L3 - 0.8125) < 1e-12
assert [f2[d_] for d_ in devs] == [1.0, 0.75, 0.75, 0.75]

W, H = 760, 604
svg = SVG(W, H, "三种均衡损失各数什么：同一张玩具路由表上的 f、f′、f″")

# ---- routing table (left)
svg.text(20, 26, "玩具路由表（K_{r} = 2，M = 2）", size=14, weight="600")
for k, (a, b) in enumerate(routes):
    y = 58 + k * 30
    svg.text(24, y, "t%d" % (k + 1), size=13, weight="600")
    for j, e in enumerate((a, b)):
        x = 58 + j * 72
        svg.rect(x, y - 16, 64, 22, fill="blue", fill_opacity=0.12, stroke="blue", width=1, rx=4)
        svg.text(x + 32, y, "e%d（%s）" % (e, dev_of[e]), size=12, anchor="middle")
svg.text(20, 318, "设备：A = {e1, e2}，B = {e3, e4}，", size=12, color="muted")
svg.text(20, 336, "C = {e5, e6}，D = {e7, e8}", size=12, color="muted")
svg.text(20, 366, "平均亲和度 P_{i}（玩具数字）：", size=12, color="muted")
svg.text(20, 384, "P_{1} = 0.22，P_{2} = 0.03，", size=12, color="muted")
svg.text(20, 402, "P_{3} = … = P_{8} = 0.125", size=12, color="muted")
svg.text(20, 430, "所以每台设备的 P′ = P″ = 0.25", size=12, color="muted")

# ---- three panels (right)
PX0, PX1 = 240, 560
BASE_H = 70   # pixels for value 1.0


def col(v):
    return "red" if v > 1 + 1e-9 else ("muted" if v < 1 - 1e-9 else "blue")


def panel(y0, title, sub, labels, vals, loss_line, alpha_line, ymax=2.0):
    svg.text(PX0, y0, title, size=13, weight="600")
    svg.text(PX0, y0 + 22, sub, size=12, color="muted")   # clear of the subscript in the title
    base = y0 + 46 + BASE_H * ymax
    n = len(vals)
    slot = (PX1 - PX0) / n
    svg.line(PX0, base, PX1, base, color="muted", width=1)
    y1 = base - BASE_H
    svg.line(PX0, y1, PX1, y1, color="ink", width=1, dash="4 3")
    svg.text(PX0 - 6, y1 + 4, "1", size=11, anchor="end", color="muted")
    for k, (lab, v) in enumerate(zip(labels, vals)):
        cx = PX0 + (k + 0.5) * slot
        bw = min(30, slot * 0.6)
        if v > 0:
            svg.rect(cx - bw / 2, base - BASE_H * v, bw, BASE_H * v, fill=col(v), fill_opacity=0.7)
        svg.text(cx, base - BASE_H * v - 5, fmt(v), size=11, anchor="middle")
        svg.text(cx, base + 15, lab, size=12, anchor="middle", color="muted")
    svg.text(PX1 + 16, y0 + 62, loss_line, size=13, weight="600")
    svg.text(PX1 + 16, y0 + 82, alpha_line, size=12, color="muted")
    return base + 22


y = panel(22, "专家级 f_{i}：每个专家被选的次数（归一化）", "e1 被 4 个 token 选中，e2 一次也没有",
          ["e%d" % i for i in range(1, 9)], [f[i] for i in range(1, 9)],
          "Σ f_{i}P_{i} = %s" % fmt(L1), "均匀时 = 1；系数 α_{1} = 0.003")
y = panel(y + 14, "设备级 f′_{m}：设备上专家的平均 f", "每台设备都分到 4 次专家计算：计算量均衡",
          devs, [f1[d_] for d_ in devs], "Σ f′_{m}P′_{m} = %s" % fmt(L2), "均匀时 = 1；系数 α_{2} = 0.05",
          ymax=1.25)
y = panel(y + 14, "通信 f″_{m}：发到该设备的 token 数（归一化）", "A 收 4 个 token，B、C、D 各收 3 个",
          devs, [f2[d_] for d_ in devs], "Σ f″_{m}P″_{m} = %.4f" % L3,
          "均匀时 = 1；系数 α_{3} = 0.02", ymax=1.25)
svg.text(PX0, H - 10, "红 = 高于均匀份额，蓝 = 恰好均匀，灰 = 低于均匀份额", size=12, color="muted")
svg.save("l10-balance-losses.svg")
