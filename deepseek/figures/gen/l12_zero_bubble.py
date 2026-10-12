"""Figure 12.11: splitting the backward pass - 1F1B vs a ZB-H1-style schedule, simulated.

P = 4 stages, m = 8 micro-batches, communication ignored. Units: forward T_F = 1, full backward
T_B = 2, split into the input-gradient half T_Bx = 1 ("B") and the weight-gradient half T_W = 1
("W"), so T_Bx + T_W = 2 T_F as in the Zero Bubble paper's analysis.

Top panel: 1F1B with the full backward (the same schedule as Figure 12.10).
Bottom panel: F and B keep the 1F1B order, but B is only the input-gradient half, so a stage can
pass the gradient upstream after T_Bx. W(s, i) only needs B(s, i) on the same stage. Rule used by
this script (the author's simple policy, not a cell-by-cell copy of the ZB paper's figure):
    - if the next F/B in the 1F1B order is not ready yet, run the oldest pending W instead;
    - stage s may keep at most s pending W's, otherwise it runs a W right away.
The second rule keeps every stage at no more than P micro-batches of stored activations
(a micro-batch's activations are freed when its W is done), the same peak as 1F1B.
The script checks that every stage idles exactly (P-1)(T_F+T_B-2T_W) = 3 units, i.e. the
ZB1P / ZB-H1 formula (12.17), and that the peak stays at P.
Writes deepseek/figures/l12-zero-bubble.svg
"""

from l12_svg import SVG

P, M, TF, TBX, TW = 4, 8, 1, 1, 1
TB = TBX + TW


def order_1f1b(s):
    w = min(P - 1 - s, M)
    ops = [("F", i) for i in range(w)]
    f, b = w, 0
    while f < M:
        ops += [("F", f), ("B", b)]
        f += 1
        b += 1
    return ops + [("B", i) for i in range(b, M)]


def dep_of(op, s, i):
    if op == "F":
        return ("F", s - 1, i) if s > 0 else None
    return ("B", s + 1, i) if s < P - 1 else ("F", s, i)


def simulate(split):
    """Event-driven simulation. split=False: B is the full backward (T_B); True: B = T_Bx, plus W."""
    orders = [order_1f1b(s) for s in range(P)]
    done, sched = {}, [[] for _ in range(P)]
    ptr, free, pend = [0] * P, [0] * P, [[] for _ in range(P)]
    n_ops = P * M * (3 if split else 2)
    count, t = 0, 0
    while count < n_ops:
        for s in range(P):
            if free[s] > t:
                continue
            nxt = orders[s][ptr[s]] if ptr[s] < len(orders[s]) else None
            ready = False
            if nxt:
                dep = dep_of(nxt[0], s, nxt[1])
                ready = dep is None or done.get(dep, 1e9) <= t
            if split and pend[s] and (len(pend[s]) > s or not ready):
                i = pend[s].pop(0)
                done[("W", s, i)] = t + TW
                sched[s].append(("W", i, t, t + TW))
                free[s] = t + TW
                count += 1
            elif ready:
                op, i = nxt
                d = TF if op == "F" else (TBX if split else TB)
                done[(op, s, i)] = t + d
                sched[s].append((op, i, t, t + d))
                free[s] = t + d
                ptr[s] += 1
                count += 1
                if op == "B" and split:
                    pend[s].append(i)
        later = [v for v in list(free) + list(done.values()) if v > t]
        if count < n_ops:
            t = min(later)
    mk = max(free)
    idle = [mk - sum(e - b for *_, b, e in row) for row in sched]
    peaks = []
    for row in sched:
        free_op = "W" if split else "B"
        ev = sorted([(e, 1) for op, _, _, e in row if op == "F"] +
                    [(e, -1) for op, _, _, e in row if op == free_op], key=lambda x: (x[0], x[1]))
        cur = best = 0
        for _, dlt in ev:
            cur += dlt
            best = max(best, cur)
        peaks.append(best)
    return sched, mk, idle, peaks


base = simulate(False)
zb = simulate(True)
print("1F1B  makespan", base[1], "idle", base[2], "peaks", base[3])
print("ZB-H1 makespan", zb[1], "idle", zb[2], "peaks", zb[3])
assert all(v == (P - 1) * (TF + TB) for v in base[2])
assert all(v == (P - 1) * (TF + TB - 2 * TW) for v in zb[2])
assert max(zb[3]) == P == max(base[3])

W, H = 760, 414
X0, U, RH = 76, 18, 26
COL = {"F": "blue", "B": "green", "W": "violet"}


def panel(y0, res, title, sub, split):
    sched, mk, idle, peaks = res
    svg.text(16, y0 - 12, title, size=13, weight="600")
    svg.text(16 + 13 * len(title) + 18, y0 - 12, sub, size=12, color="muted")
    for s in range(P):
        y = y0 + s * RH
        svg.text(X0 - 10, y + 17, "GPU %d" % s, size=12, anchor="end")
        svg.rect(X0, y, mk * U, RH - 4, fill="red", fill_opacity=0.10)
        for op, i, st, en in sched[s]:
            col = COL[op]
            svg.rect(X0 + st * U, y, (en - st) * U, RH - 4, fill=col, fill_opacity=0.22, stroke=col, width=0.9)
            lab = ("B%d" % i) if (op == "B" and not split) else "%s%d" % (op, i)
            svg.text(X0 + (st + en) / 2 * U, y + 15, lab, size=11, anchor="middle", color=col)
        svg.text(X0 + mk * U + 8, y + 16, "空闲 %g" % idle[s], size=12, color="red")
    return mk


svg = SVG(W, H, "把反向拆成两半：对权重的那一半 W 推迟去填气泡，每段空闲从 9 降到 3")
mk0 = panel(34, base, "1F1B", "反向不拆（B 占 2 格）", False)
mk1 = panel(34 + P * RH + 50, zb, "拆开反向", "B 只对输入（1 格），W 对权重（1 格），W 插空做", True)
# where the second panel ends: a short tick above its last column (t = mk1)
y_tick = 34 + P * RH + 50 - 8
svg.line(X0 + mk1 * U, y_tick - 2, X0 + mk1 * U, y_tick + 5, color="ink", width=1.2)
svg.text(X0 + mk1 * U, y_tick - 6, "%g" % mk1, size=12, anchor="middle")
ay = 34 + 2 * P * RH + 70
for t in range(0, mk0 + 1, 3):
    svg.line(X0 + t * U, ay - 6, X0 + t * U, ay, color="muted", width=1)
    svg.text(X0 + t * U, ay + 14, str(t), size=11, anchor="middle", color="muted")
svg.line(X0, ay - 6, X0 + mk0 * U, ay - 6, color="muted", width=1.2)
svg.text(X0 + mk0 * U, ay + 30, "时间（单位 = 一次前向 T_{F}；下面板的 B、W 各 1 格）", size=12, anchor="end", color="muted")
lx, ly = 16, ay + 50
for col, lab in [("blue", "F 前向"), ("green", "B 反向（上：完整；下：只对输入）"), ("violet", "W 对权重求梯度"),
                 ("red", "气泡")]:
    svg.rect(lx, ly - 11, 14, 12, fill=col, fill_opacity=0.22 if col != "red" else 0.10,
             stroke=None if col == "red" else col, width=0.9)
    svg.text(lx + 20, ly, lab, size=12)
    lx += 34 + 12.5 * len(lab)
svg.text(16, ly + 24, "两块面板每段同时存着的激活都不超过 4 份（W 做完，这个 micro-batch 的激活才能释放）。",
         size=12, color="muted")
svg.text(16, ly + 43, "每段空闲：(P − 1)(T_{F} + T_{B}) = 3 × 3 = 9   →   (P − 1)(T_{F} + T_{B} − 2T_{W}) = 3 × 1 = 3",
         size=12, color="muted")
svg.save("l12-zero-bubble.svg")
