"""Figure 12.10: pipeline bubbles from scratch - the naive schedule vs 1F1B, simulated.

P = 4 pipeline stages (one per GPU), m = 8 micro-batches, forward chunk T_F = 1 unit,
full backward chunk T_B = 2 units, communication ignored. Each stage executes its own list of
operations in order; an operation starts when the stage is free and its dependency is done:
    F(s, i) needs F(s-1, i);   B(s, i) needs B(s+1, i) (the last stage needs its own F(P-1, i)).
Naive order on every stage: F0..F7 then B0..B7.  1F1B (PipeDream-style): P-1-s warm-up
forwards, then alternate one forward and one backward, then drain the remaining backwards.
The script checks that both schedules idle (P-1)(T_F+T_B) = 9 units per stage and reports the
peak number of micro-batches whose activations are held (forward done, backward not yet).
Writes deepseek/figures/l12-pipeline-1f1b.svg
"""

from l12_svg import SVG

P, M, TF, TB = 4, 8, 1, 2


def orders(kind):
    out = []
    for s in range(P):
        if kind == "naive":
            ops = [("F", i) for i in range(M)] + [("B", i) for i in range(M)]
        else:
            w = min(P - 1 - s, M)
            ops = [("F", i) for i in range(w)]
            f, b = w, 0
            while f < M:
                ops += [("F", f), ("B", b)]
                f += 1
                b += 1
            ops += [("B", i) for i in range(b, M)]
        out.append(ops)
    return out


def simulate(kind):
    ords = orders(kind)
    done = {}                       # (op, s, i) -> finish time
    sched = [[] for _ in range(P)]  # (op, i, start, end)
    free = [0] * P
    ptr = [0] * P
    while any(ptr[s] < len(ords[s]) for s in range(P)):
        progressed = False
        for s in range(P):
            if ptr[s] >= len(ords[s]):
                continue
            op, i = ords[s][ptr[s]]
            if op == "F":
                dep = ("F", s - 1, i) if s > 0 else None
            else:
                dep = ("B", s + 1, i) if s < P - 1 else ("F", s, i)
            if dep is not None and dep not in done:
                continue
            start = max(free[s], done[dep] if dep else 0)
            end = start + (TF if op == "F" else TB)
            done[(op, s, i)] = end
            sched[s].append((op, i, start, end))
            free[s] = end
            ptr[s] += 1
            progressed = True
        assert progressed, "deadlock"
    makespan = max(free)
    return sched, makespan


def peak_inflight(rows):
    ev = []
    for op, i, st, en in rows:
        if op == "F":
            ev.append((en, 1))
        else:
            ev.append((en, -1))
    ev.sort()
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


res = {}
for kind in ("naive", "1f1b"):
    sched, mk = simulate(kind)
    idle = [mk - sum(e - s for _, _, s, e in row) for row in sched]
    peaks = [peak_inflight(row) for row in sched]
    res[kind] = (sched, mk, idle, peaks)
    print(kind, "makespan", mk, "idle per stage", idle, "peak activations", peaks)
    assert all(v == (P - 1) * (TF + TB) for v in idle)
assert res["naive"][3] == [M] * P and res["1f1b"][3] == [P - s for s in range(P)]

W, H = 760, 388
X0, U, RH = 76, 18, 26


def panel(y0, kind, title, sub):
    sched, mk, idle, peaks = res[kind]
    svg.text(16, y0 - 12, title, size=13, weight="600")
    svg.text(16 + 13 * len(title) + 18, y0 - 12, sub, size=12, color="muted")
    for s in range(P):
        y = y0 + s * RH
        svg.text(X0 - 10, y + 17, "GPU %d" % s, size=12, anchor="end")
        svg.rect(X0, y, mk * U, RH - 4, fill="red", fill_opacity=0.10)
        for op, i, st, en in sched[s]:
            col = "blue" if op == "F" else "green"
            svg.rect(X0 + st * U, y, (en - st) * U, RH - 4, fill=col, fill_opacity=0.22, stroke=col, width=0.9)
            svg.text(X0 + (st + en) / 2 * U, y + 15, "%s%d" % (op, i), size=11, anchor="middle", color=col)
        svg.text(X0 + mk * U + 8, y + 16, "峰值 %d 份" % peaks[s], size=12, color="amber")
    return mk


svg = SVG(W, H, "流水线气泡：朴素调度与 1F1B 的空闲一样多，但 1F1B 同时存着的激活少得多")
panel(34, "naive", "朴素调度", "先做完 8 个前向，再做 8 个反向")
mk = panel(34 + P * RH + 50, "1f1b", "1F1B", "预热几次前向后，一前一后交替")
ay = 34 + 2 * P * RH + 70
for t in range(0, mk + 1, 3):
    svg.line(X0 + t * U, ay - 6, X0 + t * U, ay, color="muted", width=1)
    svg.text(X0 + t * U, ay + 14, str(t), size=11, anchor="middle", color="muted")
svg.line(X0, ay - 6, X0 + mk * U, ay - 6, color="muted", width=1.2)
svg.text(X0 + mk * U, ay + 30, "时间（单位 = 一次前向 T_{F}；反向 T_{B} = 2）", size=12, anchor="end", color="muted")
svg.rect(16, ay + 44, 14, 12, fill="red", fill_opacity=0.10)
svg.text(36, ay + 55, "气泡（空闲）：每块 GPU 都空等 (P − 1)(T_{F} + T_{B}) = 3 × 3 = 9 个单位；右侧是同时存着的激活份数",
         size=12)
svg.save("l12-pipeline-1f1b.svg")
