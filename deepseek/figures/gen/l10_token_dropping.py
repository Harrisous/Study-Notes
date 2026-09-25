"""Figure 10.7: device-level token dropping with capacity factor 1.0 (toy numbers, author's construction).

Four devices receive 14, 10, 9, 7 (token, expert) assignments in one step; the average, 10, is the budget
(capacity factor 1.0, V2 sec. 2.2.4). Device 1 is over budget by 4 and drops its 4 lowest-affinity
assignments, skipping one that belongs to a never-drop sequence (V2 keeps ~10 % of training sequences
undropped). Dropped share 4 / 40 = 10 %; the slowest device goes from 14 to 10 units of work.
Writes deepseek/figures/l10-token-dropping.svg
"""

from l10_svg import SVG

loads = [14, 10, 9, 7]
aff1 = [0.31, 0.27, 0.22, 0.19, 0.17, 0.15, 0.12, 0.11, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03]
protected = {0.04}
cap = sum(loads) / len(loads)
assert cap == 10
over = loads[0] - int(cap)
droppable = sorted([a for a in aff1 if a not in protected])
dropped = set(droppable[:over])
assert sorted(dropped) == [0.03, 0.05, 0.06, 0.08]
kept = sorted([a for a in aff1 if a not in dropped], reverse=True)
kept = [a for a in kept if a not in protected] + [a for a in kept if a in protected]
stack1 = kept + sorted(dropped, reverse=True)

W, H = 760, 440
svg = SVG(W, H, "设备级 token 丢弃：容量 = 平均负载（容量因子 1.0）")
X0, BASE, SQH, SQW, PITCH = 92, 400, 24, 62, 92
for k, n in enumerate(loads):
    x = X0 + k * PITCH
    for j in range(n):
        y = BASE - (j + 1) * SQH
        if k == 0:
            a = stack1[j]
            if a in dropped:
                svg.rect(x, y + 1, SQW, SQH - 2, fill="red", fill_opacity=0.16, stroke="red", width=1.2, rx=3,
                         dash="3 2")
                svg.text(x + SQW / 2, y + 16, "%.2f 丢" % a, size=11, anchor="middle", color="red")
            elif a in protected:
                svg.rect(x, y + 1, SQW, SQH - 2, fill="amber", fill_opacity=0.18, stroke="amber", width=1.4, rx=3)
                svg.text(x + SQW / 2, y + 16, "%.2f 保" % a, size=11, anchor="middle", color="amber")
            else:
                svg.rect(x, y + 1, SQW, SQH - 2, fill="blue", fill_opacity=0.16, stroke="blue", width=1, rx=3)
                svg.text(x + SQW / 2, y + 16, "%.2f" % a, size=11, anchor="middle")
        else:
            svg.rect(x, y + 1, SQW, SQH - 2, fill="blue", fill_opacity=0.16, stroke="blue", width=1, rx=3)
    svg.text(x + SQW / 2, BASE + 18, "设备 %d" % (k + 1), size=12, anchor="middle")
    svg.text(x + SQW / 2, BASE - n * SQH - 8, ("%d → 10" % n) if n > cap else "%d" % n, size=13,
             anchor="middle", weight="600", color="red" if n > cap else "ink")
ycap = BASE - cap * SQH
svg.line(X0 - 14, ycap, X0 + 3 * PITCH + SQW + 14, ycap, color="ink", width=1.6, dash="6 4")
svg.text(X0 - 18, ycap + 4, "容量 10", size=12, anchor="end", weight="600")
svg.line(X0 - 10, BASE, X0 + 3 * PITCH + SQW + 10, BASE, color="muted", width=1.2)

RX = 456
lines = [
    ("一步里 4 台设备收到的（token, 专家）分配：", None),
    ("14 + 10 + 9 + 7 = 40，平均 10", "ink"),
    ("容量因子 1.0 ⇒ 每台的预算 = 平均 = 10", "ink"),
    ("", None),
    ("设备 1 超出 4 个：按亲和度从低到高丢", None),
    ("0.03、0.05、0.06、0.08（红）", "red"),
    ("0.04 属于永不丢弃的序列：跳过（黄）", "amber"),
    ("", None),
    ("丢弃比例 4 / 40 = 10%", "ink"),
    ("最慢设备的工作量 14 → 10（少 29%）", "ink"),
    ("设备 3、4 仍闲着：丢弃只削峰，不填谷", "muted"),
]
y = 56
for s, colr in lines:
    if s:
        svg.text(RX, y, s, size=13, color=colr or "ink", weight="600" if colr in ("ink", "red", "amber") else None)
    y += 26 if s else 12
svg.text(RX, 382, "玩具数字；每个方块是一次（token, 专家）计算，", size=12, color="muted")
svg.text(RX, 400, "方块里的数是它的亲和度 s_{i,t}", size=12, color="muted")
svg.save("l10-token-dropping.svg")
