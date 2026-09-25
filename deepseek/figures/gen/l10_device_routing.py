"""Figure 10.4: one device-limited routing decision on a toy router (4 devices x 4 experts, K = 4, M = 2).

Affinities are the author's toy numbers from lesson-10-doc sec. 5.2 (they sum to 1, as a softmax would).
Device score = the largest affinity on the device (HF modeling_deepseek.py, group_limited_greedy).
The script recomputes every selection and asserts the numbers quoted in the lesson.
Writes deepseek/figures/l10-device-routing.svg
"""

from l10_svg import SVG


def num(v):
    return "%.2f" % v if abs(v * 100 - round(v * 100)) < 1e-9 else "%.3f" % v

S = {  # device -> affinities of its 4 experts (e1..e16 numbered device by device)
    "A": [0.20, 0.03, 0.025, 0.05],
    "B": [0.04, 0.12, 0.10, 0.015],
    "C": [0.15, 0.01, 0.02, 0.035],
    "D": [0.08, 0.055, 0.03, 0.04],
}
K, M = 4, 2
devs = list(S)
experts = [(dv, j, 4 * k + j + 1, S[dv][j]) for k, dv in enumerate(devs) for j in range(4)]
assert abs(sum(e[3] for e in experts) - 1) < 1e-9


def topk(pool, k):
    return sorted(pool, key=lambda e: -e[3])[:k]


unres = topk(experts, K)
score_max = {dv: max(S[dv]) for dv in devs}
dev_max = sorted(devs, key=lambda dv: -score_max[dv])[:M]
res = topk([e for e in experts if e[0] in dev_max], K)
score_top2 = {dv: sum(sorted(S[dv])[-2:]) for dv in devs}
dev_top2 = sorted(devs, key=lambda dv: -score_top2[dv])[:M]
alt = topk([e for e in experts if e[0] in dev_top2], K)
assert [e[2] for e in unres] == [1, 9, 6, 7] and abs(sum(e[3] for e in unres) - 0.57) < 1e-9
assert dev_max == ["A", "C"] and [e[2] for e in res] == [1, 9, 4, 12] and abs(sum(e[3] for e in res) - 0.435) < 1e-9
assert dev_top2 == ["A", "B"] and sorted(e[2] for e in alt) == [1, 4, 6, 7] and abs(sum(e[3] for e in alt) - 0.47) < 1e-9

W, H = 760, 468
svg = SVG(W, H, "设备受限路由的一次决策：4 台设备 × 4 个专家，K = 4，M = 2")
GX, GY, CW, CH = 64, 82, 84, 36
res_ids = {e[2] for e in res}
unres_ids = {e[2] for e in unres}

svg.text(30, 24, "玩具路由器：16 个专家的亲和度 s（和为 1）", size=14, weight="600")
for c, dv in enumerate(devs):
    x = GX + c * CW
    chosen = dv in dev_max
    if not chosen:
        svg.rect(x + 2, GY - 30, CW - 4, 4 * CH + 34, fill="muted", fill_opacity=0.10, rx=6)
    else:
        svg.rect(x + 2, GY - 30, CW - 4, 4 * CH + 34, fill="blue", fill_opacity=0.06, stroke="blue",
                 width=1.2, rx=6)
    svg.text(x + CW / 2, GY - 10, "设备 %s" % dv, size=13, anchor="middle",
             color="blue" if chosen else "muted", weight="600")
    for j in range(4):
        eid = 4 * c + j + 1
        y = GY + j * CH
        v = S[dv][j]
        if eid in res_ids:
            svg.rect(x + 8, y + 3, CW - 16, CH - 6, fill="blue", fill_opacity=0.22, stroke="blue", width=1.4, rx=4)
        if eid in unres_ids:
            svg.rect(x + 5, y + 1, CW - 10, CH - 2, stroke="ink", width=1.3, rx=5, dash="4 3")
        svg.text(x + 14, y + CH / 2 + 5, "e%d" % eid, size=12, color="muted")
        svg.text(x + CW - 14, y + CH / 2 + 5, num(v), size=13, anchor="end",
                 color="blue" if eid in res_ids else "ink", weight="600" if eid in res_ids else None)
# device score row
sy = GY + 4 * CH + 30
svg.text(GX - 8, sy, "组内最大", size=12, anchor="end", color="muted")
rank = {dv: i + 1 for i, dv in enumerate(sorted(devs, key=lambda d_: -score_max[d_]))}
for c, dv in enumerate(devs):
    svg.text(GX + c * CW + CW / 2, sy, "%.2f（第 %d）" % (score_max[dv], rank[dv]), size=12, anchor="middle",
             color="blue" if dv in dev_max else "muted", weight="600" if dv in dev_max else None)

# steps on the right
RX = 440
steps = [
    ("第 1 步", "每台设备打一个分："),
    ("", "设备上最高的那个亲和度"),
    ("第 2 步", "取分数最高的 M = 2 台：A、C"),
    ("第 3 步", "只在 A、C 的 8 个专家里取 Top-4"),
    ("", "（蓝底 = 最终选中）"),
    ("对照", "虚线框 = 不受限 Top-4 会选的专家"),
]
yy = 92
for head, body in steps:
    if head:
        svg.text(RX, yy, head, size=13, weight="600", color="blue" if head != "对照" else "ink")
    svg.text(RX + 58, yy, body, size=13)
    yy += 26
svg.text(RX, yy + 8, "设备 B 有两个强专家（0.12、0.10），", size=12, color="muted")
svg.text(RX, yy + 26, "但它的“最大值”只排第 3，整台被排除。", size=12, color="muted")


def chips(y, title, sel, note, col):
    svg.text(30, y, title, size=13, weight="600")
    x = 30
    for e in sorted(sel, key=lambda e: -e[3]):
        lab = "e%d  %s" % (e[2], num(e[3]))
        svg.rect(x, y + 10, 92, 26, fill=col, fill_opacity=0.14, stroke=col, width=1.1, rx=13)
        svg.text(x + 46, y + 28, lab, size=12, anchor="middle")
        x += 100
    svg.text(x + 10, y + 28, note, size=13, weight="600")


chips(300, "不受限 Top-4", unres, "门控和 %.2f · 跨 A、B、C 三台设备" % sum(e[3] for e in unres), "muted")
chips(358, "V2 的规则：按组内最大值选 2 台（A、C）", res, "门控和 %.3f · 只去 2 台" % sum(e[3] for e in res), "blue")
chips(416, "另一种规则：按组内前 2 名之和选 2 台（A 0.25、B 0.22）", alt,
      "门控和 %.2f · 只去 2 台" % sum(e[3] for e in alt), "violet")
svg.save("l10-device-routing.svg")
