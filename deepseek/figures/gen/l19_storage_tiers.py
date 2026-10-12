"""Figure 19.11: where each kind of KV lives, V4 deployment vs V4.1 deployment, plus the 1/8 bar.

Facts (notes C33-C35, 4-D): V4 keeps global KV and SWA KV (saved at the end of the prompt and of the
output) in an SSD persistent cache sized for >72 h residency, SWA ~ half of it; V4.1 keeps only global KV
there (>= 72 h) and puts SWA KV in a distributed pool made of 10% of each machine's host DRAM (TTL minutes).
Bottom bars: persistent bytes per cached token relative to V4's global KV, V4 = G + S with S ~ G (the
paper's "nearly half" workload statistic), V4.1 = G41 / G4F = 890 / 3514.25. Computed from l19_num.
Writes deepseek/figures/l19-storage-tiers.svg
"""

from l19_svg import SVG
from l19_num import G41, G4F, SWA_STATE_4F, SWA_STATE_41

W, H = 760, 450
svg = SVG(W, H, "三层存储里各放什么：V4 部署与 V4.1 部署")

LX, C1, C2, CW = 20, 110, 438, 312
ROWS = [("GPU HBM", 44, 86), ("主机内存", 142, 86), ("SSD", 240, 86)]
svg.text(C1 + CW / 2, 30, "V4-Flash 部署", size=13, anchor="middle", weight="700")
svg.text(C2 + CW / 2, 30, "V4.1-Flash 部署", size=13, anchor="middle", weight="700")
for name, y, h in ROWS:
    svg.text(LX, y + h / 2 + 4, name, size=13, weight="700", color="amber")
    svg.rect(C1, y, CW, h, fill="amber", fill_opacity=0.07, stroke="amber", width=1, rx=6)
    svg.rect(C2, y, CW, h, fill="amber", fill_opacity=0.07, stroke="amber", width=1, rx=6)


def lines(x, y, rows):
    for i, (t, col) in enumerate(rows):
        svg.text(x + 12, y + 22 + i * 19, t, size=12, color=col)


mib = lambda b: b / 2 ** 20
lines(C1, 44, [("全局 KV：%s 字节/token" % "{:,.2f}".format(G4F), "ink"),
               ("SWA 状态：%.2f MiB/请求（43 层 × 128 × 584 B）" % mib(SWA_STATE_4F), "ink"),
               ("活跃请求才在这里", "muted")])
lines(C2, 44, [("全局 KV：%d 字节/token" % G41, "ink"),
               ("SWA 状态：%.2f MiB/请求（40 层 × 128 × 528 B）" % mib(SWA_STATE_41), "ink"),
               ("活跃请求才在这里", "muted")])
lines(C1, 142, [("（论文没有描述这一层的 KV 用途）", "muted")])
lines(C2, 142, [("分布式 SWA 池：每台机器拿出 10% 的主机内存", "ink"),
                ("只存编码器的 SWA KV；寿命几分钟，过期立即回收", "ink"),
                ("未命中 → 只回放最近 128 个 token", "red"),
                ("Engram 表也放在主机内存，经 RDMA 预取", "muted")])
lines(C1, 240, [("持久化 KV：全局 KV + SWA KV", "ink"),
                ("SWA 在“提示结尾”“输出结尾”两处保存", "ink"),
                ("SWA 约占一半容量；典型负载下都能驻留 72 小时以上", "muted")])
lines(C2, 240, [("持久化 KV：只有全局 KV", "ink"),
                ("保证至少驻留 72 小时", "ink"),
                ("前缀命中时从这里读回 HBM", "muted")])

# ---------------------------------------------------------------- bottom: persistent bytes
BY = 352
svg.text(LX, BY, "持久化缓存的容量（以 V4-Flash 其中的全局 KV 部分为 1）", size=13, weight="700")
BX, SC = 150, 150.0
g4 = 1.0
s4 = 1.0                       # "nearly half" -> SWA ~ global
g41 = G41 / G4F
rows = [("V4-Flash", [(g4, "amber", 0.45, "全局 KV 1.0"), (s4, "muted", 0.35, "SWA ≈ 1.0")], "≈ 2.0"),
        ("V4.1-Flash", [(g41, "orange", 0.6, "")], "%.3f ≈ 1/%.1f" % (g41 / 2, 2 / g41))]
for i, (name, parts, tot) in enumerate(rows):
    y = BY + 18 + i * 38
    svg.text(BX - 10, y + 15, name, size=12, anchor="end")
    x = BX
    for v, col, op, lab in parts:
        svg.rect(x, y, v * SC, 22, fill=col, fill_opacity=op, stroke=col, width=0.8)
        if lab:
            svg.text(x + v * SC / 2, y + 15, lab, size=11, anchor="middle")
        x += v * SC
    svg.text(x + 8, y + 15, tot if i == 0 else "%.3f（相对 V4 的 2.0：%.3f，约 1/8）" % (g41, g41 / 2), size=12,
             color="ink" if i == 0 else "orange")
svg.text(BX + 2.0 * SC + 70, BY + 33, "× 1/2：SWA 不再落盘", size=12, color="muted")
svg.text(BX + 2.0 * SC + 70, BY + 51, "× %.3f：全局 KV 本身变小" % g41, size=12, color="muted")

svg.save("l19-storage-tiers.svg")
