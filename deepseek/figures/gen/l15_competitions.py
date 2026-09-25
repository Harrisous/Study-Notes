"""Figure 15.14: competition results of DeepSeek-V3.2-Speciale and DeepSeekMath-V2 against official cut-offs.

Speciale (V3.2 paper Table 4): IMO 2025 7,7,7,7,7,0; CMO 2025 18,18,9,21,18,18; IOI 2025 100,82,72,100,55,83;
ICPC WF 2025 10 of 12 problems solved (14 submissions in total).
Official data: IMO 2025 gold cut-off 35 (imo-official.org); IOI 2025 gold cut-off 438.30, 9th / 10th place
502.78 / 491.65 (stats.ioinformatics.org); ICPC WF 2025 winner solved 11, 2nd and 3rd solved 10.
DeepSeekMath-V2 (its Table 1): IMO 2025 5 of 6 solved (83.3%), CMO 2024 73.8% (= 93 of 126, reverse-computed),
Putnam 2024 118 of 120 (best human 90). CMO gold cut-offs were not verified, so no line is drawn.
Each bar is on its own scale (fraction of the maximum score).
Writes deepseek/figures/l15-competitions.svg
"""

from l15_svg import SVG

IMO = [7, 7, 7, 7, 7, 0]
CMO25 = [18, 18, 9, 21, 18, 18]
IOI = [100, 82, 72, 100, 55, 83]
assert sum(IMO) == 35 and sum(CMO25) == 102 and sum(IOI) == 492
assert abs(sum(IOI) - 438.30 - 53.70) < 1e-9 and 491.65 < 492 < 502.78
assert round(0.738 * 126) == 93 and abs(35 / 42 - 0.8333) < 1e-4 and abs(118 / 120 - 0.98333) < 1e-4

W, H = 760, 540
svg = SVG(W, H, "V3.2-Speciale 与 DeepSeekMath-V2 的竞赛成绩，对照官方奖牌线")
BX0, BX1 = 196, 596


def bx(frac):
    return BX0 + frac * (BX1 - BX0)


def stacked(y, parts, total, label, sub, right, lines=()):
    svg.text(16, y + 14, label, size=12, weight="700")
    svg.text(16, y + 30, sub, size=11, color="muted")
    svg.rect(BX0, y, BX1 - BX0, 22, fill="muted", fill_opacity=0.06, stroke="rule", width=1)
    acc = 0.0
    for k, v in enumerate(parts):
        if v:
            svg.rect(bx(acc / total), y, bx((acc + v) / total) - bx(acc / total), 22, fill="blue", fill_opacity=0.40,
                     stroke="blue", width=1)
            if (bx((acc + v) / total) - bx(acc / total)) > 26:
                svg.text((bx(acc / total) + bx((acc + v) / total)) / 2, y + 15, str(v), size=11, anchor="middle")
        acc += v
    for val, lab, col, below in lines:
        x = bx(val / total)
        svg.line(x, y - 6, x, y + 28, color=col, width=2)
        svg.text(x, y + 42 if below else y - 9, lab, size=11, anchor="middle", color=col)
    svg.text(BX1 + 12, y + 16, right, size=12, weight="700")


svg.text(16, 22, "DeepSeek-V3.2-Speciale（V3.2 论文表 4；无工具、无联网、最长生成 128k）", size=13, weight="700")
stacked(46, IMO, 42, "IMO 2025", "6 题 × 7 分", "35 / 42", [(35, "金牌线 35", "green", True)])
stacked(106, CMO25, 126, "CMO 2025", "6 题 × 21 分（英文版）", "102 / 126")
stacked(166, IOI, 600, "IOI 2025", "6 题 × 100 分", "492 / 600",
        [(438.30, "金牌线 438.3", "green", True), (502.78, "第 9 名", "muted", False)])
# ICPC: 12 cells
y = 232
svg.text(16, y + 14, "ICPC WF 2025", size=12, weight="700")
svg.text(16, y + 30, "12 题，按解题数排名", size=11, color="muted")
solved = set("ACDEFHIJKL")
cw = (BX1 - BX0) / 12
for k, ch in enumerate("ABCDEFGHIJKL"):
    on = ch in solved
    svg.rect(BX0 + k * cw + 1, y, cw - 2, 22, fill="blue" if on else "muted", fill_opacity=0.40 if on else 0.06,
             stroke="blue" if on else "rule", width=1)
    svg.text(BX0 + k * cw + cw / 2, y + 15, ch, size=11, anchor="middle", color="ink" if on else "muted")
svg.text(BX0, y + 42, "解出 10 题（共提交 14 次）；官方记分牌：第 1 名解 11 题，第 2、3 名各解 10 题", size=11, color="muted")
svg.text(BX1 + 12, y + 16, "10 / 12", size=12, weight="700")

svg.line(12, 292, 748, 292, color="rule", width=1)
svg.text(16, 314, "DeepSeekMath-V2（其论文表 1；放大测试时计算，自家专家评分）", size=13, weight="700")
stacked(334, IMO, 42, "IMO 2025", "5 / 6 题完全解出", "35 / 42", [(35, "金牌线 35", "green", True)])


def plain(y, frac, label, sub, right, marks=(), dashed=False):
    svg.text(16, y + 14, label, size=12, weight="700")
    svg.text(16, y + 30, sub, size=11, color="muted")
    svg.rect(BX0, y, BX1 - BX0, 22, fill="muted", fill_opacity=0.06, stroke="rule", width=1)
    svg.rect(BX0, y, bx(frac) - BX0, 22, fill="blue", fill_opacity=0.25 if dashed else 0.40, stroke="blue", width=1,
             dash="4 3" if dashed else None)
    for val, lab, col in marks:
        svg.line(bx(val), y - 6, bx(val), y + 28, color=col, width=2)
        svg.text(bx(val), y - 9, lab, size=11, anchor="middle", color=col)
    svg.text(BX1 + 12, y + 16, right, size=12, weight="700")


plain(398, 0.738, "CMO 2024", "4 题满分 + 1 题部分分", "73.8%", dashed=True)
svg.text(BX1 + 12, 434, "≈ 93 / 126（笔者反推）", size=11, color="muted")
plain(466, 118 / 120, "Putnam 2024", "12 题 × 10 分", "118 / 120", marks=[(90 / 120, "人类最高 90", "muted")])
svg.text(16, 532, "每根条按各自的满分画成比例。CMO 的官方金牌线未能核实，所以不画线；IOI 第 10 名为 491.65。", size=11, color="muted")

svg.save("l15-competitions.svg")
