"""Figure 17.7: sparsity allocation and memory scaling (redrawn after Engram fig. 3, arXiv v2).

(a) Validation loss vs allocation ratio rho (share of the inactive-parameter budget given to MoE
    experts), at C = 2e20 and 6e20 FLOPs with P_tot/P_act ~ 10. To put both compute points on one
    axis we plot L(rho) - L(rho = 100%). Only (100%, 1.7248) and (~80%, 1.7109) at 6e20 are numbers
    printed in the paper; all other points were read back from the vector figure scaling_law.svg
    (notes §4.9, about +-0.0001). Quadratic fits (numpy.polyfit) put the vertices at 0.738 and 0.724.
(b) "Infinite memory": fixed 3B / 568M MoE backbone, 100B tokens, Engram slots 2.58e5 .. 1.0e7.
    Engram (8 points), OverEncoding (4 points), pure MoE (1.8082), all read from the same figure.
    Least-squares line in log10(slots): -0.0284 loss per decade (log-log slope -0.00698).
Writes deepseek/figures/l17-allocation.svg
"""

import math

import numpy as np

from l17_svg import SVG

R6 = [39.7, 50.5, 60.2, 69.8, 75.2, 79.5, 84.9, 90.3, 94.6, 100]
L6 = [1.7247, 1.7196, 1.7136, 1.7122, 1.7126, 1.7109, 1.7095, 1.7125, 1.7130, 1.7248]
R2 = [40, 50, 60, 70, 75, 80, 85, 90, 95, 100]
L2 = [1.8275, 1.8227, 1.8190, 1.8172, 1.8177, 1.8170, 1.8155, 1.8172, 1.8201, 1.8292]
TEXT6 = {9: 1.7248, 5: 1.7109}            # indices of the two numbers printed in the paper text

ENG = [(2.586e5, 1.7883), (6.464e5, 1.7760), (1.293e6, 1.7678), (1.939e6, 1.7654), (2.586e6, 1.7603),
       (3.878e6, 1.7527), (6.464e6, 1.7481), (1.034e7, 1.7432)]
OVE = [(1.293e6, 1.7820), (2.586e6, 1.7711), (5.171e6, 1.7670), (1.293e7, 1.7505)]
MOE = 1.8082

fits = []
for r, l in ((R6, L6), (R2, L2)):
    a, b, c = np.polyfit(np.array(r) / 100, np.array(l), 2)
    fits.append((a, b, c, -b / (2 * a)))
assert round(fits[0][3], 3) == 0.738 and round(fits[1][3], 3) == 0.724
slope, icpt = np.polyfit(np.log10([s for s, _ in ENG]), [v for _, v in ENG], 1)
assert round(slope, 4) == -0.0284
assert round(L6[-1] - L6[5], 4) == 0.0139

W, H = 760, 722
svg = SVG(W, H, "稀疏分配的 U 形曲线与“无限内存”下的损失-槽数关系（据 Engram 论文图 3 重绘）")

# ================================================================= (a)
X0, X1, Y0, Y1 = 78, 610, 44, 300
RA, RB = 35, 102
DA, DB = -0.0165, 0.0015


def xa(r):
    return X0 + (r - RA) / (RB - RA) * (X1 - X0)


def ya(d):
    return Y1 - (d - DA) / (DB - DA) * (Y1 - Y0)


svg.text(20, 22, "(a) 固定总参数与激活参数，把不激活的参数在 MoE 专家与 Engram 之间分配", size=13, weight="700")
svg.rect(xa(75), Y0, xa(80) - xa(75), Y1 - Y0, fill="amber", fill_opacity=0.14)
svg.text(xa(77.5), Y0 + 14, "论文所说最优区", size=11.5, anchor="middle", color="amber")
svg.text(xa(77.5), Y0 + 30, "ρ ≈ 75%–80%", size=11.5, anchor="middle", color="amber")
for d in (-0.015, -0.010, -0.005, 0.0):
    svg.line(X0, ya(d), X1, ya(d), color="grid", width=1)
    svg.text(X0 - 8, ya(d) + 4, ("%+.3f" % d).replace("-", "−").replace("+0.000", "0"), size=11,
             anchor="end", color="muted")
for r in range(40, 101, 10):
    svg.line(xa(r), Y0, xa(r), Y1, color="grid", width=1)
    svg.text(xa(r), Y1 + 17, "%d%%" % r, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "分配比 ρ：不激活参数里给 MoE 专家的比例（100% = 纯 MoE）", size=12, anchor="middle",
         color="muted")
svg.text(X0 - 56, (Y0 + Y1) / 2, "相对纯 MoE 的损失变化", size=12, anchor="middle", color="muted", rotate=-90)

for (r, l, col, lab), (a, b, c, v) in zip(((R6, L6, "violet", "C = 6×10²⁰（9.9B / 993M）"),
                                           (R2, L2, "blue", "C = 2×10²⁰（5.7B / 568M）")), fits):
    base = l[-1]
    rr = [RA + 5 + i * 0.5 for i in range(int((100 - RA - 5) / 0.5) + 1)]
    svg.path("M" + " L".join("%.1f %.1f" % (xa(t), ya(a * (t / 100) ** 2 + b * t / 100 + c - base)) for t in rr),
             stroke=col, width=1.3, dash="5 4")
    for i, (ri, li) in enumerate(zip(r, l)):
        solid = col == "violet" and i in TEXT6
        if col == "violet":
            svg.circle(xa(ri), ya(li - base), 5 if solid else 4, fill=col if solid else "surface", stroke=col,
                       width=1.6)
        else:
            s = 4.4
            svg.polygon([(xa(ri), ya(li - base) - s), (xa(ri) + s, ya(li - base)), (xa(ri), ya(li - base) + s),
                         (xa(ri) - s, ya(li - base))], fill="surface", stroke=col, width=1.6)
    svg.line(xa(v * 100), ya(a * v * v + b * v + c - base) - 7, xa(v * 100), ya(a * v * v + b * v + c - base) + 7,
             color=col, width=2)

LX = 622
svg.circle(LX + 6, 70, 4.5, fill="surface", stroke="violet", width=1.6)
svg.text(LX + 16, 74, "6×10²⁰ FLOPs", size=12, color="violet", weight="700")
svg.text(LX + 16, 91, "9.9B / 993M", size=11.5, color="muted")
s = 4.4
svg.polygon([(LX + 6, 112 - s), (LX + 6 + s, 112), (LX + 6, 112 + s), (LX + 6 - s, 112)], fill="surface",
            stroke="blue", width=1.6)
svg.text(LX + 16, 116, "2×10²⁰ FLOPs", size=12, color="blue", weight="700")
svg.text(LX + 16, 133, "5.7B / 568M", size=11.5, color="muted")
svg.line(LX, 154, LX + 12, 154, color="ink", width=1.3, dash="5 4")
svg.text(LX + 16, 158, "二次拟合", size=11.5, color="muted")
svg.line(LX + 6, 172, LX + 6, 186, color="ink", width=2)
svg.text(LX + 16, 183, "拟合顶点", size=11.5, color="muted")
svg.text(LX, 212, "顶点 ρ：", size=11.5, color="muted")
svg.text(LX, 229, "%.3f（6×10²⁰）" % fits[0][3], size=11.5, color="violet")
svg.text(LX, 246, "%.3f（2×10²⁰）" % fits[1][3], size=11.5, color="blue")

svg.text(xa(100) - 6, ya(0) - 10, "纯 MoE", size=11.5, anchor="end", color="muted")
# starts right of the steep left arm of the violet fit and ends before the shaded band's label
svg.text(xa(44.5), ya(0) + 4, "ρ≈40% 时与纯 MoE 持平（46/43 个专家）", size=11.5, color="ink")
svg.text(xa(79.5) - 6, ya(L6[5] - L6[-1]) + 19, "1.7109", size=11.5, anchor="end", color="violet", weight="700")
svg.text(xa(84.9) + 6, ya(L6[6] - L6[-1]) + 4, "最低观测点在 ρ ≈ 85%", size=11.5, color="violet")

# ================================================================= (b)
BX0, BX1, BY0, BY1 = 78, 610, 400, 630
S0, S1 = 5.2, 7.25          # log10 slots
V0, V1 = 1.738, 1.812


def xb(sl):
    return BX0 + (math.log10(sl) - S0) / (S1 - S0) * (BX1 - BX0)


def yb(v):
    return BY1 - (v - V0) / (V1 - V0) * (BY1 - BY0)


svg.text(20, 376, "(b) 固定 3B / 568M 的 MoE 骨干，只加大 Engram 表：损失对槽数近似对数线性", size=13, weight="700")
for v in (1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.80, 1.81):
    svg.line(BX0, yb(v), BX1, yb(v), color="grid", width=1)
    svg.text(BX0 - 8, yb(v) + 4, "%.2f" % v, size=11, anchor="end", color="muted")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
for e in (5.5, 6, 6.5, 7):
    sl = 10 ** e
    svg.line(xb(sl), BY0, xb(sl), BY1, color="grid", width=1)
    lab = ("10" + str(int(e)).translate(SUP)) if e == int(e) else ("3×10" + str(int(e)).translate(SUP))
    if e == 5.5:
        lab = "3.2×10⁵"
    if e == 6.5:
        lab = "3.2×10⁶"
    svg.text(xb(sl), BY1 + 17, lab, size=11, anchor="middle", color="muted")
    svg.text(xb(sl), BY1 + 32, "%.2gB 参数" % (sl * 1280 / 1e9), size=11, anchor="middle", color="muted")
svg.line(BX0, BY1, BX1, BY1, color="muted", width=1.2)
svg.line(BX0, BY0, BX0, BY1, color="muted", width=1.2)
svg.text((BX0 + BX1) / 2, BY1 + 54, "Engram 槽数 S（对数轴；每槽 1280 维，第二行是换算的参数量）", size=12, anchor="middle",
         color="muted")
svg.text(BX0 - 56, (BY0 + BY1) / 2, "验证损失", size=12, anchor="middle", color="muted", rotate=-90)

svg.line(BX0, yb(MOE), BX1, yb(MOE), color="blue", width=1.6, dash="6 4")
svg.text(BX1 - 4, yb(MOE) - 7, "纯 MoE 骨干 1.8082", size=12, anchor="end", color="blue")

xs = [10 ** (S0 + 0.05), 10 ** (S1 - 0.05)]
svg.line(xb(xs[0]), yb(slope * math.log10(xs[0]) + icpt), xb(xs[1]), yb(slope * math.log10(xs[1]) + icpt),
         color="amber", width=1.3, dash="5 4")
svg.path("M" + " L".join("%.1f %.1f" % (xb(s_), yb(v)) for s_, v in OVE), stroke="violet", width=1.6)
for s_, v in OVE:
    svg.rect(xb(s_) - 4, yb(v) - 4, 8, 8, fill="surface", stroke="violet", width=1.6)
svg.path("M" + " L".join("%.1f %.1f" % (xb(s_), yb(v)) for s_, v in ENG), stroke="amber", width=2)
for s_, v in ENG:
    svg.circle(xb(s_), yb(v), 4.2, fill="amber", stroke="surface", width=1.2)

svg.text(LX, 470, "Engram", size=12, color="amber", weight="700")
svg.text(LX, 487, "每十倍槽数", size=11.5, color="amber")
svg.text(LX, 504, "损失 −%.4f" % -slope, size=11.5, color="amber")
svg.text(LX, 536, "OverEncoding", size=12, color="violet", weight="700")
svg.text(LX, 553, "（嵌入放在输入层）", size=11.5, color="muted")
svg.text(LX, 590, "双对数斜率", size=11.5, color="muted")
svg.text(LX, 607, "只有 −0.007", size=11.5, color="muted")

svg.text(20, 714, "除 (a) 中两个实心点（1.7248、1.7109）外，所有点都是笔者从 arXiv v2 图 3 的矢量坐标读出的（约 ±0.0001），不是论文正文数字。",
         size=11.5, color="muted")
svg.save("l17-allocation.svg")
