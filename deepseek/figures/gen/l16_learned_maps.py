"""Figure 16.13: the residual mappings the 27B models actually learned (redrawn after mHC Fig. 8).

Values: the two-decimal numbers printed on mHC (arXiv 2512.24880 v2) Fig. 8, transcribed by the research notes'
author (section 5.3 item 10; cross-checked against the printed row/column-sum labels). Each matrix is an average
over the tokens of one selected sequence. Top row HC, bottom row mHC: single sublayers 1, 30, 60 and the composites
over sublayers 1..30, 31..60, 1..60. Below each matrix: max |row sum| (forward) and max |column sum| (backward)
as printed on the figure labels (row sums shown for HC only; mHC rows sum to 1 by construction).
Also printed: singular values of the HC 1..60 composite (nearly rank 1) and max |P - J_4| for mHC 1..60 (notes N29).
Writes deepseek/figures/l16-learned-maps.svg
"""

import math

import numpy as np

from l16_svg import SVG

HC = {
    "H_{1}": [[5.43, 4.43, 4.43, 4.43], [-4.07, -3.07, -4.07, -4.07], [-3.95, -3.95, -2.95, -3.95],
              [-4.22, -4.22, -4.22, -3.22]],
    "H_{30}": [[0.94, -0.07, -0.05, 0.02], [-0.08, 0.89, -0.07, -0.07], [-0.10, -0.14, 0.81, -0.07],
               [0.06, 0.05, -0.03, 0.87]],
    "H_{60}": [[-5.58, -3.74, -5.71, -6.60], [-6.06, -2.27, -5.33, -6.57], [6.08, 3.12, 6.53, 6.77],
               [-6.41, -3.97, -5.72, -5.49]],
    "∏ 1..30": [[-0.38, -0.33, -0.34, -0.31], [1.81, 1.56, 1.58, 1.51], [0.01, -0.00, 0.01, 0.01],
                [-0.23, -0.19, -0.20, -0.19]],
    "∏ 31..60": [[-69.9, -68.3, -255.3, 142.1], [-69.1, -66.1, -247.4, 139.6], [74.8, 72.7, 268.9, -151.8],
                 [-71.2, -71.8, -255.2, 143.3]],
    "∏ 1..60": [[-132.8, -112.2, -117.0, -113.3], [-129.3, -109.3, -113.9, -110.3], [142.3, 120.2, 125.3, 121.3],
                [-139.3, -117.8, -122.6, -118.7]],
}
MHC = {
    "H_{1}": [[0.67, 0.09, 0.03, 0.22], [0.26, 0.48, 0.26, 0.00], [0.03, 0.24, 0.00, 0.73], [0.03, 0.20, 0.69, 0.09]],
    "H_{30}": [[0.96, 0.01, 0.00, 0.04], [0.00, 0.97, 0.03, 0.00], [0.00, 0.00, 1.00, 0.00], [0.00, 0.04, 0.01, 0.95]],
    "H_{60}": [[0.92, 0.06, 0.01, 0.01], [0.05, 0.81, 0.01, 0.13], [0.00, 0.01, 0.97, 0.02], [0.03, 0.13, 0.00, 0.84]],
    "∏ 1..30": [[0.30, 0.25, 0.22, 0.24], [0.24, 0.25, 0.25, 0.26], [0.20, 0.24, 0.28, 0.28],
                [0.17, 0.32, 0.18, 0.34]],
    "∏ 31..60": [[0.35, 0.28, 0.17, 0.20], [0.03, 0.62, 0.29, 0.07], [0.01, 0.17, 0.80, 0.02],
                 [0.02, 0.42, 0.25, 0.31]],
    "∏ 1..60": [[0.24, 0.26, 0.23, 0.27], [0.23, 0.25, 0.26, 0.27], [0.21, 0.25, 0.27, 0.28],
                [0.21, 0.27, 0.24, 0.29]],
}

# max |row sum| and max |column sum| as PRINTED on the figure's labels (notes section 5.3 item 10)
LABEL_HC = {"H_{1}": ("18.73", "6.81"), "H_{30}": ("0.96", "0.83"), "H_{60}": ("22.50", "11.97"),
            "∏ 1..30": ("6.47", "1.22"), "∏ 31..60": ("264.6", "489.0"), "∏ 1..60": ("509.1", "259.2")}
LABEL_MHC = {"H_{1}": "1.04", "H_{30}": "1.04", "H_{60}": "1.01", "∏ 1..30": "1.11", "∏ 31..60": "1.50",
             "∏ 1..60": "1.11"}
P = np.array(HC["∏ 1..60"])
print("HC prod 1..60 singular values:", np.round(np.linalg.svd(P)[1], 3))
Q = np.array(MHC["∏ 1..60"])
print("mHC prod 1..60 max |P - J4| = %.2f" % abs(Q - 0.25).max())
for name, M in list(HC.items()) + list(MHC.items()):
    A = np.array(M)
    print("%-9s fwd %.2f  bwd %.2f" % (name, abs(A.sum(1)).max(), abs(A.sum(0)).max()))

W, H = 760, 450
svg = SVG(W, H, "27B 模型学到的残差映射：HC 与 mHC 各 6 个 4×4 矩阵（据 mHC 论文图 8 重绘）")
CW, CH, GAP, X0 = 29, 22, 10, 12


def fmt_hc(v):
    a = abs(v)
    if a >= 99.5:
        return "%d" % round(v)
    if a >= 9.95:
        return "%.0f" % v
    s = "%.1f" % v
    return "0" if s in ("-0.0", "0.0") else s


def draw_row(y, mats, kind):
    for k, (name, M) in enumerate(mats.items()):
        x = X0 + k * (4 * CW + GAP)
        svg.text(x + 2 * CW, y - 8, name, size=12, anchor="middle", weight="600")
        A = np.array(M)
        for i in range(4):
            for j in range(4):
                v = A[i, j]
                if kind == "hc":
                    op = 0.10 + 0.70 * math.log10(1 + abs(v)) / math.log10(1 + 270)
                    col = "blue" if v >= 0 else "red"
                    txt = fmt_hc(v)
                else:
                    op = 0.06 + 0.74 * v
                    col = "green"
                    txt = "%.2f" % v
                svg.rect(x + j * CW + 0.5, y + i * CH + 0.5, CW - 1, CH - 1, fill=col, fill_opacity=op, rx=2)
                svg.text(x + j * CW + CW / 2, y + i * CH + CH / 2 + 4, txt, size=11, anchor="middle")
        if kind == "hc":
            fw, bw = LABEL_HC[name]
            svg.text(x + 2 * CW, y + 4 * CH + 16, "前 %s · 反 %s" % (fw, bw), size=11, anchor="middle",
                     color="red" if max(float(fw), float(bw)) > 2 else "muted")
        else:
            svg.text(x + 2 * CW, y + 4 * CH + 16, "反 %s" % LABEL_MHC[name], size=11, anchor="middle", color="muted")


svg.text(12, 26, "(a) HC：任意实数矩阵（蓝正、红负，颜色深浅按 log(1+|值|)）", size=14, weight="700")
draw_row(58, HC, "hc")
svg.text(12, 218, "(b) mHC：双随机矩阵（绿色深浅 = 数值，0 到 1）", size=14, weight="700")
draw_row(250, MHC, "mhc")
svg.text(12, 396, "每个矩阵都是对一条序列所有 token 的平均。“前/反”= 最大 |行和| / 最大 |列和|，取自论文图上印出的标签。", size=12,
         color="muted")
svg.text(12, 416, "HC 的 60 层之积几乎是秩 1（奇异值 488、0.15、0.06、0.003）；mHC 的 60 层之积与 J_{4}（全 0.25）最大只差 0.04。",
         size=12)
svg.text(12, 436, "mHC 只标反向：行和按构造为 1（∏ 1..60 印出的行和为 1.00、1.01、1.01、1.00）。", size=12, color="muted")

svg.save("l16-learned-maps.svg")
