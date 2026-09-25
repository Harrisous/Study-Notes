"""Figure 16.4: how to read the (n+1)x(n+1) Hyper-Connections matrix, its initialisation, and
Pre-Norm / Post-Norm / sequential / parallel arrangements as special cases.

Mechanism figure after HC (Zhu et al.) eq. (1), (14)-(19) and Fig. 2/4 (research notes section 2.2):
  HC = [[0, B], [A_m, A_r]];  H^pre = A_m^T, H^post = B, H^res = A_r^T  (notes: mapping derived by the
  author of the notes, shapes checked).
  init: B = 1, A_m = e_{k mod n}, A_r = I.   Pre-Norm (n=1): [[0,1],[1,1]].
  Post-Norm (n=1): [[0,c],[1,c]], c = 1/sqrt(s_i^2+s_o^2+2 s_io) = 1/std(x + F)
  (= 1/sqrt 2 = 0.707 when s_i=s_o=1, s_io=0; the lesson writes it as 1/std(x+F) to keep sigma for sigmoid).
  sequential (n=2): [[0,1,1],[1,1,0],[0,0,1]];  parallel: odd [[0,1,0],[1,1,1],[1,1,1]], even [[0,0,1],[0,1,0],[1,0,1]].
Writes deepseek/figures/l16-hc-matrix.svg
"""

from l16_svg import SVG

W, H = 760, 480
svg = SVG(W, H, "HC 矩阵的分块读法、初始化与四个特例")

# ---------------- (a) block reading
svg.text(20, 26, "(a) HC 矩阵的分块读法（n = 2，第 k 层）", size=14, weight="700")
GX, GY, CW, CH = 44, 62, 62, 40
cells = [["0", "β_{1}", "β_{2}"], ["α_{1,0}", "α_{1,1}", "α_{1,2}"], ["α_{2,0}", "α_{2,1}", "α_{2,2}"]]
tint = [["muted", "green", "green"], ["blue", "violet", "violet"], ["blue", "violet", "violet"]]
for i in range(3):
    for j in range(3):
        x, y = GX + j * CW, GY + i * CH
        svg.rect(x + 2, y + 2, CW - 4, CH - 4, fill=tint[i][j], fill_opacity=0.14 if (i, j) != (0, 0) else 0.08,
                 stroke=tint[i][j], width=1.1, rx=5)
        svg.text(x + CW / 2, y + CH / 2 + 5, cells[i][j], size=14, anchor="middle")
# column / row headers
svg.text(GX + CW / 2, GY - 8, "层输入", size=11, anchor="middle", color="muted")
svg.text(GX + 1.5 * CW, GY - 8, "新流 1", size=11, anchor="middle", color="muted")
svg.text(GX + 2.5 * CW, GY - 8, "新流 2", size=11, anchor="middle", color="muted")
svg.text(GX - 6, GY + CH / 2 + 4, "层输出", size=11, anchor="end", color="muted")
svg.text(GX - 6, GY + 1.5 * CH + 4, "流 1", size=11, anchor="end", color="muted")
svg.text(GX - 6, GY + 2.5 * CH + 4, "流 2", size=11, anchor="end", color="muted")
LX = GX + 3 * CW + 18
svg.text(LX, GY + 16, "第 0 行 β：写入权重，= H^{post}", size=13, color="green", weight="600")
svg.text(LX, GY + 36, "𝓕 的输出按 β_{j} 加到第 j 条流上", size=12, color="muted")
svg.text(LX, GY + 70, "右下块 α_{i,j}：流间混合，= (H^{res})^{⊤}", size=13, color="violet", weight="600")
svg.text(LX, GY + 90, "旧流 i 按 α_{i,j} 流进新流 j", size=12, color="muted")
svg.text(LX, GY + 124, "第 0 列 α_{i,0}：读出权重，= (H^{pre})^{⊤}", size=13, color="blue", weight="600")
svg.text(LX, GY + 144, "𝓕 的输入 = Σ_{i} α_{i,0} × 流 i", size=12, color="muted")
svg.text(GX, GY + 3 * CH + 44, "左上角恒为 0：层的输出不直接喂回它自己的输入。", size=12, color="muted")

# ---------------- (b) initialisation
BX = 520
svg.line(BX - 14, 40, BX - 14, 240, color="rule", width=1)
svg.text(BX, 26, "(b) 初始化 = Pre-Norm 残差", size=14, weight="700")
svg.matrix(BX + 60, 50, [["0", "1", "1"], ["1", "1", "0"], ["0", "0", "1"]], cell=32, size=14,
           highlight={(0, 1), (0, 2), (1, 0), (1, 1), (2, 2)}, hl_color="green")
svg.text(BX, 166, "β = (1, 1)：写回两条流", size=12)
svg.text(BX, 188, "α_{·,0} = *e*_{k mod n}：只读一条流", size=12)
svg.text(BX, 210, "右下 = I：两条流互不混合", size=12)
svg.text(BX, 232, "⇒ 两条流永远相同（图为 k 读流 1）", size=12, color="green", weight="600")

# ---------------- (c) special cases
svg.line(20, 266, 740, 266, color="rule", width=1)
svg.text(20, 288, "(c) 旧结构都是“不可训练的 HC”：矩阵固定，只剩结构", size=14, weight="700")


def case(x, title, sub, vals, cell, foot1, foot2=None, hl=None):
    svg.text(x, 316, title, size=13, weight="600")
    svg.text(x, 333, sub, size=11, color="muted")
    svg.matrix(x + 8, 346, vals, cell=cell, size=13, highlight=hl, hl_color="blue")
    rows = len(vals)
    svg.text(x, 346 + rows * cell + 22, foot1, size=12)
    if foot2:
        svg.text(x, 346 + rows * cell + 44, foot2, size=12, color="muted")


case(20, "Pre-Norm", "n = 1", [["0", "1"], ["1", "1"]], 36, "捷径系数 1", "读、写都是 1")
case(150, "Post-Norm", "n = 1", [["0", "c"], ["1", "c"]], 36, "c = 1/std(*x* + 𝓕)",
     "独立、方差各为 1 时 c ≈ 0.707")
case(345, "串行（n = 2）", "每层读流 1，写回两条", [["0", "1", "1"], ["1", "1", "0"], ["0", "0", "1"]], 30,
     "与 (b) 相同的矩阵")
case(495, "并行（奇数层）", "读两流之和，只写流 1", [["0", "1", "0"], ["1", "1", "1"], ["1", "1", "1"]], 30,
     "右下全 1：两流相加")
case(630, "并行（偶数层）", "只读、只写流 2", [["0", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]], 30,
     "两层读到同一输入")

svg.save("l16-hc-matrix.svg")
