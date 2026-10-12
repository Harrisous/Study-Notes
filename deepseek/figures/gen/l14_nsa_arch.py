"""Figure 14.3: NSA's three branches and the gate (redrawn after Yuan et al. 2025, Fig. 2, left).

Schematic; one cell = 16 tokens, so compression blocks (2 cells) and selection blocks (4 cells)
are to scale, but the 512-token window (32 cells) is drawn as 8 cells and labelled "not to
scale", and fewer than 16 selected blocks are shown. The real hyperparameters
are written on the figure: l_c = 32, delta_c = 16, l_s = 64, n_s = 16 (1 initial + 2 local
blocks fixed), w = 512 (NSA sections 4.1 and 5).
Writes deepseek/figures/l14-nsa-arch.svg
"""

from l14_svg import SVG

W, H = 760, 500
svg = SVG(W, H, "NSA 的三个分支：压缩、选择、滑动窗口，经门控相加（据 NSA 图 2 重绘）")

N = 24            # cells in a strip
CW = 13.5         # cell width  -> strip 324 px
SX = 34           # strip x
AX, AW = 410, 118  # attention box
OX = 566          # output label x
GX = 624          # gate circle x
SUMX, SUMY = 700, 262


def strip(y, colors=None, blocks=None):
    """A strip of N cells; colors: dict cell -> (color, opacity)."""
    colors = colors or {}
    for i in range(N):
        col, op = colors.get(i, (None, None))
        if col:
            svg.rect(SX + i * CW, y, CW, 14, fill=col, fill_opacity=op)
    svg.rect(SX, y, N * CW, 14, stroke="rule", width=1)
    for i in range(1, N):
        major = blocks and i % blocks == 0
        svg.line(SX + i * CW, y, SX + i * CW, y + 14, color="muted" if major else "rule",
                 width=1.4 if major else 0.6)


def lane_label(y, text, color):
    svg.text(SX - 20, y, text, size=13, weight="700", color=color)


# ------------------------------------------------------------------ header
svg.text(SX - 20, 22, "键值序列：位置 1…t 的 token（一格 = 16 个，示意）", size=12, color="muted")
svg.box(AX - 24, 8, AW + 48, 30, ["查询 *q*_{t}（三个分支共用）"], color="blue", size=12)

# ------------------------------------------------------------------ lane 1: compression
y1 = 62
lane_label(y1, "① 压缩 cmp", "orange")
strip(y1 + 10)
# overlapping blocks: length 2 cells (32 tokens), stride 1 cell (16 tokens)
for i in range(0, N - 1):
    row = i % 2
    yy = y1 + 30 + row * 8
    svg.rect(SX + i * CW + 1.5, yy, 2 * CW - 3, 6, fill="orange", fill_opacity=0.35, stroke="orange",
             width=0.8, rx=2)
# compressed tokens
for i in range(0, N - 1):
    cx = SX + (i + 1) * CW
    svg.rect(cx - 4.5, y1 + 66, 9, 9, fill="orange", fill_opacity=0.85, rx=1.5)
svg.text(SX, y1 + 60, "每块经 φ（带块内位置编码的 MLP）压成 1 个键值", size=11, color="muted")
svg.text(SX, y1 + 94, "块长 ℓ_{c} = 32，步长 δ_{c} = 16（相邻块重叠一半）", size=12, color="orange")
svg.text(SX, y1 + 115, "→ 每 16 个 token 产生 1 个压缩键值", size=12, color="orange")
svg.arrow(SX + N * CW + 8, y1 + 70, AX - 4, y1 + 70, color="orange", width=1.6)
svg.box(AX, y1 + 50, AW, 40, ["Attn(*q*_{t}, K̃^{cmp}, Ṽ^{cmp})"], color="orange", size=12)
svg.arrow(AX + AW, y1 + 70, OX - 4, y1 + 70, color="ink", width=1.4)
svg.text(OX, y1 + 74, "*o*^{cmp}", size=13)

# ------------------------------------------------------------------ lane 2: selection
y2 = 196
lane_label(y2, "② 选择 slc", "violet")
sel_fixed = {0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23}
sel_pick = {8, 9, 10, 11}
cols = {i: ("green", 0.55) for i in sel_fixed}
cols.update({i: ("violet", 0.6) for i in sel_pick})
strip(y2 + 10, cols, blocks=4)
svg.text(SX + 2 * CW, y2 + 42, "开头块", size=11, anchor="middle", color="green")
svg.text(SX + 10 * CW, y2 + 42, "按分数选中", size=11, anchor="middle", color="violet")
svg.text(SX + 20 * CW, y2 + 42, "本地 2 块", size=11, anchor="middle", color="green")
svg.text(SX, y2 + 64, "块长 ℓ_{s} = 64，每个查询取 n_{s} = 16 块（其中开头 1 块、", size=12, color="violet")
svg.text(SX, y2 + 85, "本地 2 块固定）→ 16 × 64 = 1024 个原始 token", size=12, color="violet")
svg.arrow(SX + N * CW + 8, y2 + 17, AX - 4, y2 + 30, color="violet", width=1.6)
svg.box(AX, y2 + 10, AW, 40, ["Attn(*q*_{t}, K̃^{slc}, Ṽ^{slc})"], color="violet", size=12)
svg.arrow(AX + AW, y2 + 30, OX - 4, y2 + 30, color="ink", width=1.4)
svg.text(OX, y2 + 34, "*o*^{slc}", size=13)

# score reuse: dashed arrow from compression attention to selection
svg.path("M%d %d C %d %d, %d %d, %d %d" % (AX + 30, y1 + 90, AX + 30, y1 + 110, 360, y2 - 30, 330, y2 + 4),
         stroke="violet", width=1.4, dash="5 4", arrow=True)
svg.text(AX + 38, y1 + 116, "压缩分支的注意力分数 *p*^{cmp}", size=11, color="violet")
svg.text(AX + 38, y1 + 131, "→ 块分数 → 组内求和 → top-n", size=11, color="violet")

# ------------------------------------------------------------------ lane 3: sliding window
y3 = 316
lane_label(y3, "③ 滑动窗口 win", "green")
cols = {i: ("green", 0.55) for i in range(N - 8, N)}
strip(y3 + 10, cols)
svg.text(SX + (N - 4) * CW, y3 + 42, "最近 w = 512 个 token（未按比例）", size=11, anchor="middle", color="green")
svg.text(SX, y3 + 64, "固定看最近的上下文，防止局部模式“抢走”", size=12, color="green")
svg.text(SX, y3 + 80, "另两个分支的学习信号", size=12, color="green")
svg.arrow(SX + N * CW + 8, y3 + 17, AX - 4, y3 + 30, color="green", width=1.6)
svg.box(AX, y3 + 10, AW, 40, ["Attn(*q*_{t}, K̃^{win}, Ṽ^{win})"], color="green", size=12)
svg.arrow(AX + AW, y3 + 30, OX - 4, y3 + 30, color="ink", width=1.4)
svg.text(OX, y3 + 34, "*o*^{win}", size=13)

# ------------------------------------------------------------------ gates and sum
lanes_y = [y1 + 70, y2 + 30, y3 + 30]
names = ["cmp", "slc", "win"]
for yy, nm in zip(lanes_y, names):
    svg.circle(GX, yy, 11, fill="surface", stroke="ink", width=1.3)
    svg.text(GX, yy + 5, "×", size=14, anchor="middle")
    svg.line(OX + 40, yy, GX - 12, yy, color="ink", width=1.3)
    svg.text(GX, yy - 17, "g^{%s}" % nm, size=12, anchor="middle", color="muted")
    svg.path("M%d %d L%d %d" % (GX + 11, yy, SUMX - 14, SUMY + (yy - SUMY) * 0.15), stroke="ink",
             width=1.3, arrow=True)
svg.circle(SUMX, SUMY, 14, fill="surface", stroke="ink", width=1.5)
svg.text(SUMX, SUMY + 6, "Σ", size=16, anchor="middle", weight="700")
svg.arrow(SUMX, SUMY + 15, SUMX, SUMY + 70, color="green", width=2)
svg.text(SUMX, SUMY + 88, "*o*_{t}^{*}", size=14, anchor="middle", weight="700", color="green")

# gate generator box
svg.box(GX - 58, 400, 150, 58, ["门控：MLP + sigmoid", "g_{t}^{c} ∈ [0, 1]，各自独立"], color="ink",
        fill_opacity=0.05, size=12)
svg.path("M%d %d L%d %d" % (GX, 400, GX, y3 + 43), stroke="muted", width=1.2, dash="4 3", arrow=True)

# footer
svg.text(SX - 20, 440, "三个分支各有独立的键和值；三个门不是 softmax，和不必为 1。", size=12)
svg.text(SX - 20, 460, "t = 8192 时每个查询看：压缩 ≈ 511 + 选择 1024 + 窗口 512 ≈ 2047 个键（稠密要看 8192 个）。",
         size=12)
svg.text(SX - 20, 480, "据 NSA 论文图 2（左）重绘，配色按本课程约定。", size=11, color="muted")

svg.save("l14-nsa-arch.svg")
