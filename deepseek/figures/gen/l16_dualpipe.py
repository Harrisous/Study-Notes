"""Figure 16.11: how mHC's kernels are overlapped inside DualPipe (schematic after mHC Fig. 4).

Order of the operations in each stream is taken from mHC (arXiv 2512.24880 v2) Fig. 4 as listed in the research
notes, section 5.3 item 15. The paper states that block lengths are illustrative only; horizontal positions of
the communication blocks relative to the compute blocks are ALSO only illustrative here (only their order, and
the placement of the two high-priority kernels between specific communication blocks, follow the notes).
F^A / F^M: mHC kernels of the attention / MLP (MoE) sublayer; (F) forward, (B) backward, (W) weight-gradient.
Writes deepseek/figures/l16-dualpipe.svg
"""

from l16_svg import SVG

W, H = 760, 350
svg = SVG(W, H, "mHC 的核在 DualPipe 里怎样与通信重叠（据 mHC 论文图 4 重绘，块长与位置仅示意）")

LX = 78
ROW_Y = {"compute": 58, "comm": 152, "hi": 246}
RH = 58

svg.text(12, 28, "一个流水线 stage 内的三条 GPU 流（时间向右）", size=14, weight="700")
for key, lines in (("compute", ["普通", "计算流"]), ("comm", ["通信流", ""]), ("hi", ["高优先级", "计算流"])):
    y = ROW_Y[key]
    svg.text(10, y + 24, lines[0], size=12, weight="600")
    if lines[1]:
        svg.text(10, y + 41, lines[1], size=12, weight="600")
    svg.line(LX - 4, y + RH + 8, W - 8, y + RH + 8, color="rule", width=1)

# compute stream: (label line1, line2, width, colour)
COMPUTE = [
    ("MLP", "(B)", 46, "blue"), ("MLP", "(W)", 46, "blue"), ("MLP", "(F)", 46, "blue"),
    ("𝓕^{M}_{pre}", "(B)", 52, "green"), ("𝓕^{A}_{post,res}", "(B)", 62, "green"), ("ATTN", "(B)", 46, "violet"),
    ("𝓕^{A}_{pre}", "(B)", 52, "green"), ("ATTN", "(W)", 46, "violet"), ("整段", "重算(B)", 58, "amber"),
    ("𝓕^{A}_{pre}", "(F)", 52, "green"), ("ATTN", "(F)", 46, "violet"), ("𝓕^{A}_{post,res}", "(F)", 62, "green"),
    ("𝓕^{M}_{pre}", "(F)", 52, "green"),
]
x = LX
y = ROW_Y["compute"]
for l1, l2, w, col in COMPUTE:
    svg.rect(x + 1, y, w - 2, RH, fill=col, fill_opacity=0.14, stroke=col, width=1.1, rx=4)
    svg.text(x + w / 2, y + 22, l1, size=12, anchor="middle")
    svg.text(x + w / 2, y + 48, l2, size=12, anchor="middle", color="muted")
    x += w
print("compute stream ends at x =", x)

# communication stream and the two high-priority kernels, in order
COMM = [("DISPATCH", "(F)", 82), ("DISPATCH", "(B)", 82), ("COMBINE", "(F)", 82), ("PP 收发", "(F)", 70),
        ("PP 收发", "(B)", 70), ("COMBINE", "(B)", 82)]
xs = [LX, LX + 88, LX + 176, LX + 348, LX + 424, LX + 584]
y = ROW_Y["comm"]
for (l1, l2, w), x0 in zip(COMM, xs):
    svg.rect(x0, y, w, RH, fill="amber", fill_opacity=0.14, stroke="amber", width=1.1, rx=4)
    svg.text(x0 + w / 2, y + 26, l1, size=12, anchor="middle")
    svg.text(x0 + w / 2, y + 48, l2, size=12, anchor="middle", color="muted")
# the two high-priority kernels sit in time between COMBINE(F) / PP(F) and PP(B) / COMBINE(B)
assert xs[2] + 82 < LX + 264 and LX + 264 + 78 < xs[3]
assert xs[4] + 70 < LX + 500 and LX + 500 + 78 < xs[5]
y = ROW_Y["hi"]
for x0, w, lab in ((LX + 264, 78, "(F)"), (LX + 500, 78, "(B)")):
    svg.rect(x0, y, w, RH, fill="red", fill_opacity=0.12, stroke="red", width=1.3, rx=4)
    svg.text(x0 + w / 2, y + 22, "𝓕^{M}_{post,res}", size=12, anchor="middle")
    svg.text(x0 + w / 2, y + 48, lab, size=12, anchor="middle", color="muted")

svg.text(LX, 336, "绿：mHC 的核；红：放到高优先级流的 MLP 层 post-res 核（在 COMBINE 与流水线收发之间）；琥珀：通信与重算。",
         size=12, color="muted")
svg.save("l16-dualpipe.svg")
