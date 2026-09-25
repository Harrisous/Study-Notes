"""Figures 13.7-13.9: DeepSeek-R1-Zero training dynamics, redrawn from data digitised out of the
vector SVG sources of the R1 paper (arXiv v2 Fig. 1a, Fig. 1b, Fig. 9 = Nature Fig. 1a/1b and
Extended Data Fig. 1). The numbers are copied from reference-docs/lesson-13-doc.md section 5.2.

  l13-aime-curve.svg          AIME 2024 pass@1 and cons@16 versus RL step
  l13-length-curve.svg        average response length (smoothed curve sampled every 400 steps)
  l13-reflection-words.svg    counts of reflective words and of "wait" versus RL step
"""

from l13_svg import SVG, Axes

AIME = """200 .156 .253 | 400 .225 .340 | 600 .246 .313 | 800 .287 .347 | 1000 .317 .447 | 1200 .371 .567
1400 .412 .553 | 1600 .377 .487 | 2000 .381 .547 | 2400 .485 .613 | 2800 .529 .733 | 3200 .512 .687
3600 .583 .767 | 4000 .600 .767 | 4200 .602 .733 | 4400 .558 .767 | 4600 .579 .693 | 4800 .569 .733
5000 .540 .707 | 5200 .631 .780 | 5400 .575 .727 | 5600 .590 .800 | 5800 .617 .800 | 6000 .644 .780
6200 .625 .800 | 6400 .688 .867 | 6800 .663 .767 | 7000 .685 .833 | 7200 .652 .800 | 7400 .677 .833
7600 .673 .813 | 7800 .669 .807 | 8000 .688 .820 | 8200 .677 .833 | 8400 .710 .867 | 8600 .738 .867
8800 .744 .867 | 9000 .735 .867 | 9200 .777 .867 | 9400 .767 .867 | 9600 .767 .867 | 9800 .752 .867
10000 .773 .867"""
aime = [tuple(float(v) for v in chunk.split()) for chunk in AIME.replace("\n", " | ").split("|")]

LENGTH = """0:550 400:950 800:1330 1200:1350 1600:1700 2000:2010 2400:2330 2800:3160 3200:3130 3600:3800
4000:3860 4400:4190 4800:4910 5200:6010 5600:5050 6000:5970 6400:5610 6800:6590 7200:6590
7600:7940 8000:9200 8400:10520 8800:13210 9200:12710 9600:14600 10000:13470 10200:13650"""
length = [tuple(float(v) for v in kv.split(":")) for kv in LENGTH.split()]

REFL = """200:1479 400:2192 600:1887 800:736 1000:635 1200:1678 1400:1243 1600:1170 1800:4744 2000:1041
2200:1616 2400:1120 2600:2173 2800:1066 3000:1019 3200:1750 3400:1604 3600:1113 3800:1118
4000:1369 4200:4026 4400:5305 4600:2900 4800:3481 5200:4528 5400:6770 5600:6520 5800:8666
6000:7018 6200:7015 6400:10183 6800:5433 7000:8891 7200:8006 7400:5352 7600:7169 7800:6781
8200:8660 8600:9720 8800:6637 9000:5343 9200:8921 9400:8408 9600:7777 9800:14289 10000:7218
10200:9811 10400:7081"""
refl = [tuple(float(v) for v in kv.split(":")) for kv in REFL.split()]
WAIT = """2000:1 2200:1 2400:2 2600:1 2800:0 3000:0 3200:1 3400:0 3600:1 3800:6 4000:11
4200:21 4400:14 4600:39 4800:81 5200:48 5400:160 5600:87 5800:88 6000:154 6200:67 6400:51
6800:20 7000:23 7200:99 7400:53 7600:143 7800:126 8200:887 8600:1500 8800:334 9000:106
9200:344 9400:420 9600:550 9800:365 10000:401 10200:1008 10400:570"""
wait = [(float(s), 0.0) for s in range(200, 1801, 200)] + [tuple(float(v) for v in kv.split(":")) for kv in WAIT.split()]

JUMP = 8200


def x_axis(svg, ax, label=True, ticks=(0, 2000, 4000, 6000, 8000, 10000)):
    for s in ticks:
        x = ax.px(s)
        svg.line(x, ax.Y0, x, ax.Y1, color="grid", width=1)
        if label:
            svg.text(x, ax.Y1 + 16, "{:,}".format(s), size=11, anchor="middle", color="muted")
    svg.line(ax.X0, ax.Y1, ax.X1, ax.Y1, color="muted", width=1.1)


def jump_line(svg, ax, text=None, y_text=None, top=None):
    x = ax.px(JUMP)
    svg.line(x, ax.Y0 if top is None else top, x, ax.Y1, color="red", width=1.2, dash="4 3")
    if text:
        for k, t in enumerate(text):
            svg.text(x - 6, (y_text or ax.Y0 + 14) + 16 * k, t, size=12, anchor="end", color="red")


# ------------------------------------------------------------------ AIME
svg = SVG(760, 392, "DeepSeek-R1-Zero 训练中的 AIME 2024 正确率（据论文图 1a 的数据重绘）")
ax = Axes(62, 736, 20, 330, (0, 10400), (0, 1))
for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
    y = ax.py(v)
    svg.line(ax.X0, y, ax.X1, y, color="grid", width=1)
    svg.text(ax.X0 - 6, y + 4, "%d%%" % round(v * 100), size=11, anchor="end", color="muted")
x_axis(svg, ax)
svg.text((ax.X0 + ax.X1) / 2, ax.Y1 + 38, "RL 训练步数", size=12, anchor="middle", color="muted")
yh = ax.py(0.378)
svg.line(ax.X0, yh, ax.X1, yh, color="muted", width=1.3, dash="6 4")
svg.text(ax.X1 - 4, yh + 17, "人类参赛者平均 37.8%", size=12, anchor="end", color="muted")
jump_line(svg, ax, ["第 8,200 步：最大长度", "32,768 → 65,536"], y_text=ax.py(0.29), top=ax.py(0.92))
svg.polyline([(ax.px(s), ax.py(c)) for s, _, c in aime], stroke="green", width=2)
svg.polyline([(ax.px(s), ax.py(p)) for s, p, _ in aime], stroke="blue", width=2.4)
for s, p, c in aime:
    svg.circle(ax.px(s), ax.py(p), 2.3, fill="blue")
# annotations
s0, p0, _ = aime[0]
svg.circle(ax.px(s0), ax.py(p0), 4.2, stroke="blue", width=1.6, fill="surface")
svg.text(ax.px(s0) + 8, ax.py(p0) + 18, "第 200 步：15.6%", size=12, color="blue", weight="600")
p84 = [p for s, p, _ in aime if s == 8400][0]
svg.circle(ax.px(8400), ax.py(p84), 4.2, stroke="ink", width=1.6, fill="surface")
svg.line(ax.px(8400) + 3, ax.py(p84) + 4, ax.px(8400) + 30, ax.py(0.60) - 4, color="ink", width=0.9)
svg.text(ax.px(8400) + 32, ax.py(0.60) + 8, "第 8,400 步：71.0%", size=12, weight="600")
svg.text(ax.px(8400) + 32, ax.py(0.60) + 24, "（v1 报告的数字）", size=12, color="muted")
pmax = max(aime, key=lambda t: t[1])
svg.circle(ax.px(pmax[0]), ax.py(pmax[1]), 4.2, stroke="blue", width=1.6, fill="surface")
svg.line(ax.px(pmax[0]), ax.py(pmax[1]) - 5, ax.px(pmax[0]) + 14, ax.py(0.955) + 4, color="blue", width=0.9)
svg.text(ax.X1, ax.py(0.955), "最高点：第 9,200 步 77.7%", size=12, anchor="end", color="blue",
         weight="600")
# legend
svg.line(80, 34, 104, 34, color="blue", width=2.4)
svg.text(110, 38, "pass@1（每题 16 个样本的平均正确率）", size=12)
svg.line(80, 54, 104, 54, color="green", width=2)
svg.text(110, 58, "cons@16（16 个样本多数投票）", size=12)
svg.text(80, 78, "Nature 版正文的 77.9% 不在这条曲线上（出处未说明）", size=12, color="muted")
svg.save("l13-aime-curve.svg")

# ------------------------------------------------------------------ length
svg = SVG(760, 372, "DeepSeek-R1-Zero 训练中每条回答的平均长度（据论文图 1b 的平滑曲线重绘）")
ax = Axes(70, 736, 38, 312, (0, 10400), (0, 36000))
for v in range(0, 36001, 5000):
    y = ax.py(v)
    svg.line(ax.X0, y, ax.X1, y, color="grid", width=1)
    svg.text(ax.X0 - 6, y + 4, "{:,}".format(v), size=11, anchor="end", color="muted")
x_axis(svg, ax)
svg.text((ax.X0 + ax.X1) / 2, ax.Y1 + 38, "RL 训练步数", size=12, anchor="middle", color="muted")
svg.text(18, (ax.Y0 + ax.Y1) / 2, "平均长度（token）", size=12, anchor="middle", color="muted", rotate=-90)
ycap = ax.py(32768)
svg.line(ax.X0, ycap, ax.px(JUMP), ycap, color="red", width=1.4, dash="6 4")
svg.text(ax.X0 + 8, ycap - 8, "最大长度上限 32,768", size=12, color="red")
xj = ax.px(JUMP)
svg.line(xj, ax.Y1, xj, ycap, color="red", width=1.2, dash="4 3")
svg.arrow(xj, ycap, xj, ax.Y0 - 20, color="red", width=1.4)
svg.text(xj - 8, ax.Y0 - 10, "第 8,200 步起上限升到 65,536（超出坐标）", size=12, anchor="end", color="red")
svg.polyline([(ax.px(s), ax.py(v)) for s, v in length], stroke="blue", width=2.6)
for s, v in length:
    svg.circle(ax.px(s), ax.py(v), 2.4, fill="blue")
for s, v, txt, anc, dx, dy in ((0, 550, "第 0 步：约 550", "start", 6, -10),
                               (4000, 3860, "第 4,000 步：约 3,860", "middle", 0, -12),
                               (8000, 9200, "第 8,000 步：约 9,200", "end", -8, -6),
                               (9600, 14600, "第 9,600 步：约 14,600", "end", -8, -8)):
    svg.text(ax.px(s) + dx, ax.py(v) + dy, txt, size=12, anchor=anc, color="blue", weight="600")
svg.circle(ax.px(9525), ax.py(19980), 4, stroke="violet", width=1.5, fill="surface")
svg.line(ax.px(9525) - 5, ax.py(19980), ax.px(JUMP) - 8, ax.py(19980), color="violet", width=0.9, dash="2 2")
svg.text(ax.px(JUMP) - 12, ax.py(19980) + 4, "未平滑曲线的最高点约 19,980", size=12, anchor="end", color="violet")
svg.save("l13-length-curve.svg")

# ------------------------------------------------------------------ reflective words
svg = SVG(760, 452, "DeepSeek-R1-Zero 训练中反思词与“wait”的出现次数（据论文图 9 的数据重绘）")
a1 = Axes(70, 736, 30, 190, (0, 10400), (0, 15000))
a2 = Axes(70, 736, 254, 400, (0, 10400), (0, 1600))
svg.text(16, 18, "(a) 10 个反思词的总次数：wait, mistake, however, but, retry, error, verify, wrong, evaluate, check",
         size=12, weight="600")
for v in (0, 5000, 10000, 15000):
    y = a1.py(v)
    svg.line(a1.X0, y, a1.X1, y, color="grid", width=1)
    svg.text(a1.X0 - 6, y + 4, "{:,}".format(v), size=11, anchor="end", color="muted")
x_axis(svg, a1, label=False)
jump_line(svg, a1)
svg.polyline([(a1.px(s), a1.py(v)) for s, v in refl], stroke="violet", width=2)
for s, v in refl:
    svg.circle(a1.px(s), a1.py(v), 2.1, fill="violet")
svg.text(16, 238, "(b) “wait” 一个词的次数", size=12, weight="600")
svg.rect(a2.px(4000), a2.Y0, a2.px(7000) - a2.px(4000), a2.Y1 - a2.Y0, fill="amber", fill_opacity=0.08)
svg.text(a2.px(5500), a2.Y0 + 16, "4,000–7,000 步：偶尔出现", size=12, anchor="middle", color="amber")
for v in (0, 400, 800, 1200, 1600):
    y = a2.py(v)
    svg.line(a2.X0, y, a2.X1, y, color="grid", width=1)
    svg.text(a2.X0 - 6, y + 4, "{:,}".format(v), size=11, anchor="end", color="muted")
x_axis(svg, a2)
jump_line(svg, a2)
svg.polyline([(a2.px(s), a2.py(v)) for s, v in wait], stroke="blue", width=2)
for s, v in wait:
    svg.circle(a2.px(s), a2.py(v), 2.1, fill="blue")
svg.text(a2.px(8600) + 8, a2.py(1500) + 4, "第 8,600 步：1,500", size=12, color="blue", weight="600")
svg.text(a2.px(8200) - 6, a2.py(887) + 4, "第 8,200 步：887", size=12, anchor="end", color="blue", weight="600")
svg.text(a2.X1, a2.Y1 + 38, "红虚线：第 8,200 步，最大长度加倍", size=12, anchor="end", color="red")
svg.text((a2.X0 + a2.X1) / 2, a2.Y1 + 38, "RL 训练步数", size=12, anchor="middle", color="muted")
svg.save("l13-reflection-words.svg")
