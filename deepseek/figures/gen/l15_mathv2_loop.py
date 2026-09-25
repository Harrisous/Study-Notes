"""Figure 15.16: DeepSeekMath-V2's verifier / meta-verifier / generator loop, with the generator's reward table.

MV2 section 2: the verifier pi_phi reads (problem X, proof Y) and writes an analysis that lists issues and gives a
score s in {0, 0.5, 1}; the meta-verifier pi_eta scores whether the analysis's issues are real (R_meta); the generator
pi_theta writes a proof Y and a self-analysis Z with self-score s'. Generator reward (MV2 eqs. 5-6):
R = R_format(Y, Z) * (0.76 * R_Y + 0.24 * R_Z), R_Y = s, R_Z = (1 - |s' - s|) * R_meta(Z).
Scaled verification compute labels new hard proofs automatically to keep improving the verifier.
Table: R for six (s, s') pairs with R_format = R_meta = 1 (lesson-15-doc.md N36).
Writes deepseek/figures/l15-mathv2-loop.svg
"""

from l15_svg import SVG

WY, WZ = 0.76, 0.24


def reward(s, s_self):
    rz = 1 - abs(s_self - s)
    return rz, WY * s + WZ * rz


ROWS = [(1, 1), (1, 0.5), (0.5, 1), (0.5, 0.5), (0, 1), (0, 0)]
VALS = [reward(s, t) for s, t in ROWS]
assert [round(r, 2) for _, r in VALS] == [1.0, 0.88, 0.5, 0.62, 0.0, 0.24]

W, H = 760, 420
svg = SVG(W, H, "DeepSeekMath-V2：验证器、元验证器与会自评的生成器")

# ---------------------------------------------------------------- loop (left)
svg.text(16, 22, "一次训练迭代里的三个角色", size=13, weight="700")
svg.box(12, 44, 104, 50, [("题目 X", {"weight": "700"}), ("需要证明的题", {"size": 11, "color": "muted"})],
        color="ink", fill_opacity=0.05)
svg.box(150, 44, 124, 50, [("生成器 π_{θ}", {"weight": "700"}), ("写证明，再自评", {"size": 11, "color": "muted"})],
        color="blue", fill_opacity=0.14, line_gap=21)
svg.box(308, 36, 168, 66, [("证明 Y", {"weight": "700", "size": 12}), ("+ 自评 Z：问题清单", {"size": 12}),
                          ("与自评分 s′", {"size": 12})], color="blue", fill_opacity=0.06, line_gap=17)
svg.arrow(118, 69, 146, 69, color="ink", width=1.4)
svg.arrow(276, 69, 304, 69, color="ink", width=1.4)

svg.box(150, 150, 124, 50, [("验证器 π_{φ}", {"weight": "700"}), ("读 (X, Y)", {"size": 11, "color": "muted"})],
        color="violet", fill_opacity=0.14, line_gap=21)
svg.box(308, 142, 168, 66, [("分析：列出问题", {"size": 12}), ("证明分 s ∈ {0, 0.5, 1}", {"weight": "700", "size": 12}),
                           ("→ R_{Y} = s", {"size": 12})], color="violet", fill_opacity=0.06, line_gap=17)
svg.path("M392 102 V118 H212 V146", stroke="ink", width=1.3, arrow=True)
svg.arrow(276, 175, 304, 175, color="ink", width=1.4)

svg.box(150, 248, 124, 50, [("元验证器 π_{η}", {"weight": "700"}), ("检查一份分析", {"size": 11, "color": "muted"})],
        color="violet", fill_opacity=0.14, line_gap=21)
svg.box(308, 240, 168, 66, [("分析里指出的问题", {"size": 12}), ("是否真实存在？", {"size": 12}),
                           ("→ 元验证分 R_{meta}", {"size": 12})], color="violet", fill_opacity=0.06, line_gap=17)
svg.path("M392 208 V224 H212 V244", stroke="ink", width=1.3, arrow=True)
svg.arrow(276, 273, 304, 273, color="ink", width=1.4)

svg.rect(12, 330, 464, 52, fill="green", fill_opacity=0.10, stroke="green", width=1.3, rx=6)
svg.text(24, 352, "生成器的奖励：R = R_{format} · (0.76 · R_{Y} + 0.24 · R_{Z})", size=13, weight="700")
svg.text(24, 372, "R_{Z} = (1 − |s′ − s|) · R_{meta}(Z)：自评越接近验证器的判断，这一项越高", size=12)
svg.path("M476 356 H494 V31 H212 V40", stroke="green", width=1.3, dash="4 3", arrow=True)

svg.text(12, 404, "迭代：先训验证器，再从验证器检查点初始化生成器；放大验证算力给新的难证明自动标注，继续训验证器。", size=11,
         color="muted")

# ---------------------------------------------------------------- reward table (right)
TX = 512
svg.text(TX, 22, "奖励表（设 R_{format} = R_{meta} = 1）", size=13, weight="700")
cols = [TX, TX + 62, TX + 128, TX + 196]
for x, h in zip(cols, ["真实 s", "自评 s′", "R_{Z}", "R"]):
    svg.text(x + 24, 52, h, size=12, anchor="middle", weight="700", color="muted")
svg.line(TX, 60, TX + 236, 60, color="rule", width=1)
for i, ((s, t), (rz, r)) in enumerate(zip(ROWS, VALS)):
    y = 84 + i * 30
    if i % 2 == 0:
        svg.rect(TX - 4, y - 18, 244, 28, fill="grid", fill_opacity=0.5)
    honest = s == t
    for x, v in zip(cols, [s, t, rz, r]):
        txt = ("%.2f" % v) if x == cols[3] else ("%g" % v)
        svg.text(x + 24, y, txt, size=12, anchor="middle", weight="700" if x == cols[3] else None,
                 color=("green" if honest else "red") if x == cols[3] and s < 1 else "ink")
svg.text(TX, 280, "同一个只值 0.5 分的证明：", size=12)
svg.text(TX, 298, "老实自评 0.5 → 0.62；夸成 1 → 0.50", size=12, color="green")
svg.text(TX, 322, "同一个错误的证明（s = 0）：", size=12)
svg.text(TX, 340, "老实自评 0 → 0.24；夸成 1 → 0", size=12, color="green")
svg.text(TX, 368, "诚实地指出自己的错误，", size=12, color="muted")
svg.text(TX, 384, "得分高于谎称正确。", size=12, color="muted")

svg.save("l15-mathv2-loop.svg")
