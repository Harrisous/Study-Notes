"""Figure 6.14: optimal allocation exponents from dsllm Table 4, and what they mean for a 10x compute increase.

Rows (a = model exponent, b = data exponent):
  OpenAI/Kaplan (labelled "OpenWebText2" in the table; actually WebText2) 0.73 / 0.27
  Chinchilla (MassiveText, approach 2)                                     0.49 / 0.51
  DeepSeek early in-house data 0.450 / 0.550; current in-house 0.524 / 0.476; OpenWebText2 0.578 / 0.422
10x multipliers are 10**a and 10**b (notes R14).
Writes deepseek/figures/l06-quality.svg
"""

from l06_svg import SVG

rows = [
    ("Kaplan 等（WebText2）", 0.73, 0.27, "muted"),
    ("Chinchilla（MassiveText）", 0.49, 0.51, "muted"),
    ("DeepSeek 早期内部数据", 0.450, 0.550, "blue"),
    ("DeepSeek 当前内部数据", 0.524, 0.476, "blue"),
    ("DeepSeek OpenWebText2", 0.578, 0.422, "blue"),
]
for _, a, b, _ in rows:
    assert abs(a + b - 1) < 1e-9

W, H = 760, 380
X0, X1 = 200, 520
BH = 26
svg = SVG(W, H, "最优分配指数 a（给模型）与 b（给数据），以及算力 ×10 时两者各乘多少")


def px(f):
    return X0 + f * (X1 - X0)


top = 70
svg.text(X0, top - 34, "新增算力的“分成”：a 给模型，b 给数据（a + b = 1）", size=12, color="muted")
for k in range(0, 11, 2):
    x = px(k / 10)
    svg.line(x, top - 14, x, top + 5 * 48 - 10, color="grid", width=1)
    svg.text(x, top + 5 * 48 + 6, "%.1f" % (k / 10), size=11, anchor="middle", color="muted")
svg.line(px(0.5), top - 18, px(0.5), top + 5 * 48 - 6, color="ink", width=1.2, dash="4 3")
svg.text(px(0.5), top - 20, "0.5", size=11, anchor="middle", color="ink")
svg.text(X1 + 20, top - 34, "算力 ×10 时", size=12, color="muted")
svg.text(X1 + 20, top - 18, "模型 ×　　数据 ×", size=12, color="muted")

for i, (name, a, b, col) in enumerate(rows):
    y = top + i * 48 + (14 if i >= 2 else 0)
    svg.text(X0 - 10, y + 18, name, size=12, anchor="end", color="ink" if col == "blue" else "muted",
             weight="600" if col == "blue" else None)
    svg.rect(px(0), y, px(a) - px(0), BH, fill="blue", fill_opacity=0.45 if col == "blue" else 0.2,
             stroke="blue", width=1)
    svg.rect(px(a), y, px(1) - px(a), BH, fill="green", fill_opacity=0.35 if col == "blue" else 0.15,
             stroke="green", width=1)
    svg.text(px(0) + 8, y + 18, "a = %s" % ("%.3f" % a if col == "blue" else "%.2f" % a), size=12, color="ink")
    svg.text(px(1) - 8, y + 18, "b = %s" % ("%.3f" % b if col == "blue" else "%.2f" % b), size=12, anchor="end",
             color="ink")
    svg.text(X1 + 20, y + 18, "%.2f" % 10 ** a, size=13, color="blue", weight="600")
    svg.text(X1 + 84, y + 18, "%.2f" % 10 ** b, size=13, color="green", weight="600")

# quality arrow for the DeepSeek rows
ya, yb = top + 2 * 48 + 14, top + 4 * 48 + 14 + BH
svg.arrow(X1 + 170, ya, X1 + 170, yb, color="ink", width=1.6)
svg.text(X1 + 180, (ya + yb) / 2 - 8, "数据质量", size=12)
svg.text(X1 + 180, (ya + yb) / 2 + 10, "由低到高", size=12)
svg.line(20, top + 2 * 48 + 2, W - 20, top + 2 * 48 + 2, color="rule", width=1)

svg.save("l06-quality.svg")
