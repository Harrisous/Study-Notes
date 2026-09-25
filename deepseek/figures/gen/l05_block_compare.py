"""Figure 5.1: the GPT-2 small block (lesson 1) next to the LLaMA-7B-style block (this lesson).

Schematic with real dimensions:
  GPT-2 small  d = 768, 12 heads x 64, FFN 3072, learned position table with 1024 rows, 12 layers
               (Hugging Face openai-community/gpt2 config.json and modeling_gpt2.py)
  LLaMA-7B     d = 4096, 32 heads x 128, SwiGLU hidden 11008, 32 layers (LLaMA Table 2 + llama_v1 code);
               DeepSeek LLM 7B has the same widths and 30 layers (DeepSeek LLM Table 2 + HF config)
Writes deepseek/figures/l05-block-compare.svg
"""

from l05_svg import SVG

W, H = 760, 640
svg = SVG(W, H, "GPT-2 small 的块与 LLaMA-7B 式块的对比")

BOXW = 236


def panel(px, title, rows, layers_label, changed, subtitle=None, colors=None):
    cx = px + 196
    xl = cx - BOXW / 2
    skip_x = px + 36
    svg.text(cx - 20, 22, title, size=15, anchor="middle", weight="700")
    if subtitle:
        svg.text(cx - 20, 40, subtitle, size=12, anchor="middle", color="muted")
    y = 50
    centers = {}
    for key, lines, h in rows:
        if key in ("add1", "add2"):
            svg.circle(cx, y + 10, 10, fill="surface", stroke="ink", width=1.5)
            svg.line(cx - 5.5, y + 10, cx + 5.5, y + 10, color="ink", width=1.6)
            svg.line(cx, y + 4.5, cx, y + 15.5, color="ink", width=1.6)
            centers[key] = y + 10
            y += 20 + 16
            continue
        if key in ("split1", "split2"):
            svg.circle(cx, y + 4, 3.6, fill="ink")
            centers[key] = y + 4
            y += 8 + 12
            continue
        col = "blue" if key in changed else "muted"
        fo = 0.10 if key in changed else 0.07
        svg.box(xl, y, BOXW, h, lines, color=col, fill_opacity=fo, size=13,
                weight="600", stroke=col, line_colors=(colors or {}).get(key))
        if key in changed:
            tag = changed[key]
            svg.circle(xl + BOXW + 14, y + 13, 10, fill="blue", fill_opacity=0.14, stroke="blue", width=1.2)
            svg.text(xl + BOXW + 14, y + 17.2, tag, size=12, anchor="middle", color="blue", weight="700")
        centers[key] = (y, y + h)
        y += h + 16
    # vertical arrows between consecutive items
    order = [r[0] for r in rows]
    for a, b in zip(order, order[1:]):
        ya = centers[a][1] if isinstance(centers[a], tuple) else centers[a] + (10 if a.startswith("add") else 3.6)
        yb = centers[b][0] if isinstance(centers[b], tuple) else centers[b] - (10 if b.startswith("add") else 3.6)
        if yb - ya > 4:
            svg.arrow(cx, ya, cx, yb, color="ink", width=1.4, shorten=1)
    # residual skips
    for s, a in (("split1", "add1"), ("split2", "add2")):
        ys, ya = centers[s], centers[a]
        svg.polyline([(cx, ys), (skip_x, ys), (skip_x, ya)], stroke="ink", width=1.6)
        svg.arrow(skip_x, ya, cx - 10, ya, color="ink", width=1.6, shorten=1)
    svg.text(skip_x + 5, (centers["split1"] + centers["add1"]) / 2 + 4, "残差", size=12, color="muted")
    # layer bracket
    y0, y1 = centers["split1"] - 6, centers["add2"] + 12
    bx = px + 14
    svg.path("M%d %d h-6 v%d h6" % (bx + 6, y0, y1 - y0), stroke="muted", width=1.2)
    svg.text(bx - 3, (y0 + y1) / 2 + 4, layers_label, size=12, anchor="middle", color="muted", rotate=-90)
    return centers


gpt2_rows = [
    ("in", ["词嵌入 + 位置嵌入", "位置表 wpe 可学习，只有 1024 行"], 44),
    ("split1", None, 0),
    ("norm1", ["LayerNorm", "减均值、除标准差、再加偏置"], 42),
    ("attn", ["多头注意力 12 头 × 64 维", "位置信息只来自输入处的 wpe", "所有投影带偏置"], 58),
    ("add1", None, 0),
    ("split2", None, 0),
    ("norm2", ["LayerNorm"], 30),
    ("ffn", ["FFN：768 → 3072 → 768", "GELU，两层都带偏置"], 44),
    ("add2", None, 0),
    ("final", ["最后一层 LayerNorm"], 30),
    ("out", ["反嵌入 → logits", "上下文最长 1024"], 42),
]
llama_rows = [
    ("in", ["词嵌入 [T, 4096]", "输入处不加任何位置向量"], 44),
    ("split1", None, 0),
    ("norm1", ["RMSNorm", "只除以均方根，没有偏置"], 42),
    ("attn", ["多头注意力 32 头 × 128 维", "每层把 q、k 按位置旋转（RoPE）", "v 不旋转；无偏置"], 58),
    ("add1", None, 0),
    ("split2", None, 0),
    ("norm2", ["RMSNorm"], 30),
    ("ffn", ["SwiGLU：4096 → 2 × 11008 → 4096", "SiLU 门控 ⊙ 线性支路，无偏置"], 44),
    ("add2", None, 0),
    ("final", ["最后一层 RMSNorm"], 30),
    ("out", ["输出投影 → logits", "预训练上下文 2048"], 42),
]

panel(8, "GPT-2 small（第 1 课）", gpt2_rows, "× 12 层", {}, subtitle="d = 768，12 层")
c = panel(388, "LLaMA-7B 式块（本课）", llama_rows, "× 32 层",
          {"in": "1", "attn": "1", "norm1": "2", "norm2": "2", "ffn": "3", "final": "2"},
          subtitle="DeepSeek LLM 7B 宽度相同，30 层",
          colors={"attn": [None, "red", None]})


# legend strip
ly = H - 70
svg.rect(12, ly - 8, W - 24, 70, fill="blue", fill_opacity=0.05, stroke="rule", width=1, rx=6)
svg.text(24, ly + 12, "蓝框 = 本课讲的改动（圆圈里的数字对应下面四项）", size=13, weight="600", color="blue")
svg.text(24, ly + 34, "1  位置：不再在输入处相加，改为每层旋转 q、k（§2–§3）", size=12.5)
svg.text(24, ly + 54, "3  前馈：GELU、4d → SwiGLU、约 8d/3（§5）", size=12.5)
svg.text(480, ly + 34, "2  归一化：LayerNorm → RMSNorm（§4）", size=12.5)
svg.text(480, ly + 54, "4  去掉所有偏置（§5）", size=12.5)

svg.save("l05-block-compare.svg")
