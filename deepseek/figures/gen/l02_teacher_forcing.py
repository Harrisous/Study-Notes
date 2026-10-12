"""Figure 2.3: teacher forcing - one forward pass over a packed sequence gives one loss per position.

Seven GPT-2 tokens (ids from the GPT-2 vocabulary, verified in lesson-01-doc.md):
  The 464 | Ġunbelievable 24479 | Ġtoken 11241 | ization 1634 | <|endoftext|> 50256 | Hello 15496 | Ġworld 995
Inputs are tokens 1..6, targets are tokens 2..7 (the same row shifted left by one), so six losses.
Documents are concatenated and separated by <|endoftext|>, with no special masking (GPT-3 appendix B).
Writes deepseek/figures/l02-teacher-forcing.svg
"""

from l02_svg import SVG

toks = [("The", 464), ("Ġunbelievable", 24479), ("Ġtoken", 11241), ("ization", 1634),
        ("<|endoftext|>", 50256), ("Hello", 15496), ("Ġworld", 995)]
n = 6
W, H = 720, 420
svg = SVG(W, H, "Teacher forcing：一次前向传播，每个位置一个损失")

X0, DX, BW = 20, 90, 84


def cx(i):
    return X0 + i * DX + BW / 2


# rows (top to bottom)
Y_TGT, Y_LOSS, Y_P, Y_M0, Y_M1, Y_IN = 34, 88, 104, 160, 222, 262

svg.text(X0, 22, "目标 = 输入整体左移一位（下一个真实 token）", size=12, color="green", weight="600")
for i in range(n):
    t, tid = toks[i + 1]
    svg.rect(X0 + i * DX, Y_TGT, BW, 30, fill="green", fill_opacity=0.10, stroke="green", rx=6)
    svg.text(cx(i), Y_TGT + 20, t, size=12, anchor="middle", color="green")
    # loss
    svg.text(cx(i) + 4, Y_LOSS, "ℓ_{%d}" % (i + 1), size=13, anchor="middle", color="red", weight="600")
    svg.arrow(cx(i) - 12, Y_TGT + 31, cx(i) - 12, Y_LOSS - 6, color="green", width=1.2)
    # predicted distribution
    svg.rect(X0 + i * DX, Y_P, BW, 28, fill="blue", fill_opacity=0.10, stroke="blue", rx=6)
    svg.text(cx(i), Y_P + 19, "p(·|x_{≤%d})" % (i + 1), size=12, anchor="middle", color="blue")
    svg.arrow(cx(i) - 12, Y_P - 1, cx(i) - 12, Y_LOSS - 2, color="blue", width=1.2)
    # model -> distribution
    svg.arrow(cx(i), Y_M0, cx(i), Y_P + 30, color="blue", width=1.2)
    # input
    t, tid = toks[i]
    svg.rect(X0 + i * DX, Y_IN, BW, 30, fill="blue", fill_opacity=0.14, stroke="blue", rx=6)
    svg.text(cx(i), Y_IN + 20, t, size=12, anchor="middle", color="ink")
    svg.text(cx(i), Y_IN + 46, "id %d" % tid, size=11, anchor="middle", color="muted")
    svg.text(cx(i), Y_IN + 64, "t = %d" % (i + 1), size=12, anchor="middle", color="muted")
    svg.arrow(cx(i), Y_IN - 1, cx(i), Y_M1 + 2, color="ink", width=1.2)


# the model block
svg.rect(X0, Y_M0, n * DX - (DX - BW), Y_M1 - Y_M0, fill="muted", fill_opacity=0.08, stroke="ink", width=1.3, rx=10)
svg.text(X0 + (n * DX - (DX - BW)) / 2, Y_M0 + 27, "GPT：一次前向传播，所有位置并行计算", size=13, anchor="middle",
         color="ink", weight="600")
svg.text(X0 + (n * DX - (DX - BW)) / 2, Y_M0 + 47, "因果掩码保证位置 t 的输出只依赖 x_{1}…x_{t}（右图）", size=12,
         anchor="middle", color="muted")

svg.text(X0, Y_IN + 90, "↑ 输入是真实 token，不是模型自己生成的", size=12, color="blue", weight="600")

# document boundary: in the input row the second document starts at column 5 ("Hello"), but in the
# target row (shifted left by one) it starts at column 4 (target "Hello"), so the boundary is drawn as
# two segments: between columns 3|4 through the target / loss / prediction rows, and between columns
# 4|5 through the input row.
xt_ = X0 + 3 * DX + BW + (DX - BW) / 2
xb = X0 + 4 * DX + BW + (DX - BW) / 2
svg.line(xt_, Y_TGT - 4, xt_, Y_P + 30, color="amber", width=1.4, dash="5 4")
svg.line(xb, Y_M1 + 4, xb, Y_IN + 70, color="amber", width=1.4, dash="5 4")
svg.text(xb + 6, Y_IN + 84, "文档边界：不另加掩码，", size=12, color="amber")
svg.text(xb + 6, Y_IN + 100, "“Hello”仍能看到上一篇", size=12, color="amber")

# causal mask inset
MX, MY, C = 580, 150, 20
svg.text(MX + 3 * C, MY - 26, "因果掩码", size=13, anchor="middle", color="ink", weight="600")
svg.text(MX + 3 * C, MY - 10, "行 t = 查询，列 j = 键", size=11, anchor="middle", color="muted")
for r in range(n):
    for c_ in range(n):
        if c_ <= r:
            svg.rect(MX + c_ * C + 1, MY + r * C + 1, C - 2, C - 2, fill="blue", fill_opacity=0.35, rx=2)
        else:
            svg.rect(MX + c_ * C + 1, MY + r * C + 1, C - 2, C - 2, fill="red", fill_opacity=0.12, rx=2)
            svg.text(MX + c_ * C + C / 2, MY + r * C + 14, "×", size=11, anchor="middle", color="red")
for k in range(n):
    svg.text(MX - 6, MY + k * C + 14, str(k + 1), size=11, anchor="end", color="muted")
    svg.text(MX + k * C + C / 2, MY + n * C + 14, str(k + 1), size=11, anchor="middle", color="muted")
svg.rect(MX, MY + n * C + 26, 12, 12, fill="blue", fill_opacity=0.35)
svg.text(MX + 17, MY + n * C + 37, "可见（j ≤ t）", size=12, color="ink")
svg.rect(MX, MY + n * C + 46, 12, 12, fill="red", fill_opacity=0.12)
svg.text(MX + 17, MY + n * C + 57, "置 −∞，权重为 0", size=12, color="ink")

svg.text(X0, H - 30, "ℓ_{t} = −log p_{θ}(x_{t+1} | x_{≤t})：7 个 token → 6 个输入、6 个目标、6 个损失，", size=12, color="ink")
svg.text(X0, H - 8, "一次前向全部得到；这条序列的损失 L = (ℓ_{1} + … + ℓ_{6}) / 6", size=12, color="ink")

svg.save("l02-teacher-forcing.svg")
