"""Figure 17.15: DeepEncoder V2 of DeepSeek-OCR 2 and its attention mask (redrawn after OCR 2 figs. 3 and 5).

Mask, OCR 2 eq. (1): M = [[1_{m x m}, 0_{m x n}], [1_{n x m}, LowerTri(n)]], n = m; drawn at m = n = 4.
The lesson writes the paper's m (visual tokens) as n_v, the symbol of eq. (17.21), and n (queries) as n_q.
Real sizes (OCR 2 §3.2.3): global view 1024^2 -> 256 visual tokens, local view 768^2 -> 144; as many causal
queries as visual tokens; only the query outputs (last n positions) go to the decoder; total fed to the
decoder k x 144 + 256 for k = 0..6 local views -> 256 .. 1120. Encoder: SAM-base + two convs (last one 896-d)
-> Qwen2-0.5B-style LLM (500M); decoder unchanged 3B MoE.
Writes deepseek/figures/l17-ocr2-mask.svg
"""

from l17_svg import SVG

M = N = 4
mask = [[1 if (i < M and j < M) or (i >= M and j < M) or (i >= M and j >= M and j - M <= i - M) else 0
         for j in range(M + N)] for i in range(M + N)]
assert sum(map(sum, mask)) == M * M + N * M + N * (N + 1) // 2
assert [256 + 144 * k for k in (0, 6)] == [256, 1120]

W, H = 760, 414
svg = SVG(W, H, "DeepEncoder V2：视觉 token 双向注意力，因果流查询单向注意力，只把查询的输出交给解码器")

# ---------------------------------------------------------------- left: data flow
svg.text(20, 22, "DeepSeek-OCR 2 的编码器", size=13, weight="700")
svg.box(20, 40, 150, 44, ["SAM-base + 2 层卷积", "（末层输出 896 维）"], color="blue", fill_opacity=0.10, size=12)
svg.arrow(95, 86, 95, 110, color="ink", width=1.4)
# sequence: m visual tokens then n queries
SX, SY, CELL = 20, 114, 34
for j in range(M + N):
    col = "blue" if j < M else "violet"
    svg.rect(SX + j * CELL + 1, SY, CELL - 3, 30, fill=col, fill_opacity=0.16, stroke=col, width=1.1, rx=4)
    svg.text(SX + j * CELL + CELL / 2 - 1, SY + 20, ("v%d" % (j + 1)) if j < M else ("q%d" % (j - M + 1)), size=12,
             anchor="middle", color=col)
svg.text(SX + M * CELL / 2, SY + 48, "n_{v} 个视觉 token", size=11.5, anchor="middle", color="blue")
svg.text(SX + M * CELL + N * CELL / 2, SY + 48, "n_{q} = n_{v} 个因果流查询", size=11.5, anchor="middle", color="violet")
svg.rect(14, SY + 60, (M + N) * CELL + 12, 50, fill="ink", fill_opacity=0.04, stroke="ink", width=1.2, rx=6)
svg.text(20 + (M + N) * CELL / 2, SY + 81, "Qwen2-0.5B 结构的 LLM 式编码器（500M）", size=12, anchor="middle",
         weight="700")
svg.text(20 + (M + N) * CELL / 2, SY + 99, "替换原 DeepEncoder 里的 CLIP；按右图的掩码", size=11.5, anchor="middle",
         color="muted")
svg.arrow(SX + M * CELL + N * CELL / 2, SY + 112, SX + M * CELL + N * CELL / 2, SY + 140, color="violet", width=1.5)
svg.text(SX + M * CELL + N * CELL / 2 + 8, SY + 132, "只取后 n_{q} 个输出", size=11.5, color="violet")
svg.box(SX + M * CELL - 20, SY + 144, N * CELL + 40, 40, ["DeepSeek-3B MoE 解码器"], color="green", fill_opacity=0.10,
        size=12)
# kept narrower than the left column (x < 390) so it never runs under the mask's notes on the right
svg.text(20, SY + 228, "送进解码器的 token：全局视图 1024² → 256，", size=12)
svg.text(20, SY + 246, "每个局部视图 768² → 144，局部视图 k = 0–6 个：", size=12)
svg.text(20, SY + 264, "k × 144 + 256 ∈ [256, 1120]。", size=12)
svg.text(20, SY + 286, "DeepSeek-OCR 的 Gundam 模式最多 1156 个。", size=12, color="muted")

# ---------------------------------------------------------------- right: the mask
MX, MY, MC = 432, 58, 28
svg.text(MX - 26, 22, "注意力掩码（n_{v} = n_{q} = 4 的缩小版）", size=13, weight="700")
labels = ["v1", "v2", "v3", "v4", "q1", "q2", "q3", "q4"]
for i in range(M + N):
    for j in range(M + N):
        on = mask[i][j]
        col = "blue" if j < M else "violet"
        svg.rect(MX + j * MC + 1, MY + i * MC + 1, MC - 2, MC - 2, fill=col if on else "grid",
                 fill_opacity=0.45 if on else 0.6, rx=2)
        svg.text(MX + j * MC + MC / 2, MY + i * MC + MC / 2 + 4.5, str(on), size=11, anchor="middle",
                 color="ink" if on else "muted")
for k, lab in enumerate(labels):
    col = "blue" if k < M else "violet"
    svg.text(MX + k * MC + MC / 2, MY - 8, lab, size=11.5, anchor="middle", color=col)
    svg.text(MX - 8, MY + k * MC + MC / 2 + 4.5, lab, size=11.5, anchor="end", color=col)
svg.line(MX, MY + M * MC, MX + (M + N) * MC, MY + M * MC, color="ink", width=1.5)
svg.line(MX + M * MC, MY, MX + M * MC, MY + (M + N) * MC, color="ink", width=1.5)
svg.text(MX + (M + N) * MC + 8, MY + 2 * MC + 4, "视觉 token", size=11.5, color="blue")
svg.text(MX + (M + N) * MC + 8, MY + 2 * MC + 20, "互相全看", size=11.5, color="blue")
svg.text(MX + (M + N) * MC + 8, MY + 6 * MC - 4, "查询：看全部", size=11.5, color="violet")
svg.text(MX + (M + N) * MC + 8, MY + 6 * MC + 12, "视觉 token +", size=11.5, color="violet")
svg.text(MX + (M + N) * MC + 8, MY + 6 * MC + 28, "之前的查询", size=11.5, color="violet")
svg.text(MX - 26, MY + (M + N) * MC + 26, "行 = 谁在看，列 = 被看的；1 = 可见。", size=11.5, color="muted")
svg.text(MX - 26, MY + (M + N) * MC + 44, "真实规模：全局视图 256 + 256 = 512 行 × 512 列。", size=11.5, color="muted")
svg.save("l17-ocr2-mask.svg")
