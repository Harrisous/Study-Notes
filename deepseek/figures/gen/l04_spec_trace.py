"""Figure 4.13: execution trace of speculative decoding with draft length gamma = 3 (schematic,
same idea as Leviathan et al. 2023 Fig. 5). Time runs left to right; a target-model call is drawn
100 units long and a draft-model call 18 units (c = 0.18, illustrative).
Round 1: drafts 山 ， 海; the target accepts 山 ， , rejects 海 and resamples 山 from the residual.
Round 2: drafts 里 有 座; all accepted, plus one bonus token 庙 sampled from p_{gamma+1}.
Plain decoding produces 5 tokens in 5 target calls; speculative decoding 7 tokens in 2 target calls.
Writes deepseek/figures/l04-spec-trace.svg
"""

from l04_svg import SVG

W, H = 720, 412
TGT, DFT, GAP = 100, 18, 4
X0 = 110
svg = SVG(W, H, "投机解码的执行轨迹（γ = 3）：小模型起草，大模型一次并行验证")


def tokbox(x, y, s, col, mark=None, strike=False):
    svg.rect(x, y, 24, 24, fill=col, fill_opacity=0.18, stroke=col, width=1.3, rx=4)
    svg.text(x + 12, y + 17, s, size=13, anchor="middle")
    if strike:
        svg.line(x + 3, y + 21, x + 21, y + 3, color="red", width=1.6)
    if mark:
        svg.text(x + 12, y + 40, mark, size=12, anchor="middle", color=col, weight="600")


# ---- row A: plain autoregressive decoding
ya = 60
svg.text(16, 28, "Ⓐ 普通解码：每调用一次目标模型，产出 1 个 token（5 次调用 → 5 个 token）", size=14, weight="600")
svg.text(X0 - 10, ya + 20, "目标模型", size=12, anchor="end", color="muted")
plain = ["山", "，", "山", "里", "有"]
for i, s in enumerate(plain):
    x = X0 + i * (TGT + GAP)
    svg.rect(x, ya, TGT, 30, fill="blue", fill_opacity=0.45, rx=4)
    svg.text(x + TGT / 2, ya + 20, "调用 %d" % (i + 1), size=12, anchor="middle")
    tokbox(x + TGT / 2 - 12, ya + 38, s, "green")

# ---- row B: speculative decoding
yb = 204
svg.text(16, 152, "Ⓑ 投机解码：草稿模型先连写 γ = 3 个，目标模型一次前向同时检查", size=14, weight="600")
svg.text(X0 - 10, yb - 22, "草稿模型", size=12, anchor="end", color="muted")
svg.text(X0 - 10, yb + 20, "目标模型", size=12, anchor="end", color="muted")
rounds = [
    (["山", "，", "海"], [("山", "green", "✓"), ("，", "green", "✓"), ("海", "red", "✗"), ("山", "blue", "修正")]),
    (["里", "有", "座"], [("里", "green", "✓"), ("有", "green", "✓"), ("座", "green", "✓"), ("庙", "blue", "附赠")]),
]
x = X0
for r, (drafts, outcome) in enumerate(rounds):
    # draft calls (small, sequential)
    for k, s in enumerate(drafts):
        svg.rect(x, yb - 36, DFT, 22, fill="blue", fill_opacity=0.18, stroke="blue", width=1, rx=3)
        svg.text(x + DFT / 2, yb - 20, s, size=12, anchor="middle")
        x += DFT + GAP
    # one target call verifying gamma + 1 positions in parallel
    svg.rect(x, yb, TGT, 30, fill="blue", fill_opacity=0.45, rx=4)
    svg.text(x + TGT / 2, yb + 20, "调用 %d：验证 4 处" % (r + 1), size=12, anchor="middle")
    ox = x + TGT / 2 - (4 * 28 - 4) / 2
    for k, (s, col, mark) in enumerate(outcome):
        if col == "red":
            tokbox(ox + k * 28, yb + 40, s, "red", mark, strike=True)
        else:
            tokbox(ox + k * 28, yb + 40, s, col, mark)
    x += TGT + GAP + 8
svg.text(x + 4, yb + 20, "2 次调用 → 7 个 token", size=12, weight="600")

yt = yb + 120
svg.line(16, yt - 18, W - 16, yt - 18, color="rule", width=1)
svg.text(16, yt, "目标模型对“前缀 + 3 个草稿”做一次前向，就同时得到 4 个位置的分布 p_{1}, …, p_{4}，", size=12)
svg.text(16, yt + 20, "像 prefill 一样并行。从左到右逐个按接受规则检查：第一个被拒的草稿（红）之后全部作废，", size=12)
svg.text(16, yt + 40, "并从残差分布补采一个 token（蓝，“修正”）；若 γ 个全部通过，再从 p_{4} 白拿一个（蓝，“附赠”）。", size=12)
svg.text(16, yt + 62, "所以每次调用目标模型至少产出 1 个 token，最多 γ + 1 个。时间比例为示意。", size=12, color="muted")
svg.save("l04-spec-trace.svg")
