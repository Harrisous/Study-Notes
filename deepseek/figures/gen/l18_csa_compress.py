"""Figure 18.3: CSA's overlapped compression with the real m = 4.

Entry s reads the 2m = 8 tokens 4(s-1) .. 4s+3: its own block via the a-branch and the previous
block via the b-branch (V4 report eqs. 9-12, research notes sec. 2.2); entry 0 has no previous
block. The inset heat map shows per-channel softmax weights for entry 2 over its 8 tokens and
6 of its 512 channels, computed from seeded random logits (illustration of the structure only).
Writes deepseek/figures/l18-csa-compress.svg
"""

import numpy as np

from l18_svg import SVG

M, NTOK = 4, 16
W, H = 760, 540
svg = SVG(W, H, "CSA 的重叠压缩：每个条目读 8 个 token、步长 4，权重逐通道 softmax")

X0, CW, TY = 96, 38, 44           # token row: x of token 0, cell width, top y


def tx(tau):                        # left edge of token tau
    return X0 + tau * CW


# ---------------------------------------------------------------- token row
svg.text(X0 - 10, TY + 22, "token", size=12, anchor="end", color="muted")
for tau in range(NTOK):
    svg.rect(tx(tau) + 2, TY, CW - 4, 30, fill="muted", fill_opacity=0.10, stroke="muted", width=1, rx=4)
    svg.text(tx(tau) + CW / 2, TY + 20, str(tau), size=12, anchor="middle")
for b in range(NTOK // M):
    svg.text(tx(b * M) + 2 * CW, TY - 8, "块 %d" % b, size=12, anchor="middle", color="muted")
    if b:
        svg.line(tx(b * M), TY - 18, tx(b * M), TY + 36, color="rule", width=1.2)
svg.text(tx(NTOK) + 8, TY + 20, "…", size=14)

# ---------------------------------------------------------------- entries with brackets
LV0, LSTEP = 104, 56
for s in range(NTOK // M):
    y = LV0 + s * LSTEP            # bracket level
    # a-branch: own block (green)
    xa0, xa1 = tx(M * s) + 4, tx(M * (s + 1)) - 4
    svg.line(xa0, y, xa1, y, color="green", width=3, cap="round")
    svg.line(xa0, y, xa0, y - 7, color="green", width=1.5)
    svg.line(xa1, y, xa1, y - 7, color="green", width=1.5)
    if s == 0:
        svg.text(tx(0) - 6, y + 4, "b 路为空", size=11, anchor="end", color="violet")
    else:
        xb0, xb1 = tx(M * (s - 1)) + 4, tx(M * s) - 4
        svg.line(xb0, y, xb1, y, color="violet", width=3, cap="round")
        svg.line(xb0, y, xb0, y - 7, color="violet", width=1.5)
        svg.line(xb1, y, xb1, y - 7, color="violet", width=1.5)
    # entry box hanging under the join point
    cx = tx(M * s)
    bx = cx - 30 if s else tx(0) + 2 * CW - 30
    joint = cx if s else tx(0) + 2 * CW
    svg.line(joint, y, joint, y + 12, color="orange", width=1.4)
    svg.rect(bx, y + 12, 60, 22, fill="orange", fill_opacity=0.16, stroke="orange", width=1.3, rx=4)
    svg.text(bx + 30, y + 27, "*e*_{%d}" % s, size=12, anchor="middle")

# legend (bottom left)
LY = 490
svg.line(24, LY, 50, LY, color="green", width=3, cap="round")
svg.text(56, LY + 4, "a 路：本块", size=12)
svg.line(150, LY, 176, LY, color="violet", width=3, cap="round")
svg.text(182, LY + 4, "b 路：前一块", size=12)
svg.rect(24, LY + 16, 26, 16, fill="orange", fill_opacity=0.16, stroke="orange", width=1.3, rx=3)
svg.text(56, LY + 29, "压缩条目（KV cache 里存的就是它）", size=12)

# ---------------------------------------------------------------- facts (bottom left)
FY = 344
facts = [
    "条目 s 读 token 4(s−1) … 4s+3，共 8 个；步长 4",
    "相邻条目重叠 4 个 token，条目数仍是 T/4",
    "条目 0 没有前一块：b 路打分填 −∞、内容填 0",
    "条目 s 在查询位置 t ≥ 4s+3 时才可见",
    "每个条目 512 维，同时当键和值",
]
svg.text(24, FY, "读图要点", size=13, weight="700")
for n, f in enumerate(facts):
    svg.text(24, FY + 24 + n * 22, "· " + f, size=12)

# ---------------------------------------------------------------- inset heat map for entry 2
rng = np.random.default_rng(18)
logits = rng.normal(0.0, 1.1, size=(8, 6))
wts = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
assert np.allclose(wts.sum(axis=0), 1.0)
HX, HY, CWH, CHH = 520, 350, 34, 18
svg.text(HX - 70, HY - 16, "条目 *e*_{2} 的权重（示意）", size=13, weight="700")
for r in range(6):
    svg.text(HX + r * CWH + CWH / 2, HY + 2, "r=%d" % (r + 1), size=11, anchor="middle", color="muted")
rows = [(4 + k, "b") for k in range(4)] + [(8 + k, "a") for k in range(4)]
for n, (tau, br) in enumerate(rows):
    y = HY + 8 + n * CHH
    svg.text(HX - 8, y + 13, "τ=%d（%s 路）" % (tau, br), size=11, anchor="end",
             color="violet" if br == "b" else "green")
    for r in range(6):
        v = wts[n, r]
        svg.rect(HX + r * CWH + 1, y + 1, CWH - 2, CHH - 2, fill="orange", fill_opacity=0.08 + 0.85 * v / wts.max(),
                 stroke=None, rx=2)
        if n == int(np.argmax(wts[:, r])):
            svg.rect(HX + r * CWH + 1, y + 1, CWH - 2, CHH - 2, stroke="ink", width=1.4, rx=2)
svg.line(HX, HY + 8 + 4 * CHH, HX + 6 * CWH, HY + 8 + 4 * CHH, color="rule", width=1.2, dash="3 3")
svg.text(HX + 3 * CWH, HY + 8 + 8 * CHH + 18, "每一列（一个通道）的 8 格之和为 1",
         size=11, anchor="middle", color="muted")
svg.text(HX + 3 * CWH, HY + 8 + 8 * CHH + 34, "黑框 = 该通道权重最大的 token",
         size=11, anchor="middle", color="muted")

svg.save("l18-csa-compress.svg")
