"""Figure 9.9: anatomy of one head's 192-dim query and key under decoupled RoPE (DeepSeek-V2).

q_{t,i} = [q^C_{t,i} (128) ; q^R_{t,i} (64)],  k_{j,i} = [k^C_{j,i} (128) ; k^R_j (64)]
(arXiv:2405.04434v5 eqs. 14-19, d_h = 128, d_h^R = 64).  k^R_j has no head index: one vector,
shared by all 128 heads.  Widths are to scale (1 dim = 2 px).
Writes deepseek/figures/l09-decoupled-key.svg
"""

from l09_svg import SVG

d_h, d_r, d_c = 128, 64, 512
U = 2.0
W, H = 740, 452
svg = SVG(W, H, "解耦 RoPE：一个头的 192 维查询与键")
X = 118
wc, wr = d_h * U, d_r * U


def seg(x, y, w, h, col, label, rot=False, shared=False, op=0.22):
    svg.rect(x, y, w, h, fill=col, fill_opacity=op, stroke="red" if rot else col, width=2.0 if rot else 1.2,
             rx=3, dash="5 3" if shared else None)
    svg.text(x + w / 2, y + h / 2 + 5, label, size=13, anchor="middle", color=col if col != "ink" else "ink",
             weight="700")


svg.text(16, 24, "第 i 个头：查询和键各 192 维 = 128 维内容 + 64 维位置", size=14, weight="700")
yq, yk, h = 50, 118, 32
svg.text(X - 12, yq + 21, "查询 *q*_{t,i}", size=13, anchor="end", color="blue", weight="600")
seg(X, yq, wc, h, "blue", "*q*^{C}_{t,i}（128）")
seg(X + wc, yq, wr, h, "blue", "*q*^{R}_{t,i}（64）", rot=True)
svg.text(X + wc / 2, yq - 6, "= *W*^{UQ}_{i}*c*^{Q}_{t}，不旋转", size=11.5, anchor="middle", color="muted")
svg.text(X + wc + wr / 2, yq - 6, "RoPE(*W*^{QR}_{i}*c*^{Q}_{t})，每头一个", size=11.5, anchor="middle", color="red")
svg.text(X - 12, yk + 21, "键 *k*_{j,i}", size=13, anchor="end", color="violet", weight="600")
seg(X, yk, wc, h, "violet", "*k*^{C}_{j,i}（128）")
seg(X + wc, yk, wr, h, "red", "*k*^{R}_{j}（64）", rot=True, op=0.16)
svg.text(X + wc / 2, yk + h + 16, "= *W*^{UK}_{i}*c*^{KV}_{j}，每头不同，不旋转", size=11.5, anchor="middle", color="muted")
svg.text(X + wc + wr / 2, yk + h + 16, "RoPE(*W*^{KR}*h*_{j})，没有下标 i", size=11.5, anchor="middle", color="red")
# score split
yb = 196
svg.path("M%s %s v8 h%s v-8" % (X + 2, yb, wc - 4), stroke="blue", width=1.4)
svg.path("M%s %s v8 h%s v-8" % (X + wc + 2, yb, wr - 4), stroke="red", width=1.4)
svg.text(X + wc / 2, yb + 26, "内容分数 (*q*^{C}_{t,i})^{⊤}*k*^{C}_{j,i}：没有旋转夹在中间 → 可吸收", size=12,
         anchor="middle", color="blue")
svg.text(X + wc + wr / 2 + 36, yb + 46, "位置分数 (*q*^{R}_{t,i})^{⊤}*k*^{R}_{j}：含 *R*_{j−t}", size=12,
         anchor="middle", color="red")
svg.text(X + wc + wr + 14, yq + 21, "拼接", size=12, color="muted")
svg.text(X + wc + wr + 14, yk + 21, "拼接", size=12, color="muted")
svg.text(X + wc + wr + 58, (yq + yk) / 2 + 26, "分数 = 两部分之和，", size=12)
svg.text(X + wc + wr + 58, (yq + yk) / 2 + 44, "再除以 √192", size=12)
svg.line(12, 262, W - 12, 262, color="rule", width=1)

# -------- sharing across heads
svg.text(16, 286, "同一个 token j 在不同头里的键：内容部分各不相同，位置部分是同一个向量", size=13.5, weight="700")
hh = 22
heads = [(1, 0.30), (2, 0.18), (128, 0.40)]
y = 302
for n, (i, op) in enumerate(heads):
    yy = y + n * 32 + (14 if i == 128 else 0)
    svg.text(X - 12, yy + 16, "头 %d" % i, size=12, anchor="end", color="muted")
    seg(X, yy, wc, hh, "violet", "", op=op)
    svg.text(X + wc / 2, yy + 16, "*W*^{UK}_{%d}*c*^{KV}_{j}" % i, size=12, anchor="middle", color="violet")
    seg(X + wc, yy, wr, hh, "red", "", rot=True, op=0.16)
    svg.text(X + wc + wr / 2, yy + 16, "*k*^{R}_{j}", size=12, anchor="middle", color="red")
svg.text(X - 12, y + 2 * 32 - 2, "⋮", size=14, anchor="end", color="muted")
svg.text(X + wc / 2, y + 2 * 32 + 4, "⋮", size=14, anchor="middle", color="muted")
# cache box
cx, cy = X + wc + wr + 24, 318
svg.rect(cx, cy, 198, 96, fill="amber", fill_opacity=0.06, stroke="amber", width=1.6, rx=6)
svg.text(cx + 10, cy + 20, "推理时每 token 每层缓存：", size=12, color="amber", weight="600")
svg.rect(cx + 10, cy + 30, 102, 18, fill="orange", fill_opacity=0.28, stroke="amber", width=2, rx=3)
svg.text(cx + 61, cy + 44, "*c*^{KV}_{j}  512", size=12, anchor="middle", color="orange", weight="700")
svg.rect(cx + 116, cy + 30, 50, 18, fill="red", fill_opacity=0.16, stroke="amber", width=2, rx=3)
svg.text(cx + 141, cy + 44, "*k*^{R}_{j} 64", size=11.5, anchor="middle", color="red", weight="700")
svg.text(cx + 10, cy + 70, "= 576 个数；", size=12)
svg.text(cx + 10, cy + 87, "同形状 MHA 要存 32,768 个", size=12, color="muted")
svg.save("l09-decoupled-key.svg")
