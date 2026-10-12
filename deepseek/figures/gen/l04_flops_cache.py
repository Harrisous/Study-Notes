"""Figure 4.5: decode FLOPs with and without a KV cache, Llama-2-7B (bf16 irrelevant: FLOPs only).

Model: l = 32 layers, n_h d_h = 4096, N_mm = 6.607e9 parameters in matrix multiplies (lesson-04-doc §4.4).
  with cache, one step at position t    : C_step(t) = 2 N_mm + 4 l n_h d_h t
  without cache, one step = full forward : C_fwd(t)  = 2 N_mm t + 2 l n_h d_h t (t + 1)
  generate T tokens from scratch         : sum_{t=1..T} C_step(t)  vs  sum_{t=1..T} C_fwd(t)
The cached total equals C_fwd(T) exactly (one prefill of T tokens); asserted below.
Writes deepseek/figures/l04-flops-cache.svg
"""

import math

from l04_data import MODELS, n_mm_llama
from l04_svg import SVG

m = MODELS["Llama-2-7B"]
L, A = m["l"], m["nh"] * m["dh"]
NMM = n_mm_llama(m)
TMAX = 4096


def step_cache(t):
    return 2 * NMM + 4 * L * A * t


def step_nocache(t):
    return 2 * NMM * t + 2 * L * A * t * (t + 1)


cum_c, cum_n = [0.0], [0.0]
for t in range(1, TMAX + 1):
    cum_c.append(cum_c[-1] + step_cache(t))
    cum_n.append(cum_n[-1] + step_nocache(t))

# checks against lesson-04-doc §4.4
assert abs(step_cache(4096) / 1.536e10 - 1) < 1e-3
assert abs(step_nocache(4096) / 5.852e13 - 1) < 1e-3
assert round(step_nocache(4096) / step_cache(4096)) == 3810
assert abs(cum_n[TMAX] / 1.169e17 - 1) < 1e-3
assert abs(cum_c[TMAX] - step_nocache(TMAX)) < 1  # cached total == one prefill of T tokens
share = 4 * L * A * 4096 / step_cache(4096)
assert abs(share - 0.140) < 0.001

W, H = 720, 610
svg = SVG(W, H, "Llama-2-7B 解码：有 KV cache 与没有 KV cache 的 FLOPs")
X0, X1 = 92, 560
LX0, LX1 = 0, 12  # log2 t from 1 to 4096


def px(t):
    return X0 + (math.log2(t) - LX0) / (LX1 - LX0) * (X1 - X0)


def panel(y0, y1, ly0, ly1, title, ylab):
    def py(v):
        return y1 - (math.log10(v) - ly0) / (ly1 - ly0) * (y1 - y0)
    svg.text(16, y0 - 16, title, size=14, weight="600")
    for e in range(ly0, ly1 + 1):
        y = py(10 ** e)
        svg.line(X0, y, X1, y, color="grid", width=1)
        svg.text(X0 - 8, y + 4, "10^{%d}" % e, size=11, anchor="end", color="muted")
    for t, lab in ((1, "1"), (4, "4"), (16, "16"), (64, "64"), (256, "256"), (1024, "1K"), (4096, "4K")):
        x = px(t)
        svg.line(x, y0, x, y1, color="grid", width=1)
        svg.text(x, y1 + 17, lab, size=11, anchor="middle", color="muted")
    svg.line(X0, y1, X1, y1, color="muted", width=1.2)
    svg.line(X0, y0, X0, y1, color="muted", width=1.2)
    svg.text(24, (y0 + y1) / 2, ylab, size=12, anchor="middle", color="muted", rotate=-90)
    return py


def poly(xs, ys, px_, py_):
    return "M" + " L".join("%.1f %.1f" % (px_(x), py_(y)) for x, y in zip(xs, ys))


ts = sorted(set([1] + [int(round(2 ** (k / 16))) for k in range(0, 16 * 12 + 1)] + [TMAX]))

# ---- panel A: one step
ya0, ya1 = 44, 250
pyA = panel(ya0, ya1, 9, 14, "Ⓐ 生成一个 token 的 FLOPs（上下文里已有 t 个 token）", "单步 FLOPs（对数）")
svg.path(poly(ts, [step_nocache(t) for t in ts], px, pyA), stroke="red", width=2.6)
svg.path(poly(ts, [step_cache(t) for t in ts], px, pyA), stroke="blue", width=2.6)
xe = px(TMAX)
for v, col in ((step_nocache(TMAX), "red"), (step_cache(TMAX), "blue")):
    svg.circle(xe, pyA(v), 4, fill=col)
svg.text(X1 + 10, pyA(step_nocache(TMAX)) + 4, "不缓存：%.2f×10^{13}" % (step_nocache(TMAX) / 1e13), size=12, color="red", weight="600")
svg.text(X1 + 10, pyA(step_nocache(TMAX)) + 20, "每步把 t 个 token", size=12, color="muted")
svg.text(X1 + 10, pyA(step_nocache(TMAX)) + 36, "整段重算一遍", size=12, color="muted")
svg.text(X1 + 10, pyA(step_cache(TMAX)) - 6, "有缓存：%.3f×10^{10}" % (step_cache(TMAX) / 1e10), size=12, color="blue", weight="600")
svg.text(X1 + 10, pyA(step_cache(TMAX)) + 10, "只算新 token，其中", size=12, color="muted")
svg.text(X1 + 10, pyA(step_cache(TMAX)) + 26, "注意力占 %.0f%%" % (share * 100), size=12, color="muted")
# ratio bracket at t = 4096
yb1, yb2 = pyA(step_nocache(TMAX)), pyA(step_cache(TMAX))
svg.line(xe - 14, yb1, xe - 14, yb2, color="ink", width=1.2)
svg.line(xe - 18, yb1, xe - 10, yb1, color="ink", width=1.2)
svg.line(xe - 18, yb2, xe - 10, yb2, color="ink", width=1.2)
svg.text(xe - 20, (yb1 + yb2) / 2 + 4, "%d 倍" % round(step_nocache(TMAX) / step_cache(TMAX)), size=13, anchor="end", weight="700")
svg.text((X0 + X1) / 2, ya1 + 36, "当前位置 t（对数刻度，1K = 1024）", size=12, anchor="middle", color="muted")

# ---- panel B: generate T tokens from scratch
yb0, yb1_ = 330, 540
pyB = panel(yb0, yb1_, 9, 18, "Ⓑ 从头生成 T 个 token 的累计 FLOPs", "累计 FLOPs（对数）")
svg.path(poly(ts, [cum_n[t] for t in ts], px, pyB), stroke="red", width=2.6)
svg.path(poly(ts, [cum_c[t] for t in ts], px, pyB), stroke="blue", width=2.6)
svg.path(poly(ts, [step_nocache(t) for t in ts], px, pyB), stroke="ink", width=1.4, dash="2 4")
for v, col in ((cum_n[TMAX], "red"), (cum_c[TMAX], "blue")):
    svg.circle(xe, pyB(v), 4, fill=col)
svg.text(X1 + 10, pyB(cum_n[TMAX]) + 4, "不缓存：%.2f×10^{17}" % (cum_n[TMAX] / 1e17), size=12, color="red", weight="600")
svg.text(X1 + 10, pyB(cum_n[TMAX]) + 20, "主项 ∝ T^{2}，更长时 T^{3}", size=12, color="muted")
svg.text(X1 + 10, pyB(cum_c[TMAX]) - 22, "有缓存：%.3f×10^{13}" % (cum_c[TMAX] / 1e13), size=12, color="blue", weight="600")
svg.text(X1 + 10, pyB(cum_c[TMAX]) - 6, "点线：一次 prefill T 个", size=12, color="muted")
svg.text(X1 + 10, pyB(cum_c[TMAX]) + 10, "token，与蓝线完全重合", size=12, color="muted")
r = cum_n[TMAX] / cum_c[TMAX]
svg.text(px(1.4), pyB(2e16), "T = 4096 时两者相差 %d 倍，约为 T/2" % round(r), size=13, weight="700")
svg.text((X0 + X1) / 2, yb1_ + 36, "生成的 token 数 T（对数刻度）", size=12, anchor="middle", color="muted")
svg.text(16, H - 12, "Llama-2-7B：l = 32，n_{h}d_{h} = 4096，N_{mm} = 6.607×10^{9}（含输出层）；曲线逐点计算。",
         size=12, color="muted")

svg.save("l04-flops-cache.svg")
