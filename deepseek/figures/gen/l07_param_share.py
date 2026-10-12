"""Figure 7.1: where the parameters of a dense LLM live, and total vs activated parameters.

(a) DeepSeek LLM 7B (config: 30 layers, d = 4096, MHA, SwiGLU d_ff = 11008, vocab 102400,
    untied embedding / output head): attention 4d^2 per layer, FFN 3 d d_ff per layer.
(b) total vs per-token activated parameters. Dense models: recomputed from config;
    Mixtral recomputed from its config (46.70B / 12.88B); DeepSeekMoE 16B recomputed from
    config (16.376B / 2.829B); DeepSeek-V2-Lite as stated in the V2 paper (15.7B / 2.4B).
Writes deepseek/figures/l07-param-share.svg
"""

from l07_svg import SVG, fmt

# ---- (a) DeepSeek LLM 7B composition (all counts computed here)
L, d, dff, V = 30, 4096, 11008, 102400
att = L * 4 * d * d
ffn = L * 3 * d * dff
emb = V * d
head = V * d
norms = L * 2 * d + d
total = att + ffn + emb + head + norms
assert abs(total - 6_910_365_696) == 0

# ---- (b) total vs activated (billions)
def mixtral():
    a = 4096 * 4096 + 2 * 4096 * 1024 + 4096 * 4096
    e = 3 * 4096 * 14336
    base = 2 * 32000 * 4096 + 4096
    per = a + 8 * 4096 + 2 * 4096
    return (32 * (per + 8 * e) + base) / 1e9, (32 * (per + 2 * e) + base) / 1e9


def dsmoe16b():
    d_, V_, L_ = 2048, 102400, 28
    e = 3 * d_ * 1408
    common = 2 * V_ * d_ + d_ + L_ * (4 * d_ * d_ + 2 * d_) + 3 * d_ * 10944
    return (common + 27 * (66 * e + 64 * d_)) / 1e9, (common + 27 * (8 * e + 64 * d_)) / 1e9


def llama2_7b():
    L_, d_, f_, V_ = 32, 4096, 11008, 32000
    t = L_ * (4 * d_ * d_ + 3 * d_ * f_ + 2 * d_) + 2 * V_ * d_ + d_
    return t / 1e9, t / 1e9


mx_t, mx_a = mixtral()
ds_t, ds_a = dsmoe16b()
ll_t, ll_a = llama2_7b()
rows = [
    ("DeepSeek LLM 7B", total / 1e9, total / 1e9, "稠密"),
    ("LLaMA2 7B", ll_t, ll_a, "稠密"),
    ("Mixtral 8x7B", mx_t, mx_a, "MoE"),
    ("DeepSeekMoE 16B", ds_t, ds_a, "MoE"),
    ("DeepSeek-V2-Lite", 15.7, 2.4, "MoE"),
]

W, H = 720, 500
svg = SVG(W, H, "稠密模型的参数分布，以及稠密与 MoE 模型的总参数和激活参数")

# ================= panel (a)
svg.text(24, 26, "(a) DeepSeek LLM 7B 的参数都在哪里（共 %.2fB，B = 10^{9}）" % (total / 1e9), size=14, weight="600")
X0, X1, Y, BH = 24, 696, 44, 38
segs = [
    ("注意力", att, "ink", 0.28),
    ("FFN", ffn, "blue", 0.78),
    ("输入嵌入", emb, "muted", 0.30),
    ("输出头", head, "muted", 0.18),
]
x = X0
for name, v, col, op in segs:
    w = (X1 - X0) * v / total
    svg.rect(x, Y, w, BH, fill=col, fill_opacity=op, stroke="surface", width=1.5)
    share = 100 * v / total
    if w > 120:
        txt_col = "surface" if col == "blue" else "ink"
        svg.text(x + w / 2, Y + 17, name, size=13, anchor="middle", color=txt_col, weight="600")
        svg.text(x + w / 2, Y + 32, "%.2fB · %.1f%%" % (v / 1e9, share), size=12, anchor="middle", color=txt_col)
    else:
        svg.text(x + w / 2, Y + 24, "%.1f%%" % share, size=12, anchor="middle", color="ink")
    x += w
small_x0 = X0 + (X1 - X0) * (att + ffn) / total
svg.line(small_x0 + 2, Y + BH + 4, X1 - 2, Y + BH + 4, color="muted", width=1)
svg.text(X1, Y + BH + 20, "输入嵌入、输出头 各 %.2fB（不共享权重）" % (emb / 1e9), size=12, anchor="end", color="muted")

# per-layer comparison, drawn to scale
py = 138
svg.text(24, py, "每一层内部（按比例）：", size=13, color="ink", weight="600")
unit = 470 / (3 * d * dff)
for i, (lab, v, col, op, formula) in enumerate((
        ("FFN", 3 * d * dff, "blue", 0.78, "3·d·d_{ff} = 3×4096×11008"),
        ("注意力", 4 * d * d, "ink", 0.28, "4·d^{2} = 4×4096^{2}"))):
    yy = py + 14 + i * 34
    svg.rect(150, yy, v * unit, 22, fill=col, fill_opacity=op)
    svg.text(140, yy + 16, lab, size=13, anchor="end", color="ink")
    tc = "surface" if col == "blue" else "ink"
    svg.text(158, yy + 16, formula, size=12, color=tc)
    svg.text(150 + v * unit + 8, yy + 16, "%.1fM" % (v / 1e6), size=12, color="ink")
svg.text(150, py + 96, "FFN : 注意力 = 3·d_{ff} : 4d = %.2f，FFN 占每层参数的 %.1f%%" %
         (3 * dff / (4 * d), 100 * 3 * d * dff / (3 * d * dff + 4 * d * d)), size=12, color="muted")

# ================= panel (b)
top = 290
svg.text(24, top, "(b) 总参数（浅色长条）与每个 token 激活的参数（深色）", size=14, weight="600")
BX0, BX1, vmax = 172, 600, 50.0
ticks = (0, 10, 20, 30, 40, 50)
ry0 = top + 18
rh = 30
for t in ticks:
    xx = BX0 + (BX1 - BX0) * t / vmax
    svg.line(xx, ry0 - 4, xx, ry0 + rh * len(rows) + 2, color="grid", width=1)
    svg.text(xx, ry0 + rh * len(rows) + 16, "%dB" % t, size=11, anchor="middle", color="muted")
for i, (name, tot, act, kind) in enumerate(rows):
    yy = ry0 + i * rh
    wt = (BX1 - BX0) * tot / vmax
    wa = (BX1 - BX0) * act / vmax
    svg.text(BX0 - 10, yy + 16, name, size=12, anchor="end", color="ink")
    if kind == "MoE":
        svg.rect(BX0, yy + 3, wt, 18, fill="blue", fill_opacity=0.14, stroke="blue", width=1)
    svg.rect(BX0, yy + 3, wa, 18, fill="blue", fill_opacity=0.85)
    label = "%.1fB（稠密：全部激活）" % tot if kind == "稠密" else "%.1fB / %.1fB = %d%%" % (tot, act, round(100 * act / tot))
    svg.text(BX0 + wt + 8, yy + 16, label, size=12, color="ink" if kind == "MoE" else "muted")
svg.text(BX0, ry0 + rh * len(rows) + 36, "MoE 的每 token 计算量跟着“激活参数”走，知识容量跟着“总参数”走：两者被拆开了。",
         size=12, color="muted")

svg.save("l07-param-share.svg")
print("DeepSeek 7B total %.4fB attention %.4fB ffn %.4fB" % (total / 1e9, att / 1e9, ffn / 1e9))
print("Mixtral %.2f/%.2f  16B %.3f/%.3f  LLaMA2 %.3f" % (mx_t, mx_a, ds_t, ds_a, ll_t))
