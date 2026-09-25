"""Figure 4.6: KV cache bytes of real models (bf16).

Panel A: bytes per token = 2 * n_g * d_h * l * 2 bytes, log axis, with the MHA-equivalent
(n_g = n_h) drawn as a dashed ghost bar for the GQA models.
Panel B: one sequence at 4K / 32K / 128K tokens (GiB, log axis). Bars beyond a model's native
context are drawn hollow and labelled hypothetical. Vertical line: 80 GB = 74.5 GiB of HBM.
All specs from lesson-04-doc §3.1 / §4.1 (via l04_data.py). Writes deepseek/figures/l04-kv-bytes.svg
"""

import math

from l04_data import L405, MODELS, V3_MLA_BYTES, kv_bytes, kv_bytes_mha
from l04_svg import SVG

rows = []
for name in ("GPT-3 175B", "Llama-2-7B", "DeepSeek LLM 7B"):
    m = MODELS[name]
    rows.append((name, m["kind"], kv_bytes(m), None, m["ctx"]))
rows.append(("LLaMA-3.1 405B", "GQA-8", kv_bytes(L405), kv_bytes_mha(L405), None))
for name in ("DeepSeek LLM 67B", "Llama-2-70B", "Llama-3-8B"):
    m = MODELS[name]
    rows.append(("Llama-2/3-70B" if name == "Llama-2-70B" else name, m["kind"], kv_bytes(m), kv_bytes_mha(m), m["ctx"]))
assert kv_bytes(MODELS["Llama-3-70B"]) == kv_bytes(MODELS["Llama-2-70B"])  # same l, n_g, d_h: one row
V3_MHA = 2 * 128 * 128 * 61 * 2
rows.append(("DeepSeek-V3", "MLA", V3_MLA_BYTES, V3_MHA, 163840))
rows.sort(key=lambda r: -r[2])
assert V3_MHA == 3_997_696 and kv_bytes_mha(L405) == 8_257_536


def human(b):
    kib = b / 1024
    if kib >= 1024:
        return "%.1f MiB" % (kib / 1024) if kib % 1024 else "%d MiB" % (kib // 1024)
    return ("%.1f KiB" % kib) if kib != int(kib) else ("%d KiB" % kib)


W, H = 720, 680
svg = SVG(W, H, "真实模型的 KV cache：每 token 字节数与单条序列在 4K/32K/128K 时的大小（bf16）")
LAB = 170            # label column right edge
X0, X1 = 182, 600

# ------------------------------------------------------------------ panel A
svg.text(16, 24, "Ⓐ 每个 token 的 KV cache（2 · n_{g} · d_{h} · l · 2 字节，对数刻度）", size=14, weight="600")
LA0, LA1 = math.log2(32), math.log2(16384)   # KiB


def pxa(b):
    return X0 + (math.log2(b / 1024) - LA0) / (LA1 - LA0) * (X1 - X0)


ya, rh = 46, 30
for k in range(5, 15):
    x = pxa(2 ** k * 1024)
    svg.line(x, ya - 4, x, ya + rh * len(rows), color="grid", width=1)
    lab = ("%d KiB" % 2 ** k) if k < 10 else ("%d MiB" % 2 ** (k - 10))
    if k % 2 == 1:
        svg.text(x, ya + rh * len(rows) + 16, lab, size=11, anchor="middle", color="muted")
for i, (name, kind, b, bmha, ctx) in enumerate(rows):
    y = ya + i * rh
    col = "orange" if kind == "MLA" else "amber"
    svg.text(LAB, y + 13, name, size=12, anchor="end")
    svg.text(LAB, y + 26, kind if kind != "MLA" else "MLA（第 9 课）", size=11, anchor="end", color="muted")
    if bmha:
        svg.rect(X0, y + 4, pxa(bmha) - X0, 18, fill=None, stroke="muted", width=1.1, dash="4 3", rx=2)
    svg.rect(X0, y + 4, pxa(b) - X0, 18, fill=col, fill_opacity=0.55, rx=2)
    svg.text(pxa(b) + 6, y + 17, human(b), size=12, weight="600", color=col if kind == "MLA" else "ink")
    if bmha:
        svg.text(pxa(bmha) + 6, y + 17, "若用 MHA：%s（%.0f×）" % (human(bmha), bmha / b), size=11, color="muted")
svg.line(X0, ya - 4, X0, ya + rh * len(rows), color="muted", width=1.2)
yA_end = ya + rh * len(rows) + 22

# ------------------------------------------------------------------ panel B
yb = yA_end + 40
svg.text(16, yb - 12, "Ⓑ 一条序列的 KV cache：上下文 4K / 32K / 128K（GiB，对数刻度）", size=14, weight="600")
LB0, LB1 = math.log10(0.1), math.log10(1000)


def pxb(g):
    return X0 + (math.log10(g) - LB0) / (LB1 - LB0) * (X1 - X0)


CTXS = (4096, 32768, 131072)
rows = [r for r in rows if r[0] != "LLaMA-3.1 405B"]   # native context not verified in the notes
OPAC = (0.30, 0.55, 0.85)
rb = 34
nrows = len(rows)
for g, lab in ((0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000")):
    x = pxb(g)
    svg.line(x, yb + 4, x, yb + 8 + rb * nrows, color="grid", width=1)
    svg.text(x, yb + rb * nrows + 24, lab, size=11, anchor="middle", color="muted")
for i, (name, kind, b, bmha, ctx) in enumerate(rows):
    y = yb + 8 + i * rb
    col = "orange" if kind == "MLA" else "amber"
    svg.text(LAB, y + 18, name, size=12, anchor="end")
    for j, L in enumerate(CTXS):
        g = b * L / 2 ** 30
        yy = y + 3 + j * 9
        native = ctx is None or L <= ctx
        if native:
            svg.rect(X0, yy, pxb(g) - X0, 8, fill=col, fill_opacity=OPAC[j], rx=1)
        else:
            svg.rect(X0, yy, pxb(g) - X0, 8, fill=None, stroke=col, width=1, dash="3 2", rx=1)
        if j == 2:
            svg.text(X1 + 14, y + 18, "128K：" + ("%.3g GiB" % g if g < 10 else ("%.1f" % g).rstrip("0").rstrip(".") + " GiB"), size=12, color="ink")
svg.line(X0, yb + 4, X0, yb + 8 + rb * nrows, color="muted", width=1.2)
x80 = pxb(80e9 / 2 ** 30)
svg.line(x80, yb - 2, x80, yb + 8 + rb * nrows, color="red", width=1.5, dash="5 3")
svg.text(x80 + 4, yb + 2, "一张 80 GB 显卡", size=11, color="red")
svg.text(x80 + 4, yb + 15, "= 74.5 GiB", size=11, color="red")
# legend
yl = yb + rb * nrows + 44
lx = 16
for j, L in enumerate(CTXS):
    svg.rect(lx, yl - 9, 22, 8, fill="amber", fill_opacity=OPAC[j], rx=1)
    svg.text(lx + 28, yl, "%dK" % (L // 1024), size=12)
    lx += 76
svg.rect(lx, yl - 9, 22, 8, fill=None, stroke="amber", width=1, dash="3 2", rx=1)
svg.text(lx + 28, yl, "超出该模型原生上下文，只是“假如能跑这么长”的账单", size=12, color="muted")
svg.text(16, yl + 22, "原生上下文：GPT-3 2048，Llama-2 与 DeepSeek LLM 4096，Llama-3 8192，DeepSeek-V3 163840（config）。", size=12, color="muted")

svg.save("l04-kv-bytes.svg")
