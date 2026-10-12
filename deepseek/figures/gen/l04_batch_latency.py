"""Figure 4.9: ideal decode step time versus batch on one A100 80GB SXM, context 4096, bf16.

  bytes = 2 * N_mm + B * kv_bytes_per_token * t        (weights read once, each sequence reads its own KV)
  flops = B * (2 * N_mm + 4 * l * n_h * d_h * t)
  step  = max(bytes / 2.039e12, flops / 312e12)         (idealised lower bound: no activations, no overheads)
  max B = (80e9 - weight bytes) / (kv_bytes_per_token * 4096)
Llama-2-7B (MHA, 512 KiB/token) vs Llama-3-8B (GQA-8, 128 KiB/token). Reproduces lesson-04-doc §4.6.
Writes deepseek/figures/l04-batch-latency.svg
"""

from l04_data import HW, MODELS, kv_bytes, n_mm_llama, params_llama
from l04_svg import SVG

A = HW["A100"]
T_CTX = 4096


def model_numbers(name):
    m = MODELS[name]
    nmm = n_mm_llama(m)
    kv = kv_bytes(m)
    wbytes = 2 * params_llama(m)
    bmax = int((A["HBM"] - wbytes) // (kv * T_CTX))

    def at(B):
        mem_w = 2 * nmm / A["BW"]
        mem_kv = B * kv * T_CTX / A["BW"]
        comp = B * (2 * nmm + 4 * m["l"] * m["nh"] * m["dh"] * T_CTX) / A["F"]
        step = max(mem_w + mem_kv, comp)
        return mem_w * 1e3, mem_kv * 1e3, comp * 1e3, step * 1e3, B / step
    return bmax, at


b2, at2 = model_numbers("Llama-2-7B")
b3, at3 = model_numbers("Llama-3-8B")
assert b2 == 30 and b3 == 119
assert abs(at2(1)[3] - 7.53) < 0.01 and abs(at2(30)[3] - 38.08) < 0.01 and round(at2(30)[4]) == 788
assert abs(at3(1)[3] - 7.62) < 0.01 and abs(at3(119)[3] - 38.69) < 0.01 and round(at3(119)[4]) == 3075
assert round(at2(1)[4]) == 133 and round(at3(1)[4]) == 131

W, H = 720, 560
X0, X1 = 70, 560
BMAX_AX = 125
svg = SVG(W, H, "一张 A100 上 decode 每步时间下界随 batch 的变化：Llama-2-7B 与 Llama-3-8B")


def px(B):
    return X0 + B / BMAX_AX * (X1 - X0)


def panel(y0, y1, name, tag, bmax, at):
    def py(ms):
        return y1 - ms / 45 * (y1 - y0)
    svg.text(16, y0 - 14, "%s（%s）：显存最多装下 %d 条 4096 token 的序列" % (name, tag, bmax), size=14, weight="600")
    for ms in (0, 10, 20, 30, 40):
        y = py(ms)
        svg.line(X0, y, X1, y, color="grid", width=1)
        svg.text(X0 - 8, y + 4, str(ms), size=11, anchor="end", color="muted")
    for B in (1, 20, 40, 60, 80, 100, 120):
        svg.line(px(B), y0, px(B), y1, color="grid", width=1)
        svg.text(px(B), y1 + 16, str(B), size=11, anchor="middle", color="muted")
    svg.line(X0, y1, X1, y1, color="muted", width=1.2)
    svg.line(X0, y0, X0, y1, color="muted", width=1.2)
    svg.text(22, (y0 + y1) / 2, "每步时间（ms）", size=12, anchor="middle", color="muted", rotate=-90)
    Bs = list(range(1, bmax + 1))
    w_top = [(px(B), py(at(B)[0])) for B in Bs]
    kv_top = [(px(B), py(at(B)[0] + at(B)[1])) for B in Bs]
    base = [(px(B), py(0)) for B in Bs]
    svg.polygon(base + w_top[::-1], fill="muted", fill_opacity=0.28)
    svg.polygon(w_top + kv_top[::-1], fill="amber", fill_opacity=0.45)
    svg.path("M" + " L".join("%.1f %.1f" % p for p in kv_top), stroke="ink", width=2)
    svg.path("M" + " L".join("%.1f %.1f" % (px(B), py(at(B)[2])) for B in Bs), stroke="blue", width=2)
    xm = px(bmax)
    svg.line(xm, y0 + 4, xm, y1, color="red", width=1.5, dash="5 3")
    svg.text(xm + 6, y0 + 16, "B = %d：显存装满" % bmax, size=12, color="red")
    mw, mkv, comp, step, thr = at(1)
    lx, ly = (px(62), py(12)) if bmax < 60 else (px(4), py(27))
    svg.line(px(1) + 3, py(step) - 3, lx - 4, ly - 4, color="ink", width=0.9)
    svg.circle(px(1), py(step), 3.2, fill="ink")
    svg.text(lx, ly, "B = 1：%.2f ms，%d token/s" % (step, round(thr)), size=12, weight="600")
    mw, mkv, comp, step, thr = at(bmax)
    svg.text(xm + 6, py(step) + 4, "%.1f ms" % step, size=12, weight="600")
    svg.text(xm + 6, py(step) + 20, "总吞吐 %d token/s" % round(thr), size=12, weight="600")
    svg.text(xm + 6, py(step) + 36, "每条序列 %.1f token/s" % (1e3 / step), size=12, color="muted")
    svg.text(xm + 6, py(step) + 52, "KV 占访存 %.0f%%" % (100 * mkv / (mw + mkv)), size=12, color="amber")
    # the grey band ends at B = bmax; with bmax = 30 the long label would cross the red dashed line
    lab = "灰色：读一遍权重（与 B 无关）" if bmax >= 60 else "灰色：读一遍权重"
    svg.text(px(3), py(at(1)[0]) + 16, lab, size=12, color="muted")
    return py


pyA = panel(46, 226, "Llama-2-7B", "MHA，512 KiB/token", b2, at2)
pyB = panel(296, 476, "Llama-3-8B", "GQA-8，128 KiB/token", b3, at3)
svg.text(px(62), pyA(24), "黄色：读 KV cache，随 B 线性增长", size=12, color="amber", weight="600")
svg.text(px(62), pyA(4.5), "蓝线：计算时间，远在访存时间之下", size=12, color="blue")
svg.text((X0 + X1) / 2, H - 42, "batch B（同时解码的序列数）", size=12, anchor="middle", color="muted")
svg.text(16, H - 14, "A100 80GB SXM：2.039 TB/s，312 TFLOP/s；上下文 4096，bf16；忽略激活、碎片与 kernel 开销的理想下界。",
         size=12, color="muted")
svg.save("l04-batch-latency.svg")
