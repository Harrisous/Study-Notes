"""Figure 19.2: the Causal Encoder-Decoder (CED) of DeepSeek-V4.1-Flash, redrawn after paper Fig. 3.

Layer groups, KV source layers and the 13-px-per-layer heights are taken from l19_num (config.json):
encoder = layers 0-19 (0-1 pure SWA; three groups of six CSA2 m=2 layers, Full + 5 Reuse),
decoder = layers 20-39 (five groups of four CSA2 m=1 layers; group 1 Full + 3 Reuse, groups 2-5
Reindex + 3 Reuse). Only layer 20 writes decoder global KV, projected from H_{L/2}.
Writes deepseek/figures/l19-ced.svg
"""

from l19_svg import SVG
from l19_num import MODES, MAIN_V41, IDX, ENGRAM

W, H = 760, 522
svg = SVG(W, H, "因果编码器–解码器：解码器的全局 KV 由编码器输出投影而来")

PER = 13          # px per layer
GAP = 7
Y0 = 372          # bottom of layer 0
ENC_X, DEC_X, CW = 150, 480, 176

name = {"swa": "纯 SWA", "full": "Full", "reindex": "Reindex", "reuse": "Reuse"}


def groups(first, sizes):
    out, l = [], first
    for n in sizes:
        out.append(list(range(l, l + n)))
        l += n
    return out


def stack(grps):
    y, boxes = Y0, []
    for g in grps:
        h = len(g) * PER
        boxes.append((g, y - h, h))
        y = y - h - GAP
    return boxes


enc = stack(groups(0, [2, 6, 6, 6]))
dec = stack(groups(20, [4, 4, 4, 4, 4]))
enc_top, dec_top = enc[-1][1], dec[-1][1]

# ---------------------------------------------------------------- embeddings
svg.box(ENC_X, Y0 + 14, CW, 30, ["嵌入：文本 token + 视觉 token"], color="muted", fill_opacity=0.10, size=12)
svg.arrow(ENC_X + CW / 2, Y0 + 14, ENC_X + CW / 2, Y0 + 2, color="ink", width=1.4)

# ---------------------------------------------------------------- encoder
svg.text(ENC_X + CW / 2, 22, "编码器：层 0–19", size=13, anchor="middle", weight="700")
for g, top, h in enc:
    if MODES[g[0]] == "swa":
        svg.box(ENC_X, top, CW, h, ["层 %d–%d · 纯 SWA" % (g[0], g[-1])], color="muted", fill_opacity=0.14, size=12)
        continue
    svg.rect(ENC_X, top, CW, h, fill="blue", fill_opacity=0.06, stroke="blue", width=1.2, rx=5)
    svg.rect(ENC_X, top + h - PER, CW, PER, fill="orange", fill_opacity=0.22, rx=3)
    svg.text(ENC_X + CW / 2, top + 22, "层 %d–%d · CSA2，m = 2" % (g[0], g[-1]), size=12, anchor="middle")
    svg.text(ENC_X + CW / 2, top + 40, "层 %d Full，其余 Reuse" % g[0], size=12, anchor="middle", color="muted")
    svg.text(ENC_X + CW / 2, top + h - 3.4, "层 %d" % g[0], size=11, anchor="middle", color="orange")
    ky = top + h - PER / 2
    svg.box(14, ky - 17, 112, 34, [("全局 KV：层 %d" % g[0], {}),
                                   ("m = 2，%d B/token" % ((MAIN_V41 + IDX) // 2), {"size": 11, "color": "muted"})],
            color="amber", fill_opacity=0.16, size=12, line_gap=15)
    svg.arrow(ENC_X, ky, 128, ky, color="amber", width=1.4)
for l in ENGRAM:
    g, top, h = next(b for b in enc if l in b[0])
    y = top + h - (l - g[0] + 0.5) * PER
    ex = ENC_X + CW - 14
    svg.polygon([(ex - 6, y), (ex, y - 6), (ex + 6, y), (ex, y + 6)], fill="green", fill_opacity=0.8)
svg.arrow(ENC_X + CW / 2, enc_top, ENC_X + CW / 2, enc_top - 22, color="ink", width=1.6)
svg.text(ENC_X + CW / 2, enc_top - 30, "*H*_{L/2}：层 19 的输出", size=13, anchor="middle", weight="700")

# ---------------------------------------------------------------- decoder
svg.text(DEC_X + CW / 2, 22, "解码器：层 20–39", size=13, anchor="middle", weight="700")
for g, top, h in dec:
    svg.rect(DEC_X, top, CW, h, fill="blue", fill_opacity=0.06, stroke="blue", width=1.2, rx=5)
    first = MODES[g[0]]
    col = "orange" if first == "full" else "violet"
    svg.rect(DEC_X, top + h - PER, CW, PER, fill=col, fill_opacity=0.22, rx=3)
    svg.text(DEC_X + CW / 2, top + 20, "层 %d–%d · CSA2，m = 1" % (g[0], g[-1]), size=12, anchor="middle")
    svg.text(DEC_X + CW / 2, top + h - 3.4, "层 %d %s，其余 Reuse" % (g[0], name[first]), size=11,
             anchor="middle", color=col)
svg.box(DEC_X, dec_top - 44, CW, 30, ["输出头 → 下一个 token"], color="green", fill_opacity=0.12, size=12)
svg.arrow(DEC_X + CW / 2, dec_top, DEC_X + CW / 2, dec_top - 13, color="ink", width=1.4)

# ---------------------------------------------------------------- the one decoder KV copy
MX = ENC_X + CW + 23
MW = DEC_X - MX - 23
ly = enc_top - 35
svg.line(ENC_X + CW / 2 + 90, ly, MX + MW / 2, ly, color="orange", width=1.6)
svg.arrow(MX + MW / 2, ly, MX + MW / 2, 148, color="orange", width=1.6)
svg.box(MX, 150, MW, 40, [("层 20 的 *W*^{KV}", {"weight": "700"}), ("投影一次", {"size": 11})],
        color="orange", fill_opacity=0.14, size=12, line_gap=16)
svg.arrow(MX + MW / 2, 190, MX + MW / 2, 212, color="amber", width=1.6)
svg.box(MX, 214, MW, 62, [("解码器全局 KV", {"weight": "700"}), "只有 1 份，m = 1",
                          ("%d B/token" % (MAIN_V41 + IDX), {"color": "muted"})],
        color="amber", fill_opacity=0.18, size=12, line_gap=17)
BUS = DEC_X - 14
ys = [top + h / 2 for g, top, h in dec]
svg.line(MX + MW, 245, BUS, 245, color="amber", width=1.4, dash="4 3")
svg.line(BUS, min(ys), BUS, max(ys), color="amber", width=1.4, dash="4 3")
for yy in ys:
    svg.arrow(BUS, yy, DEC_X - 1, yy, color="amber", width=1.3)

# ---------------------------------------------------------------- path bars along the columns
svg.line(ENC_X + CW + 7, Y0, ENC_X + CW + 7, enc_top, color="blue", width=3)
svg.line(ENC_X + CW + 14, Y0, ENC_X + CW + 14, enc_top, color="green", width=3)
svg.line(DEC_X + CW + 7, Y0, DEC_X + CW + 7, dec_top, color="blue", width=2, dash="3 4")
svg.line(DEC_X + CW + 14, Y0, DEC_X + CW + 14, dec_top, color="green", width=3)
svg.text(DEC_X + CW + 22, Y0 - 60, "prefill 只有", size=11, color="blue")
svg.text(DEC_X + CW + 22, Y0 - 45, "最后 128 个", size=11, color="blue")
svg.text(DEC_X + CW + 22, Y0 - 30, "token 走这里", size=11, color="blue")

# ---------------------------------------------------------------- legend
LY = 438
svg.line(20, LY, 56, LY, color="blue", width=3)
svg.text(64, LY + 4, "prefill：T 个提示 token 走编码器 20 层 + 层 20 的 KV 投影；只有最后 128 个再走解码器（回放）。约 8B 激活/token",
         size=12)
svg.line(20, LY + 23, 56, LY + 23, color="green", width=3)
svg.text(64, LY + 27, "decode：每个新 token 走全部 40 层和输出头。约 16B 激活/token", size=12)
svg.rect(20, LY + 39, 36, 12, fill="orange", fill_opacity=0.22)
svg.text(64, LY + 50, "Full 层（自己写全局 KV）", size=12)
svg.rect(250, LY + 39, 36, 12, fill="violet", fill_opacity=0.22)
svg.text(294, LY + 50, "Reindex 层（借 KV，自己重新选）", size=12)
svg.polygon([(520, LY + 45), (526, LY + 39), (532, LY + 45), (526, LY + 51)], fill="green", fill_opacity=0.8)
svg.text(540, LY + 50, "Engram（层 1、14）", size=12)
svg.text(20, LY + 74, "每一层（两半都算）都保留自己的 SWA：最近 128 个 token，FP8，每条目 528 字节，不跨层共享。",
         size=12, color="amber")

svg.save("l19-ced.svg")
