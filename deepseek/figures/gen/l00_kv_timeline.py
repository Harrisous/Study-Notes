"""Figure 0.2: global KV-cache bytes per token of every DeepSeek main-line model, against its release date.

Bytes are COMPUTED from the architecture, with the formulas (0.1)-(0.5) of lesson 0 (research notes
lesson-00-doc.md section 2.5 and N1):
  dense MHA/GQA      2 * n_kv * d_h * l * b
  MLA (V2/V3)        (d_c + d_h^R) * l * b
  MLA + DSA (V3.2)   l * (656 + 132)                       FlashMLA FP8 row + FP8 indexer key
  CSA/HCA (V4)       sum_CSA (584 + 68)/m + sum_HCA 584/m'
  CSA2 (V4.1)        sum_Full (288 + 68)/m
The right axis converts bytes/token into GiB for one 2^20-token request (bytes * 2^20 / 2^30).

    python deepseek/figures/gen/l00_kv_timeline.py
"""

from __future__ import annotations

import datetime as dt
import math

from l00_svg import SVG, fmt


def dense(n_kv, d_h, l, b=2):
    return 2 * n_kv * d_h * l * b


def mla(l, d_c=512, d_r=64, b=2):
    return (d_c + d_r) * l * b


def v4(n_csa, n_hca, m=4, m2=128, main=584, idx=68):
    return n_csa * (main + idx) / m + n_hca * main / m2


def v41(full_m=(2, 2, 2, 1), main=288, idx=68):
    return sum((main + idx) / m for m in full_m)


# key, label, date, bytes, style, label (dx, dy, anchor)
POINTS = [
    ("7b", "7B（MHA）", "2023-11-29", dense(32, 128, 30), "dense", (10, -6, "start")),
    # label offsets keep every label off the dashed chain (checked against the segment geometry)
    ("67b", "67B（GQA-8）", "2023-11-29", dense(8, 128, 95), "main", (10, 4, "start")),
    ("16b", "MoE 16B（MHA）", "2024-01-11", dense(16, 128, 28), "dense", (10, 1, "start")),
    ("v2", "V2", "2024-05-06", mla(60), "main", (-9, 18, "end")),
    ("v2q", "V2 部署（平均 6 bit）", "2024-05-06", (512 + 64) * 60 * 6 / 8, "deploy", (10, 5, "start")),
    ("v3", "V3", "2024-12-26", mla(61), "main", (6, 19, "start")),
    ("v32", "V3.2", "2025-12-01", 61 * (656 + 132), "main", (10, 4, "start")),
    ("v4pro", "V4-Pro（推算）", "2026-04-24", v4(30, 31), "dense", (-29, 1, "end")),
    ("v4f", "V4-Flash", "2026-04-24", v4(21, 20), "main", (-9, 14, "end")),
    ("v41", "V4.1-Flash", "2026-09-10", v41(), "main", (-9, 17, "end")),
]
CHAIN = ["67b", "v2", "v3", "v32", "v4f", "v41"]
# where to print the ratio of each chain step: (dx, dy, anchor) from the segment midpoint
RATIO_POS = {("67b", "v2"): (-10, 8, "end"), ("v3", "v32"): (0, 20, "middle"),
             ("v32", "v4f"): (-8, -6, "end"), ("v4f", "v41"): (8, 4, "start")}
GPT3 = dense(96, 128, 96)          # 4,718,592

W, H = 740, 440
L, R, T, B = 78, 610, 30, 380      # plot box
D0, D1 = dt.date(2023, 10, 1), dt.date(2026, 10, 1)
Y0, Y1 = 2.5, 7.0                  # log10 bytes


def xd(d):
    return L + (R - L) * (d - D0).days / (D1 - D0).days


def yb(v):
    return B - (B - T) * (math.log10(v) - Y0) / (Y1 - Y0)


def pdate(s):
    return dt.date(*map(int, s.split("-")))


def main():
    s = SVG(W, H, title="每 token 全局 KV 字节随发布时间的变化")
    for e in range(2, 8):
        for k in range(1, 10):
            v = k * 10 ** e
            if 10 ** Y0 <= v <= 10 ** Y1:
                y = yb(v)
                s.line(L, y, R, y, color="grid" if k > 1 else "rule", width=0.8 if k > 1 else 1)
        v = 10 ** e
        if 10 ** Y0 <= v <= 10 ** Y1:
            s.text(L - 8, yb(v) + 4, "10^{%d}" % e, size=12, anchor="end", color="muted")
    s.text(L - 54, (T + B) / 2, "每 token 全局 KV（字节，对数轴）", size=12.5, anchor="middle", rotate=-90)
    for g in (1, 10, 100, 1000):
        v = g * 1024
        y = yb(v)
        s.line(R, y, R + 5, y, color="muted", width=1)
        s.text(R + 8, y + 4, "%s GiB" % g, size=12, color="muted")
    s.text(R + 104, (T + B) / 2, "一个 1M（2^{20}）token 请求的 KV", size=12.5, anchor="middle", rotate=90)
    s.line(L, B, R, B, color="muted", width=1)
    for yr in (2023, 2024, 2025, 2026):
        for mo in (1, 4, 7, 10):
            d = dt.date(yr, mo, 1)
            if D0 <= d <= D1:
                x = xd(d)
                s.line(x, B, x, B + (6 if mo == 1 else 3), color="muted", width=1)
                if mo == 1:
                    s.text(x, B + 20, str(yr), size=12.5, anchor="middle", weight=600)
                    s.line(x, T, x, B, color="rule", width=0.8)
                else:
                    s.text(x, B + 17, "%d 月" % mo, size=11, anchor="middle", color="muted")
    s.text((L + R) / 2, B + 44, "公开发布日", size=12.5, anchor="middle")

    v80 = 80e9 / 2 ** 20
    y80 = yb(v80)
    s.line(L, y80, R, y80, color="red", width=1.3, dash="5 4")
    s.text(R - 4, y80 - 6, "80 GB 显存恰好放下一个 1M token 请求的 KV（%s 字节/token）" % f"{round(v80):,}",
           size=11.5, anchor="end", color="red")
    yg = yb(GPT3)
    s.line(L, yg, R, yg, color="muted", width=1.2, dash="2 3")
    s.text(R - 4, yg - 6, "参照：GPT-3 175B（2020，MHA）%s" % f"{GPT3:,}", size=11.5, anchor="end", color="muted")

    pos = {k: (xd(pdate(ds)), yb(v), v) for k, _, ds, v, _, _ in POINTS}
    s.path("M" + " L".join("%s %s" % (fmt(pos[k][0]), fmt(pos[k][1])) for k in CHAIN), stroke="amber", width=1.6,
           dash="6 4", opacity=0.8)
    for a, b in zip(CHAIN, CHAIN[1:]):
        if (a, b) not in RATIO_POS:
            continue
        (x0, y0, v0), (x1, y1, v1) = pos[a], pos[b]
        dx, dy, anc = RATIO_POS[(a, b)]
        s.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, "÷%.2f" % (v0 / v1), size=12, anchor=anc, color="amber",
               weight=700)
    for key, lab, ds, v, style, (dx, dy, anc) in POINTS:
        x, y, _ = pos[key]
        if style == "main":
            s.circle(x, y, 5.5, fill="amber", stroke="amber")
        elif style == "deploy":
            s.circle(x, y, 5.5, fill="surface", stroke="amber", width=1.8)
        else:
            s.circle(x, y, 4.5, fill="surface", stroke="muted", width=1.6)
        s.text(x + dx, y + dy, "%s %s" % (lab, f"{round(v):,}"), size=12, anchor=anc,
               color="ink" if style != "dense" else "muted", weight=600 if style == "main" else None)
    s.save("l00-kv-timeline.svg")
    for key, lab, _, v, *_ in POINTS:
        print("%-6s %12.2f" % (key, v))


if __name__ == "__main__":
    main()
