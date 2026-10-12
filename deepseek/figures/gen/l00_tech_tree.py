"""Figure 0.1: the DeepSeek tech tree.

Time axis 2017-01 -> 2026-10 (piecewise linear: 2017-01..2023-09 compressed, 2023-10..2026-09 expanded),
one header row of model releases and six lanes for the six threads (A-F). Every node is one paper or one
model, drawn as a card whose LEFT EDGE sits on its date; the circled number is the lesson that teaches it
(0 = only mentioned in this overview). Solid arrows = "builds on" within a thread; thin arrows in the model
row = weight lineage (continued training from the earlier checkpoint).

Every date is taken from deepseek/reference-docs/lesson-00-doc.md (T*, G*, S10) or from the research notes
of the lesson named on the card (lesson-01/02/03/04/05/07/11/12/16/18-doc.md). Models use the public release
date, papers without a model use the arXiv v1 date.

    python deepseek/figures/gen/l00_tech_tree.py      # writes deepseek/figures/l00-tech-tree.svg
"""

from __future__ import annotations

import datetime as dt

from l00_svg import SVG, text_width

# ---------------------------------------------------------------- data table
# lane key, label lines, colour
LANES = [
    ("M", ["模型发布"], "ink"),
    ("A", ["A 注意力与", "KV cache"], "blue"),
    ("B", ["B 前馈层稀疏化", "（MoE）"], "violet"),
    ("C", ["C 数值精度", "与训练系统"], "orange"),
    ("D", ["D 训练目标", "与后训练"], "green"),
    ("E", ["E 残差流", "与记忆"], "amber"),
    ("F", ["F 规模化规律", "与优化器"], "ink"),
]

# id, lane, date (YYYY-MM-DD), label, lessons, flagship
NODES = [
    # --- model releases (public release date; GPT side: announcement / arXiv v1, lesson-00-doc G1-G6)
    ("transformer", "M", "2017-06-12", "Transformer", [1], False),
    ("gpt1", "M", "2018-06-11", "GPT-1", [2], False),
    ("gpt2", "M", "2019-02-14", "GPT-2", [1], False),
    ("gpt3", "M", "2020-05-28", "GPT-3", [2], False),
    ("instructgpt", "M", "2022-01-27", "InstructGPT", [3], False),
    ("chatgpt", "M", "2022-11-30", "ChatGPT", [0], False),
    ("llama", "M", "2023-02-27", "LLaMA", [5], False),
    ("coder", "M", "2023-11-02", "DeepSeek-Coder", [6], False),
    ("llm", "M", "2023-11-29", "DeepSeek LLM 7B/67B", [6], True),
    ("moe16b", "M", "2024-01-11", "DeepSeekMoE 16B", [7], False),
    ("math", "M", "2024-02-05", "DeepSeekMath 7B", [8], False),
    ("v2", "M", "2024-05-06", "V2", [9, 10], True),
    ("v3", "M", "2024-12-26", "V3", [11, 12], True),
    ("r1", "M", "2025-01-20", "R1", [13], True),
    ("v31", "M", "2025-08-21", "V3.1", [15], False),
    ("v32exp", "M", "2025-09-29", "V3.2-Exp", [14], False),
    ("v32", "M", "2025-12-01", "V3.2", [15], True),
    ("v4", "M", "2026-04-24", "V4 预览版", [18], True),
    ("v4pro", "M", "2026-08-13", "V4-Pro 正式版", [18], False),
    ("v41", "M", "2026-09-10", "V4.1-Flash", [19], True),
    # --- A attention and KV cache
    ("mha", "A", "2017-06-12", "多头注意力 MHA", [1], False),
    ("mqa", "A", "2019-11-06", "MQA", [4], False),
    ("rope", "A", "2021-04-20", "RoPE", [5], False),
    ("gqa", "A", "2023-05-22", "GQA", [4], False),
    ("yarn", "A", "2023-08-31", "YaRN", [5], False),
    ("gqa67", "A", "2023-11-29", "67B 用 GQA-8", [6], False),
    ("mla", "A", "2024-05-06", "MLA 潜在注意力", [9], True),
    ("nsa", "A", "2025-02-16", "NSA 原生稀疏注意力", [14], False),
    ("dsa", "A", "2025-09-29", "DSA 稀疏注意力", [14], True),
    ("csa", "A", "2026-04-24", "CSA / HCA 序列压缩", [18], True),
    ("csa2", "A", "2026-09-10", "CSA2 跨层共享 KV", [19], True),
    # --- B MoE
    ("sgmoe", "B", "2017-01-23", "稀疏门控 MoE", [7], False),
    ("gshard", "B", "2020-06-30", "GShard", [7], False),
    ("switch", "B", "2021-01-11", "Switch", [7], False),
    ("dsmoe", "B", "2024-01-11", "细粒度 + 共享专家", [7], True),
    ("devroute", "B", "2024-05-06", "设备受限路由", [10], False),
    ("alf", "B", "2024-08-28", "无辅助损失均衡（论文）", [11], False),
    ("v3moe", "B", "2024-12-26", "偏置均衡 · 节点受限", [11], True),
    ("v4moe", "B", "2026-04-24", "哈希路由 · 去节点限制", [18], False),
    ("v41moe", "B", "2026-09-10", "1+384 专家 · 去哈希", [19], False),
    # --- C numerics and systems
    ("fp8fmt", "C", "2022-09-12", "FP8 格式", [12], False),
    ("fireflyer", "C", "2024-08-26", "Fire-Flyer 集群", [12], False),
    ("fp8train", "C", "2024-12-26", "FP8 训练 · DualPipe", [12], True),
    ("openweek", "C", "2025-02-24", "开源周", [12], False),
    ("insights", "C", "2025-05-14", "V3 硬件反思", [12], False),
    ("ue8m0", "C", "2025-08-21", "UE8M0 缩放", [15], False),
    ("fp4", "C", "2026-04-24", "FP4 专家与 indexer", [18], False),
    ("fp4kv", "C", "2026-09-10", "FP4 主 KV", [19], True),
    # --- D training objective and post-training
    ("ppo", "D", "2017-07-20", "PPO", [3], False),
    ("rlhf", "D", "2022-01-27", "RLHF 三步法", [3], False),
    ("dpo", "D", "2023-05-29", "DPO", [3], False),
    ("grpo", "D", "2024-02-05", "GRPO", [8], True),
    ("mtp_par", "D", "2024-04-30", "并行多 token 预测", [11], False),
    ("grpo2", "D", "2024-05-06", "两阶段 GRPO 对齐", [10], False),
    ("mtp", "D", "2024-12-26", "MTP 因果链", [11], False),
    ("r1rl", "D", "2025-01-20", "R1-Zero / R1 纯 RL", [13], True),
    ("hybrid", "D", "2025-08-21", "一模两式（思考开关）", [15], False),
    ("mathv2", "D", "2025-11-27", "Math-V2 自验证", [15], False),
    ("scalerl", "D", "2025-12-01", "规模化 RL · 智能体", [15], True),
    ("opd", "D", "2026-04-24", "专家 + 在线蒸馏", [18], False),
    ("dspark", "D", "2026-06-26", "DSpark 投机解码", [19], False),
    # --- E residual stream and memory
    ("preln", "E", "2019-02-14", "Pre-LN（GPT-2）", [1], False),
    ("xiong", "E", "2020-02-12", "Pre-LN 梯度分析", [1], False),
    ("hc", "E", "2024-09-29", "Hyper-Connections", [16], False),
    ("ocr", "E", "2025-10-20", "OCR 光学压缩", [17], False),
    ("mhc", "E", "2025-12-31", "mHC 双随机约束", [16], True),
    ("engram", "E", "2026-01-12", "Engram 条件记忆", [17], True),
    ("ocr2", "E", "2026-01-27", "OCR 2", [17], False),
    ("v4mhc", "E", "2026-04-24", "mHC 进入 V4", [18], False),
    ("v41mem", "E", "2026-09-10", "单遍 mHC · Engram", [19], False),
    # --- F scaling laws and optimizers
    ("adamw", "F", "2017-11-14", "AdamW", [2], False),
    ("kaplan", "F", "2020-01-23", "Kaplan 规模化规律", [2], False),
    ("chinchilla", "F", "2022-03-29", "Chinchilla", [2], False),
    ("hpscale", "F", "2024-01-05", "超参数规模化规律", [6], True),
    ("muon", "F", "2024-12-08", "Muon（博客）", [18], False),
    ("v4muon", "F", "2026-04-24", "Muon 训练 V4", [18], True),
    ("sinkupd", "F", "2026-09-10", "Sinkhorn 平衡更新", [17], False),
]

# "builds on" (thread lanes) and weight lineage (model row)
EDGES = [
    # (LLM 7B -> Coder-v1.5 -> Math 7B is real lineage but cannot be drawn without passing behind the
    #  DeepSeek-Coder card; the caption states it in words instead)
    ("gpt3", "instructgpt"), ("v3", "r1"), ("v3", "v31"), ("v31", "v32exp"), ("v31", "v32"),
    ("mha", "mqa"), ("mqa", "gqa"), ("gqa", "gqa67"), ("gqa67", "mla"), ("rope", "yarn"), ("yarn", "mla"),
    ("mla", "dsa"), ("nsa", "dsa"), ("dsa", "csa"), ("csa", "csa2"),
    ("sgmoe", "gshard"), ("gshard", "switch"), ("switch", "dsmoe"), ("dsmoe", "devroute"),
    ("devroute", "v3moe"), ("alf", "v3moe"), ("v3moe", "v4moe"), ("v4moe", "v41moe"),
    ("fp8fmt", "fp8train"), ("fireflyer", "fp8train"), ("fp8train", "openweek"), ("fp8train", "insights"),
    ("fp8train", "ue8m0"), ("ue8m0", "fp4"), ("fp4", "fp4kv"),
    ("ppo", "rlhf"), ("rlhf", "dpo"), ("rlhf", "grpo"), ("grpo", "grpo2"), ("grpo2", "r1rl"),
    ("mtp_par", "mtp"), ("mtp", "dspark"), ("r1rl", "hybrid"), ("hybrid", "scalerl"), ("scalerl", "opd"),
    ("preln", "xiong"), ("xiong", "hc"), ("hc", "mhc"), ("mhc", "v4mhc"), ("v4mhc", "v41mem"),
    ("ocr", "ocr2"), ("engram", "v41mem"),
    ("kaplan", "chinchilla"), ("chinchilla", "hpscale"), ("muon", "v4muon"), ("adamw", "muon"),
    ("v4muon", "sinkupd"),
]

# ---------------------------------------------------------------- geometry
T0 = dt.date(2017, 1, 1)
TB = dt.date(2023, 10, 1)     # axis break: scale changes here
T1 = dt.date(2026, 10, 1)
LW = 150                      # lane-label column
X0 = LW + 14
W1 = 340                      # 2017-01 .. 2023-09 (81 months)
GAP = 18                      # visual break
W2 = 1020                     # 2023-10 .. 2026-09 (36 months)
X1 = X0 + W1 + GAP + W2

CARD_H = 38
PITCH = 44
LANE_PAD = 9
NAME_SIZE = 13.5
DATE_SIZE = 11.5
BADGE_R = 9.5


def months(d: dt.date) -> float:
    return (d - T0).days / 30.436875


def xof(d: dt.date) -> float:
    if d < TB:
        return X0 + W1 * months(d) / months(TB)
    return X0 + W1 + GAP + W2 * (months(d) - months(TB)) / (months(T1) - months(TB))


def parse(s: str) -> dt.date:
    return dt.date(*map(int, s.split("-")))


class Card:
    def __init__(self, nid, lane, date, label, lessons, strong):
        self.id, self.lane, self.date, self.label, self.lessons, self.strong = nid, lane, parse(date), label, lessons, strong
        self.datestr = date
        self.x = xof(self.date)
        nb = len(lessons)
        self.text_x = 6 + 2 * BADGE_R * nb + 4 * (nb - 1) + 6
        wname = text_width(label, NAME_SIZE, 600 if strong else 500)
        wdate = text_width(date, DATE_SIZE)
        self.w = self.text_x + max(wname, wdate) + 9
        self.row = 0
        self.y = 0.0


def layout():
    cards = {n[0]: Card(*n) for n in NODES}
    lanes = {}
    for key, _, _ in LANES:
        lane_cards = sorted((c for c in cards.values() if c.lane == key), key=lambda c: c.date)
        ends: list[float] = []
        for cd in lane_cards:
            for r, e in enumerate(ends):
                if e + 8 <= cd.x:
                    cd.row = r
                    ends[r] = cd.x + cd.w
                    break
            else:
                cd.row = len(ends)
                ends.append(cd.x + cd.w)
        lanes[key] = (lane_cards, max(1, len(ends)))
    return cards, lanes


def main():
    cards, lanes = layout()
    top_axis_h = 40
    y = 16
    lane_top = {}
    for key, _, _ in LANES:
        _, nrows = lanes[key]
        h = 2 * LANE_PAD + (nrows - 1) * PITCH + CARD_H
        lane_top[key] = (y, h)
        y += h
        if key == "M":
            y += top_axis_h       # time axis between the model row and the thread lanes
        else:
            y += 6
    lanes_bottom = y
    legend_y = lanes_bottom + 76
    H = int(legend_y + 40)
    # the canvas ends where the right-most card ends (cards start at their date and extend to the right)
    W = int(max(X1, max(cd.x + cd.w for cd in cards.values())) + 14)

    s = SVG(W, H, title="DeepSeek 科技树：2017 年到 2026 年 9 月的六条主线", min_width=W)

    for key, _, _ in LANES:
        top, _ = lane_top[key]
        for cd in lanes[key][0]:
            cd.y = top + LANE_PAD + cd.row * PITCH

    # lane bands
    for i, (key, label, col) in enumerate(LANES):
        top, h = lane_top[key]
        s.rect(4, top, W - 8, h, fill=col, fill_opacity=0.045 if key != "M" else 0.03, rx=8)
        if key != "M":
            s.circle(20, top + h / 2 - 9, 11, fill=col)
            s.text(20, top + h / 2 - 4.5, key, size=13, anchor="middle", color="surface", weight=700)
            for j, ln in enumerate(label):
                txt = ln[2:] if j == 0 else ln
                s.text(38, top + h / 2 - 5 + j * 17, txt, size=13, color=col, weight=600)
        else:
            s.text(14, top + h / 2 + 5, label[0], size=13.5, color="ink", weight=700)

    grid_top = lane_top["M"][0]
    # year / quarter grid
    def vline(d, strong):
        x = xof(d)
        s.line(x, grid_top, x, lanes_bottom, color="grid" if not strong else "rule", width=1 if strong else 0.8)

    for yr in range(2017, 2027):
        d = dt.date(yr, 1, 1)
        if d < TB:
            vline(d, True)
        elif d <= T1:
            vline(d, True)
    for yr in (2023, 2024, 2025, 2026):
        for mo in (4, 7, 10):
            d = dt.date(yr, mo, 1)
            if TB < d < T1:
                vline(d, False)

    # the break band
    bx = X0 + W1
    s.rect(bx + 2, grid_top, GAP - 4, lanes_bottom - grid_top, fill="surface", fill_opacity=0.9)

    def axis(ybase, labels_above):
        s.line(X0, ybase, bx + 1, ybase, color="muted", width=1)
        s.line(bx + GAP - 1, ybase, X1, ybase, color="muted", width=1)
        for dx in (0, 7):
            s.line(bx + 2 + dx, ybase + 6, bx + 8 + dx, ybase - 6, color="muted", width=1.2)
        ty = ybase - 7 if labels_above else ybase + 17
        for yr in range(2017, 2024):
            x = xof(dt.date(yr, 1, 1))
            s.line(x, ybase - 4, x, ybase + 4, color="muted", width=1)
            s.text(x + 3, ty, str(yr), size=12, color="muted")
        for yr in (2024, 2025, 2026):
            x = xof(dt.date(yr, 1, 1))
            s.line(x, ybase - 6, x, ybase + 6, color="ink", width=1.3)
            s.text(x + 4, ty, str(yr), size=13, color="ink", weight=700)
        for yr in (2023, 2024, 2025, 2026):
            for mo in (4, 7, 10):
                d = dt.date(yr, mo, 1)
                if TB < d < T1:
                    x = xof(d)
                    s.line(x, ybase - 3, x, ybase + 3, color="muted", width=1)
                    s.text(x + 3, ty, "%d 月" % mo, size=11, color="muted")

    mtop, mh = lane_top["M"]
    axis(mtop + mh + top_axis_h / 2 + 4, labels_above=False)
    axis(lanes_bottom + 8, labels_above=False)

    # scale notes under the bottom axis
    zoom = (W2 / (months(T1) - months(TB))) / (W1 / months(TB))
    s.text(bx - 6, lanes_bottom + 46, "← 2017-01 至 2023-09：刻度压缩", size=12, anchor="end", color="muted")
    s.text(bx + GAP + 6, lanes_bottom + 46, "2023-10 起：同样一个月画宽 %.1f 倍 →" % zoom, size=12, color="muted")

    # flagship release lines through the thread lanes
    for cd in lanes["M"][0]:
        if cd.strong:
            s.line(cd.x, cd.y + CARD_H, cd.x, lanes_bottom, color="ink", width=1, dash="3 4", opacity=0.45)

    # edges
    for a, b in EDGES:
        ca, cb = cards[a], cards[b]
        col = dict((k, c) for k, _, c in LANES)[ca.lane]
        if ca.lane == "M":
            col = "muted"
        sx, sy = ca.x + ca.w, ca.y + CARD_H / 2
        tx, ty = cb.x, cb.y + CARD_H / 2
        wdt = 1.5 if ca.lane != "M" else 1.2
        if tx - sx >= 14:
            if abs(ty - sy) < 0.5:
                s.arrow(sx, sy, tx - 1, ty, color=col, width=wdt)
            else:
                dx = min(46, (tx - sx) / 2)
                others = [c for c in lanes[ca.lane][0] if c is not ca and c is not cb]

                def behind(x, y):
                    return any(c.x - 2 <= x <= c.x + c.w + 2 and c.y - 2 <= y <= c.y + CARD_H + 2 for c in others)

                def bez(t):
                    u = 1 - t
                    return (u ** 3 * sx + 3 * u * u * t * (sx + dx) + 3 * u * t * t * (tx - dx) + t ** 3 * (tx - 1),
                            u ** 3 * sy + 3 * u * u * t * sy + 3 * u * t * t * ty + t ** 3 * ty)

                if abs(cb.row - ca.row) == 1 and any(behind(*bez(i / 200)) for i in range(201)):
                    # the S-curve would disappear behind another card and look like it passes through it:
                    # run along the 6-unit gutter between the two rows instead
                    down = cb.row > ca.row
                    yg = cb.y - (PITCH - CARD_H) / 2 if down else ca.y - (PITCH - CARD_H) / 2
                    xa, xb, r = sx + 5, tx - 12, 3
                    sg = 1 if down else -1
                    s.path("M%.1f %.1f L%.1f %.1f Q%.1f %.1f %.1f %.1f L%.1f %.1f Q%.1f %.1f %.1f %.1f "
                           "L%.1f %.1f Q%.1f %.1f %.1f %.1f L%.1f %.1f Q%.1f %.1f %.1f %.1f L%.1f %.1f"
                           % (sx, sy, xa - r, sy, xa, sy, xa, sy + sg * r, xa, yg - sg * r, xa, yg, xa + r, yg,
                              xb - r, yg, xb, yg, xb, yg + sg * r, xb, ty - sg * r, xb, ty, xb + r, ty, tx - 1, ty),
                           stroke=col, width=wdt, arrow=True)
                    continue
                s.path("M%.1f %.1f C%.1f %.1f %.1f %.1f %.1f %.1f" % (sx, sy, sx + dx, sy, tx - dx, ty, tx - 1, ty),
                       stroke=col, width=wdt, arrow=True)
        else:
            # target starts before the source card ends: leave from the source card's lower/upper edge
            down = cb.y > ca.y
            y0 = ca.y + CARD_H if down else ca.y
            lo, hi = sorted((ca.row, cb.row))
            others = [c for c in lanes[ca.lane][0] if c is not ca and c is not cb]

            def crosses_between(x):
                return any(lo < c.row < hi and c.x - 4 <= x <= c.x + c.w + 4 for c in others)

            def leg_blocked(x):
                # the horizontal leg along the target's row, from x to the target's left edge
                return any(c.row == cb.row and c.x - 4 < tx and c.x + c.w + 4 > x for c in others)

            # 1) an L-shaped route whose vertical and horizontal legs both avoid the other cards
            x0 = None
            for cand in range(int(ca.x + 12), int(tx - 11), 3):
                if cand <= ca.x + ca.w - 6 and not crosses_between(cand) and not leg_blocked(cand):
                    x0 = cand
                    break
            xv = cb.x + 12
            if x0 is None and hi - lo >= 2 and ca.x + 6 <= xv <= ca.x + ca.w - 6 and not crosses_between(xv):
                # 2) rows far enough apart: a straight vertical arrow into the target's edge
                ye = cb.y if down else cb.y + CARD_H
                s.path("M%.1f %.1f L%.1f %.1f" % (xv, y0, xv, ye - (1 if down else -1)), stroke=col, width=wdt,
                       arrow=True)
                continue
            if x0 is None:
                x0 = ca.x + 12
                if crosses_between(x0) or leg_blocked(x0):
                    print("warning: edge %s -> %s passes behind a card" % (a, b))
            if tx - x0 >= 10:
                r = min(8, (tx - x0) / 2)
                yb = ty - r if down else ty + r
                s.path("M%.1f %.1f L%.1f %.1f Q%.1f %.1f %.1f %.1f L%.1f %.1f"
                       % (x0, y0, x0, yb, x0, ty, x0 + r, ty, tx - 1, ty), stroke=col, width=wdt, arrow=True)
            else:
                ye = cb.y if down else cb.y + CARD_H
                s.path("M%.1f %.1f L%.1f %.1f" % (x0, y0, x0, ye - (1 if down else -1)), stroke=col, width=wdt,
                       arrow=True)

    # cards
    for key, _, col in LANES:
        for cd in lanes[key][0]:
            ccol = col
            s.rect(cd.x, cd.y, cd.w, CARD_H, fill="surface", stroke=ccol, width=2.0 if cd.strong else 1.1, rx=6)
            s.rect(cd.x, cd.y, cd.w, CARD_H, fill=ccol, fill_opacity=0.10 if cd.strong else 0.04, rx=6)
            s.rect(cd.x, cd.y + 3, 3, CARD_H - 6, fill=ccol, rx=1.5)
            for i, les in enumerate(cd.lessons):
                bx0 = cd.x + 6 + BADGE_R + i * (2 * BADGE_R + 4)
                s.circle(bx0, cd.y + CARD_H / 2, BADGE_R, fill=ccol if les else None, stroke=ccol, width=1.2)
                s.text(bx0, cd.y + CARD_H / 2 + 4, str(les), size=11.5 if les < 10 else 11, anchor="middle",
                       color="surface" if les else ccol, weight=700)
            s.text(cd.x + cd.text_x, cd.y + 16, cd.label, size=NAME_SIZE, color="ink",
                   weight=700 if cd.strong else 500)
            s.text(cd.x + cd.text_x, cd.y + 31, cd.datestr, size=DATE_SIZE, color="muted")

    # legend
    ly = legend_y
    lx = X0
    s.rect(lx, ly - 13, 30, 22, fill="surface", stroke="blue", width=1.1, rx=5)
    s.rect(lx, ly - 10, 3, 16, fill="blue", rx=1.5)
    s.text(lx + 38, ly + 3, "卡片左边缘 = 日期（模型取公开发布日，论文取 arXiv v1 日期）", size=12.5, color="ink")
    lx2 = lx + 470
    s.circle(lx2 + 9, ly - 2, BADGE_R, fill="blue")
    s.text(lx2 + 9, ly + 2, "9", size=11.5, anchor="middle", color="surface", weight=700)
    s.text(lx2 + 24, ly + 3, "讲它的课（0 = 只在本课提到）", size=12.5)
    lx3 = lx2 + 250
    s.rect(lx3, ly - 13, 30, 22, fill="blue", fill_opacity=0.10, stroke="blue", width=2.0, rx=5)
    s.text(lx3 + 38, ly + 3, "粗框 = DeepSeek 主干模型及其关键创新；虚线竖线 = 主干模型的发布日", size=12.5)
    ly2 = ly + 26
    s.arrow(lx, ly2 - 4, lx + 30, ly2 - 4, color="green", width=1.5)
    s.text(lx + 38, ly2 + 1, "主线内的箭头 = “在前者基础上改进”", size=12.5)
    s.arrow(lx2, ly2 - 4, lx2 + 30, ly2 - 4, color="muted", width=1.2)
    s.text(lx2 + 38, ly2 + 1, "模型行的箭头 = 权重血缘（在前者的检查点上继续训练）", size=12.5)

    s.save("l00-tech-tree.svg")
    # report lane rows so the source text can mention them if needed
    for key, _, _ in LANES:
        print(key, "rows", lanes[key][1])
    print("size", W, H)


if __name__ == "__main__":
    main()
