"""Figure 19.1: the shape of an input-heavy agent session (illustrative workload, real prices).

Assumed workload (stated in the lesson): an initial context of 20,000 tokens; each of 50 turns
emits 400 output tokens and receives a 2,000-token tool result. Turn k re-reads the whole context
T_k = T_0 + (k-1)(a+o). Prices: DeepSeek API off-peak list prices for deepseek-flash (notes C60).
Every bar length and every total is computed in l19_num.
Writes deepseek/figures/l19-agent-session.svg
"""

from l19_svg import SVG
from l19_num import (SESS_T0, SESS_OUT, SESS_TOOL, SESS_K, SESS_CTX, SESS_IN, SESS_OUT_TOT, SESS_MISS,
                     SESS_HIT, SESS_COST_CACHED, SESS_COST_NOCACHE, ACT_PREFILL, ACT_DECODE)

W, H = 760, 452
svg = SVG(W, H, "一次智能体会话的形状：每一轮都重读全部上下文，输出很短")

X0, Y0 = 58, 52          # plot origin (top-left of first bar)
XW = 400                 # px for 140K tokens
XMAX = 140000
BH, BG = 4.2, 1.4        # bar height and gap


def xs(tok):
    return X0 + tok / XMAX * XW


svg.text(20, 22, "(a) 50 轮会话：每一行是一次请求（第 1 轮在最上面）", size=13, weight="700")
for k, ctx in enumerate(SESS_CTX, start=1):
    y = Y0 + (k - 1) * (BH + BG)
    cached = 0 if k == 1 else ctx - SESS_TOOL
    new = SESS_T0 if k == 1 else SESS_TOOL
    if cached:
        svg.rect(xs(0), y, xs(cached) - xs(0), BH, fill="amber", fill_opacity=0.55)
    svg.rect(xs(cached), y, xs(cached + new) - xs(cached), BH, fill="blue", fill_opacity=0.85)
    svg.rect(xs(ctx), y, max(xs(ctx + SESS_OUT) - xs(ctx), 1.2), BH, fill="green", fill_opacity=1)
yb = Y0 + SESS_K * (BH + BG) + 4
svg.line(X0, yb, xs(XMAX), yb, color="muted", width=1)
for t in range(0, XMAX + 1, 20000):
    svg.line(xs(t), yb, xs(t), yb + 4, color="muted", width=1)
    svg.text(xs(t), yb + 17, "%dK" % (t // 1000), size=11, anchor="middle", color="muted")
svg.text(xs(XMAX / 2), yb + 35, "本轮请求的上下文长度（token）", size=12, anchor="middle", color="muted")
for k in (1, 10, 20, 30, 40, 50):
    y = Y0 + (k - 1) * (BH + BG) + BH / 2 + 4
    svg.text(X0 - 8, y, "%d" % k, size=11, anchor="end", color="muted")
svg.text(18, Y0 + 140, "轮次", size=12, color="muted", rotate=-90)
# callout in the empty triangle above the diagonal
yl = Y0 + (SESS_K - 1) * (BH + BG) + BH / 2
cx, cy = xs(72000), Y0 + 70
svg.text(cx, cy, "第 50 轮重读 %s token：" % "{:,}".format(SESS_CTX[-1]), size=12)
svg.text(cx, cy + 17, "其中只有 %s 个是新的，" % "{:,}".format(SESS_TOOL), size=12, color="blue")
svg.text(cx, cy + 34, "而它只输出 %d 个" % SESS_OUT, size=12, color="green")
svg.line(cx + 110, cy + 42, xs(SESS_CTX[-1]) + 2, yl - 3, color="muted", width=1, dash="3 3")

# legend under the plot
LY = yb + 52
svg.rect(20, LY - 10, 14, 11, fill="amber", fill_opacity=0.55)
svg.text(40, LY, "前缀：上一轮已算过的 KV（缓存命中）", size=12)
svg.rect(20, LY + 10, 14, 11, fill="blue", fill_opacity=0.85)
svg.text(40, LY + 20, "新输入：工具返回的 %s token（要 prefill）" % "{:,}".format(SESS_TOOL), size=12)
svg.rect(20, LY + 30, 14, 11, fill="green")
svg.text(40, LY + 40, "输出：模型的动作 %d token（decode）" % SESS_OUT, size=12)

# ---------------------------------------------------------------- right panel: the totals
RX = 492
svg.text(RX, 22, "(b) 全会话合计", size=13, weight="700")


def bar_row(y, label, parts, total_w, scale_max, note):
    svg.text(RX, y, label, size=12, weight="700")
    x = RX
    for val, col, op in parts:
        w = val / scale_max * total_w
        svg.rect(x, y + 7, max(w, 1.2), 12, fill=col, fill_opacity=op)
        x += w
    svg.text(RX, y + 36, note, size=12, color="muted")


BW = 196
bar_row(62, "token 数", [(SESS_HIT, "amber", 0.55), (SESS_MISS, "blue", 0.85), (SESS_OUT_TOT, "green", 1)],
        BW, SESS_IN + SESS_OUT_TOT,
        "输入 %.2fM（命中 %.1f%%）· 输出 %.2fM" % (SESS_IN / 1e6, SESS_HIT / SESS_IN * 100, SESS_OUT_TOT / 1e6))
svg.text(RX, 118, "输入 : 输出 = %d : 1" % round(SESS_IN / SESS_OUT_TOT), size=12)

pre = 2 * ACT_PREFILL * SESS_MISS
dec = 2 * ACT_DECODE * SESS_OUT_TOT
bar_row(152, "主干计算（2 × 激活参数 × token）", [(pre, "blue", 0.85), (dec, "green", 1)], BW + 40,
        pre + dec, "")
svg.text(RX, 188, "prefill %.2f × 10¹⁵ · decode %.2f × 10¹⁵" % (pre / 1e15, dec / 1e15), size=12, color="muted")
svg.text(RX, 206, "（只算未命中的 %s 个输入 token）" % "{:,}".format(SESS_MISS), size=12, color="muted")

svg.text(RX, 242, "API 账单（off-peak 标价）", size=12, weight="700")
hit_c, miss_c, out_c = SESS_HIT * 0.003 / 1e6, SESS_MISS * 0.15 / 1e6, SESS_OUT_TOT * 0.6 / 1e6
scale = SESS_COST_NOCACHE
x = RX
for val, col, op in ((hit_c, "amber", 0.55), (miss_c, "blue", 0.85), (out_c, "green", 1)):
    w = val / scale * BW
    svg.rect(x, 250, w, 12, fill=col, fill_opacity=op)
    x += w
svg.text(RX + BW + 6, 261, "$%.3f" % SESS_COST_CACHED, size=12)
svg.text(RX, 284, "有前缀缓存", size=12, color="muted")
svg.rect(RX, 294, SESS_IN * 0.15 / 1e6 / scale * BW, 12, fill="blue", fill_opacity=0.85)
svg.rect(RX + SESS_IN * 0.15 / 1e6 / scale * BW, 294, out_c / scale * BW, 12, fill="green")
svg.text(RX + BW + 6, 305, "$%.3f" % SESS_COST_NOCACHE, size=12)
svg.text(RX, 328, "没有前缀缓存（全按未命中计）：账单 ×%.1f" % (SESS_COST_NOCACHE / SESS_COST_CACHED), size=12,
         color="muted")
svg.text(RX, 362, "价格：命中 $0.003、未命中 $0.15、", size=12, color="muted")
svg.text(RX, 378, "输出 $0.60（每百万 token）", size=12, color="muted")
svg.text(RX, 404, "会话参数是假设的；价格、激活量是真实的", size=12, color="red")

svg.save("l19-agent-session.svg")
