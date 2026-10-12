"""Figure 17.16: one shared Engram table versus one KV cache per request (V4.1-Flash numbers).

x = number of concurrent one-million-token sessions, y = bytes held (GB).
  Engram table: 768,022,850 rows x (256 FP8 bytes + 8 UE8M0 scale bytes) = 202.8 GB, shared by all
                requests (notes §4.3, recomputed in l17_num.py);
  KV cache:     890 bytes/token (dsv41) x 1e6 tokens = 0.89 GB per session;
  optical:      the DeepSeek-OCR proposal at about 10x (visual tokens assumed to cost the same KV as
                text tokens -- an assumption of the lesson, never measured by the paper).
Crossing: 202.8 / 0.89 = 227.8 sessions (text) and about 2,278 sessions (10x optical).
Writes deepseek/figures/l17-shared-vs-kv.svg
"""

from l17_svg import SVG
from l17_num import V41_BYTES

GB = 1e9
TABLE = V41_BYTES / GB                 # 202.76 GB
KV_TOKEN = 890                         # bytes per token, V4.1-Flash global KV
SESSION = 1_000_000                    # tokens per session
KV_SESSION = KV_TOKEN * SESSION / GB   # 0.89 GB
OPT_SESSION = KV_SESSION / 10          # about 10x optical compression (proposal)
CROSS = TABLE / KV_SESSION
assert round(TABLE, 1) == 202.8 and round(CROSS) == 228 and round(TABLE / OPT_SESSION, -1) == 2280

W, H = 760, 420
svg = SVG(W, H, "共享一份的 Engram 表与每个请求一份的 KV cache（V4.1-Flash 的数字）")

X0, X1, Y0, Y1 = 72, 520, 50, 318
XM, YM = 300, 300


def x(v):
    return X0 + v / XM * (X1 - X0)


def y(v):
    return Y1 - v / YM * (Y1 - Y0)


svg.text(20, 22, "同时服务 N 个一百万 token 的会话，要占多少内存", size=13, weight="700")
for v in range(0, YM + 1, 50):
    svg.line(X0, y(v), X1, y(v), color="grid", width=1)
    svg.text(X0 - 8, y(v) + 4, "%d" % v, size=11, anchor="end", color="muted")
for v in range(0, XM + 1, 50):
    svg.line(x(v), Y0, x(v), Y1, color="grid", width=1)
    svg.text(x(v), Y1 + 17, "%d" % v, size=11, anchor="middle", color="muted")
svg.line(X0, Y1, X1, Y1, color="muted", width=1.2)
svg.line(X0, Y0, X0, Y1, color="muted", width=1.2)
svg.text((X0 + X1) / 2, Y1 + 38, "同时服务的百万 token 会话数 N", size=12, anchor="middle", color="muted")
svg.text(24, (Y0 + Y1) / 2, "占用（GB）", size=12, anchor="middle", color="muted", rotate=-90)

# the three lines
svg.line(x(0), y(TABLE), x(XM), y(TABLE), color="amber", width=2.6)
svg.line(x(0), y(0), x(XM), y(KV_SESSION * XM), color="ink", width=2.4)
svg.line(x(0), y(0), x(XM), y(OPT_SESSION * XM), color="orange", width=2.2, dash="7 5")

# direct labels at the right end of each line
LX = X1 + 12
svg.text(LX, y(KV_SESSION * XM) + 4, "KV cache：每个会话 0.89 GB", size=12, weight="700")
svg.text(LX, y(KV_SESSION * XM) + 21, "每个请求一份，随 N 线性涨", size=11.5, color="muted")
svg.text(LX, y(TABLE) + 4, "Engram 表：%.1f GB" % TABLE, size=12, weight="700", color="amber")
svg.text(LX, y(TABLE) + 21, "所有请求共用一份，与 N 无关", size=11.5, color="muted")
svg.text(LX, y(OPT_SESSION * XM) - 14, "光学压缩约 10×（设想）", size=12, weight="700", color="orange")
svg.text(LX, y(OPT_SESSION * XM) + 3, "每个会话约 0.089 GB", size=11.5, color="muted")

# the crossing
svg.line(x(CROSS), y(TABLE), x(CROSS), Y1, color="muted", width=1, dash="3 3")
svg.circle(x(CROSS), y(TABLE), 5, fill="ink", stroke="surface", width=1.4)
svg.text(x(CROSS) - 10, y(TABLE) - 12, "N ≈ 228：KV 超过整张表", size=12, anchor="end", weight="700")
svg.text(x(CROSS) + 6, y(OPT_SESSION * CROSS) - 9, "N = 228", size=11.5, color="muted")
svg.text(x(8), y(TABLE) + 19, "202.8 GB ÷ 0.89 GB ≈ 228", size=11.5, color="amber")

svg.text(20, H - 30, "Engram 表 = 768,022,850 行 × (256 + 8) 字节（FP8 数值 + 每 32 维 1 字节缩放）；KV 按 V4.1-Flash 全局 890 字节/token。",
         size=11.5, color="muted")
svg.text(20, H - 12, "橙色虚线按视觉 token 与文本 token 的 KV 同价计（笔者假设），要约 2,280 个会话才追上 Engram 表；论文没有做过这种实验。",
         size=11.5, color="muted")

svg.save("l17-shared-vs-kv.svg")
