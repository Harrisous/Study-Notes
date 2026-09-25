"""Figure 17.1: four forms of memory in a language model (schematic, numbers from the notes).

Rows: what the earlier lessons had / what this lesson adds. Columns: parametric memory
(written at training time, shared by every request) / context memory (written at inference
time, one copy per request). Numbers: V4.1-Flash 196B Engram, 48 rows per token (l17_num);
V4.1-Flash global KV 890 B/token (dsv41 abstract); DeepSeek-OCR table 2 (100 tokens, 9.7x, 96.8%).
Writes deepseek/figures/l17-memory-forms.svg
"""

from l17_svg import SVG
from l17_num import V41_PARAMS

assert round(V41_PARAMS / 1e9, 1) == 196.6

W, H = 760, 452
svg = SVG(W, H, "语言模型里的四种记忆：两种旧的，两种本课新讲的")

CX = [20, 396]
CW = 344
svg.text(CX[0] + CW / 2, 24, "参数化记忆：训练时写入，所有请求共用", size=13, anchor="middle", weight="700")
svg.text(CX[1] + CW / 2, 24, "上下文记忆：推理时写入，每个请求一份", size=13, anchor="middle", weight="700")
svg.line(380, 12, 380, 440, color="rule", width=1.2, dash="4 4")


def card(x, y, w, h, color, title, lines, thick=False, tag=None):
    svg.rect(x, y, w, h, fill=color, fill_opacity=0.08, stroke=color, width=2.2 if thick else 1.2, rx=8)
    svg.text(x + 14, y + 24, title, size=14, weight="700", color=color)
    if tag:
        svg.text(x + w - 12, y + 24, tag, size=12, anchor="end", color="muted")
    for i, (k, v) in enumerate(lines):
        yy = y + 50 + i * 22
        svg.text(x + 14, yy, k, size=12, color="muted")
        svg.text(x + 80, yy, v, size=12.5)


Y1, Y2, HC = 44, 262, 150
card(CX[0], Y1, CW, HC, "blue", "FFN / MoE 专家权重", [
    ("怎样想起", "隐藏状态逐层计算，知识分散在权重里"),
    ("放在哪里", "GPU 显存（HBM）"),
    ("代价", "拼出一个多 token 实体要耗掉前几层"),
    ("选谁", "MoE 由隐藏状态决定：算到那层才知道"),
], tag="第 1、7 课")
card(CX[1], Y1, CW, HC, "amber", "KV cache", [
    ("怎样记", "每个 token 每层留下完整的键和值"),
    ("放在哪里", "HBM，随上下文长度线性增长"),
    ("例", "V4.1-Flash 全局 890 B/token"),
    ("", "一百万 token ≈ 0.89 GB（一个请求）"),
], tag="第 4、9、14 课")
card(CX[0], Y2, CW, HC + 20, "amber", "Engram：查表式条件记忆", [
    ("怎样想起", "用最近 2–4 个 token 的 ID 算出行号"),
    ("放在哪里", "可以放主机内存，提前异步预取"),
    ("例", "V4.1-Flash：196.6B 参数，两个模块"),
    ("", "每 token 只取 48 行（与表大小无关）"),
    ("状态", "已进入 V4.1-Flash（生产模型）"),
], thick=True, tag="本课 §2–§5")
card(CX[1], Y2, CW, HC + 20, "orange", "光学压缩：把旧上下文拍成图", [
    ("怎样记", "文本渲染成图片，再编码成视觉 token"),
    ("放在哪里", "仍是 KV，但条目约少 10 倍"),
    ("例", "约 970 文本 token → 100 视觉 token"),
    ("", "压缩 9.7 倍，Fox 上 OCR 精度 96.8%"),
    ("状态", "OCR 已实测；当对话记忆用仍是设想"),
], thick=True, tag="本课 §6–§7")

for x, lab1, lab2 in ((CX[0], "静态的局部搭配", "不必每次用深度重算"), (CX[1], "旧的上下文", "不必逐字全精度保存")):
    xm = x + CW / 2
    svg.arrow(xm, Y1 + HC + 6, xm, Y2 - 6, color="ink", width=1.8)
    svg.text(xm + 12, Y1 + HC + 30, lab1, size=12.5, color="ink")
    svg.text(xm + 12, Y1 + HC + 48, lab2, size=12.5, color="ink")

svg.save("l17-memory-forms.svg")
