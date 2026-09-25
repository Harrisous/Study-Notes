"""Figure 0.9: the colour legend shared by every figure of the course (style-guide.md section 4).

Each row: a swatch drawn with the colour token itself (so it follows light/dark theme), the token name,
its meaning, and a tiny example of how it appears in the lessons' figures.

    python deepseek/figures/gen/l00_colors.py
"""

from __future__ import annotations

from l00_svg import SVG

ROWS = [
    ("blue", "蓝", "--fig-blue", "查询 Q、当前 token、“正在计算的东西”、被激活的路由专家"),
    ("violet", "紫", "--fig-violet", "键 K、路由打分（含 indexer）、共享专家"),
    ("green", "绿", "--fig-green", "值 V、输出、正奖励、“好的 / 保留的”"),
    ("amber", "黄褐", "--fig-amber", "缓存、存储、记忆（KV cache、状态、内存层级）"),
    ("orange", "橙", "--fig-orange", "压缩后的潜在表示（c^{KV}）、量化后的低精度数据"),
    ("red", "红", "--fig-red", "被掩掉、被丢弃、错误、负奖励；细线表示 RoPE 位置分支"),
    ("muted", "灰", "--fig-muted / rule / grid", "未激活的专家、背景结构、坐标网格"),
    ("ink", "墨", "--fig-ink", "文字、主要线条、箭头"),
]

W, H = 740, 8 * 46 + 26
X_SW = 18
X_TXT = 80
X_EX = 640


def example(s, col, x, y):
    """A tiny glyph per colour, centred near (x, y)."""
    if col == "blue":
        s.rect(x - 22, y - 11, 44, 22, fill="blue", fill_opacity=0.15, stroke="blue", width=1.4, rx=4)
        s.text(x, y + 5, "q_{t}", size=13, anchor="middle", color="blue", weight=700)
    elif col == "violet":
        s.rect(x - 22, y - 11, 44, 22, fill="violet", fill_opacity=0.15, stroke="violet", width=1.4, rx=4)
        s.text(x, y + 5, "k_{j}", size=13, anchor="middle", color="violet", weight=700)
    elif col == "green":
        s.rect(x - 22, y - 11, 44, 22, fill="green", fill_opacity=0.15, stroke="green", width=1.4, rx=4)
        s.text(x, y + 5, "v_{j}", size=13, anchor="middle", color="green", weight=700)
    elif col == "amber":
        for k in range(4):
            s.rect(x - 24 + k * 12, y - 11, 10, 22, fill="amber", fill_opacity=0.25 + 0.15 * k, stroke="amber",
                   width=1, rx=2)
    elif col == "orange":
        s.rect(x - 8, y - 12, 16, 24, fill="orange", fill_opacity=0.3, stroke="orange", width=1.4, rx=3)
        s.text(x + 13, y + 5, "c", size=13, color="orange", weight=700)
    elif col == "red":
        s.rect(x - 11, y - 11, 22, 22, fill="red", fill_opacity=0.12, stroke="red", width=1.2, rx=2)
        s.line(x - 7, y - 7, x + 7, y + 7, color="red", width=1.8)
        s.line(x - 7, y + 7, x + 7, y - 7, color="red", width=1.8)
    elif col == "muted":
        for k in range(3):
            s.rect(x - 25 + k * 17, y - 8, 14, 16, fill="muted", fill_opacity=0.25, stroke="muted", width=1, rx=2)
    else:
        s.arrow(x - 24, y, x + 22, y, color="ink", width=1.8)


def main():
    s = SVG(W, H, title="全课统一的图色语义")
    for i, (col, cn, token, meaning) in enumerate(ROWS):
        y = 28 + i * 46
        if i % 2 == 0:
            s.rect(6, y - 21, W - 12, 44, fill="grid", fill_opacity=0.5, rx=6)
        s.rect(X_SW, y - 12, 44, 24, fill=col, rx=4)
        s.text(X_TXT, y - 4, cn, size=13.5, color=col if col != "muted" else "ink", weight=700)
        s.text(X_TXT + 42, y - 4, token, size=11.5, color="muted")
        s.text(X_TXT, y + 15, meaning, size=12.5)
        example(s, col, X_EX + 40, y + 1)
    s.save("l00-colors.svg")


if __name__ == "__main__":
    main()
