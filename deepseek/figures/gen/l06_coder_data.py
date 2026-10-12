"""Figure 6.15: DeepSeek-Coder's data pipeline (after dscoder Fig. 2) and a worked example of the
repository-level dependency ordering (dscoder Algorithm 1: edge B -> A when A depends on B; per disconnected
subgraph, repeatedly take the file with the minimal in-degree, which also handles cycles).

Pipeline numbers (dscoder §2): repos created before Feb 2023, 87 languages; rule filtering keeps 32.8 %;
final code corpus 798 GB / 603 M files; 10-gram decontamination.
The five-file repository is a made-up example; its ordering is computed here by the same rule.
Writes deepseek/figures/l06-coder-data.svg
"""

from l06_svg import SVG, box

# ---------- the ordering, computed
deps = {"model.py": ["utils.py"], "train.py": ["model.py", "utils.py"], "a.py": ["b.py"], "b.py": ["a.py"],
        "utils.py": []}
files = ["utils.py", "model.py", "train.py", "a.py", "b.py"]
graph = {f: [] for f in files}
indeg = {f: 0 for f in files}
for A in files:
    for B in deps[A]:
        graph[B].append(A)
        indeg[A] += 1
start_indeg = dict(indeg)
subgraphs = [["utils.py", "model.py", "train.py"], ["a.py", "b.py"]]
orders = []
for sg in subgraphs:
    res = []
    while len(res) < len(sg):
        f = min((x for x in sg if x not in res), key=lambda x: (indeg[x], sg.index(x)))
        for n in graph[f]:
            indeg[n] -= 1
        res.append(f)
    orders.append(res)
assert orders == [["utils.py", "model.py", "train.py"], ["a.py", "b.py"]]

W, H = 760, 500
svg = SVG(W, H, "DeepSeek-Coder 的数据管线与仓库级依赖排序示例")

# ---------- top: five steps
bw, gap, ty, th = 130, 12, 30, 92
x0 = 16
steps = [
    (["① 爬取", "GitHub 公开仓库", "2023 年 2 月前创建", "只留 87 种语言"], "muted"),
    (["② 规则过滤", "行长、字母比例、", "文件类型等规则", "剩原始量 32.8%"], "red"),
    (["③ 依赖解析", "按 import 等关系", "排列仓库内文件"], "blue"),
    (["④ 仓库级去重", "以仓库为单位去重", "不拆散仓库结构"], "violet"),
    (["⑤ 质量筛选", "质量模型 + 去污染", "（与测试题有 10-gram", "重合即删）"], "ink"),
]
for i, (lines, col) in enumerate(steps):
    x = x0 + i * (bw + gap)
    box(svg, x, ty, bw, th, lines, color=col, fill_opacity=0.07, size=12, line_gap=17, title_size=13)
    if i < 4:
        svg.arrow(x + bw + 1, ty + th / 2, x + bw + gap - 1, ty + th / 2, color="ink", width=1.4)
svg.text(x0, ty - 12, "据 DeepSeek-Coder 报告图 2 重绘：最终代码语料 798 GB、6.03 亿个文件", size=12, color="muted")

# ---------- bottom-left: dependency graph
gy0 = 170
svg.text(16, gy0, "示例仓库（虚构）：箭头 B → A 表示 A import 了 B", size=13, weight="600")
pos = {"utils.py": (80, 250), "model.py": (220, 250), "train.py": (150, 350), "a.py": (80, 440),
       "b.py": (220, 440)}
for f, (x, y) in pos.items():
    col = "blue" if f in subgraphs[0] else "violet"
    svg.rect(x - 44, y - 16, 88, 32, fill=col, fill_opacity=0.10, stroke=col, width=1.4, rx=6)
    svg.text(x, y + 5, f, size=12, anchor="middle")
    svg.text(x + 50, y - 16, "入度 %d" % start_indeg[f], size=11, color="muted")


def edge(a, b, bend=0):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    if a == "utils.py" and b == "model.py":
        svg.arrow(x1 + 44, y1, x2 - 46, y2, color="ink", width=1.4)
    elif a == "utils.py" and b == "train.py":
        svg.arrow(x1 + 10, y1 + 16, x2 - 20, y2 - 18, color="ink", width=1.4)
    elif a == "model.py" and b == "train.py":
        svg.arrow(x1 - 10, y1 + 16, x2 + 20, y2 - 18, color="ink", width=1.4)
    elif a == "a.py":
        svg.path("M%.1f %.1f Q %.1f %.1f %.1f %.1f" % (x1 + 30, y1 - 16, 150, y1 - 44, x2 - 32, y2 - 18),
                 stroke="ink", width=1.4, arrow=True)
    else:
        svg.path("M%.1f %.1f Q %.1f %.1f %.1f %.1f" % (x1 - 30, y1 + 16, 150, y1 + 44, x2 + 32, y2 + 18),
                 stroke="ink", width=1.4, arrow=True)


edge("utils.py", "model.py")
edge("utils.py", "train.py")
edge("model.py", "train.py")
edge("a.py", "b.py")
edge("b.py", "a.py")
svg.text(300, 445, "a、b 互相 import：", size=12, color="violet")
svg.text(300, 463, "一个环，入度都是 1", size=12, color="violet")

# ---------- bottom-right: the ordering and the samples
rx = 440
svg.text(rx, gy0, "每个连通子图单独排序，每次取入度最小的文件", size=13, weight="600")
lines = [
    ("子图 1：", "ink"),
    ("取 utils（入度 0）→ model 降为 0，train 降为 1", "blue"),
    ("取 model（入度 0）→ train 降为 0", "blue"),
    ("取 train", "blue"),
    ("子图 2：a、b 入度都是 1，取其一（a）", "violet"),
    ("→ b 降为 0，再取 b。环不会卡住排序", "violet"),
]
for i, (s, col) in enumerate(lines):
    svg.text(rx, gy0 + 28 + i * 20, s, size=12, color=col)
sy = gy0 + 160
svg.text(rx, sy, "拼成两条训练样本，每个文件前加路径注释：", size=12)
for k, (order, col) in enumerate(zip(orders, ("blue", "violet"))):
    y = sy + 14 + k * 76
    h = 16 * len(order) + 12
    svg.rect(rx, y, 300, h, fill=col, fill_opacity=0.06, stroke=col, width=1.2, rx=6)
    for j, f in enumerate(order):
        svg.text(rx + 12, y + 20 + j * 16, "# %s    ……文件内容……" % f, size=12, color=col)

svg.save("l06-coder-data.svg")
