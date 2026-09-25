"""Figure 4.4: why the cache is exact. Dependency graph of hidden states h_j^(l) (positions x layers).

Left, causal (decoder): the node (l, j) receives edges only from layer l-1 positions <= j. When x_5
arrives, only the new column is computed; every old node keeps its value, so its K and V can be reused.
Right, bidirectional (encoder): (l, j) receives edges from every position, so x_5 flows into every old
node from layer 1 upwards and every cached K, V would be stale. Schematic.
Writes deepseek/figures/l04-causal-exact.svg
"""

from l04_svg import SVG

W, H = 720, 372
NP, NL = 5, 4          # positions, layers (0 = embedding)
DX, DY = 54, 64
R = 10
svg = SVG(W, H, "缓存为什么精确：因果注意力与双向注意力的依赖图")


def node_xy(ox, oy, l, j):
    return ox + (j - 1) * DX, oy + (NL - 1 - l) * DY


def draw(ox, oy, causal):
    # faint background edges for the old part (structure only)
    for l in range(1, NL):
        for j in range(1, NP):
            for i in range(1, NP):
                if causal and i > j:
                    continue
                x1, y1 = node_xy(ox, oy, l - 1, i)
                x2, y2 = node_xy(ox, oy, l, j)
                svg.line(x1, y1 - R, x2, y2 + R, color="rule", width=0.8)
    # edges touching the new column
    for l in range(1, NL):
        x2, y2 = node_xy(ox, oy, l, NP)
        for i in range(1, NP + 1):          # into the new node: from all positions <= 5 (both cases)
            x1, y1 = node_xy(ox, oy, l - 1, i)
            svg.line(x1, y1 - R, x2, y2 + R, color="blue", width=1.2, opacity=0.8)
        if not causal:                       # out of the new column into old nodes: only bidirectional
            x1, y1 = node_xy(ox, oy, l - 1, NP)
            for j in range(1, NP):
                x2b, y2b = node_xy(ox, oy, l, j)
                svg.line(x1, y1 - R, x2b, y2b + R, color="red", width=1.4)
    # nodes
    for l in range(NL):
        for j in range(1, NP + 1):
            x, y = node_xy(ox, oy, l, j)
            if j == NP:
                svg.circle(x, y, R, fill="blue", fill_opacity=0.35, stroke="blue", width=1.6)
            elif causal or l == 0:
                svg.circle(x, y, R, fill="amber", fill_opacity=0.30, stroke="amber", width=1.3)
            else:
                svg.circle(x, y, R, fill="red", fill_opacity=0.25, stroke="red", width=1.6)
    for j in range(1, NP + 1):
        x, y = node_xy(ox, oy, 0, j)
        svg.text(x, y + 30, "x_{%d}" % j, size=12, anchor="middle", color="blue" if j == NP else "ink",
                 weight="600" if j == NP else None)
    for l in range(NL):
        x, y = node_xy(ox, oy, l, 1)
        svg.text(x - 18, y + 4, "嵌入" if l == 0 else "第 %d 层" % l, size=12, anchor="end", color="muted")


svg.text(16, 24, "Ⓐ 因果注意力（GPT 式的解码器）", size=14, weight="600")
svg.text(376, 24, "Ⓑ 双向注意力（编码器式）", size=14, weight="600")
draw(88, 64, True)
draw(450, 64, False)
yt = 64 + (NL - 1) * DY + 58
svg.text(16, yt, "新 token 只向自己这一列连线（蓝）：", size=12)
svg.text(16, yt + 18, "旧节点的输入没有变，值也不变（黄）", size=12)
svg.text(16, yt + 36, "→ 它们的 K、V 算一次就是终值，缓存精确", size=12, weight="600", color="amber")
svg.text(376, yt, "新 token 还连向每一个旧位置（红线）：", size=12)
svg.text(376, yt + 18, "第 1 层起所有旧节点都被改写（红）", size=12)
svg.text(376, yt + 36, "→ 旧的 K、V 全部作废，只能整段重算", size=12, weight="600", color="red")
svg.save("l04-causal-exact.svg")
