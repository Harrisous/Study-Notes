"""Figure 13.15: the two abandoned routes - process reward models and Monte Carlo tree search (schematic).

The difficulties listed are the ones the R1 paper gives (v1 section 4.2 / v2 section G.2, recorded in
reference-docs/lesson-13-doc.md C30, C31). Scores drawn on steps and nodes are illustrative only.
Writes deepseek/figures/l13-prm-mcts.svg
"""

from l13_svg import SVG

W, H = 760, 420
svg = SVG(W, H, "两条被放弃的路线：过程奖励模型（PRM）与蒙特卡洛树搜索（MCTS）")

# ------------------------------------------------------------ (a) ORM vs PRM
svg.text(16, 22, "(a) 结果奖励（R1 用的）与过程奖励（PRM）", size=13, weight="600")
steps = ["步骤 1：两边平方", "步骤 2：整理成四次方程", "步骤 3：因式分解", "步骤 4：检验增根", "答案：\\boxed{…}"]
prm = ["0.9", "0.8", "0.7", "0.4", ""]
x0, w = 100, 170
for k, s in enumerate(steps):
    y = 40 + k * 40
    last = k == len(steps) - 1
    svg.rect(x0, y, w, 30, fill="green" if last else "surface", fill_opacity=0.12 if last else None,
             stroke="green" if last else "ink", width=1.2, rx=6)
    svg.text(x0 + w / 2, y + 20, s, size=12, anchor="middle")
    if not last:
        svg.arrow(x0 + w / 2, y + 31, x0 + w / 2, y + 39, width=1.1)
        # PRM score on the right
        svg.rect(x0 + w + 14, y + 4, 44, 22, fill="violet", fill_opacity=0.14, stroke="violet", width=1.1, rx=11)
        svg.text(x0 + w + 36, y + 20, prm[k], size=12, anchor="middle", color="violet", weight="600")
svg.text(x0 + w + 36, 34, "PRM", size=12, anchor="middle", color="violet", weight="700")
# ORM score on the left of the answer
ya = 40 + 4 * 40
svg.rect(28, ya + 4, 52, 22, fill="green", fill_opacity=0.16, stroke="green", width=1.1, rx=11)
svg.text(54, ya + 20, "1 / 0", size=12, anchor="middle", color="green", weight="700")
svg.arrow(82, ya + 15, 98, ya + 15, color="green", width=1.2)
svg.text(54, ya - 8, "结果奖励", size=12, anchor="middle", color="green", weight="600")
svg.text(28, 70, "每一步", size=12, color="muted")
svg.text(28, 86, "都要打分 →", size=12, color="muted")
# problems
py0 = 262
svg.rect(16, py0, 354, 142, fill="red", fill_opacity=0.05, stroke="red", width=1.1, rx=8, dash="5 4")
svg.text(28, py0 + 22, "PRM 的三个难处（论文 §G.2）", size=12.5, color="red", weight="600")
for k, s in enumerate(["① 通用推理里，“一步”很难明确定义",
                       "② 判断中间一步对不对很难：模型自动标注",
                       "    效果不理想，人工标注又无法规模化",
                       "③ 基于模型的 PRM 必然招来 reward hacking，",
                       "    重训它要额外资源，流程也更复杂"]):
    svg.text(28 + (16 if s.startswith("    ") else 0), py0 + 46 + k * 19, s.strip(), size=12)

# ------------------------------------------------------------ (b) MCTS
svg.text(392, 22, "(b) MCTS：在 token 序列上做树搜索", size=13, weight="600")
root = (566, 58)
svg.rect(root[0] - 40, root[1] - 14, 80, 28, fill="blue", fill_opacity=0.14, stroke="blue", width=1.2, rx=6)
svg.text(root[0], root[1] + 5, "题目 q", size=12, anchor="middle", weight="600")
lvl1 = [(446, 130, "0.62"), (566, 130, "0.35"), (686, 130, "0.48")]
for x, y, v in lvl1:
    svg.line(root[0], root[1] + 14, x, y - 14, color="ink", width=1.1)
    svg.rect(x - 38, y - 14, 76, 28, fill="surface", stroke="ink", width=1.1, rx=6)
    svg.text(x, y + 5, "部分推理", size=12, anchor="middle")
    svg.rect(x + 22, y - 30, 38, 18, fill="violet", fill_opacity=0.14, stroke="violet", width=1, rx=9)
    svg.text(x + 41, y - 17, v, size=11, anchor="middle", color="violet")
lvl2 = [(410, 200, "0.71"), (482, 200, "0.40")]
for x, y, v in lvl2:
    svg.line(446, 144, x, y - 14, color="ink", width=1.1)
    svg.rect(x - 32, y - 14, 64, 28, fill="surface", stroke="ink", width=1.1, rx=6)
    svg.text(x, y + 5, "……", size=12, anchor="middle")
    svg.text(x, y + 30, v, size=11, anchor="middle", color="violet")
svg.text(566, 176, "未展开", size=12, anchor="middle", color="muted")
svg.text(686, 176, "未展开", size=12, anchor="middle", color="muted")
svg.text(586, 212, "紫色数字 = 价值模型对", size=12, color="violet")
svg.text(586, 229, "“从这里继续能答对”的估计", size=12, color="violet")
svg.text(392, 250, "每个节点最多扩展 K 个子节点（论文设了上限）", size=12, color="muted")
py1 = 262
svg.rect(390, py1, 354, 142, fill="red", fill_opacity=0.05, stroke="red", width=1.1, rx=8, dash="5 4")
svg.text(402, py1 + 22, "MCTS 的两个难处（论文 §G.2）", size=12.5, color="red", weight="600")
for k, s in enumerate(["① 不像棋类，token 生成的搜索空间指数级大：",
                       "    每步 129,280 种可能；限制扩展数又容易",
                       "    卡在局部最优",
                       "② 价值模型决定每一步搜索的方向，而细粒度",
                       "    价值模型本身很难训练，难以迭代提升"]):
    svg.text(402 + (16 if s.startswith("    ") else 0), py1 + 46 + k * 19, s.strip(), size=12)
svg.save("l13-prm-mcts.svg")
