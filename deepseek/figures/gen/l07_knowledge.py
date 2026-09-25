"""Figure 7.7: schematic of knowledge hybridity and redundancy (the DeepSeekMoE hypothesis).

A schematic, not a measurement: topic labels are illustrative. Left: a few large experts, each
holding several unrelated topics plus its own copy of common knowledge. Right: many small
experts holding one topic each, with common knowledge moved into one shared expert.
Writes deepseek/figures/l07-knowledge.svg
"""

from l07_svg import SVG

W, H = 740, 348
svg = SVG(W, H, "知识混杂与知识冗余示意：少而大的专家 vs 细粒度专家 + 共享专家")

topics = ["代码", "数学", "诗词", "法律", "医学", "体育", "历史", "化学", "金融", "地理", "音乐", "烹饪"]

# ---------------- (a) coarse experts
svg.text(20, 24, "(a) 传统 MoE：少而大的专家（画出 4 个）", size=13, weight="600")
bw, gap, x0, y0 = 72, 12, 24, 44
sh = 30
for e in range(4):
    x = x0 + e * (bw + gap)
    svg.rect(x, y0, bw, 4 * sh + 30, fill="surface", stroke="blue", width=1.4, rx=6)
    svg.text(x + bw / 2, y0 + 18, "专家 %d" % (e + 1), size=12, anchor="middle", color="blue", weight="600")
    for k in range(3):
        yy = y0 + 26 + k * sh
        svg.rect(x + 6, yy, bw - 12, sh - 4, fill="blue", fill_opacity=0.10 + 0.08 * k, rx=3)
        svg.text(x + bw / 2, yy + 18, topics[e * 3 + k], size=12, anchor="middle")
    yy = y0 + 26 + 3 * sh
    svg.rect(x + 6, yy, bw - 12, sh - 4, fill="violet", fill_opacity=0.22, rx=3)
    svg.text(x + bw / 2, yy + 18, "公共知识", size=12, anchor="middle", color="violet")
# annotations
ay = y0 + 4 * sh + 66
svg.text(24, ay, "混杂：一个专家要同时装下 3 类互不相干的知识，", size=12, color="ink")
svg.text(24, ay + 18, "它们很难在同一组参数里被同时用好。", size=12, color="ink")
svg.text(24, ay + 44, "冗余：“公共知识”（语法、常识……）在每个专家里", size=12, color="violet")
svg.text(24, ay + 62, "都要各学一份，参数花在了重复的东西上。", size=12, color="violet")
# brace across the 4 violet cells
by = y0 + 4 * sh + 36
svg.path("M%d %d L%d %d L%d %d L%d %d" % (x0 + 6, by, x0 + 6, by + 6, x0 + 4 * bw + 3 * gap - 6, by + 6,
                                          x0 + 4 * bw + 3 * gap - 6, by), stroke="violet", width=1.2)

# ---------------- (b) fine-grained + shared
RX = 392
svg.text(RX, 24, "(b) DeepSeekMoE：细粒度专家 + 共享专家", size=13, weight="600")
svg.rect(RX, 44, 330, 40, fill="violet", fill_opacity=0.16, stroke="violet", width=1.4, rx=6)
svg.text(RX + 165, 62, "共享专家：公共知识只存一份", size=12, anchor="middle", color="violet", weight="600")
svg.text(RX + 165, 78, "每个 token 都经过它，不参与路由", size=12, anchor="middle", color="violet")
sbw, sgap = 50, 6
for k, tname in enumerate(topics):
    r, c = divmod(k, 6)
    x = RX + c * (sbw + sgap)
    y = 100 + r * 52
    svg.rect(x, y, sbw, 44, fill="blue", fill_opacity=0.12, stroke="blue", width=1.1, rx=5)
    svg.text(x + sbw / 2, y + 17, "%d" % (k + 1), size=11, anchor="middle", color="muted")
    svg.text(x + sbw / 2, y + 35, tname, size=12, anchor="middle")
svg.text(RX, 228, "每个小专家只装一类知识，一个 token 同时激活几个", size=12, color="ink")
svg.text(RX, 246, "相关的小专家，按需组合（§5）；公共知识交给", size=12, color="ink")
svg.text(RX, 264, "共享专家，路由专家不必各存一份（§6）。", size=12, color="ink")

svg.line(20, 306, W - 20, 306, color="rule", width=1)
svg.text(20, 330, "示意图：这是 DeepSeekMoE 论文提出的假说。真实专家“装了什么”是训练中自发形成的，论文只给出间接证据（§8）。",
         size=12, color="muted")

svg.save("l07-knowledge.svg")
