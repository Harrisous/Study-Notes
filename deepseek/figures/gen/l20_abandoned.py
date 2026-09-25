"""Figure 20.8: routes DeepSeek used and later replaced, or tried and did not adopt, on a date axis.

Positions are computed from release dates (l20_num.DATES, lesson-00-doc section 3.2). A solid band
means "in a released model"; a hollow dashed band means "published or tried, not in a DeepSeek
production model"; a diamond marks a failed attempt reported by a paper (R1 appendix: PRM, MCTS,
RL on small bases). Every row is sourced in the lesson table on page 7.
Writes deepseek/figures/l20-abandoned.svg
"""

import datetime as dt

from l20_svg import SVG
from l20_num import DATES

D0, D1 = dt.date(2023, 10, 1), dt.date(2026, 11, 1)
W, H = 760, 590
X0, X1 = 250, 745
svg = SVG(W, H, "被放弃或被替换的路线：什么时候上、什么时候下、换成了什么")


def xd(key_or_date):
    d = dt.date(*DATES[key_or_date]) if isinstance(key_or_date, str) else key_or_date
    return X0 + (d - D0).days / (D1 - D0).days * (X1 - X0)


# (label, start, end, replacement, kind)   kind: used / tried / point
ROWS = [
    ("逐头缓存 K、V（MHA / GQA）", "DeepSeek LLM", "V2", "MLA", "used"),
    ("以辅助损失为主的均衡", "DeepSeekMoE", "V3", "偏置均衡", "used"),
    ("softmax 门控", "DeepSeekMoE", "V3", "sigmoid", "used"),
    ("设备级 token 丢弃", "V2", "V3", "不丢 token", "used"),
    ("设备 / 节点受限路由", "V2", "V4", "取消限制", "used"),
    ("神经奖励模型给推理打分", "DeepSeekMath", "R1", "规则验证器", "used"),
    ("过程奖励模型 PRM", "R1", None, "试过未用", "point"),
    ("蒙特卡洛树搜索 MCTS", "R1", None, "试过未用", "point"),
    ("小底座直接做 RL", "R1", None, "改用蒸馏", "point"),
    ("NSA：按块选择 + 压缩分支", dt.date(2025, 2, 16), "V3.2-Exp", "DSA", "tried"),
    ("无约束的超连接 HC", dt.date(2024, 9, 29), "mHC", "mHC", "tried"),   # arXiv v1 (lesson-16-doc)
    ("预训练里的 MTP 模块", "V3", "V4.1", "DSpark", "used"),
    ("稠密注意力预热", "V3.2-Exp", "V4.1", "从头稀疏", "used"),
    ("HCA 与重叠压缩", "V4", "V4.1", "纯 CSA2", "used"),
    ("前 3 层哈希路由", "V4", "V4.1", "标准路由", "used"),
    ("SWA 状态落盘 / 精确重建", "V4", "V4.1", "有界回放", "used"),
]

TOP, RH = 58, 30
# year grid
for y in (2024, 2025, 2026):
    x = xd(dt.date(y, 1, 1))
    svg.line(x, TOP - 12, x, TOP + len(ROWS) * RH - 4, color="rule", width=1)
    svg.text(x + 3, TOP - 16, str(y), size=11, color="muted")
for key, lab in [("V2", "V2"), ("V3", "V3"), ("R1", "R1"), ("V3.2-Exp", "V3.2-Exp"), ("V4", "V4"), ("V4.1", "V4.1")]:
    x = xd(key)
    svg.line(x, TOP - 4, x, TOP + len(ROWS) * RH - 4, color="grid", width=1, dash="2 3")
svg.text(X0, 20, "横轴：发布日期（2023-10 至 2026-10）；虚竖线是主干模型发布日", size=12, color="muted")
for key, lab in [("V2", "V2"), ("V3", "V3"), ("V3.2-Exp", "V3.2-Exp"), ("V4", "V4"), ("V4.1", "V4.1")]:
    svg.text(xd(key), TOP + len(ROWS) * RH + 12, lab, size=11, anchor="middle", color="muted")
svg.text(xd("R1") + 2, TOP + len(ROWS) * RH + 26, "R1", size=11, anchor="middle", color="muted")

for i, (lab, a, b, rep, kind) in enumerate(ROWS):
    y = TOP + i * RH
    svg.text(X0 - 10, y + 15, lab, size=12, anchor="end")
    xa = xd(a)
    if kind == "point":
        svg.polygon([(xa, y + 4), (xa + 6, y + 11), (xa, y + 18), (xa - 6, y + 11)], fill="red", fill_opacity=0.15,
                    stroke="red", width=1.2)
        svg.text(xa - 10, y + 15, rep, size=11, anchor="end", color="red")
        continue
    xb = xd(b)
    if kind == "used":
        svg.rect(xa, y + 5, xb - xa, 12, fill="red", fill_opacity=0.28, stroke="red", width=0.8)
    else:
        svg.rect(xa, y + 5, xb - xa, 12, fill="red", fill_opacity=0.06, stroke="red", width=1, dash="3 2")
    svg.line(xb, y + 2, xb, y + 20, color="ink", width=1.6)
    tx = xb + 5
    anchor = "start"
    if tx + 14 * len(rep) > X1 + 10:
        tx, anchor = xa - 5, "end"
    svg.text(tx, y + 15, "→ " + rep if anchor == "start" else rep + " ←", size=11, anchor=anchor, color="green")

# thin continuation: the aux-loss route survives as a 1e-4 sequence-wise guard loss
yy = TOP + 1 * RH
svg.line(xd("V3") + 82, yy + 11, xd("V4.1") - 150, yy + 11, color="red", width=1, dash="2 2")
svg.text(xd("V4.1") - 146, yy + 15, "只留 10^{-4} 的序列级护栏", size=11, color="red")
# device-limited (M = 3 devices, V2) became node-limited (M = 4 nodes, V3) before V4 removed it
yy = TOP + 4 * RH
svg.line(xd("V3"), yy + 5, xd("V3"), yy + 17, color="ink", width=1)
svg.text((xd("V2") + xd("V3")) / 2, yy + 15, "3 台设备", size=11, anchor="middle")
svg.text((xd("V3") + xd("V4")) / 2, yy + 15, "4 个节点", size=11, anchor="middle")

LY = TOP + len(ROWS) * RH + 40
svg.rect(10, LY - 10, 22, 12, fill="red", fill_opacity=0.28, stroke="red", width=0.8)
svg.text(38, LY, "用在发布的模型里", size=12)
svg.rect(160, LY - 10, 22, 12, fill="red", fill_opacity=0.06, stroke="red", width=1, dash="3 2")
svg.text(188, LY, "发表或试验过，未进生产", size=12)
svg.polygon([(360, LY - 11), (366, LY - 4), (360, LY + 3), (354, LY - 4)], fill="red", fill_opacity=0.15, stroke="red",
            width=1.2)
svg.text(372, LY, "论文报告的失败尝试", size=12)
svg.text(520, LY, "绿字：换成了什么", size=12, color="green")

svg.save("l20-abandoned.svg")
