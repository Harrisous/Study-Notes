"""Figure 13.10: the multi-stage pipeline of DeepSeek-R1 (redrawn after R1 paper arXiv v2 Fig. 2 = Nature
Fig. 2), annotated with the data sizes and settings recorded in reference-docs/lesson-13-doc.md
(C20-C24, section 3.2-3.4, 3.8). Models blue, data amber, training steps outlined, final model green.
Writes deepseek/figures/l13-pipeline.svg
"""

from l13_svg import SVG

W, H = 760, 588
svg = SVG(W, H, "DeepSeek-R1 的四阶段训练流水线：冷启动、推理 RL、拒绝采样与 SFT、全场景 RL")

COLS = [16, 204, 392, 580]
CW = 164


def model(col, y, name, sub=None, color="blue"):
    x = COLS[col]
    svg.rect(x, y, CW, 40 if not sub else 48, fill=color, fill_opacity=0.14, stroke=color, width=1.5, rx=20)
    svg.text(x + CW / 2, y + (25 if not sub else 21), name, size=13.5, anchor="middle",
             color="ink", weight="700")
    if sub:
        svg.text(x + CW / 2, y + 38, sub, size=12, anchor="middle", color="ink")


def step(col, y, lines, h=None, color="ink", dash=None):
    x = COLS[col]
    h = h or (22 + 17 * len(lines))
    svg.rect(x, y, CW, h, fill="surface", stroke=color, width=1.2, rx=6, dash=dash)
    for k, s in enumerate(lines):
        svg.text(x + CW / 2, y + 21 + 17 * k, s, size=12.5 if k == 0 else 12, anchor="middle",
                 color="ink", weight="600" if k == 0 else None)
    return y + h


def data(col, y, lines, h=None):
    x = COLS[col]
    h = h or (22 + 17 * len(lines))
    svg.rect(x, y, CW, h, fill="amber", fill_opacity=0.12, stroke="amber", width=1.3, rx=6)
    for k, s in enumerate(lines):
        svg.text(x + CW / 2, y + 21 + 17 * k, s, size=12.5 if k == 0 else 12, anchor="middle",
                 color="ink", weight="600" if k == 0 else None)
    return y + h


def down(col, y1, y2, color="ink"):
    x = COLS[col] + CW / 2
    svg.arrow(x, y1 + 2, x, y2 - 2, color=color, width=1.4)


# column headers
heads = ["① 造冷启动数据", "② 冷启动 SFT + 推理 RL", "③ 拒绝采样 + 第二次 SFT", "④ 全场景 RL"]
for col, h in enumerate(heads):
    svg.text(COLS[col] + CW / 2, 22, h, size=13, anchor="middle", weight="700")
    svg.line(COLS[col], 30, COLS[col] + CW, 30, color="rule", width=1)

# ---- column 1: R1-Zero -> cold start data
model(0, 44, "DeepSeek-V3-Base")
down(0, 84, 102)
e = step(0, 102, ["RL（GRPO）", "规则奖励：准确性 + 格式"])
down(0, e, e + 16)
model(0, e + 16, "DeepSeek-R1-Zero")
y = e + 56
down(0, y, y + 16)
e = step(0, y + 16, ["采样：温度 T = 1.0", "每题多条推理轨迹"])
down(0, e, e + 16)
e = step(0, e + 16, ["过滤", "答案正确 · 格式可读", "去重复、去语言混杂"])
down(0, e, e + 16)
e = step(0, e + 16, ["改写与核验", "V3 改写推理与总结", "人工二次核验"])
down(0, e, e + 16)
cold_y = e + 16
e_cold = data(0, cold_y, ["冷启动数据", "“数千条”长思维链"])

# ---- column 2
model(1, 44, "DeepSeek-V3-Base")
down(1, 84, 102)
e = step(1, 102, ["SFT（冷启动）", "2–3 轮，lr 5×10^{-5} → 5×10^{-6}"])
# arrow from cold-start data to the SFT box
svg.path("M%d %d H%d V%d H%d" % (COLS[0] + CW, cold_y + 30, COLS[0] + CW + 12, 130, COLS[1] - 2),
         stroke="amber", width=1.4, arrow=True)
down(1, e, e + 16)
model(1, e + 16, "Dev1")
y = e + 56
down(1, y, y + 16)
e = step(1, y + 16, ["推理导向的 RL", "规则奖励 + 语言一致性", "ε = 10，T = 1，G = 16"])
down(1, e, e + 16)
model(1, e + 16, "Dev2")
dev2_y = e + 16

# ---- column 3
e = step(2, 44, ["拒绝采样（用 Dev2）", "每题采多个回答", "只保留正确的"])
svg.path("M%d %d H%d V%d H%d" % (COLS[1] + CW, dev2_y + 20, COLS[1] + CW + 12, 80, COLS[2] - 2),
         stroke="blue", width=1.4, arrow=True)
down(2, e, e + 14)
e = data(2, e + 14, ["推理数据 626,933 条", "数学 · 代码 · STEM · 逻辑"])
e2 = data(2, e + 10, ["非推理数据 177,812 条", "写作、问答、翻译等"])
down(2, e2, e2 + 14)
e = data(2, e2 + 14, ["合计 804,745 条", "平均 5,355 token / 条"])
tot_y = e
down(2, e, e + 16)
model(2, e + 16, "DeepSeek-V3-Base", sub="从底座重新开始")
y = e + 64
down(2, y, y + 16)
e = step(2, y + 16, ["SFT（第二次）", "2–3 轮，上下文 32,768"])
down(2, e, e + 16)
model(2, e + 16, "Dev3")
dev3_y = e + 16

# ---- column 4
e = step(3, 44, ["全场景 RL（1,700 步）", "推理题：规则奖励", "一般题：偏好奖励模型", "T = 0.7；偏好奖励只在", "最后 400 步加入"], h=110)
svg.path("M%d %d H%d V%d H%d" % (COLS[2] + CW, dev3_y + 20, COLS[2] + CW + 12, 100, COLS[3] - 2),
         stroke="blue", width=1.4, arrow=True)
down(3, e, e + 18)
model(3, e + 18, "DeepSeek-R1", color="green")
# distillation branch
dy = e + 100
svg.rect(COLS[3], dy, CW, 72, fill="surface", stroke="violet", width=1.3, rx=6, dash="5 4")
svg.text(COLS[3] + CW / 2, dy + 22, "蒸馏（§6）", size=12.5, anchor="middle", weight="600")
svg.text(COLS[3] + CW / 2, dy + 40, "同一份 804,745 条数据", size=12, anchor="middle")
svg.text(COLS[3] + CW / 2, dy + 57, "SFT 6 个稠密小模型", size=12, anchor="middle")
svg.path("M%d %d H%d V%d H%d" % (COLS[2] + CW, tot_y - 20, COLS[2] + CW + 12, dy + 36, COLS[3] - 2),
         stroke="violet", width=1.3, dash="5 4", arrow=True)

# legend
ly = H - 12
svg.rect(16, ly - 12, 30, 16, fill="blue", fill_opacity=0.14, stroke="blue", width=1.3, rx=8)
svg.text(52, ly, "模型检查点", size=12)
svg.rect(140, ly - 12, 30, 16, fill="amber", fill_opacity=0.12, stroke="amber", width=1.3, rx=4)
svg.text(176, ly, "数据", size=12)
svg.rect(226, ly - 12, 30, 16, fill="surface", stroke="ink", width=1.2, rx=4)
svg.text(262, ly, "训练或处理步骤", size=12)
svg.text(744, ly, "据 R1 论文 arXiv v2 图 2（Nature 图 2）重绘，数字见正文", size=12, anchor="end", color="muted")
svg.save("l13-pipeline.svg")
