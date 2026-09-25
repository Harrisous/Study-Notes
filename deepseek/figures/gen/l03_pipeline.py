"""Figure 3.1: InstructGPT's three steps (redrawn after Ouyang et al. 2022, Figure 2).

Dataset sizes are arXiv v1 Table 6 (train split, counted in prompts):
  SFT 11,295 labeler + 1,430 customer = 12,725;  RM 6,623 + 26,584 = 33,207;  PPO 31,144 (API only).
beta = 0.02 and gamma_ptx = 27.8 are from Appendix C.4. Writes deepseek/figures/l03-pipeline.svg
"""
from l03_svg import SVG, box

assert 11295 + 1430 == 12725 and 6623 + 26584 == 33207

W, H = 750, 480
svg = SVG(W, H, "InstructGPT 的三步：监督微调、训练奖励模型、用 PPO 做强化学习")
cw = 226
xs = [12, 262, 512]
heads = [("① 监督微调（SFT）", "ink"), ("② 训练奖励模型（RM）", "green"), ("③ 用 PPO 做强化学习", "blue")]
for (t, col), x in zip(heads, xs):
    svg.text(x + cw / 2, 24, t, size=15, anchor="middle", color=col, weight="700")

ys = [42, 132, 216, 302]
hs = [72, 58, 62, 56]


def col_boxes(x, specs):
    for k, (y, h) in enumerate(zip(ys, hs)):
        lines, kw = specs[k]
        box(svg, x, y, cw, h, lines, **kw)
        if k < 3:
            svg.arrow(x + cw / 2, y + h + 2, x + cw / 2, ys[k + 1] - 3, color="muted", width=1.5)


human = dict(stroke="ink", dash="5 3")
col_boxes(xs[0], [
    (["抽一个提示 *q*", "12,725 个训练提示", "标注员写 11,295 + API 1,430"], dict(weight="600")),
    (["标注员写出示范回答 *o*", "（人工，约 40 名标注员）"], dict(weight="600", **human)),
    (["有监督微调 GPT-3", "在示范上做最大似然"], dict(weight="600")),
    (["SFT 模型 π^{SFT}"], dict(stroke="blue", fill="blue", fill_opacity=0.10, color="blue", weight="700")),
])
col_boxes(xs[1], [
    (["同一个 *q* 采样 K = 4…9 个回答", "33,207 个训练提示", "标注员写 6,623 + API 26,584"], dict(weight="600")),
    (["标注员从好到差排序", "例：D > C > A > B（人工）"], dict(weight="600", **human)),
    (["拆成 C(K,2) 对两两比较", "Bradley–Terry 损失，训 1 个 epoch"], dict(weight="600")),
    (["奖励模型 r_{φ}(q, o)：输出一个数"], dict(stroke="green", fill="green", fill_opacity=0.10, color="green", weight="700")),
])
col_boxes(xs[2], [
    (["抽一个新提示 *q*", "31,144 个 API 提示", "只有提示，没有人工标签"], dict(weight="600")),
    (["策略 π_{θ} 生成回答 *o*"], dict(weight="600", stroke="blue")),
    (["奖励 = r_{φ}(q,o) − β·log(π_{θ}/π_{ref})", "β = 0.02"], dict(weight="600", stroke="green")),
    (["PPO 更新 π_{θ}", "PPO-ptx：另混预训练梯度"], dict(stroke="blue", fill="blue", fill_opacity=0.10, color="blue", weight="700")),
])
# RL loop on the right of column 3
xr = xs[2] + cw
svg.path("M%d %d h10 V%d h-7" % (xr, ys[3] + hs[3] / 2, ys[1] + hs[1] / 2), stroke="blue", width=1.5, arrow=True)
# RM feeds the reward box in column 3
x1 = xs[1] + cw
ym = ys[3] + hs[3] / 2
svg.path("M%d %d h12 V%d h9" % (x1, ym, ys[2] + hs[2] / 2), stroke="green", width=1.6, arrow=True)
# SFT model initialises policy and reference (route under the columns)
xa = xs[0] + cw / 2
xb = xs[2] + cw / 2
yb = ys[3] + hs[3]
svg.path("M%d %d V%d H%d V%d" % (xa, yb + 2, yb + 26, xb, yb + 5), stroke="blue", width=1.6, arrow=True)
svg.text((xa + xb) / 2, yb + 46, "初始化策略 π_{θ}，并冻结一份作参考 π_{ref}", size=12.5, anchor="middle", color="blue")
# iterate steps 2-3
svg.path("M%d %d C %d %d, %d %d, %d %d" % (xs[2] + 30, ys[0], xs[2] - 10, ys[0] - 22, xs[1] + cw + 10, ys[0] - 22,
                                           xs[1] + cw - 30, ys[0] - 2), stroke="muted", width=1.3, dash="4 3", arrow=True)
# legend / notes
yl = 436
svg.rect(14, yl - 10, 22, 12, fill="surface", stroke="ink", dash="5 3", width=1.2, rx=3)
svg.text(42, yl, "虚线框：人工完成的步骤。上方虚线弧：第 2、3 步可以反复迭代（用新策略的回答再收集比较）。", size=12, color="muted")
svg.text(14, yl + 22, "数字为 InstructGPT 训练集规模（按提示计，arXiv v1 表 6）；β、PPO-ptx 见附录 C.4。", size=12, color="muted")
svg.save("l03-pipeline.svg")
