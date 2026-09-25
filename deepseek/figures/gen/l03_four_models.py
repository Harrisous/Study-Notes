"""Figure 3.11: the four networks of PPO-based RLHF and one iteration of data flow.

Policy pi_theta (trained), reference pi_ref (frozen), reward model r_phi (frozen), value network V_psi
(trained). Sizes in the side notes are InstructGPT's (arXiv v1, sec. 3.5 and App. C.3-C.4): policy
1.3B / 6B / 175B, reference = RL-init model of the same size, one 6B RM, one 6B value function
initialised from the RM. Schematic; writes deepseek/figures/l03-four-models.svg

Layout note: every multi-line box uses a line gap of about 1.75 x the font size, so a subscript on one
line (pi_theta, r_phi, V_psi) never touches the line below; no connector crosses a label.
"""
from l03_svg import SVG, box

W, H = 740, 424
svg = SVG(W, H, "PPO-RLHF 的四个网络与一轮迭代的数据流")
trained = dict(width=2.4)
frozen = dict(dash="6 4", width=1.4)
GAP = 21

# column 1: prompt and policy
box(svg, 16, 40, 96, 40, "提示 *q*", stroke="rule", size=13.5, weight="600")
svg.arrow(64, 82, 64, 101, color="muted", width=1.5)
box(svg, 16, 104, 136, 120, ["策略 π_{θ}（训练）", "① 生成回答 *o*", "记下每个 token 的", "log π_{old}(o_{t})"],
    stroke="blue", fill="blue", fill_opacity=0.10, color="blue", weight="700", size=12.5, line_gap=GAP,
    **trained)
svg.text(84, 242, "InstructGPT：", size=11.5, anchor="middle", color="muted")
svg.text(84, 258, "1.3B / 6B / 175B", size=11.5, anchor="middle", color="muted")

# fan-out bus to the three scorers
xb = 168
svg.line(152, 165, xb, 165, color="ink", width=1.5)
svg.line(xb, 67, xb, 263, color="ink", width=1.5)
for y in (67, 165, 263):
    svg.arrow(xb, y, 184, y, color="ink", width=1.5)
svg.text(176, 30, "② 打分：三个网络各读一遍 (*q*, *o*)", size=12.5, weight="600")

# column 2: scorers (boxes 62 high, centres at 67 / 165 / 263)
box(svg, 184, 36, 212, 62, ["参考模型 π_{ref}（冻结）", "每个 token 的 log π_{ref}(o_{t})"],
    stroke="muted", color="ink", weight="700", size=12.5, line_gap=GAP + 1, **frozen)
box(svg, 184, 134, 212, 62, ["奖励模型 r_{φ}（冻结）", "整段回答一个分数"],
    stroke="green", fill="green", fill_opacity=0.08, color="green", weight="700", size=12.5,
    line_gap=GAP + 1, **frozen)
box(svg, 184, 232, 212, 62, ["价值网络 V_{ψ}（训练）", "每个 token 的 V_{ψ}(s_{t})"],
    stroke="violet", fill="violet", fill_opacity=0.08, color="violet", weight="700", size=12.5,
    line_gap=GAP + 1, **trained)
svg.text(290, 114, "InstructGPT：与策略同尺寸", size=11.5, anchor="middle", color="muted")
svg.text(290, 212, "InstructGPT：6B", size=11.5, anchor="middle", color="muted")
svg.text(190, 312, "InstructGPT：6B，由 RM 初始化", size=11.5, color="muted")

# column 3: per-token reward and GAE
box(svg, 432, 84, 150, 84, ["③ 逐 token 奖励 r_{t}", "−β·log(π_{old}/π_{ref})", "末位再加 r_{φ}，式 (3.8)"],
    stroke="ink", weight="700", size=12, line_gap=GAP + 1)
svg.arrow(396, 70, 431, 100, color="ink", width=1.5)
svg.arrow(396, 163, 431, 150, color="green", width=1.5)
# log pi_old from the policy, routed over the top (clear of the prompt box and of every label)
svg.path("M132 104 V14 H500 V81", stroke="blue", width=1.4, dash="5 3", arrow=True)
svg.text(512, 40, "log π_{old}", size=11.5, color="blue")
svg.text(508, 66, "（采样时已记下）", size=11.5, color="blue")
box(svg, 432, 232, 150, 62, ["④ GAE → 优势 Â_{t}", "式 (3.24)"], stroke="ink", weight="700", size=12,
    line_gap=GAP + 1)
svg.arrow(507, 168, 507, 229, color="ink", width=1.5)
svg.arrow(396, 263, 429, 263, color="violet", width=1.5)

# column 4: update
box(svg, 600, 220, 132, 86, ["⑤ 更新", "π_{θ}：裁剪目标 (3.17)", "V_{ψ}：回归 (3.25)"],
    stroke="blue", weight="700", size=12, line_gap=GAP + 2)
svg.arrow(582, 263, 597, 263, color="ink", width=1.5)
# loops back to the two trained networks
svg.path("M666 306 V352 H34 V227", stroke="blue", width=1.6, arrow=True)
svg.path("M388 352 V297", stroke="violet", width=1.6, arrow=True)
svg.text(404, 345, "反向传播，只更新两个“训练”的网络", size=12, color="blue")

# legend
svg.rect(16, 380, 26, 14, fill="surface", stroke="ink", width=2.4, rx=3)
svg.text(48, 392, "训练：前向 + 反向 + 优化器状态", size=12)
svg.rect(262, 380, 26, 14, fill="surface", stroke="muted", width=1.4, dash="6 4", rx=3)
svg.text(294, 392, "冻结：只做前向", size=12)
svg.text(16, 414, "π_{old} 不是第五个网络：只需要采样时顺手记下的逐 token 对数概率。", size=12, color="muted")
svg.save("l03-four-models.svg")
