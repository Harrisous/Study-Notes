"""Figure M9.11: the ladder from REINFORCE to GRPO, one change per rung (schematic, hand-laid-out).

Every estimator has the form  grad J ~ E[ sum_t Psi_t * grad log pi(a_t | s_t) ];  each rung changes Psi_t
(or how the samples are reused) and states what it buys and what it costs.
Writes math/figures/m09-psi-ladder.svg
"""

from m09_svg import SVG

rungs = [
    ("REINFORCE", "Ψ_{t} = R(τ)", "整条轨迹的回报", "无偏，只用样本", "方差大；过去的奖励也算进来", "§2"),
    ("只用之后的回报", "Ψ_{t} = R_{t}", "因果性：去掉动作之前的奖励", "期望不变，方差通常变小", "仍背着之后所有 token 的运气", "§2"),
    ("减去基线", "Ψ_{t} = R_{t} − b(s_{t})", "b 不依赖当前动作", "期望不变，方差可降几个数量级", "要有一个好的 b", "§3"),
    ("价值网络 + GAE", "Ψ_{t} = Σ (γλ)^{l} δ_{t+l}", "δ_{t} = r_{t} + γV(s_{t+1}) − V(s_{t})", "逐 token 记功，λ 调偏差与方差", "要训练与策略同规模的价值网络", "§4"),
    ("PPO", "ρ_{t} Â_{t}，再裁剪", "ρ_{t} = π_{θ} / π_{θold}（逐 token）", "一批样本能更新多步", "只在旧策略附近可信，要限步长", "§5–6"),
    ("GRPO", "Ψ_{i,t} = (r_{i} − mean) / std", "同一题 G 个回答组内标准化", "不要价值网络", "同一回答的 token 共享一个数", "§7"),
]

W, H = 720, 478
svg = SVG(W, H, "从 REINFORCE 到 GRPO：每一级只改一处")
svg.text(14, 22, "所有方法都是 ∇J ≈ E[ Σ_{t} Ψ_{t} ∇ log π_{θ}(a_{t} | s_{t}) ]，区别只在 Ψ_{t} 和样本怎么用", size=13, weight="600")

X_M, W_M = 14, 124        # method box
X_F, W_F = 150, 250       # formula box
X_G = 414                 # gains / costs text
RH, Y0, GAP = 62, 40, 11

for i, (name, psi, sub, gain, cost, page) in enumerate(rungs):
    y = Y0 + i * (RH + GAP)
    col = "violet" if name in ("PPO", "GRPO") else "blue"
    svg.rect(X_M, y, W_M, RH, fill=col, fill_opacity=0.12, stroke=col, width=1.3, rx=8)
    svg.text(X_M + W_M / 2, y + 27, name, size=13.5, anchor="middle", weight="700")
    svg.text(X_M + W_M / 2, y + 47, page, size=12, anchor="middle", color="muted")
    svg.rect(X_F, y, W_F, RH, fill="surface", fill_opacity=1.0, stroke="rule", width=1.1, rx=8)
    svg.text(X_F + 12, y + 26, psi, size=14, weight="600")
    svg.text(X_F + 12, y + 47, sub, size=12, color="muted")
    svg.text(X_G, y + 25, "＋ " + gain, size=12.5, color="green", weight="600")
    svg.text(X_G, y + 47, "－ " + cost, size=12.5, color="red")
    if i < len(rungs) - 1:
        svg.arrow(X_M + W_M / 2, y + RH + 1, X_M + W_M / 2, y + RH + GAP - 1, color="muted", width=1.4)

svg.save("m09-psi-ladder.svg")
