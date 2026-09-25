"""Figure M9.1: a language model generating an answer, read as a Markov decision process.

Top: the two-token toy example used throughout M09 (prompt "2+3=", first token 5/6, second
token ！/吧), drawn as state -> policy -> action -> next state. Bottom: the dictionary
between RL terms and LLM terms. Probabilities are the toy values (author's construction).
Writes math/figures/m09-llm-mdp.svg
"""

from m09_svg import SVG

W, H = 720, 420
svg = SVG(W, H, "语言模型生成回答 = 在一个马尔可夫决策过程里按策略行动")

# ---------- top: the chain ----------
BY, BH, BW = 44, 74, 128          # state boxes
PW = 124                          # policy panels
xs0 = 14
xp0 = xs0 + BW + 14
xs1 = xp0 + PW + 14
xp1 = xs1 + BW + 14
xs2 = xp1 + PW + 14

svg.text(14, 24, "一次生成 = 一个回合（例子：M09 通篇使用的“2+3=”玩具模型）", size=13, weight="600")


def state_box(x, label, content, sub=None, terminal=False):
    svg.rect(x, BY, BW, BH, fill="blue" if not terminal else "green", fill_opacity=0.08,
             stroke="blue" if not terminal else "green", width=1.4, rx=8)
    svg.text(x + 10, BY + 20, label, size=12, color="muted")
    svg.text(x + BW / 2, BY + 46, content, size=16, anchor="middle", weight="600")
    if sub:
        svg.text(x + BW / 2, BY + 64, sub, size=12, anchor="middle", color="muted")


state_box(xs0, "状态 s_{0}", "“2+3=”", "只有提示 q")
state_box(xs1, "状态 s_{1}", "“2+3=5”", "q + 已生成的 “5”")
state_box(xs2, "状态 s_{2}（终止）", "“2+3=5！”", "遇到结束符", terminal=True)


def policy_panel(x, title, items, chosen):
    # items: list of (token, prob); draw horizontal bars
    svg.text(x + PW / 2, BY - 6, title, size=12, anchor="middle", color="muted")
    y = BY + 2
    for tok, p in items:
        bw = 70 * p
        col = "blue" if tok == chosen else "muted"
        svg.text(x + 4, y + 13, "“%s”" % tok, size=13)
        svg.rect(x + 34, y + 2, bw, 15, fill=col, fill_opacity=0.85 if tok == chosen else 0.35, rx=2)
        svg.text(x + 34 + bw + 4, y + 14, "%.1f" % p, size=12, color=col)
        y += 23
    # the sampled action: an arrow from this state box to the next one
    ay = BY + BH - 14
    svg.arrow(x - 14, ay, x + PW + 13, ay, color="blue", width=1.6)
    svg.text(x + PW / 2, ay + 18, "采样 a = “%s”" % chosen, size=12, anchor="middle", color="blue",
             weight="600")


policy_panel(xp0, "策略 π_{θ}(·|s_{0})", [("5", 0.6), ("6", 0.4)], "5")
policy_panel(xp1, "策略 π_{θ}(·|s_{1})", [("！", 0.7), ("吧", 0.3)], "！")

# reward under the terminal box
ry = BY + BH + 16
svg.arrow(xs2 + BW / 2, BY + BH + 2, xs2 + BW / 2, ry + 10, color="green", width=1.6)
svg.rect(xs2 + 6, ry + 12, BW - 12, 44, fill="green", fill_opacity=0.12, stroke="green", width=1.2, rx=6)
svg.text(xs2 + BW / 2, ry + 30, "奖励 r = 1", size=13, anchor="middle", color="green", weight="600")
svg.text(xs2 + BW / 2, ry + 48, "（答对且语气肯定）", size=12, anchor="middle", color="green")

svg.text(14, BY + BH + 44, "这条回答出现的概率 = 0.6 × 0.7 = 0.42；", size=12, color="muted")
svg.text(14, BY + BH + 62, "转移是确定性的：下一个状态就是把 token 接到前缀末尾。", size=12, color="muted")

# ---------- bottom: dictionary ----------
TY = 214
rows = [
    ("强化学习", "语言模型（RLHF / 可验证奖励的 RL）"),
    ("状态 s_{t}", "提示加上已经生成的前缀 (q, o_{<t})"),
    ("动作 a_{t}", "下一个 token o_{t}：从整个词表里选一个"),
    ("策略 π_{θ}(a | s)", "语言模型最后一层 softmax 给出的下一个 token 分布"),
    ("转移 P(s′ | s, a)", "确定性：s_{t+1} = s_{t} 接上 a_{t}，没有外部随机性"),
    ("奖励 r_{t}", "奖励模型或规则在回答结束时打一次分（RLHF 另有每 token 的 KL 项）"),
    ("回合、折扣 γ", "一段回答，遇到结束符 EOS 停止；长度有限，通常取 γ = 1"),
]
C0, C1, C2 = 14, 170, 706
RH = 26
for i, (a, b) in enumerate(rows):
    y = TY + i * RH
    if i == 0:
        svg.rect(C0, y, C2 - C0, RH, fill="grid", fill_opacity=1.0)
    svg.line(C0, y + RH, C2, y + RH, color="rule", width=1)
    w = "600" if i == 0 else None
    svg.text(C0 + 8, y + 18, a, size=13, weight=w, color="ink")
    svg.text(C1 + 8, y + 18, b, size=13, weight=w, color="ink")
svg.line(C1, TY, C1, TY + RH * len(rows), color="rule", width=1)
svg.rect(C0, TY, C2 - C0, RH * len(rows), stroke="rule", width=1)

svg.save("m09-llm-mdp.svg")
