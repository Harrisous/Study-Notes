"""Figure 15.13: test-time context management for search agents (V3.2 paper section 4.4).

Facts (lesson-15-doc.md C44, S4, N27, F12):
  - V3.2's context window is 128K = 131,072 tokens; management is triggered when the token usage exceeds 80%
    of the window: 0.8 * 131,072 = 104,857.6 tokens.
  - Summary "summarizes the overflowed trajectory and re-initiates the rollout"; Discard-75% "discards the first
    75% tool call history in the trajectory"; Discard-all "resets the context by discarding all previous tool call
    history". Baseline Parallel-fewest-step samples N trajectories and keeps the one with the fewest steps.
  - BrowseComp: 51.4 without context management (Table 2 note); Summary up to 60.2 with 364 steps on average;
    Discard-all 67.6, "comparable to parallel scaling while using significantly fewer steps". The paper gives no
    single number for Discard-75% (only a curve in its Figure 6), so none is drawn.
Only the window size and the 80% trigger are to scale; the prompt block and the per-step blocks are schematic.
Writes deepseek/figures/l15-context-management.svg
"""

from l15_svg import SVG, fmt

WINDOW = 131_072
TRIGGER = 0.8 * WINDOW
assert abs(TRIGGER - 104_857.6) < 1e-6

W, H = 760, 404
svg = SVG(W, H, "测试时的上下文管理：触发点与三种策略")

BX0, BX1 = 150, 560          # the 131,072-token window
BH = 24


def bx(tok):
    return BX0 + tok / WINDOW * (BX1 - BX0)


PROMPT = 4_000               # schematic: system prompt + question
STEP = 10_000                # schematic: one tool call with its result
N_STEPS = int((TRIGGER - PROMPT) // STEP) + 1


def window_frame(y):
    svg.rect(bx(0), y, bx(WINDOW) - bx(0), BH, fill="surface", stroke="rule", width=1, rx=3)


def prompt_block(y):
    svg.rect(bx(0), y, bx(PROMPT) - bx(0), BH, fill="muted", fill_opacity=0.35, stroke="muted", width=1)


def steps(y, t0, t1, label_first=None):
    """Tool-call history from token t0 to t1, one alternating block per step."""
    t, k = t0, 0
    while t < t1 - 1e-6:
        t_end = min(t + STEP, t1)
        col = "violet" if k % 2 == 0 else "amber"
        svg.rect(bx(t), y, bx(t_end) - bx(t), BH, fill=col, fill_opacity=0.22, stroke=col, width=0.8)
        t, k = t_end, k + 1


def deleted(y, t0, t1, label):
    svg.rect(bx(t0), y + 1, bx(t1) - bx(t0), BH - 2, fill="red", fill_opacity=0.06, stroke="red", width=1.2,
             dash="4 3", rx=2)
    svg.text((bx(t0) + bx(t1)) / 2, y + 16.5, label, size=12, anchor="middle", color="red")


# ------------------------------------------------------------------ header of the score column
SX = 584
svg.text(SX, 22, "BrowseComp 得分", size=12, weight="700")

# ------------------------------------------------------------------ row 0: the trajectory at the trigger
Y0 = 58
svg.text(16, 22, "一条搜索轨迹逐步填满 128K 窗口", size=13, weight="700")
svg.text(16, Y0 + 11, "触发时的上下文", size=12, weight="700")
window_frame(Y0)
prompt_block(Y0)
steps(Y0, PROMPT, TRIGGER)
xt = bx(TRIGGER)
svg.line(xt, Y0 - 16, xt, 330, color="red", width=1.3, dash="5 4")
svg.text(xt, Y0 - 20, "80%：0.8 × 131,072 = 104,858", size=11, anchor="middle", color="red")
svg.text(bx(WINDOW), Y0 + BH + 14, "131,072", size=11, anchor="end", color="muted")
svg.text(bx(0), Y0 + BH + 14, "0", size=11, color="muted")
svg.text(bx(PROMPT) + 2, Y0 + BH + 14, "系统提示 + 问题", size=11, color="muted")
svg.text(bx(PROMPT) + 96, Y0 + BH + 14, "紫、黄每块 = 一步工具调用与结果", size=11, color="muted")
svg.text(SX, Y0 + 11, "不做管理：51.4", size=12)
svg.text(SX, Y0 + 27, "约 20%+ 的题超过 128K", size=11, color="muted")

# ------------------------------------------------------------------ three strategies
svg.line(16, 112, W - 16, 112, color="rule", width=1)
svg.text(16, 132, "触发之后，三种策略把上下文变成：", size=12, color="muted")

HIST = TRIGGER - PROMPT
ROWS = [
    ("Summary", "把溢出的轨迹写成摘要，", "重新开始 rollout"),
    ("Discard-75%", "丢掉最早 75% 的", "工具调用历史"),
    ("Discard-all", "丢掉全部工具调用历史", "（类似 new context 工具）"),
]
for i, (name, d1, d2) in enumerate(ROWS):
    y = 152 + i * 58
    svg.text(16, y + 11, name, size=13, weight="700")
    svg.text(16, y + 27, d1, size=11, color="muted")
    svg.text(16, y + 41, d2, size=11, color="muted")
    window_frame(y)
    prompt_block(y)
    if name == "Summary":
        s1 = PROMPT + 6_000
        svg.rect(bx(PROMPT), y, bx(s1) - bx(PROMPT), BH, fill="orange", fill_opacity=0.25, stroke="orange", width=1.2)
        svg.text(bx(s1) + 6, y + 16.5, "摘要", size=12, color="orange")
        svg.arrow(bx(s1) + 36, y + 12, bx(s1) + 96, y + 12, color="muted", width=1.2, dash="3 3")
        svg.text(bx(s1) + 102, y + 16.5, "在摘要之后继续搜索", size=11, color="muted")
        svg.text(SX, y + 11, "最高 60.2", size=12)
        svg.text(SX, y + 27, "平均步数延长到 364", size=11, color="muted")
    elif name == "Discard-75%":
        cut = PROMPT + 0.75 * HIST
        deleted(y, PROMPT, cut, "最早 75% 删掉")
        steps(y, cut, TRIGGER)
        svg.text(bx(TRIGGER) + 6, y + 16.5, "后 25% 保留", size=11, color="muted")
        svg.text(SX, y + 11, "论文只给了曲线", size=12, color="muted")
        svg.text(SX, y + 27, "（图 6），没有单独数字", size=11, color="muted")
    else:
        deleted(y, PROMPT, TRIGGER, "工具调用历史全部删掉")
        svg.text(SX, y + 11, "67.6", size=12, weight="700", color="green")
        svg.text(SX, y + 27, "与并行扩展相当，步数少得多", size=11, color="muted")

svg.text(16, 348, "对照（Parallel-fewest-step）：独立采 N 条轨迹，选步数最少的一条。V3.2 表 2 的 BrowseComp 67.6* 就是 Discard-all 的成绩。",
         size=11, color="muted")
svg.text(16, 366, "只有窗口长度与 80% 触发点按比例；“系统提示 + 问题”与每一步的方块长度是示意。", size=11, color="muted")
svg.text(16, 384, "据 DeepSeek-V3.2 论文 §4.1、§4.4 绘制。", size=11, color="muted")

svg.save("l15-context-management.svg")
print("trigger at", fmt(TRIGGER), "tokens; steps drawn:", N_STEPS)
