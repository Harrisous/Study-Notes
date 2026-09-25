"""Figure 16.14: V4's multi-pass mHC vs V4.1-Flash's Single-Pass mHC (mechanism figure).

After DeepSeek-V4.1-Flash (arXiv 2609.19969 v1) section 2.4.1 and its inference code (research notes C41, 2.8):
multi-pass (V4, per sublayer l):
  K1  X_l = B_{l-1} X_{l-1} + C_{l-1} Y_{l-1}      read (n+1)d
  K2  (A_l, B_l, C_l) = H(X_l)                     read nd   -- needs a reduction over all of X_l
  K3  X^_l = A_l X_l                               read nd
  three kernels write (n+1)d in total; plus the sublayer's pre-norm -> (4n+4)d = 20d for n = 4
Single-Pass: X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l): the input mix uses the coefficients predicted one sublayer
earlier, so one streaming pass can update the residual, apply the mix and accumulate the next coefficients.
Lower bound (2n+2)d = 10d; read (n+1)d + write (n+1)d is the author's reading of that bound.
Writes deepseek/figures/l16-single-pass.svg
"""

from l16_svg import SVG

W, H = 760, 470
svg = SVG(W, H, "V4 的多趟 mHC 与 V4.1-Flash 的 Single-Pass mHC")

# ---------------- (a) multi-pass
svg.text(14, 26, "(a) V4：三个串行的核（每个子层）", size=14, weight="700")
Y = 44
boxes = [
    (14, 200, "blue", ["核 1：更新残差流", "*X*_{l} = B_{l−1}*X*_{l−1} + C_{l−1}*Y*_{l−1}"], "读 (n+1)d"),
    (262, 200, "violet", ["核 2：算系数", "(A_{l}, B_{l}, C_{l}) = 𝓗(*X*_{l})"], "读 nd"),
    (510, 236, "blue", ["核 3：输入混合", "*X̂*_{l} = A_{l}*X*_{l}"], "读 nd"),
]
for x, w, col, lines, io in boxes:
    svg.box(x, Y, w, 58, [(lines[0], {"weight": "600"}), lines[1]], color=col, fill_opacity=0.10, size=13)
    svg.text(x + w / 2, Y + 78, io, size=12, anchor="middle", color="amber", weight="600")
svg.arrow(214, Y + 29, 260, Y + 29)
svg.arrow(462, Y + 29, 508, Y + 29, color="red")
svg.text(485, Y + 108, "必须等待：A_{l} 要先对整条 *X*_{l}", size=12, anchor="middle", color="red")
svg.text(485, Y + 126, "（16384 个数）做完归约才能得到", size=12, anchor="middle", color="red")
svg.text(14, Y + 108, "三个核共写 (n+1)d；再加子层自己的 pre-norm 一读一写", size=12)
svg.text(14, Y + 126, "合计 (4n+4)d = 20d：残差流被从显存读了三遍", size=12, weight="600")

# ---------------- (b) single pass
svg.line(14, 196, 746, 196, color="rule", width=1)
svg.text(14, 222, "(b) V4.1-Flash：Single-Pass，一次遍历（按 tile 流式处理）", size=14, weight="700")
YB = 240
svg.rect(14, YB, 732, 126, fill="green", fill_opacity=0.06, stroke="green", width=1.4, rx=8)
svg.text(28, YB + 22, "一个核：对 *X*_{l−1} 的每一小块（tile）依次做三件事，中间结果留在片上", size=13, weight="600")
TX = [30, 270, 510]
steps = [
    ("① 更新残差流", "*X*_{l} = B_{l−1}*X*_{l−1} + C_{l−1}*Y*_{l−1}", "blue"),
    ("② 用上一层给的 A_{l−1} 混合", "*X̂*_{l} = A_{l−1}*X*_{l}", "blue"),
    ("③ 累加下一层的系数", "*X*_{l}*W* 的部分和 → (A_{l}, B_{l}, C_{l})", "violet"),
]
for x, (t1, t2, col) in zip(TX, steps):
    svg.box(x, YB + 36, 216, 56, [(t1, {"weight": "600"}), t2], color=col, fill_opacity=0.10, size=12)
svg.arrow(246, YB + 64, 268, YB + 64)
svg.arrow(486, YB + 64, 508, YB + 64)
svg.text(380, YB + 114, "每个 tile 只读一次、写一次：读 (n+1)d（*X*_{l−1} 与 *Y*_{l−1}），写 (n+1)d（*X*_{l} 与子层输入）",
         size=12, anchor="middle", color="amber")

svg.text(14, 396, "代价：子层 ~l~ 的输入混合用的是子层 ~l~−1 预测的 A_{l−1}，“推迟一格”。论文称实测性能损失可以忽略（未给数字）。",
         size=12)
svg.text(14, 416, "合计 (2n+2)d = 10d = 访存下界，是多趟实现的一半；第一个子层没有“上一层”，代码用 one-hot 的 *e*_{0}（只读第 0 条流）。",
         size=12)
svg.text(14, 436, "预训练仍用多核实现；部署时用单核的 Mega-mHC。“读、写各 (n+1)d”据 V4.1 论文 §2.4.1。", size=12,
         color="muted")

svg.save("l16-single-pass.svg")
