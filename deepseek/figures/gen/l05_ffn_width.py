"""Figure 5.9: FFN width ratio d_ff / d across models (research notes R13, R14, and GPT-2 config).

GPT-2 small: 3072 / 768 (HF config: inner dim defaults to 4 x hidden); PaLM: d_ff = 4 d (Table 1 note);
LLaMA 7B/13B/33B/65B: 11008/4096, 13824/5120, 17920/6656, 22016/8192 (code rule, huggyllama configs);
DeepSeek LLM 7B / 67B: 11008/4096 = 22016/8192; DeepSeek-V2 dense layer 12288/5120; V3 dense 18432/7168;
MoE experts: V2 1536/5120, V3 2048/7168. Reference line 8/3.
Writes deepseek/figures/l05-ffn-width.svg
"""

from l05_svg import SVG

rows = [
    ("GPT-2 small（GELU）", 3072, 768, "muted"),
    ("PaLM（SwiGLU，仍取 4d）", 4, 1, "blue"),
    ("LLaMA-7B", 11008, 4096, "blue"),
    ("LLaMA-13B", 13824, 5120, "blue"),
    ("LLaMA-33B", 17920, 6656, "blue"),
    ("LLaMA-65B", 22016, 8192, "blue"),
    ("DeepSeek LLM 7B / 67B", 11008, 4096, "blue"),
    ("DeepSeek-V2 稠密层（第 1 层）", 12288, 5120, "blue"),
    ("DeepSeek-V3 稠密层（前 3 层）", 18432, 7168, "blue"),
    ("DeepSeek-V2 每个 MoE 专家", 1536, 5120, "violet"),
    ("DeepSeek-V3 每个 MoE 专家", 2048, 7168, "violet"),
]
W, H = 760, 470
X0, X1 = 250, 700
Y0 = 44
RH = 30
VMAX = 4.4


def X(v):
    return X0 + v / VMAX * (X1 - X0)


svg = SVG(W, H, "各模型前馈层的宽度比 d_ff / d")
svg.text(X0, 24, "前馈层隐藏维与模型宽度之比 d_{ff} / d", size=13, weight="700")
for v in (0, 1, 2, 3, 4):
    svg.line(X(v), Y0 - 6, X(v), Y0 + len(rows) * RH, color="grid", width=1)
    svg.text(X(v), Y0 + len(rows) * RH + 16, str(v), size=11, anchor="middle", color="muted")
for n, (lab, a, b, col) in enumerate(rows):
    y = Y0 + n * RH
    r = a / b
    svg.text(X0 - 10, y + 18, lab, size=12.5, anchor="end")
    svg.rect(X0, y + 6, X(r) - X0, RH - 12, fill=col, fill_opacity=0.55 if col != "muted" else 0.45)
    txt = ("%.4f" % r).rstrip("0").rstrip(".")
    # a bar that stops just short of the 8/3 line would have its label cut by the red dashes:
    # start such labels on the far side of the line instead
    lx = X(8 / 3) + 6 if 2.0 < r < 8 / 3 else X(r) + 6
    svg.text(lx, y + 19, txt + ("" if a == 4 else "（%s / %s）" % ("{:,}".format(a), "{:,}".format(b))),
             size=11.5, color="ink")
ref = 8 / 3
svg.line(X(ref), Y0 - 10, X(ref), Y0 + len(rows) * RH + 2, color="red", width=1.4, dash="5 3")
svg.text(X(ref), Y0 + len(rows) * RH + 34, "红色虚线 8/3 ≈ 2.667：与 4d 的 GELU 前馈层参数相等", size=12, anchor="middle", color="red")
svg.text(20, H - 30, "灰：GELU 前馈层；蓝：SwiGLU 稠密前馈层；", size=12, color="muted")
svg.text(20, H - 12, "紫：MoE 里的单个专家（第 7、10、11 课），专家本来就设计得很窄，不适用 8/3 规则。", size=12, color="muted")
svg.save("l05-ffn-width.svg")
