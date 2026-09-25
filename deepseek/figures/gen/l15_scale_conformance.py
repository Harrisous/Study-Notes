"""Figure 15.5: are the FP8 weight scales in the released checkpoints powers of two?

Data: lesson-15-doc.md N32 and 补充核实 S7 (HTTP-Range scan of the safetensors `weight_scale_inv` tensors;
a tensor counts as conforming when every np.frexp mantissa is exactly 0.5). Fraction of scales that are
powers of two, per module and release. Layer 0 is a dense layer (first_k_dense_replace = 3), so its
mlp.down_proj is the dense FFN; experts / shared experts / indexer were read at layers 30 and 60.
S7 re-read every cell below for all four releases (the earlier N32 scan had left V3 and Terminus cells
unmeasured); every tensor read at the listed layers gave the same fraction, so one value per cell.
None = not measured; "n/a" = the module does not exist in that release.
Writes deepseek/figures/l15-scale-conformance.svg
"""

from l15_svg import SVG

VERS = ["V3", "V3.1", "V3.1-Terminus", "V3.2"]
ROWS = [  # module, layers read, values per release
    ("self_attn.q_a_proj", "第 0、30、60 层", [0.0, 1.0, 1.0, 1.0]),
    ("self_attn.kv_b_proj", "第 0、30、60 层", [0.0, 1.0, 1.0, 1.0]),
    ("self_attn.o_proj", "第 0、30、60 层", [0.0, 0.0, 0.0, 0.0]),
    ("mlp.down_proj（稠密层）", "第 0 层", [0.0, 1.0, 1.0, 1.0]),
    ("mlp.experts.7.down_proj", "第 30、60 层", [0.0, 1.0, 1.0, 1.0]),
    ("mlp.shared_experts.up_proj", "第 30、60 层", [0.0, 1.0, 1.0, 1.0]),
    ("indexer.wq_b", "第 30、60 层", ["n/a", "n/a", "n/a", 1.0]),
    ("indexer.wk", "第 30、60 层", ["n/a", "n/a", "n/a", 1.0]),
]

W, H = 760, 380
svg = SVG(W, H, "各版本检查点里 FP8 权重的缩放因子是否全是 2 的幂")

CX0, CW, RH, TOP = 300, 112, 30, 58
svg.text(16, 22, "缩放因子中 2 的整数次幂所占比例（1.000 = 全部是；0.000 = 一个也不是）", size=13, weight="700")
svg.text(16, TOP - 12, "模块（读取的层）", size=12, color="muted")
for j, v in enumerate(VERS):
    svg.text(CX0 + j * CW + CW / 2, TOP - 12, v, size=13, anchor="middle", weight="700")

for i, (mod, layers, vals) in enumerate(ROWS):
    y = TOP + i * RH
    if i % 2 == 0:
        svg.rect(10, y, CX0 + 4 * CW - 6, RH, fill="grid", fill_opacity=0.5)
    svg.text(16, y + 19, mod, size=12, color="ink")
    svg.text(CX0 - 10, y + 19, layers, size=11, anchor="end", color="muted")
    for j, v in enumerate(vals):
        x = CX0 + j * CW + 6
        if v is None:
            svg.rect(x, y + 4, CW - 12, RH - 8, fill="muted", fill_opacity=0.06, stroke="rule", width=1, rx=4,
                     dash="3 3")
            svg.text(x + (CW - 12) / 2, y + 19, "未测", size=11, anchor="middle", color="muted")
        elif v == "n/a":
            svg.text(x + (CW - 12) / 2, y + 19, "无此模块", size=11, anchor="middle", color="muted")
        else:
            col = "green" if v == 1.0 else "red"
            svg.rect(x, y + 4, CW - 12, RH - 8, fill=col, fill_opacity=0.16, stroke=col, width=1.2, rx=4)
            svg.text(x + (CW - 12) / 2, y + 19, "%.3f" % v, size=12, anchor="middle", color=col, weight="700")

y = TOP + len(ROWS) * RH + 22
svg.text(16, y, "所有版本的缩放因子张量都以 FP32（F32）存储；打包成 1 字节 UE8M0 是 DeepGEMM 在新硬件（SM100）上的做法。", size=12, color="muted")
svg.text(16, y + 18, "V3.1 起，除 o_proj 外抽查到的权重缩放全是 2 的幂；Terminus 的模型卡承认 o_proj 不符合，V3.2 仍未改。", size=12, color="muted")
svg.text(16, y + 36, "数据：用 HTTP Range 读取 safetensors 头与缩放张量（2026-09-24 读取，09-25 补全），只抽查了图中列出的层。", size=12, color="muted")

svg.save("l15-scale-conformance.svg")
