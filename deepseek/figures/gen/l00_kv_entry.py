"""Figure 0.4: the bytes inside ONE cached KV entry, generation by generation (thread C, numerics).

Byte layouts from lesson-00-doc.md K3, K4, K8 (FlashMLA README; V4 paper section 2.3.4; V4.1 paper
section 2.4.4 and inference/model.py):
  V2/V3  MLA entry, BF16: 512-dim latent + 64-dim RoPE key, 2 bytes each               = 1,152 B
  V3.2   FP8 row: 512 FP8 + 4 FP32 scales (16 B) + 64 BF16 RoPE (128 B)                = 656 B, + indexer key 128 FP8 + 4 = 132 B
  V4     448 FP8 (NoPE) + 64 BF16 RoPE (128 B) + 7 E8M0 scales + 1 pad (8 B)          = 584 B, + indexer key FP4 64 + 4 = 68 B
  V4.1   512 E2M1 (256 B) + 32 E4M3 scales (one per 16 channels, 32 B)                = 288 B, + indexer key 68 B
Bits per dimension = 8 * main bytes / dims (576 for MLA, 512 for V4/V4.1).

    python deepseek/figures/gen/l00_kv_entry.py
"""

from __future__ import annotations

from l00_svg import SVG, text_width

# title, [(bytes, colour, label)], dims, note
ROWS = [
    ("V2 / V3：MLA 条目（BF16）",
     [(512 * 2, "orange", "潜向量 512 维 × BF16"), (64 * 2, "red", "RoPE")], 576,
     "每层 1 条、每条对应 1 个 token；576 维全用 BF16"),
    ("V3.2：MLA 条目（FP8）+ indexer 键",
     [(512, "orange", "潜向量 512 维 × FP8"), (16, "muted", ""), (128, "red", "RoPE"), (132, "violet", "索引键")], 576,
     "缩放 4 个 FP32 = 16 B；RoPE 64 维仍用 BF16；索引键 128 维 FP8 + 4 B 缩放"),
    ("V4：CSA/HCA 压缩条目 + indexer 键",
     [(448, "orange", "448 维 × FP8"), (128, "red", "RoPE"), (8, "muted", ""), (68, "violet", "")], 512,
     "CSA 一条管 4 个 token、HCA 一条管 128 个；缩放 8 B；索引键 FP4，只在 CSA 层"),
    ("V4.1：CSA2 条目（FP4）+ indexer 键",
     [(256, "orange", "512 维 × FP4"), (32, "muted", ""), (68, "violet", "")], 512,
     "40 层里只有 4 层存，一条管 2 或 1 个 token；每 16 维一个 E4M3 缩放 = 32 B"),
]

W, H = 760, 440
L, R = 20, 640
X_MAX = 1200.0
TOP = 44
PITCH = 80
BAR_H = 28


def xb(v):
    return L + (R - L) * v / X_MAX


def main():
    s = SVG(W, H, title="一个 KV 条目里的字节")
    axis_y = TOP + PITCH * len(ROWS) + 6
    for v in range(0, 1201, 200):
        x = xb(v)
        s.line(x, TOP - 8, x, axis_y - 12, color="grid", width=1)
        s.text(x, axis_y + 2, "%d B" % v, size=12, anchor="middle", color="muted")
    for i, (title, segs, dims, note) in enumerate(ROWS):
        y = TOP + i * PITCH
        s.text(L, y, title, size=13, weight=700)
        x = xb(0)
        main_bytes = 0
        for b, col, lab in segs:
            w = xb(b) - xb(0)
            s.rect(x, y + 8, w, BAR_H, fill=col, fill_opacity=0.85 if col != "muted" else 0.6, stroke="surface",
                   width=1)
            if lab and w > text_width(lab, 12, 600) + 10:
                s.text(x + w / 2, y + 8 + BAR_H / 2 + 4.5, lab, size=12, anchor="middle", color="surface",
                       weight=600)
            if col != "violet":
                main_bytes += b
            x += w
        idx = sum(b for b, col, _ in segs if col == "violet")
        lab = ("%s B" % f"{main_bytes:,}") if not idx else "%d + %d = %d B" % (main_bytes, idx, main_bytes + idx)
        s.text(x + 8, y + 8 + BAR_H / 2 + 4.5, lab, size=12.5, weight=700)
        bits = 8 * main_bytes / dims
        s.text(R + 114, y + 8 + BAR_H / 2 + 4.5, ("%.1f 位/维" % bits) if bits % 1 else ("%d 位/维" % bits), size=13,
               anchor="end", color="orange", weight=700)
        s.text(L, y + 8 + BAR_H + 17, note, size=12, color="muted")
    ly = H - 16
    items = [("orange", "内容维（低精度数据）"), ("red", "RoPE 维（BF16）"), ("muted", "缩放因子"),
             ("violet", "indexer 键（索引键）")]
    lx = L
    for col, lab in items:
        s.rect(lx, ly - 11, 14, 14, fill=col, fill_opacity=0.85 if col != "muted" else 0.6, rx=2)
        s.text(lx + 20, ly + 1, lab, size=12)
        lx += 34 + text_width(lab, 12)
    s.text(R + 114, ly + 1, "“位/维”只算主 KV", size=11.5, anchor="end", color="muted")
    s.save("l00-kv-entry.svg")


if __name__ == "__main__":
    main()
