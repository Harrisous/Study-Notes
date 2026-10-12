"""Numbers shared by the lesson-20 figure scripts (the retrospective).

Lesson 20 introduces no new facts: every constant below is copied from a finished lesson
(deepseek/src/lesson-NN-*.md) or from deepseek/reference-docs/lesson-00-doc.md, and the comment
names the page it comes from. Every derived number is recomputed here and checked with an assert
against the value the lesson text quotes, so a figure cannot drift from the prose.

Run it directly to print the tables used in the lesson text:  python l20_num.py
"""

from __future__ import annotations

import math

# =================================================================== thread A: KV bytes per token
# (name, layers that hold global KV, global-KV entries per token, numbers per token, bytes per token)
# 67B: 95 layers x GQA-8 x d_h 128 x (K,V), BF16 (lesson 4 p5, lesson 9 p8)
# V2 / V3: 60 / 61 layers x (512 + 64), BF16 (lesson 9 p8, lesson 11 p8)
# V3.2: 61 layers x (656 B MLA row + 132 B indexer key); 576 + 128 numbers (lesson 14 p7)
# V4-Flash: 21 CSA (m=4, 584 B entry + 68 B indexer key) + 20 HCA (m=128, 584 B) (lesson 18 p7)
# V4.1-Flash: 4 Full layers, m = 2, 2, 2, 1, 288 B + 68 B (lesson 19 p3, p5, p8)
KV = [
    ("DeepSeek LLM 67B", 95, 95, 95 * 2048, 2 * 95 * 8 * 128 * 2),
    ("DeepSeek-V2", 60, 60, 60 * 576, 60 * 576 * 2),
    ("DeepSeek-V3", 61, 61, 61 * 576, 61 * 576 * 2),
    ("DeepSeek-V3.2", 61, 61, 61 * (576 + 128), 61 * (656 + 132)),
    ("DeepSeek-V4-Flash", 41, 21 / 4 + 20 / 128, 21 / 4 * (512 + 128) + 20 / 128 * 512,
     (21 / 4 + 20 / 128) * 584 + 21 / 4 * 68),
    ("DeepSeek-V4.1-Flash", 4, 3 / 2 + 1, 2.5 * (512 + 128), 2.5 * (288 + 68)),
]
BYTES = {k[0]: k[4] for k in KV}
assert BYTES["DeepSeek LLM 67B"] == 389_120 and BYTES["DeepSeek-V2"] == 69_120
assert BYTES["DeepSeek-V3"] == 70_272 and BYTES["DeepSeek-V3.2"] == 48_068
assert BYTES["DeepSeek-V4-Flash"] == 3514.25 and BYTES["DeepSeek-V4.1-Flash"] == 890
# V4.1 would hold 38 copies if its 38 CSA2 layers did not share (lesson 19 exercise 6)
V41_UNSHARED_COPIES = 38

DIMS = ["层数", "跨层共享", "序列维", "条目维", "精度"]


def factors(a, b):
    """Split bytes(a)/bytes(b) into the five multiplicative factors of (20.2)."""
    _, ca, ea, na, ba = a
    _, cb, eb, nb, bb = b
    share = 1.0
    depth = ca / cb
    if b[0] == "DeepSeek-V4.1-Flash":                    # 41 -> 38 is depth, 38 -> 4 is sharing
        depth, share = ca / V41_UNSHARED_COPIES, V41_UNSHARED_COPIES / cb
    seq = (ea / ca) / (eb / cb)                            # global-KV entries per token per copy
    dim = (na / ea) / (nb / eb)                            # numbers per entry
    prec = (ba / na) / (bb / nb)                           # bytes per number (scales amortised)
    return {"层数": depth, "跨层共享": share, "序列维": seq, "条目维": dim, "精度": prec}


STEPS = []
for a, b in zip(KV, KV[1:]):
    f = factors(a, b)
    prod = math.prod(f.values())
    assert abs(prod - a[4] / b[4]) < 1e-9 * prod
    STEPS.append((a[0], b[0], f, a[4] / b[4]))

TOTAL = {d: math.prod(s[2][d] for s in STEPS) for d in DIMS}
TOTAL_RATIO = BYTES["DeepSeek LLM 67B"] / BYTES["DeepSeek-V4.1-Flash"]
assert abs(math.prod(TOTAL.values()) - TOTAL_RATIO) < 1e-6 and round(TOTAL_RATIO, 1) == 437.2
SHARE = {d: math.log(TOTAL[d]) / math.log(TOTAL_RATIO) for d in DIMS}
assert round(TOTAL["层数"], 3) == 2.5 and round(TOTAL["跨层共享"], 2) == 9.5
assert round(TOTAL["序列维"], 3) == 1.6 and round(TOTAL["条目维"], 3) == 3.2
assert round(TOTAL["精度"], 3) == 3.596
# lesson 19 (19.21) splits the last step as structure 2.156 x precision 1.831 using a 652 B V4 entry;
# with the averaged V4-Flash entry (650.03 B, HCA entries carry no indexer key) it is 2.163 x 1.826
LAST = STEPS[-1][2]
assert round(LAST["层数"] * LAST["跨层共享"] * LAST["序列维"], 3) == 2.163
assert round(LAST["条目维"] * LAST["精度"], 3) == 1.826

# =================================================================== thread B: total vs activated
# (name, total B, activated B, experts note) - lesson 7 p8, 10 p1/p8, 11 p8, 18 p7, 19 p1
MOE = [
    ("DeepSeek LLM 67B", 67.4, 67.4, "稠密"),
    ("DeepSeekMoE 16B", 16.4, 2.8, "2 共享 + 64 选 6"),
    ("DeepSeek-V2-Lite", 15.7, 2.4, "2 + 64 选 6"),
    ("DeepSeek-V2", 236, 21, "2 + 160 选 6"),
    ("DeepSeek-V3", 671, 37, "1 + 256 选 8"),
    ("DeepSeek-V4-Flash", 284, 13, "1 + 256 选 6"),
    ("DeepSeek-V4-Pro", 1600, 49, "1 + 384 选 6"),
    ("DeepSeek-V4.1-Flash", 552, 16, "1 + 384 选 6"),
]
ROUTED = {"DeepSeekMoE 16B": (6, 64), "DeepSeek-V2-Lite": (6, 64), "DeepSeek-V2": (6, 160),
          "DeepSeek-V3": (8, 256), "DeepSeek-V4-Flash": (6, 256), "DeepSeek-V4-Pro": (6, 384),
          "DeepSeek-V4.1-Flash": (6, 384)}
V41_PREFILL, V41_ENGRAM = 8, 196
ACT = {m[0]: m[2] / m[1] for m in MOE}
assert round(ACT["DeepSeekMoE 16B"] * 100, 1) == 17.1 and round(ACT["DeepSeek-V3"] * 100, 1) == 5.5
assert round(ACT["DeepSeek-V4.1-Flash"] * 100, 1) == 2.9 and round(ACT["DeepSeek-V4-Pro"] * 100, 1) == 3.1

# =================================================================== thread C: bits per stored number
# (label, bits per number incl. amortised scales, lesson/page, note)
BITS = [
    ("BF16（混合精度训练的计算格式）", 16.0, "第 2 课 §6"),
    ("V3 激活 FP8：E4M3 + 每 1×128 一个 FP32", 8 + 32 / 128, "第 12 课 §3"),
    ("V3 权重 FP8：E4M3 + 每 128×128 一个 FP32", 8 + 32 / (128 * 128), "第 12 课 §3"),
    ("V3.2 MLA 缓存行：656 B / 576 维", 656 * 8 / 576, "第 14 课 §7"),
    ("V4 KV 条目：584 B / 512 维", 584 * 8 / 512, "第 18 课 §7"),
    ("V2 部署时的 KV（平均）", 6.0, "第 9 课 §8"),
    ("V4.1 SWA 条目：528 B / 512 维", 528 * 8 / 512, "第 19 课 §5"),
    ("V4.1 主 KV：E2M1 + 每 16 个一个 E4M3", 288 * 8 / 512, "第 19 课 §5"),
    ("V4.1 索引器键：MXFP4（每 32 个一个 E8M0）", 68 * 8 / 128, "第 19 课 §5"),
]
assert [round(b[1], 3) for b in BITS] == [16, 8.25, 8.002, 9.111, 9.125, 6, 8.25, 4.5, 4.25]

# GPU hours per 10^12 training tokens (H800) and MFU - lesson 10 p5, lesson 12 p1
GPUH_PER_T = [("DeepSeek LLM 67B", 300.6, 67.4, 0.38), ("DeepSeek-V2", 172.8, 21, 0.20),
              ("DeepSeek-V3", 180.0, 37, 0.3894)]
# V3 cross-check (lesson 12 p1): 62,914,560 tokens per 19.926 s step on 2048 GPUs
V3_TOKENS_PER_DAY = 15360 * 4096 * 86400 / 19.926
V3_GPUH_PER_T = 1e12 / V3_TOKENS_PER_DAY * 2048 * 24
assert round(V3_TOKENS_PER_DAY / 1e9, 1) == 272.8 and round(V3_GPUH_PER_T) == 180_176

# =================================================================== thread D: post-training
# model-state memory, 7B each, (3.26): 16 B per trained parameter, 2 B per frozen one (lesson 3 p6)
N7 = 7


def mem(trained, frozen):
    return 16 * N7 * trained + 2 * N7 * frozen


MEM = [("SFT", 1, 0, 0), ("DPO", 1, 1, 0), ("GRPO", 1, 2, 0), ("PPO", 1, 2, 1)]   # (name, policy, frozen, value)
MEM_GB = {n: mem(p + v, f) for n, p, f, v in MEM}
assert MEM_GB == {"SFT": 112, "DPO": 126, "GRPO": 140, "PPO": 252}
# post-training GPU hours relative to V3 pre-training (2,664K H800 hours) - lesson 12 p1, 13 p8, 15 p1
PRE_V3 = 2664
POST = [("V3 的后训练", 5, "第 12 课 §1"), ("R1 公开账单（R1-Zero + SFT 数据 + R1）", 101 + 5 + 41, "第 13 课 §8"),
        ("V3.2（“超过预训练成本的 10%”，下限）", 0.10 * PRE_V3, "第 15 课 §1")]
assert round(5 / PRE_V3 * 100, 3) == 0.188 and round(147 / PRE_V3 * 100, 2) == 5.52

# =================================================================== thread E: composite gain over 60 sub-layers
# lesson 16 p1 (scalar), p3 (HC 27B, digitised 2825), p7 (mHC 27B, digitised 1.65), p4 (bound n = 4)
GAIN = [
    ("每层 ×0.95 的标量捷径", 0.95 ** 60, "计算"),
    ("恒等捷径（每层 ×1）", 1.0, "计算"),
    ("mHC 实测（27B，反向，读图）", 1.65, "第 16 课 §7"),
    ("行和为 1 的非负矩阵之积（反向上界 n）", 4.0, "第 16 课 §4"),
    ("每层 ×1.05 的标量捷径", 1.05 ** 60, "计算"),
    ("HC 实测（27B，反向，读图）", 2825.0, "第 16 课 §3"),
    ("每层 ×1.1428 的标量捷径", (3000 ** (1 / 60)) ** 60, "计算"),
]
assert round(0.95 ** 60, 3) == 0.046 and round(1.05 ** 60, 1) == 18.7 and round(3000 ** (1 / 60), 4) == 1.1428
# residual-stream memory traffic per sub-layer, multiples of d (lesson 16 p7 table)
TRAFFIC = [("普通残差（Pre-LN）", 3), ("HC，未融合", 34), ("mHC，融合后", 18), ("V4 的多趟实现", 20),
           ("V4.1 Single-Pass", 10)]

# =================================================================== thread F: tokens per parameter, 6ND
# (name, total params B, activated B, training tokens T) - lesson-00-doc N20; GPT-3 from lesson 2 p8
TOK = [
    ("GPT-3 175B", 175, 175, 0.3),
    ("DeepSeek LLM 67B", 67, 67, 2.0),
    ("DeepSeekMoE 16B", 16.4, 2.8, 2.0),
    ("DeepSeek-V2", 236, 21, 8.1),
    ("DeepSeek-V3", 671, 37, 14.8),
    ("DeepSeek-V4-Flash", 284, 13, 32),
    ("DeepSeek-V4-Pro", 1600, 49, 33),
    ("DeepSeek-V4.1-Flash", 552, 16, 45),
]
DN = {t[0]: (t[3] * 1e3 / t[1], t[3] * 1e3 / t[2], 6 * t[2] * 1e9 * t[3] * 1e12) for t in TOK}
assert round(DN["DeepSeek-V3"][1]) == 400 and round(DN["DeepSeek-V4.1-Flash"][1], 1) == 2812.5
assert round(DN["DeepSeek-V3"][2] / 1e24, 2) == 3.29 and round(DN["GPT-3 175B"][0], 2) == 1.71
assert round(DN["DeepSeek-V4-Pro"][0], 1) == 20.6 and round(DN["DeepSeek LLM 67B"][0], 1) == 29.9

# =================================================================== release dates (lesson-00-doc 3.2)
DATES = {
    "DeepSeek LLM": (2023, 11, 29), "DeepSeekMoE": (2024, 1, 11), "DeepSeekMath": (2024, 2, 5),
    "V2": (2024, 5, 6), "V3": (2024, 12, 26), "R1": (2025, 1, 20), "V3.1": (2025, 8, 21),
    "V3.2-Exp": (2025, 9, 29), "V3.2": (2025, 12, 1), "mHC": (2025, 12, 31), "V4": (2026, 4, 24),
    "V4.1": (2026, 9, 10),
}

if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("thread A: per-step factors")
    for a, b, f, r in STEPS:
        print("  %-20s -> %-20s x%.4f  " % (a, b, r) + "  ".join("%s %.4f" % (d, f[d]) for d in DIMS))
    print("  total x%.2f  " % TOTAL_RATIO + "  ".join("%s %.4f (%.1f%%)" % (d, TOTAL[d], 100 * SHARE[d]) for d in DIMS))
    for k in KV:
        print("  %-20s copies %d entries/token %.5f numbers/entry %.2f bytes/number %.5f bytes %.2f"
              % (k[0], k[1], k[2], k[3] / k[2], k[4] / k[3], k[4]))
    print("thread B")
    for m in MOE:
        print("  %-20s %.1f%%" % (m[0], 100 * ACT[m[0]]), ROUTED.get(m[0]))
    print("thread C bits", [(b[0], round(b[1], 3)) for b in BITS])
    for n, h, a, mfu in GPUH_PER_T:
        print("  %-18s %.1fK GPUh/T  %.2fK per B act" % (n, h, h / a))
    print("thread D", MEM_GB, [(p[0], round(p[1] / PRE_V3 * 100, 3)) for p in POST])
    print("thread E", [(g[0], round(g[1], 3)) for g in GAIN])
    print("thread F")
    for t in TOK:
        print("  %-20s D/N %.1f  D/Nact %.1f  6ND %.3g" % ((t[0],) + DN[t[0]]))
