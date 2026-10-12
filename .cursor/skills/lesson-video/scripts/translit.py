"""Latin words and letters in narration, rewritten the way a Mandarin speaker reads them (W → 达布溜, GQA → 吉丘诶).

Voices fine-tuned only on Chinese speech (the GPT-SoVITS voices, e.g. 曼波) garble English: in their "zh" mode an
English word goes through the English phoneme frontend, which the fine-tuned model never learned. Spelling the word
in Chinese characters keeps it on the Chinese side, where the voice is clear. Subtitles are untouched: this only
changes what is sent to the engine. Kokoro reads English well and does not use this.

    to_spoken("合头靠 W O，GPT 二 small 有 12 个头")  ->  "合头靠达布溜欧，吉批踢二斯莫尔有 12 个头"

Unknown words are spelled letter by letter and reported by `unknown()`, so they can be added to WORDS.
"""
from __future__ import annotations

import re

# English letter names as commonly said in Mandarin
LETTERS = {
    "a": "诶", "b": "比", "c": "西", "d": "迪", "e": "伊", "f": "艾夫", "g": "吉", "h": "艾尺", "i": "艾", "j": "杰",
    "k": "开", "l": "艾偶", "m": "艾姆", "n": "恩", "o": "欧", "p": "批", "q": "丘", "r": "阿尔", "s": "艾斯", "t": "踢",
    "u": "优", "v": "维", "w": "达布溜", "x": "艾克斯", "y": "歪", "z": "贼",
}
DIGITS = "零一二三四五六七八九"

# whole words (lower case) read as words, not spelled
WORDS = {
    # lesson 1 §4
    "softmax": "索夫特麦克斯", "token": "托肯", "tokens": "托肯", "the": "泽", "cat": "凯特", "sat": "萨特", "on": "昂",
    "small": "斯莫尔", "cache": "卡什", "bleu": "布鲁", "deepseek": "迪普西克", "transformer": "穿斯佛么",
    "flash": "弗莱什", "delta": "德尔塔", "we": "维", "suspect": "瑟斯拜克特", "layernorm": "雷尔诺姆",
    # the course map (lesson 0)
    "pro": "普肉", "lite": "莱特", "exp": "伊克斯皮", "speciale": "斯佩夏莱", "terminus": "特米纳斯", "zero": "贼肉",
    "coder": "扣得尔", "math": "麦斯", "prover": "普肉沃", "chat": "柴特", "base": "贝斯", "instruct": "因斯特拉克特",
    "chatgpt": "柴特吉批踢", "instructgpt": "因斯特拉克特吉批踢", "llama": "拉玛", "gshard": "吉沙德", "switch": "斯维奇",
    "yarn": "亚恩", "rope": "罗普", "muon": "缪昂", "adamw": "阿当达布溜", "adam": "阿当", "chinchilla": "钦奇拉",
    "kaplan": "卡普兰", "engram": "恩格拉姆", "dualpipe": "杜尔派普", "deepep": "迪普伊批", "deepgemm": "迪普杰姆",
    "flashmla": "弗莱什艾姆艾偶诶", "janus": "杰纳斯", "hyper": "海珀", "connections": "康耐克申斯",
    "scaling": "斯凯林", "law": "劳", "laws": "劳斯", "attention": "阿腾申", "norm": "诺姆", "rmsnorm": "阿尔艾姆艾斯诺姆",
    "swiglu": "斯维格鲁", "gelu": "吉鲁", "relu": "瑞鲁", "lightning": "莱特宁", "indexer": "因代克瑟", "cuda": "库达",
    "nvidia": "英伟达", "hugging": "哈金", "face": "费斯", "github": "吉特哈布", "arxiv": "阿凯夫", "nature": "内彻",
    "bytedance": "字节跳动", "google": "谷歌", "openai": "欧喷诶艾", "meta": "梅塔", "anthropic": "安斯若匹克",
    "sinkhorn": "辛克霍恩", "knopp": "克诺普", "birkhoff": "伯克霍夫", "aha": "啊哈", "moment": "莫门特",
    "agent": "埃金特", "agents": "埃金茨", "thinking": "辛金", "mode": "莫德", "fp": "艾夫批",
    # units and jargon said as words
    "bit": "比特", "bits": "比特", "byte": "拜特", "bytes": "拜特", "flops": "弗洛普斯", "top": "托普",
    "sigmoid": "西格莫伊德", "all": "奥", "to": "图", "deepseekmoe": "迪普西克艾姆欧伊", "moe": "艾姆欧伊",
    "max": "麦克斯", "isca": "艾斯卡", "fire": "法尔", "flyer": "弗莱尔", "dsec": "迪赛克", "pcie": "批西艾伊",
    "deepseekmath": "迪普西克麦斯", "dspark": "迪斯帕克", "maj": "马吉", "at": "艾特", "pass": "帕斯",
    "pre": "普瑞", "post": "波斯特", "res": "瑞斯", "warm": "沃姆", "up": "阿普", "gram": "格拉姆",
    "batch": "拜奇", "isoflop": "艾索弗洛普", "markdown": "马克当", "python": "派森", "grep": "格瑞普", "html": "艾尺踢艾姆艾偶",
    "full": "富尔", "openai": "欧喷诶艾", "pdf": "批迪艾夫",
}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.\-][A-Za-z0-9]+)*")
_seen_unknown: set[str] = set()


def _num(run: str) -> str:
    """Digits inside a token: short runs as a number (16 → 十六), long ones digit by digit."""
    if 3 <= len(run) <= 4 and run.endswith("00") and run[0] != "0":   # H800 → 艾尺八百, A100 → 诶一百
        n = int(run)
        k, h = divmod(n // 100, 10)
        return (DIGITS[k] + "千" if k else "") + (DIGITS[h] + "百" if h else "")
    if len(run) <= 2:
        n = int(run)
        if n < 10:
            return DIGITS[n]
        tens, ones = divmod(n, 10)
        return ("" if tens == 1 else DIGITS[tens]) + "十" + (DIGITS[ones] if ones else "")
    return "".join(DIGITS[int(c)] for c in run)


def _spell(tok: str) -> str:
    out = []
    for piece in re.findall(r"[A-Za-z]+|\d+|[.\-]", tok):
        if piece.isdigit():
            out.append(_num(piece))
        elif piece == ".":
            out.append("点")
        elif piece == "-":
            continue
        elif piece.lower() in WORDS:
            out.append(WORDS[piece.lower()])
        else:
            out.append("".join(LETTERS[c] for c in piece.lower()))
            if len(piece) > 3 and not piece.isupper():
                _seen_unknown.add(piece)
    return "".join(out)


def _word(tok: str) -> str:
    low = tok.lower()
    if low in WORDS:
        return WORDS[low]
    # a word glued to a version or a suffix: DeepSeek-V4, V3.2-Exp, GPT-2
    return _spell(tok)


def to_spoken(text: str) -> str:
    """Replace every Latin token; spaces around a replaced token are dropped so the Chinese reads continuously."""
    out = _TOKEN.sub(lambda m: _word(m.group(0)), text)
    return re.sub(r"(?<=[一-鿿]) +(?=[一-鿿])", "", out)


def unknown() -> list[str]:
    return sorted(_seen_unknown)


if __name__ == "__main__":
    import sys
    for line in sys.argv[1:] or ["合头靠 W O。还是 The cat sat，看 sat 这一行。GPT 二 small，d k 等于六十四，GQA、MLA、FP8、BF16、V3.2-Exp、mHC、YaRN。"]:
        print(to_spoken(line))
    print("unknown:", unknown())
