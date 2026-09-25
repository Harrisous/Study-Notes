# 已发布的 Artifact（可评论版本）

每一课都有一个私有的 claude.ai Artifact 页面，用来在页面上直接划词评论。本地 Markdown 源文件
（`deepseek/src/`、`math/src/`）是唯一的正本；Artifact 由它编译而来。机器可读的映射在 `artifacts.json`
（仓库相对路径 → URL），构建器用它把课与课之间的链接改写成 Artifact 链接。首次发布于 2026-09-25。

## 怎么更新

```bash
S=.cursor/skills/lesson-markdown/scripts
python $S/build_artifacts.py <scratch>/artifacts --link-map deepseek/reference-docs/artifacts.json
```

然后把 `<scratch>/artifacts/<track>-<slug>.html` 重新发布到下表对应的 URL（新会话里发布时带上 `url` 参数，
否则会生成一个新的 Artifact）。新增一课：先不带 `url` 发布，拿到 URL 后加进 `artifacts.json` 和本表，
再带 link map 重建并重发所有链接到它的课。

## 目录

| 页面 | URL |
|---|---|
| DeepSeek 科技树目录 | https://claude.ai/artifact/Dsrd1uuGzLMQ9Cbja9rmwr |
| 数学补课目录 | https://claude.ai/artifact/FRTcqR8vBKDJepW7i81RLz |

## DeepSeek 科技树

| 课 | URL |
|---|---|
| 第 0 课 · 科技树总览 | https://claude.ai/artifact/9gteUBcVfU497P9PSshNtT |
| 第 1 课 · Transformer 解剖 | https://claude.ai/artifact/JPjrAfp7YpgC6p6CYdEWDc |
| 第 2 课 · GPT 的训练 | https://claude.ai/artifact/DwqR6jgzQGwewzmz84dJ5D |
| 第 3 课 · RLHF 与 DPO | https://claude.ai/artifact/3rt139CtJ6uXThJV9iDstj |
| 第 4 课 · KV cache 与解码 | https://claude.ai/artifact/AgAXFtymhcfPprD1pAbD93 |
| 第 5 课 · 现代 Transformer 积木 | https://claude.ai/artifact/HU8SY2BhqF2nC8grX3viJx |
| 第 6 课 · DeepSeek LLM | https://claude.ai/artifact/EVQ62EyasG9EF3teW5fFJF |
| 第 7 课 · DeepSeekMoE | https://claude.ai/artifact/3FEcA5fSEFfazhMVDeUFXt |
| 第 8 课 · DeepSeekMath 与 GRPO | https://claude.ai/artifact/FwGnRGdapAHNvFfVKQu34A |
| 第 9 课 · MLA 潜在注意力 | https://claude.ai/artifact/GN9faE5jtzv8HR7Qts8vQ5 |
| 第 10 课 · DeepSeek-V2 训练 | https://claude.ai/artifact/MFFKd7xL4VWFzyhkxRcpyj |
| 第 11 课 · DeepSeek-V3 架构 | https://claude.ai/artifact/FNgkLNNwfjHEq6kSisw97h |
| 第 12 课 · DeepSeek-V3 训练基建 | https://claude.ai/artifact/JJrRfBxVYzSNxQw2RFdFnj |
| 第 13 课 · DeepSeek-R1 | https://claude.ai/artifact/NoAaGEBGJEoxpC4PRQZFmJ |
| 第 14 课 · DSA 稀疏注意力 | https://claude.ai/artifact/PQenRNvSnTUfVwmtUtfAuu |
| 第 15 课 · DeepSeek-V3.2 与智能体 | https://claude.ai/artifact/7g5oiF8seR8XjT8Mr9aTkC |
| 第 16 课 · mHC 超连接 | https://claude.ai/artifact/QdF5xWD6RRoiYMTFnTG6da |
| 第 17 课 · Engram 与 OCR | https://claude.ai/artifact/9dXs68Ec2RZUadeFYZd8YK |
| 第 18 课 · DeepSeek-V4 | https://claude.ai/artifact/2BQ8CNQnQ9C5jwgGsvzsPp |
| 第 19 课 · DeepSeek-V4.1-Flash | https://claude.ai/artifact/4eehy9jF7rRXjTykxfEd9a |
| 第 20 课 · 科技树复盘 | https://claude.ai/artifact/TGgjFjsPmyUi1ovWhFBqz1 |

## 数学补课

| 课 | URL |
|---|---|
| M01 · 矩阵与形状思维 | https://claude.ai/artifact/1CqL2VxLnRAnVnex615YUW |
| M02 · 低秩与 SVD | https://claude.ai/artifact/S6xaFLFBP2K3xyHXWvFT1c |
| M03 · 旋转与复数 | https://claude.ai/artifact/YEDEzrpwdR3s7XSy7qdHH2 |
| M04 · 概率与方差 | https://claude.ai/artifact/NCimu1XNLf83duFKyborzd |
| M05 · 熵与 KL 散度 | https://claude.ai/artifact/4BuWBwtrtfmwfNzFVPyjBB |
| M06 · 反向传播推导 | https://claude.ai/artifact/1u9faduh3pUv69PLkLVu3e |
| M07 · 从 SGD 到 Muon | https://claude.ai/artifact/9L27g7KwQTV9LX2MtVvjFU |
| M08 · 浮点数与量化 | https://claude.ai/artifact/1mwQ3aqrtK1VkqXAUXYEpE |
| M09 · 策略梯度数学 | https://claude.ai/artifact/CCpFMTCCVeaVuAi52LJwZB |
| M10 · 算力算术 | https://claude.ai/artifact/RJiCnG8x3sHBCYpe4YJ3fs |
| M11 · Sinkhorn 与双随机 | https://claude.ai/artifact/MJmoYXWEFrW42WNVubGP4M |
