# 第 07 课研究笔记：MoE 背景与 DeepSeekMoE（细粒度专家 + 共享专家）

> 服务于 `deepseek/src/lesson-07-deepseekmoe.md`（课程地图第 7 课）。建立于 2026-09-24。
> 本文件只收录**核对过**的引文、公式、规格、复算数字与勘误；课文作者不得绕过本文件凭记忆写数字。
> 所有派生数字都在 Python 中重算过（表达式见 §4）。引文保持英文原文。
> 核对手段：arXiv LaTeXML / ar5iv 全文 grep；ACL Anthology PDF 文本层；Jacobs 1991 扫描页逐页目读；
> Hugging Face `config.json` 与 `modeling_deepseek.py`、`configuration_deepseek.py` 原文。

---

## 0. 覆盖范围与所读版本

| 短键 | 标题 | 作者 | 编号 / 所读版本 | 发表状态 | 日期 |
|---|---|---|---|---|---|
| `jacobs1991` | Adaptive Mixtures of Local Experts | Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton | 无 arXiv；读的是 Hinton 主页扫描 PDF（`cs.toronto.edu/~hinton/absps/jjnh91.pdf`，9 页，页眉印 “Neural Computation 3, 79–87 (1991)”） | Neural Computation 3(1):79–87，doi 10.1162/neco.1991.3.1.79 | 1991 |
| `shazeer2017` | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean | arXiv:1701.06538 **v1**（唯一版本；ar5iv 渲染） | ICLR 2017（poster；OpenReview B1ckMDqlg。OpenReview 页面本次被人机验证拦截，venue 由检索结果与 Hinton 主页 PDF 页眉 “Published as a conference paper at ICLR 2017” 交叉确认） | 2017-01-23 |
| `gshard2020` | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen | arXiv:2006.16668 **v1**（唯一版本；ar5iv 渲染 + arXiv PDF 文本层） | ICLR 2021（iclr.cc 2021 poster 3196；dblp `conf/iclr/LepikhinLXCFHKS21`） | 2020-06-30 |
| `switch2021` | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | William Fedus, Barret Zoph, Noam Shazeer | arXiv:2101.03961 **v3**（2022-06-16，LaTeXML） | JMLR 23(120):1–39, 2022（jmlr.org/papers/v23/21-0998） | v1 2021-01-11 |
| `stmoe2022` | ST-MoE: Designing Stable and Transferable Sparse Expert Models | Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus | arXiv:2202.08906 **v2**（2022-04-29，LaTeXML） | arXiv 预印本（未查到正式 venue，见 §8） | v1 2022-02-17 |
| `dsmoe2024` | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y.K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, Wenfeng Liang | arXiv:2401.06066 **v1**（唯一 arXiv 版本，LaTeXML 全文） | **ACL 2024 Long**，pp. 1280–1297，doi 10.18653/v1/2024.acl-long.70（ACL PDF 文本层也读了，**与 arXiv v1 有实质差异**，见 §7） | 2024-01-11 |
| `mixtral2024` | Mixtral of Experts | Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford 等（共 26 人） | arXiv:2401.04088 **v1**（LaTeXML） | arXiv 预印本 | 2024-01-08 |
| `dsv2` | DeepSeek-V2 | DeepSeek-AI | arXiv:2405.04434 **v5**（2024-06-19） | 预印本 | — |
| `dsv3` | DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv:2412.19437（只 grep 了一句门控差异；核查时 arXiv 最新版为 **v2**，grep 的是最新版全文） | 预印本 | 2024-12-27（v1） |
| `dsmoe_residual` | DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale | Samyam Rajbhandari 等 | arXiv:2201.05596 **v2** | ICML 2022（PMLR v162） | 2022-01-14 |
| `dsllm` | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | DeepSeek-AI | arXiv:2401.02954 v1（只读 Table 2 规格） | 预印本 | 2024-01-05 |
| `hashlayer` | Hash Layers For Large Sparse Models | Roller, Sukhbaatar, Szlam, Weston | arXiv:2106.04426 v3（**只读了元数据**，作为 DeepSeekMoE 基线出现） | — | 2021-06-08 |
| `llama2` | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron 等 | arXiv:2307.09288 v2（只读元数据；规格用 LLaMA 式公开配置推算，见 §4） | 预印本 | 2023-07-18 |

制品（无论文，`kind: artefact`）：

| 短键 | 内容 | 读取方式 |
|---|---|---|
| `hf_dsmoe16b` | `deepseek-ai/deepseek-moe-16b-base` 的 `config.json`、`modeling_deepseek.py`、`configuration_deepseek.py`；`deepseek-moe-16b-chat/config.json` | `huggingface.co/.../raw/main/...` 原文 |
| `gh_dsmoe` | GitHub `deepseek-ai/DeepSeek-MoE` README | raw.githubusercontent.com |
| `hf_mixtral` | `mistralai/Mixtral-8x7B-v0.1/config.json` | 原文 |
| `hf_v2lite` | `deepseek-ai/DeepSeek-V2-Lite/config.json` | 原文 |
| `hf_dsllm` | `deepseek-llm-7b-base`、`deepseek-llm-67b-base` 的 `config.json` | 原文 |

---

## 1. 核心论断与原文引文

### 1.A 前史：1991 年的“局部专家混合”（`jacobs1991`，扫描版逐页核对）

**C1. MoE 的原始动机是“减少干扰”，而不是“省算力”。**
§1 第一段：“If backpropagation is used to train a single, multilayer network to perform different subtasks on different occasions, there will generally be strong interference effects that lead to slow learning and poor generalization.”（p. 79）
脚注 1：“This idea was first presented by Jacobs and Hinton at the Connectionist Summer School in Pittsburgh in 1988.”

**C2. 线性混合的误差函数会让专家“合作”而非“专精”。**
p. 80，式 (1.1) 之后：“This strong coupling between the experts causes them to cooperate nicely, but tends to lead to solutions in which many experts are used for each case.”

**C3. 改成“随机选一个专家”的期望误差，专家之间转为竞争。**
p. 80：“Instead of linearly combining the outputs of the separate experts, we imagine that the gating network makes a stochastic decision about which single expert to use on each occasion (see Fig. 1).”
p. 80（式 (1.2) 之后，整句都在 p. 80）：“Notice that in this new error function, each expert is required to produce the whole of the output vector rather than a residual.”
p. 81：“If both the gating network and the local experts are trained by gradient descent in this new error function, the system tends to devote a single expert to each training case.”

**C4. 实际用的是负对数混合高斯似然 (1.3)，因为它让“最合适的专家学得最快”。**
p. 82：“The error defined in equation 1.3 is simply the negative log probability of generating the desired output vector under the mixture of gaussians model described at the end of the next section.”
p. 82：“Equation 1.4 will adapt the best-fitting expert the *slowest*, whereas equation 1.5 will adapt it the fastest.”（前提：“the gating network initially gives equal weights to all experts and ‖d^c − o_i^c‖ > 1 for all the experts”）

**C5. 门控是 softmax，且 1991 年的 MoE 是“稠密”的（所有专家都计算）。**
图 1 标题（p. 81）：“It has normalized outputs p_j = exp(x_j)/Σ_i exp(x_i), where x_j is the total weighted input received by output unit j of the gating network. The selector acts like a multiple input, single output stochastic switch; the probability that the switch will select the output from expert j is p_j.”

**C6. 实验：四类元音、75 位说话人、专家数 4 或 8，收敛所需 epoch 约为反传网络的一半。**
p. 83：“The data consisted of the first and second formants of the vowels [i], [I], [a], and [A] (usually denoted [Λ]) from 75 speakers (males, females, and children) uttered in a hVd context (Peterson and Barney 1952).”
p. 84：“All models were trained with data from the first 50 speakers and tested with data from the remaining 25 speakers.”
p. 85：“The mixtures of experts reach the error criterion significantly faster than the backpropagation networks (p ≫ 0.999), requiring only about half as many epochs on average (Table 1).”（“p ≫ 0.999” 照原文印刷）
Table 1（p. 85）：4 Experts 1124 epochs（SD 23）；8 Experts 1083（SD 12）；BP 6 Hid 2209（SD 83）；BP 12 Hid 2435（SD 124）；四者 Train 88% / Test 90%。
图 2 标题（p. 84）：“Although the mixture in this case contained 4 experts, one of these experts made no significant contribution to the final mixture since its mixing proportion p_i was effectively 0 for all cases.”

### 1.B 稀疏门控 MoE 进入深度学习（`shazeer2017`，arXiv v1）

**C7. 稀疏 MoE 层的定义与目标：条件计算，容量扩大 >1000 倍。**
Abstract：“achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks.”
Abstract：“We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers.”（注意：是 LSTM 之间，不是 Transformer）

**C8. 稀疏靠 top-k + −∞，噪声用于负载均衡。**
§2.1 Noisy Top-K Gating：“Before taking the softmax function, we add tunable Gaussian noise, then keep only the top k values, setting the rest to −∞ (which causes the corresponding gate values to equal 0).”
同段：“While this form of sparsity creates some theoretically scary discontinuities in the output of gating function, we have not yet observed this to be a problem in practice. The noise term helps with load balancing, as will be discussed in Appendix A.”

**C9. k>1 才有路由梯度（这是后来 Switch 反驳的“猜想”）。**
§2.1 Training the Gating Network：“If we choose k>1, the gate values for the top k experts have nonzero derivatives with respect to the weights of the gating network.”

**C10. “收缩批量”问题。**
§3.1：“If the gating network chooses k out of n experts for each example, then for a batch of b examples, each expert receives a much smaller batch of approximately kb/n ≪ b examples.”

**C11. 路由坍缩是自我强化的。**
§4：“We have observed that the gating network tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus are selected even more by the gating network.”

**C12. 两个辅助损失：importance（门值之和）与 load（被分到的样本数的光滑估计），都用变异系数平方。**
§4：“This loss is equal to the square of the coefficient of variation of the set of importance values, multiplied by a hand-tuned scaling factor w_importance.”
§4：“For example, one expert may receive a few examples with large weights, and another may receive many examples with small weights. This can cause memory and performance problems on distributed hardware.”
Appendix A：“Unfortunately, the number of examples received by an expert is a discrete quantity, so it can not be used in back-propagation. Instead, we define a smooth estimator Load(X) … This is the purpose of the noise term in the gating function.”
Appendix A 初始化：“we initialize the matrices W_g and W_noise to all zeros, which yields no signal and some noise.”

**C13. 辅助损失的效果（Appendix A, Table 6，MoE-256 模型）。**
无任何损失：test perplexity 39.8，CV(Importance) 3.04，CV(Load) 3.01，max/mean load 17.80；
w_imp=0.1, w_load=0.1：35.6 / 0.06 / 0.05 / 1.14；w_imp=0.01, w_load=0.01：35.7 / 0.48 / 0.11 / 1.37；w=1.0/1.0：35.7 / 0.03 / 0.02 / 1.07。
结论原文：“All the combinations containing at least one the two losses led to very similar model quality, where having no loss was much worse.”（“one the two” 照原文）
超参：LM 实验 “we set w_importance = 0.1 and w_load = 0.1”（Appendix C）；MT 实验 “w_importance = 0.01 and w_load = 0.01”（Appendix E）。

**C14. 规模结果。** §5.2：“When training over the full 100 billion words, test perplexity improves significantly up to 65536 experts (68 billion parameters), dropping 39% lower than the computationally matched baseline, but degrades at 131072 experts, possibly a result of too much sparsity.”

### 1.C Transformer 里的 MoE：GShard（`gshard2020`，arXiv v1）

**C15. 每隔一层 FFN 换成 MoE，top-2。** §2.1：“We sparsely scale Transformer with conditional computation by replacing every other feed-forward layer with a Position-wise Mixture of Experts (MoE) layer [16] with a variant of top-2 gating in both the encoder and the decoder (Figure 3).”

**C16. 专家容量与溢出 token 走残差。** §2.2 Expert capacity：“When both experts selected by a token already exceed their capacity, the token is considered as an overflowed token, where 𝒢_{s,E} degenerates into a zero vector. Such tokens have their representation x_s passed on to the next layer via residual connections.”

**C17. 本地分组调度：每组容量 2N/(G·E)。** §2.2：“Each group is given a fractional capacity of each expert, 2N/(G·E).”

**C18. 辅助损失的设计理由：用平均门值 m_e 代替不可导的 c_e/S。** §2.2：“the term c_e/S represents the fraction of input routed to each expert, and we want to minimize mean square of c_e/S. But because c_e is derived from top-2 operation and is not differentiable, we use the mean gates per expert m_e as a differentiable approximation and replace (c_e/S)² with m_e(c_e/S), which can now be optimized with gradient descent.”

**C19. 随机路由第二专家。** §2.2：“GATE(·) dispatches to the 2nd-best expert with the probability proportional to its weight g_2.”（Algorithm 1 注释写的是 “probability ∝ 2·g2”）

**C20. 规模：600B、2048 TPU v3、4 天。** Abstract：“GShard enabled us to scale up multilingual neural machine translation Transformer model with Sparsely-Gated Mixture-of-Experts beyond 600 billion parameters using automatic sharding. We demonstrate that such a giant model can efficiently be trained on 2048 TPU v3 accelerators in 4 days”。§4.3 另述 1T 模型（MoE(2048E, 60L)）“encountered several trainability issues with numerical stability, hence did not include the results”。

### 1.D Switch Transformer（`switch2021`，arXiv v3 = JMLR 版）

**C21. top-1 足够。** §2.1：“Shazeer et al. (2017) conjectured that routing to k>1 experts was necessary in order to have non-trivial gradients to the routing functions. … Contrary to these ideas, we instead use a simplified strategy where we route to only a single expert. We show this simplification preserves model quality, reduces routing computation and performs better.”
同节：“Note that for both MoE and Switch Routing, the gate value p_i(x) in Equation 2 permits differentiability of the router.”

**C22. 容量因子。** §2.2 Eq. (3) 之后：“A capacity factor greater than 1.0 creates additional buffer to accommodate for when tokens are not perfectly balanced across experts. If too many tokens are routed to an expert (referred to later as dropped tokens), computation is skipped and the token representation is passed directly to the next layer through the residual connection.”
同节：“Throughout our experiments we didn't notice any dependency on the number of experts for the number of tokens dropped (typically <1%).”

**C23. 可微负载均衡损失与 α = 10⁻²。** §2.2 “A Differentiable Load Balancing Loss”：“As in Shazeer et al. (2018); Lepikhin et al. (2020), Switch Transformers simplifies the original design in Shazeer et al. (2017) which had separate load-balancing and importance-weighting losses.”
“The objective can also be differentiated as the P-vector is differentiable, but the f-vector is not.”
“The final loss is multiplied by expert count N to keep the loss constant as the number of experts varies since under uniform routing Σ(f_i·P_i) = Σ(1/N·1/N) = 1/N.”
“throughout this work we use an α = 10⁻² which was sufficiently large to ensure load balancing while small enough to not to overwhelm the primary cross-entropy objective. We swept hyper-parameter ranges of α from 10⁻¹ to 10⁻⁵ in powers of 10”。

**C24. 低容量因子下 Switch 更好（Table 1，128 experts，每隔一层）。** Table 1：Switch-Base CF 1.0 → Neg. Log Perp. −1.561、1000 examples/sec；CF 1.25 → −1.553、910；CF 2.0 → −1.554、860；MoE-Base（top-2）CF 2.0 → −1.547、840；CF 1.25 → −1.559、790；CF 1.0 → −1.572、860；T5-Base −1.731、1600；T5-Large −1.550、470。
“(3) Switch Transformers perform better at lower capacity factors (1.0, 1.25).”

**C25. 稳定性技巧：路由器局部用 float32；初始化缩小 10 倍。** §2.4：“we cast the router input to float32 precision”；“we recommend reducing the default Transformer initialization scale s = 1.0 by a factor of 10.”（Table 2：Switch-Base bfloat16 “−3.780 [diverged]”，selective precision −1.716）

**C26. 规模：Switch-C 1571B 参数、2048 专家、每层都是专家层（Expert Freq. 1）、15 层；Switch-XXL 395B、64 专家。** Table 9；§5.6：“the larger Switch-C model, with 1.6T parameters and 2048 experts, exhibits no training instability at all.”

### 1.E ST-MoE 的 router z-loss（`stmoe2022`，v2，选讲）

**C27.** §3.3：“we introduce the router z-loss … This penalizes large logits into the gating network”；“We choose a value of c_z = 0.001 based on the best model quality after pre-training with a hyperparameter sweep.” Table 4：Baseline 4/6 stable, −1.755；update clipping (0.1) 3/3, −4.206；Router Z-Loss 3/3, −1.741。
Appendix A 再次使用 α·N·Σf_iP_i 且 “we use an α = 10⁻²”。

### 1.F DeepSeekMoE（`dsmoe2024`，arXiv v1 为主，ACL 版差异单列）

**C28. 问题陈述：专家专精（expert specialization）不足，来自两个原因。** §1：
“(1) Knowledge Hybridity: existing MoE practices often employ a limited number of experts (e.g., 8 or 16), and thus tokens assigned to a specific expert will be likely to cover diverse knowledge. Consequently, the designated expert will intend to assemble vastly different types of knowledge in its parameters, which are hard to utilize simultaneously.”
“(2) Knowledge Redundancy: tokens assigned to different experts may require common knowledge. As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters, thereby leading to redundancy in expert parameters.”
专精的定义（Abstract）：“expert specialization, i.e. each expert acquires non-overlapping and focused knowledge.”

**C29. 细粒度专家切分：把中间维切成 1/m，激活数乘 m，参数与计算不变。** §3.1：“we segment each expert FFN into m smaller experts by reducing the FFN intermediate hidden dimension to 1/m times its original size. Since each expert becomes smaller, in response, we also increase the number of activated experts to m times to keep the same computation cost”。

**C30. 组合数例子（已复算，正确）。** §3.1：“As an illustrative example, we consider the case where N=16. A typical top-2 routing strategy can yield C(16,2)=120 possible combinations. By contrast, if each expert is split into 4 smaller experts, the fine-grained routing strategy can yield C(64,8)=4,426,165,368 potential combinations.”

**C31. 共享专家隔离：确定性激活，路由激活数相应减 K_s。** §3.2：“Regardless of the router module, each token will be deterministically assigned to these shared experts. In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by K_s”。
来源致谢（§3.2，arXiv v1）：“It is worth noting that the prototype of shared expert isolation can be credited to Rajbhandari et al. (2022). The key distinction lies in the fact that they derive this strategy from an engineering perspective, while we approach it from an algorithmic standpoint.”
DeepSpeed-MoE 原文（`dsmoe_residual` §4.1.1）：“fixing one expert and varying the second expert across different experts (referred to as Residual-MoE)”；“The main intuition is to treat the expert from MoE module as an error correction term of the dense MLP module.”

**C32. 负载不均衡的两个危害。** §3.3：“Firstly, there is a risk of routing collapse (Shazeer et al., 2017), i.e., the model always selects only a few experts, preventing other experts from sufficient training. Secondly, if experts are distributed across multiple devices, load imbalance can exacerbate computation bottlenecks.”

**C33. 设备级均衡的理由：严格的专家级均衡伤性能。** §3.3 Device-Level Balance Loss：“When aiming to alleviate computation bottlenecks, it becomes unnecessary to enforce strict balance constraints at the expert level, because excessive constraints on load balance will compromise model performance. Instead, our primary objective is to ensure balanced computation across the devices.”
“In practice, we set a small expert-level balance factor to mitigate the risk of routing collapse, and meanwhile set a larger device-level balance factor to promote balanced computation across the devices.”

**C34. 2B 验证实验设置。** §4.1.3：9 层、隐藏维 1280、10 头×128、init std 0.006；“ensure that the total number of expert parameters equals 16 times that of a standard FFN”；“keep the activated expert parameters, including shared expert parameters and activated routed expert parameters, as 2 times that of a standard FFN”；“approximately 2B total parameters, with the number of activated parameters around 0.3B.” 训练：100B tokens、词表 8K、AdamW(0.9, 0.95, wd 0.1)、峰值 LR 1.08×10⁻³、2K warmup、在 80%/90% 步处各乘 0.316、batch 2K×2K = 4M tokens、25,000 步；“we do not drop any tokens during training and do not employ the device-level balance loss. In order to prevent routing collapse, we set an expert-level balance factor of 0.01.”
DeepSeekMoE 2B 构成（§4.2）：“DeepSeekMoE has 1 shared expert and 63 routed experts, where each expert is 0.25 times the size of a standard FFN.”（§4.5：“1 shared expert and 7 out of 63 routed experts being activated”）

**C35. 同等总参数与激活参数下显著优于 GShard（Table 1）；追平 1.5 倍专家的 GShard 与 Dense×16。** §4.2：“With the same number of total parameters and activated parameters, DeepSeekMoE demonstrates overwhelming advantages over GShard.”
§4.3：“DeepSeekMoE nearly approaches the performance of Dense×16, which sets the strict upper bound of MoE models in terms of the model capacity. These results suggest that, at least at the scale of about 2B parameters and 100B training tokens, the performance of DeepSeekMoE aligns closely with the theoretical upper bound of MoE models.”
Dense×16 的构造：“we configure 16 shared experts where each expert has the same number of parameters as a standard FFN.”

**C36. 消融：两项策略各有贡献；共享 : 路由比例影响很小，选 1:3。** §4.4：“1, 2, and 4 shared experts achieve a Pile loss of 1.808, 1.806, and 1.811, respectively. Considering that the ratio of 1:3 yields a marginally better Pile loss, when scaling up DeepSeekMoE, we keep the ratio between shared experts and activated routed experts as 1:3.”
（换算说明，据同段 “keeping the number of total experts and activated experts constant”，即 64 个专家、每 token 激活 8 个：1 共享 = 1 + 7 路由，比例 1:7；**2 共享 = 2 + 6 路由，比例 1:3，Pile 1.806**；4 共享 = 4 + 4，比例 1:1。所以“1:3”对应的是 2 个共享专家那一档，不是 1 个共享专家的 2B 主模型。16B 的 2 + 6、145B 的 4 + 12 都是 1:3。）
消融图（arXiv 图 3 / ACL 图 2）四条配置的图例原文（取自 ACL PDF 文本层）：“0 shared expert + 2 out of 16 routed experts (GShard)”；“1 shared expert + 1 out of 15 routed experts (+ shared expert isolation)”；“1 shared expert + 3 out of 31 routed experts (+ fine-grained expert segmentation)”；“1 shared expert + 7 out of 63 routed experts (+ finer expert segmentation)”。

**C37. 专精性证据（三条）。** §4.5：
- 禁用 top 路由专家更敏感 ⇒ 冗余更低：“This sensitivity suggests a lower level of parameter redundancy in DeepSeekMoE, since each routed expert is more irreplaceable.”（对照 GShard×1.5，因两者未禁用时 Pile loss 相同）
- 共享专家不可替代：“we disable it and activate one more routed expert. The evaluation on Pile shows a significant increase in the Pile loss, rising from 1.808 to 2.414, even though we maintain the same computational cost.”
- 更少的激活专家也够：“even with only 4 routed experts activated, DeepSeekMoE achieves a Pile loss comparable with GShard.”；从零训练 1 共享 + 63 路由中只激活 3 个：“even with the same total expert parameters and only half of the activated expert parameters, DeepSeekMoE still outperforms GShard.”

**C38. DeepSeekMoE 16B 规格（§5.1.2，逐字）。** “we set the number of Transformer layers to 28 and the hidden dimension to 2048. We employ the multi-head attention mechanism with a total of 16 attention heads, where each head has a dimension of 128.”；“We substitute all FFNs except for the first layer with MoE layers, since we observe that the load balance status converges especially slower for the first layer. Each MoE layer consists of 2 shared experts and 64 routed experts, where each expert is 0.25 times the size of a standard FFN. Each token will be routed to these 2 shared experts and 6 out of 64 routed experts.”；“DeepSeekMoE 16B has approximately 16.4B total parameters, with the number of activated parameters around 2.8B.”
为什么不更细：“An even finer expert segmentation granularity is not employed due to the potential reduction in computational efficiency associated with excessively small expert sizes.”
训练：2T tokens、词表 “100K”、峰值 LR 4.2×10⁻⁴、batch 4.5K × 4K = 18M tokens、106,449 步；“for each layer, all the experts will be deployed on the same device. Therefore, we also do not drop any tokens during training and do not employ the device-level balance loss. In order to prevent routing collapse, we set a quite small expert-level balance factor of 0.001 because we find that under our parallelization strategy, a higher expert-level balance factor cannot increase the computation efficiency, but instead, it will compromise the model performance.”

**C39. 16B vs DeepSeek 7B / LLaMA2 7B：约 40% 计算量。** Table 3 标题：“With only 40.5% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B.” Table 4 标题：“With only 39.6% of computations, DeepSeekMoE 16B outperforms LLaMA2 7B on the majority of benchmarks.” §5.2.2：“Compared with LLaMA2 7B, DeepSeekMoE has 245% of total parameters but only needs 39.6% of computations.”

**C40. 弱项：多选题（归因于注意力参数少）。** §5.2.1：“This inadequacy stems from the limited attention parameters in DeepSeekMoE 16B (DeepSeekMoE 16B has only about 0.5B attention parameters, while DeepSeek 7B has 2.5B attention parameters).”（2.5B 与规格复算不符，见 §7-E13。**ACL 2024 版删掉了这个括号里的两个数字**，只写 “which may stem from the limited attention parameters in DeepSeekMoE 16B”）

**C41. 部署：单卡 40GB、约 2.5 倍推理速度。** §1：“this model can be deployed on a single GPU with 40GB of memory without the need for quantization.” §5.2.1：“With appropriate operator optimizations, it can achieve nearly 2.5 times the inference speed of a 7B dense model.”（未给测量细节）

**C42. 145B 初步结果（仅 arXiv v1 §7，标题 “DeepSeekMoE 145B Ongoing”）。** “In this initial study, DeepSeekMoE 145B is trained on 245B tokens”；62 层、隐藏维 4096、32 头×128；“Each MoE layer consists of 4 shared experts and 128 routed experts, where each expert is 0.125 times the size of a standard FFN. Each token will be routed to these 4 shared experts and 12 out of 128 routed experts.”；“approximately 144.6B total parameters, with the number of activated parameters around 22.2B.”；warmup-and-constant LR，峰值 3.0×10⁻⁴，13,000 步；“all the routed experts will be uniformly deployed on 4 devices”；“we set the device-level balance factor to 0.05 … we still set a small expert-level balance factor of 0.003”。
Table 6 标题：“with 28.5% of computations, DeepSeekMoE 145B achieves comparable performance with DeepSeek 67B.” §7.2(3)：“DeepSeekMoE 142B (Half Activated) still match the performance of DeepSeek 67B (Dense), with only 18.2% of computations.”
GShard 137B 与 145B 的尺寸差：“DeepSeekMoE 145B aligns the intermediate hidden dimension in each expert to a multiple of 64 for computation efficiency, so its model size is 6% larger than GShard 137B.”

**C43. 对 MoE 做 SFT 也有收益（arXiv v1 §6，ACL 版删除）。** SFT 数据 1.4M 条、batch 1024、8 epochs、常数 LR 10⁻⁵；Table 5 标题：“Compared with both 7B dense models, DeepSeekMoE Chat 16B still achieves comparable or better performance on the majority of benchmarks with only 40% of computations.”

**C44. DeepSeek-V2 把 DeepSeekMoE 改写成 N_s / N_r / K_r 记号（课程采用这一套）。** `dsv2` §2.2.1 式 (20)–(22) 后：“where N_s and N_r denote the numbers of shared experts and routed experts, respectively; … K_r denotes the number of activated routed experts; … e_i is the centroid of the i-th routed expert in this layer”。

**C45. V2-Lite 沿用 DeepSeekMoE 16B 的 FFN 配置。** `dsv2` Appendix B.1：“DeepSeek-V2-Lite has 27 layers and a hidden dimension of 2048.”；“Each MoE layer consists of 2 shared experts and 64 routed experts, where the intermediate hidden dimension of each expert is 1408. Among the routed experts, 6 experts will be activated for each token. Under this configuration, DeepSeek-V2-Lite comprises 15.7B total parameters, of which 2.4B are activated for each token.”；“we only employ a small expert-level balance loss with α₁ = 0.001”；5.7T tokens。

**C46.（前瞻，只核对了一句）V3 改门控。** `dsv3` §2.1.2：“Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values.”——第 11 课详谈。

### 1.G 对照组：Mixtral 8x7B（`mixtral2024`，v1）

**C47. 每层 8 个 SwiGLU 专家，top-2，softmax 作用在 top-K logits 上。** §2.1：“a simple and performant one is implemented by taking the softmax over the Top-K logits of a linear layer [28]”；“For Mixtral we use the same SwiGLU architecture as the expert function E_i(x) and set K = 2.”
与 GShard 的差别（作者自述）：“This formulation is similar to the GShard architecture [21], with the exceptions that we replace all FFN sub-blocks by MoE layers while GShard replaces every other block, and that GShard uses a more elaborate gating strategy for the second expert assigned to each token.”

**C48. 47B 总参数 / 13B 激活参数；32k 上下文。** Abstract：“each token has access to 47B parameters, but only uses 13B active parameters during inference. Mixtral was trained with a context size of 32k tokens”。

**C49. 路由没有明显的“按主题”专精。** §5：“Surprisingly, we do not observe obvious patterns in the assignment of experts based on the topic.”；“This suggests that the router does exhibit some structured syntactic behavior.”；“consecutive tokens are often assigned the same experts.”
（可与 DeepSeekMoE 的“知识混杂”论点并读：8 个大专家确实没按领域分工——但这是不同论文不同分析方法的观察，不能直接当作 DeepSeekMoE 论点的证据，见 §7-E20。）

---

## 2. 公式（原文转录 + 课程记号对照）

### 2.1 Jacobs et al. 1991（p. 80–83，编号照原文）

$$E^c = \Big\|\mathbf d^c - \sum_i p_i^c\,\mathbf o_i^c\Big\|^2 \tag{1.1}$$

$$E^c = \big\langle \|\mathbf d^c-\mathbf o_i^c\|^2\big\rangle = \sum_i p_i^c\,\|\mathbf d^c-\mathbf o_i^c\|^2 \tag{1.2}$$

$$E^c = -\log \sum_i p_i^c\, e^{-\frac12\|\mathbf d^c-\mathbf o_i^c\|^2} \tag{1.3}$$

$$\frac{\partial E^c}{\partial \mathbf o_i^c} = -2p_i^c(\mathbf d^c-\mathbf o_i^c) \tag{1.4}$$

$$\frac{\partial E^c}{\partial \mathbf o_i^c} = -\left[\frac{p_i^c e^{-\frac12\|\mathbf d^c-\mathbf o_i^c\|^2}}{\sum_j p_j^c e^{-\frac12\|\mathbf d^c-\mathbf o_j^c\|^2}}\right](\mathbf d^c-\mathbf o_i^c) \tag{1.5}$$

门控（图 1 标题）：$p_j=\exp(x_j)/\sum_i\exp(x_i)$。另有 (2.1) $\log P^c=\log\sum_i p_i k e^{-\frac12\|\boldsymbol\mu_i-\mathbf o^c\|^2}$（竞争学习的混合高斯）。
**课程对照**：$p_i^c$ ↔ 后来的门值 $g_{i,t}$（但 1991 年是**稠密**的，所有专家都算）；(1.5) 方括号里就是“后验责任”（EM 的 E 步），课文可在此提示“软分配 vs 硬分配”。

### 2.2 Shazeer et al. 2017（§2–§4、Appendix A）

$$y=\sum_{i=1}^{n}G(x)_iE_i(x) \tag{1}$$
$$G_\sigma(x)=\mathrm{Softmax}(x\cdot W_g) \tag{2}$$
$$G(x)=\mathrm{Softmax}(\mathrm{KeepTopK}(H(x),k)) \tag{3}$$
$$H(x)_i=(x\cdot W_g)_i+\mathrm{StandardNormal}()\cdot\mathrm{Softplus}((x\cdot W_{noise})_i) \tag{4}$$
$$\mathrm{KeepTopK}(v,k)_i=\begin{cases}v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v.\\ -\infty & \text{otherwise.}\end{cases} \tag{5}$$
$$\mathrm{Importance}(X)=\sum_{x\in X}G(x) \tag{6}$$
$$L_{importance}(X)=w_{importance}\cdot CV(\mathrm{Importance}(X))^2 \tag{7}$$
$$P(x,i)=\Phi\!\left(\frac{(x\cdot W_g)_i-kth\_excluding(H(x),k,i)}{\mathrm{Softplus}((x\cdot W_{noise})_i)}\right) \tag{9}$$
（(8) 是 (9) 化简前的概率式；$\Phi$ 为标准正态 CDF。）
$$\mathrm{Load}(X)_i=\sum_{x\in X}P(x,i) \tag{10}$$
$$L_{load}(X)=w_{load}\cdot CV(\mathrm{Load}(X))^2 \tag{11}$$

**课程对照**：$x$（行向量，$x\cdot W_g$）↔ $\mathbf u_t^\top$；$n$ ↔ 专家数 $N$；$G(x)_i$ ↔ $g_{i,t}$；$W_g$ 的第 $i$ 列 ↔ $\mathbf e_i$。注意 Shazeer 是 **先 top-k 再 softmax**（门值在 k 个专家上归一化，和为 1）。CV 未说明用总体还是样本标准差（§8）。

### 2.3 GShard（§2.2 式 (1)–(3)；Algorithm 1 用 **PDF 行号**）

$$\mathcal G_{s,E}=\mathrm{GATE}(x_s)\tag{1}\qquad \mathrm{FFN}_e(x_s)=wo_e\cdot\mathrm{ReLU}(wi_e\cdot x_s)\tag{2}\qquad y_s=\sum_{e=1}^{E}\mathcal G_{s,e}\cdot\mathrm{FFN}_e(x_s)\tag{3}$$

Algorithm 1（Group-level top-2 gating with auxiliary loss，PDF 行号）：
- (2) $g_{S,E}\leftarrow \mathrm{softmax}(wg\cdot x_S)$；(3) $m_E\leftarrow\frac1S\sum_{s=1}^{S}g_{s,E}$
- (5) $g1,e1,g2,e2=\mathrm{top\_2}(g_{s,E})$；(6) $g1\leftarrow g1/(g1+g2)$；(8)–(11) 若 $c_{e1}<C$ 则 $\mathcal G_{s,e1}\leftarrow g1$；无论是否溢出都 $c_{e1}\leftarrow c+1$
- (13) $\displaystyle \ell_{aux}=\frac1E\sum_{e=1}^{E}\frac{c_e}{S}\cdot m_e$
- (16) $g2\leftarrow g2/(g1+g2)$；(17) $rnd\leftarrow\mathrm{uniform}(0,1)$；之后若 $c<C\wedge 2\cdot g2>rnd$ 才派发第二专家
- 总损失 $\mathcal L=\ell_{nll}+k*\ell_{aux}$；每组容量 $2N/(G\cdot E)$

**课程对照**：$E$ ↔ $N$（专家数）；$c_e/S$ ↔ Switch 的 $f_i$；$m_e$ ↔ $P_i$。**关键细节**：(13) 在第一个循环（只处理首选专家 $e1$）之后、第二个循环之前计算，所以 $c_e$ 只数了**首选**分配。

### 2.4 Switch Transformer（§2.1–§2.2）

$$p_i(x)=\frac{e^{h(x)_i}}{\sum_j^N e^{h(x)_j}},\quad h(x)=W_r\cdot x \tag{1}$$
$$y=\sum_{i\in\mathcal T}p_i(x)E_i(x) \tag{2}$$
$$\text{expert capacity}=\Big(\frac{\text{tokens per batch}}{\text{number of experts}}\Big)\times\text{capacity factor} \tag{3}$$
$$\text{loss}=\alpha\cdot N\cdot\sum_{i=1}^{N}f_i\cdot P_i \tag{4}$$
$$f_i=\frac1T\sum_{x\in\mathcal B}\mathbb 1\{\mathrm{argmax}\,p(x)=i\} \tag{5}$$
$$P_i=\frac1T\sum_{x\in\mathcal B}p_i(x) \tag{6}$$

**课程对照**：Switch 的 $T$ 是**一个 batch 里的 token 数**（DeepSeekMoE 的 $T$ 在 §2 被定义为序列长度，二者不同）；$N$ ↔ $N_r$；$p_i(x)$ ↔ $s_{i,t}$。注意 Switch 是 **先 softmax（对全部 N 个）再取 top-1**，门值 $p_i(x)<1$，这正是 top-1 时路由器仍有梯度的原因。

### 2.5 ST-MoE（§3.3）

$$L_z(x)=\frac1B\sum_{i=1}^{B}\Big(\log\sum_{j=1}^{N}e^{x_j^{(i)}}\Big)^2 \tag{5}$$
$$L_{tot}=L_{CE}+c_BL_B+c_zL_Z \tag{6}$$
$c_z=0.001$；$L_B$ 用 Appendix A 的 $\alpha N\sum f_iP_i$，$\alpha=10^{-2}$。

### 2.6 DeepSeekMoE（arXiv v1，编号照原文）

标准块（省略 LayerNorm）：
$$\mathbf u_{1:T}^{l}=\operatorname{Self-Att}\big(\mathbf h_{1:T}^{l-1}\big)+\mathbf h_{1:T}^{l-1} \tag{1}$$
$$\mathbf h_t^l=\operatorname{FFN}\big(\mathbf u_t^l\big)+\mathbf u_t^l \tag{2}$$

传统 top-K MoE：
$$\mathbf h_t^l=\sum_{i=1}^{N}\big(g_{i,t}\operatorname{FFN}_i(\mathbf u_t^l)\big)+\mathbf u_t^l \tag{3}$$
$$g_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}\in\operatorname{Topk}(\{s_{j,t}\mid 1\leqslant j\leqslant N\},K),\\ 0,&\text{otherwise},\end{cases} \tag{4}$$
$$s_{i,t}=\operatorname{Softmax}_i\big({\mathbf u_t^l}^{T}\mathbf e_i^l\big) \tag{5}$$

细粒度切分：
$$\mathbf h_t^l=\sum_{i=1}^{mN}\big(g_{i,t}\operatorname{FFN}_i(\mathbf u_t^l)\big)+\mathbf u_t^l \tag{6}$$
$$g_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}\in\operatorname{Topk}(\{s_{j,t}\mid 1\leqslant j\leqslant mN\},mK),\\ 0,&\text{otherwise},\end{cases} \tag{7}$$
$$s_{i,t}=\operatorname{Softmax}_i\big({\mathbf u_t^l}^{T}\mathbf e_i^l\big) \tag{8}$$

完整 DeepSeekMoE（+ 共享专家）：
$$\mathbf h_t^l=\sum_{i=1}^{K_s}\operatorname{FFN}_i(\mathbf u_t^l)+\sum_{i=K_s+1}^{mN}\big(g_{i,t}\operatorname{FFN}_i(\mathbf u_t^l)\big)+\mathbf u_t^l \tag{9}$$
$$g_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}\in\operatorname{Topk}(\{s_{j,t}\mid K_s+1\leqslant j\leqslant mN\},mK-K_s),\\ 0,&\text{otherwise},\end{cases} \tag{10}$$
$$s_{i,t}=\operatorname{Softmax}_i\big({\mathbf u_t^l}^{T}\mathbf e_i^l\big) \tag{11}$$
（(11) 的 softmax 下标范围原文未写明；代码中只对 $N_r$ 个路由专家做 softmax——`MoEGate.weight` 形状为 `(n_routed_experts, hidden_size)`。）

专家级均衡损失：
$$\mathcal L_{\mathrm{ExpBal}}=\alpha_1\sum_{i=1}^{N'}f_iP_i \tag{12}$$
$$f_i=\frac{N'}{K'T}\sum_{t=1}^{T}\mathbb 1(\text{Token } t \text{ selects Expert } i) \tag{13}$$
$$P_i=\frac1T\sum_{t=1}^{T}s_{i,t} \tag{14}$$
其中 $N'=mN-K_s$，$K'=mK-K_s$。

设备级均衡损失（路由专家分成 $D$ 组 $\{\mathcal E_1,\dots,\mathcal E_D\}$，每组一台设备）：
$$\mathcal L_{\mathrm{DevBal}}=\alpha_2\sum_{i=1}^{D}f_i'P_i' \tag{15}$$
$$f_i'=\frac{1}{|\mathcal E_i|}\sum_{j\in\mathcal E_i}f_j \tag{16}$$
$$P_i'=\sum_{j\in\mathcal E_i}P_j \tag{17}$$

**ACL 2024 版**只有一个损失：$\mathcal L_{Bal}=\alpha\sum_{i=1}^{N'}f_iP_i$ (12)，(13)(14) 同上，**无设备级损失**。

### 2.7 课程记号对照表（DeepSeekMoE ↔ V2/V3 ↔ 课程）

| 论文（DeepSeekMoE） | DeepSeek-V2/V3 | 课程写法 | 说明 |
|---|---|---|---|
| $\mathbf u_t^l$ | $\mathbf u_t$ | $\mathbf u_t\in\R^{d}$ | FFN/MoE 的输入（注意力 + 残差之后） |
| $\mathbf h_t^l$ | $\mathbf h_t'$ | $\mathbf h_t'$ 或 $\mathbf h_t^{l}$ | MoE 层输出（含残差）；DeepSeekMoE 用 $\mathbf h_t^l$，V2 用 $\mathbf h_t'$，课文需说明二者同指 |
| $N$（切分前的标准专家数）、$K$、$m$ | — | 仅在“细粒度切分”推导中用 $N,K,m$ | **与全局符号表的 $N$（参数量）冲突**：本课开头声明“本节 $N$ 指专家数”，推导结束后改用 $N_r$ |
| $K_s$ | $N_s$ | $N_s$ | 共享专家数（总是激活，所以“激活的共享专家数”= $N_s$） |
| $mN-K_s$（= $N'$） | $N_r$ | $N_r$ | 路由专家数 |
| $mK-K_s$（= $K'$） | $K_r$ | $K_r$ | 每 token 激活的路由专家数 |
| $s_{i,t}$ | $s_{i,t}$ | $s_{i,t}$ | 亲和度（DeepSeekMoE/V2 为 softmax；V3 为 sigmoid） |
| $g_{i,t}$ | $g_{i,t}$ | $g_{i,t}$ | 门值；DeepSeekMoE/V2 = 未归一化的 $s_{i,t}$ |
| $\mathbf e_i^l$ | $\mathbf e_i$ | $\mathbf e_i\in\R^{d}$ | 专家中心向量 = 路由矩阵第 $i$ 行 |
| $T$ | $T$ | $T$ | 序列长度（DeepSeekMoE §2 定义）；Switch 的 $T$ 是 batch token 数 |
| $\alpha_1,\alpha_2$ | $\alpha_1,\alpha_2,\alpha_3$ | $\alpha_1,\alpha_2$ | 专家级、设备级系数（V2 另有通信均衡 $\alpha_3$，第 10 课） |
| Shazeer $G(x)_i$ / Switch $p_i(x)$ / GShard $g_{s,e}$ / Mixtral $G(x)_i$ | | $s_{i,t}$ 或 $g_{i,t}$ | 必须逐篇说明“softmax 在 top-k 前还是后” |
| Switch $f_i$ / GShard $c_e/S$ | $f_i$ 定义不同 | 课文区分 $\tilde f_i$（比例）与 $f_i$（DeepSeek 归一化版） | $f_i^{DS}=\frac{N_r}{K_r}\tilde f_i^{\,sel}$，见 §4-R10 |

---

## 3. 规格与超参数表

### 3.1 DeepSeekMoE 三个规模（论文 Table 7 为主，单元格注明来源）

| 项 | 2B（验证） | 16B | 145B（初步） |
|---|---|---|---|
| 总参数 | 2.0B（T7） | 16.4B（T7；复算 16.376B） | 144.6B（T7） |
| 激活参数 | ~0.3B（§4.1.3） | ~2.8B（§5.1.2；复算 2.829B，含 embedding 与输出头） | ~22.2B（§7.1） |
| 层数 | 9（T7） | 28（T7；config `num_hidden_layers` 28） | 62（T7） |
| 隐藏维 $d$ | 1280（T7） | 2048（T7；config 2048） | 4096（T7） |
| 注意力 | MHA 10 头 × 128（§4.1.3） | MHA 16 头 × 128（§5.1.2；config 16/16） | MHA 32 头 × 128（§7.1） |
| 共享专家 $N_s$ | 1（T7） | 2（T7；config `n_shared_experts` 2） | 4（T7） |
| 路由专家 $N_r$（激活 $K_r$） | 63（7）（T7） | 64（6）（T7；config 64 / `num_experts_per_tok` 6） | 128（12）（T7） |
| 相对专家大小 | 0.25（T7） | 0.25（T7；config `moe_intermediate_size` 1408 ⇒ 标准 FFN 5632） | 0.125（T7） |
| 首层 | 全部 FFN 换 MoE（§4.1.3） | 首层为稠密 FFN（§5.1.2；config `first_k_dense_replace` 1，`intermediate_size` 10944） | 首层稠密（§7.1） |
| 序列长度 | 2048（T7） | 4096（T7；config `max_position_embeddings` 4096） | 4096（T7） |
| Batch（序列） | 2048（T7） | 4608（T7；正文 “4.5K”） | 4608（T7） |
| 峰值 LR | 1.08e-3（T7） | 4.2e-4（T7） | 3.0e-4（T7） |
| LR 调度 | warmup 2K + 80%/90% 处 ×0.316 | 同左 | warmup 2K + 常数 |
| 训练 tokens / 步数 | 100B / 25,000 | 2T / 106,449 | 245B / 13,000 |
| 词表 | 8K（§4.1.1） | “100K”（§5.1.1）；config 102,400 | 同 16B（§7.1） |
| init std | 0.006 | 0.006（config `initializer_range` 0.02，见 §7-E15） | 0.006 |
| $\alpha_1$（专家级） | 0.01 | 0.001（config 默认 `aux_loss_alpha` 0.001） | 0.003 |
| $\alpha_2$（设备级） | 不用 | 不用 | 0.05（4 台设备） |
| token 丢弃 | 不丢 | 不丢 | 未说明 |
| 以标准 FFN 计的专家总量 / 激活量 | 64×0.25 = 16 / 8×0.25 = 2 | 66×0.25 = 16.5 / 8×0.25 = 2 | 132×0.125 = 16.5 / 16×0.125 = 2 |

### 3.2 DeepSeekMoE 16B：config.json 与代码（`hf_dsmoe16b`）

| 字段 | 值 | 与论文的关系 |
|---|---|---|
| `hidden_size` | 2048 | 一致 |
| `num_hidden_layers` | 28 | 一致 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 16 | MHA，一致 |
| `n_routed_experts` / `n_shared_experts` / `num_experts_per_tok` | 64 / 2 / 6 | 一致 |
| `moe_intermediate_size` | 1408 | 论文只说 0.25×标准 FFN；1408×4 = 5632 = 2.75 $d$ |
| `intermediate_size`（首层稠密 FFN） | 10944 | 论文未给；**不等于** 5632，也不等于 8×1408 = 11264 |
| `first_k_dense_replace` / `moe_layer_freq` | 1 / 1 | 首层稠密，其余全 MoE，一致 |
| `scoring_func` | softmax | 一致 |
| `norm_topk_prob` | false | 与式 (10) 一致：门值不重新归一化 |
| `aux_loss_alpha` | base 版未写（代码默认 0.001）；chat 版显式 0.001 | 与 §5.1.2 一致 |
| `seq_aux` | base 版未写（代码默认 True）；chat 版显式 True | 按序列计算 $f_i,P_i$ 后对 batch 取平均 |
| `vocab_size` | 102400 | 论文 “100K” |
| `max_position_embeddings` | 4096 | 一致 |
| `rope_theta` | 10000 | 论文未提 |
| `hidden_act` | silu（SwiGLU 三矩阵 gate/up/down） | 论文未提 |
| `initializer_range` | 0.02 | 论文 0.006（见 §7-E15） |
| `tie_word_embeddings` | false | 输入 embedding 与输出头各 0.21B |

代码事实（`modeling_deepseek.py`）：
- `MoEGate.forward`：`scores = logits.softmax(dim=-1)`；`torch.topk(scores, k=self.top_k, dim=-1, sorted=False)`；仅当 `self.top_k > 1 and self.norm_topk_prob` 才除以和。
- 非 `seq_aux` 分支：`ce = mask_ce.float().mean(0)`（对 $T\cdot K_r$ 行取平均），`fi = ce * self.n_routed_experts`，`aux_loss = (Pi * fi).sum() * self.alpha` ——与式 (12)–(14) 完全一致（$\text{ce}_i=\text{count}_i/(K_rT)$）。
- `seq_aux` 分支：对每条序列算 `ce`（`div_(seq_len * aux_topk / self.n_routed_experts)`）与 `scores.mean(dim=1)`，求和后对 batch `mean()`。
- 共享专家：`intermediate_size = config.moe_intermediate_size * config.n_shared_experts`，即**一个**中间维 2816 的 MLP（与两个 1408 专家相加数学上等价，§4-R12）。
- 辅助损失通过自定义 `AddAuxiliaryLoss` autograd 函数挂到输出上（前向恒等，反向注入 aux loss 梯度）。
- 路由器无偏置、无噪声；`reset_parameters` 用 `kaiming_uniform_`（只影响随机初始化）。

### 3.3 经典 MoE 对照表

| 模型 | 年份 | 专家数 / 每 token 激活 | 门控 | MoE 层位置 | 均衡手段 | 容量/丢弃 | 规模（来源） |
|---|---|---|---|---|---|---|---|
| Jacobs et al. | 1991 | 4 或 8（实验）/ 全部（稠密混合） | softmax，随机单选的概率解释 | 整个模型就是一个 MoE | 无 | 无 | 玩具任务（Table 1） |
| Shazeer et al. | 2017 | 最多 131,072（层级 MoE）/ LM 与单语对 MT：k=4（平坦）或每级 k=2（Appendix C、E）；多语 MT：平坦 n=512、k=2（Appendix E） | Noisy Top-K：KeepTopK 后 softmax | LSTM 层之间 | $w_{imp}\,CV^2$ + $w_{load}\,CV^2$；LM 0.1/0.1，MT 0.01/0.01 | “strictly-balanced gating”（MT，Appendix F，未细读） | MoE 层最多 137B 参数（Abstract） |
| GShard | 2020 | 128/512/2048 每层 / top-2（第二专家随机派发） | softmax → top-2 → 重新归一化 | 每隔一层 FFN | $\ell_{aux}=\frac1E\sum\frac{c_e}{S}m_e$，系数 $k$ 未给 | 每组容量 $2N/(GE)$，溢出走残差 | 600B（2048E, 36L），1T 未报告结果 |
| Switch | 2021 | 128（Base/Large）、64（XXL）、2048（C）/ top-1 | softmax（全部 N）→ top-1，不归一化 | Base/Large/XXL 每隔一层；C 每层 | $\alpha N\sum f_iP_i$，$\alpha=10^{-2}$ | CF 1.0–2.0，溢出走残差，典型丢弃 <1% | Switch-C 1571B（Table 9） |
| ST-MoE | 2022 | 32 / top-2 | 同 Switch 类 | 每 4 层 FFN 一个 | $\alpha=10^{-2}$ + router z-loss $c_z=10^{-3}$ | train CF 1.25，eval CF 2.0 | 269B，FLOP 相当于 32B 稠密 |
| Mixtral 8x7B | 2024-01 | 8 / 2 | Softmax(Top2(logits)) | 每层 | 论文未说明（HF config `router_aux_loss_coef` 0.02） | 论文未说明 | 46.7B / 12.9B（复算；论文 47B/13B） |
| DeepSeekMoE 16B | 2024-01 | 2 共享 + 64 路由 / 2 + 6 | softmax（全部路由）→ top-6，不归一化 | 除首层外每层 | $\alpha_1=0.001$ 专家级 | 不丢 token | 16.4B / 2.8B |
| DeepSeek-V2-Lite | 2024-05 | 2 + 64 / 2 + 6 | 同上（config `topk_method` greedy） | 除首层外每层（27 层） | $\alpha_1=0.001$ | — | 15.7B / 2.4B（论文） |

### 3.4 Mixtral 8x7B（论文 Table 1 与 config.json 一致）

| 字段 | 论文 Table 1 | config.json |
|---|---|---|
| dim | 4096 | `hidden_size` 4096 |
| n_layers | 32 | 32 |
| head_dim | 128 | （4096/32） |
| hidden_dim | 14336 | `intermediate_size` 14336 |
| n_heads / n_kv_heads | 32 / 8 | 32 / 8（GQA） |
| context_len | 32768 | `max_position_embeddings` 32768 |
| vocab_size | 32000 | 32000 |
| num_experts / top_k | 8 / 2 | `num_local_experts` 8 / `num_experts_per_tok` 2 |
| 其它 | — | `rope_theta` 1e6，`router_aux_loss_coef` 0.02，`sliding_window` null |

---

## 4. 复算的数字（全部 Python 重算）

**R1 组合数（C30 核对）**
- `comb(16,2)` = **120** ✓；`comb(64,8)` = **4,426,165,368** ✓；比值 `comb(64,8)/comb(16,2)` = 36,884,711.4。
- 切分粒度 $m$ 下（$N=16,K=2$）：`comb(16m,2m)`，$m=1..8$ → 120；35,960；12,271,512；4,426,165,368；1,646,492,110,120；624,668,654,531,480；240,260,199,935,164,200；93,343,021,201,262,177,400。`log10` 分别为 2.079、4.556、7.089、9.646、12.217、14.796、17.381、19.970。
- 真实模型（只数路由专家的无序集合）：2B `comb(63,7)` = 553,270,671；16B `comb(64,6)` = 74,974,368；145B `comb(128,12)` = 23,726,045,489,546,400；142B-half `comb(128,6)` = 5,423,611,200；Mixtral `comb(8,2)` = 28；GShard 2048 专家 top-2 `comb(2048,2)` = 2,096,128；V2 `comb(160,6)` = 21,193,254,160；V3 `comb(256,8)` = 409,663,695,276,000（V2/V3 的 160/6 与 256/8 取自课程地图待第 10/11 课核对，这里只是算术）。
- 消融四配置：GShard `comb(16,2)`=120；+共享 `comb(15,1)`=15；+细粒度 `comb(31,3)`=4,495；+更细 `comb(63,7)`=553,270,671。**注意**：加共享专家那一步组合数从 120 降到 15，但性能反而上升——组合数不是性能的充分解释（练习素材）。

**R2 切分不改变参数与计算（SwiGLU，$d=2048$，标准 FFN 中间维 5632）**
`3*2048*(5632//m)*m` 对 $m=1,2,4,8$ 都 = 34,603,008；每专家参数 34,603,008 / 17,301,504 / 8,650,752 / 4,325,376。路由器参数 `16*m*2048` = 32,768 / 65,536 / 131,072 / 262,144——**随 $m$ 线性增长**（但与专家参数相比可忽略）。

**R3 DeepSeekMoE 16B 参数复算（config.json）**
- 每专家 `3*2048*1408` = 8,650,752；每 MoE 层总 `66*8650752 + 64*2048` = 571,080,704；激活 `8*8650752 + 64*2048` = 69,337,088。
- 总参数 = `2*102400*2048 + 2048 + 28*(4*2048**2 + 2*2048) + 3*2048*10944 + 27*571080704` = **16,375,728,128 ≈ 16.4B** ✓。
- 激活参数（含输入 embedding 与输出头）= **2,828,650,496 ≈ 2.8B** ✓；去掉输入 embedding 2.619B；去掉 embedding 与输出头 2.409B。
- 注意力参数 `28*4*2048**2` = 469,762,048 ≈ **0.47B**（论文 “about 0.5B” ✓）。
- 参数构成（占总参数）：路由专家 14.9485B（91.28%）、共享专家 0.4671B（2.85%）、注意力 0.4698B（2.87%）、embedding 0.2097B（1.28%）、输出头 0.2097B（1.28%）、首层稠密 FFN 0.0672B（0.41%）、路由器 0.0035B（0.02%）。
- 激活参数构成：路由专家 1.4014B（49.54%）、注意力 16.61%、共享专家 16.51%、embedding 7.41%、输出头 7.41%、首层 FFN 2.38%、路由器 0.13%。

**R4 DeepSeek 7B / LLaMA2 7B / DeepSeek 67B（公开 config）**
- DeepSeek 7B（30 层、4096、MHA、11008、词表 102400）：总 6.910B ✓（论文 6.9B）；注意力 `30*4*4096**2` = **2.013B**（论文写 2.5B，✗ 见 §7-E13）。
- LLaMA2 7B（32 层、4096、11008、词表 32000）：6.738B ✓（论文 6.7B）。

**R5 计算量比值（C39、C42 核对）**
- `74.4/183.5` = 0.4054 → 40.5% ✓；`74.4/187.9` = 0.3960 → 39.6% ✓；`16.4/6.7` = 2.448 → 245% ✓。
- `585.6/2057.5` = 0.2846 → 28.5% ✓；`374.6/2057.5` = 0.1821 → 18.2% ✓；`572.7/2057.5` = 0.2783（GShard 137B）。
- 激活参数比（对照，**不是**论文 40% 的来源）：`2.8/6.7` = 0.418，`6.7/2.8` = 2.39（论文 “approximately 2.5 times the activated parameters”）；`6.9/2.8` = 2.46。
- 2B：`4.3/5.8` = 0.741（DeepSeekMoE 是 GShard×1.5 的 74% 计算）；`4.3/24.6` = 0.175（Dense×16 的 17.5%）；`2.9/4.3` = 0.674。
- `144.6/136.5` = 1.0593 → “6% larger” ✓。

**R6 论文 FLOPs 列的重建（笔者推测的公式，非论文给出）**
假设每 token 训练 FLOPs ≈ $6\,(N_{act}^{non\text{-}emb}+Vd)+12\,L\,d\,s$，其中注意力投影一律按 $4d^2$/层计（即使 67B 是 GQA），$s$ 为序列长度：
- 2B 表（$s=2048$，$V=8000$ 或 8192，标准 FFN 按 $2\cdot1280\cdot5120$ 参数）：DeepSeekMoE/GShard 4.33T（表 4.3T）、Switch/Hash/Dense 2.88T（2.9T）、GShard×1.5 5.78T（5.8T）、Dense×16 24.62T（24.6T）——全部吻合。
- DeepSeek 7B 44.98 vs 44.80 GF/token（+0.4%）；LLaMA2 7B 46.08 vs 45.87（+0.5%）；DeepSeek 67B（按 MHA 计注意力）504.7 vs 502.3（+0.5%）；145B（专家中间维取 1408 推断值）143.1 vs 143.0（+0.1%）；142B-half 92.5 vs 91.5（+1.1%）；**16B 18.53 vs 18.16（+2.0%，残差未解释）**。
- 课文若用此公式，必须标“笔者重建，论文未给出公式”。

**R7 训练 token 数**
- 16B：`4608*4096` = 18,874,368 tokens/batch（“18M”）；×106,449 = 2.009T ✓。
- 2B：`2048*2048` = 4,194,304（“4M”）；×25,000 = 104.9B（“100B”）✓。
- 145B：`18874368*13000` = 245.4B ✓。
- 学习率阶梯：`0.316**2` = 0.0999；`1/sqrt(10)` = 0.3162（0.316 即 $10^{-1/2}$，两次后降到约 10%，与 DeepSeek LLM 的 “31.6% … 10%” 一致）。

**R8 显存**
- 16B 权重 BF16：`16375728128*2` = 32.75 GB（30.50 GiB）→ 放得进 40GB 单卡 ✓，余约 7 GB 给 KV cache 与激活。
- 16B KV cache（MHA，BF16）：`2*16*128*28*2` = 229,376 B/token = 224 KiB/token；4K 上下文 0.94 GB/序列。
- V2-Lite（MLA，$d_c=512$，$d_h^R=64$）：`(512+64)*27*2` = 31,104 B/token；16B : V2-Lite = 7.37×（第 9 课素材）。
- Mixtral（GQA 8 组）：`2*8*128*32*2` = 131,072 B/token。

**R9 Mixtral 参数复算（config）**
注意力/层 `4096*4096 + 2*4096*1024 + 4096*4096` = 41,943,040；专家 `3*4096*14336` = 176,160,768；总 = **46.70B**（论文 47B ✓）；激活 = **12.88B**（论文 13B ✓）；字面的 “8×7B” = 56B、或“8 个同配置稠密模型”（按同一 config 只留 1 个专家复算为 7.24B，×8 = 57.9B）都是错的，因为注意力和 embedding 在专家之间共享。

**R10 均衡损失的取值范围**
- Switch：均匀时 $\alpha N\cdot N\cdot\frac1N\cdot\frac1N=\alpha$；全部坍缩到 1 个专家时 $\alpha N$（$N=128$ 时 128α）。
- DeepSeekMoE（式 12–14）：$\sum_i f_i = N'$；均匀时 $f_i=1,P_i=1/N'$ ⇒ $\mathcal L=\alpha_1$；坍缩到固定 $K'$ 个专家时 $f_i=N'/K'$、$P_i\to 1/K'$ ⇒ $\alpha_1N'/K'$：2B 9α、16B 10.667α、145B 10.667α、（N=16,K=2 时 8α）。
- 线性插值 $\lambda$（$\lambda$ 比例的 token 坍缩到固定 6 个专家，$N'=64$）：λ = 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1 → $\mathcal L/\alpha_1$ = 1.0000, 1.0967, 1.6042, 3.4167, 6.4375, 8.8300, 10.6667。
- 与 Switch 的关系：若 $K'=1$，$f_i^{DS}=N'\tilde f_i$，$\mathcal L_{ExpBal}=\alpha_1N'\sum\tilde f_iP_i$，**正是** Switch 式 (4)。DeepSeekMoE 的 $1/K'$ 把“每 token 选 $K'$ 个”带来的计数放大抵消掉。
- GShard：均匀时 $\ell_{aux}=\frac1E\cdot E\cdot\frac1E\cdot\frac1E=1/E^2$；Switch 损失 $=\alpha E^2\,\ell_{aux}$（同一 $f,P$ 下）。
- 梯度：$\partial\mathcal L_{ExpBal}/\partial P_i=\alpha_1 f_i$——负载越高的专家，其平均概率被压得越狠；$f_i$ 本身不可导。
- 设备级均匀时：$f_i'=1$，$P_i'=1/D$ ⇒ $\mathcal L_{DevBal}=\alpha_2$。

**R11 其它小数字**
- 共享专家被禁用：`2.414-1.808` = 0.606（+33.5%）。
- Jacobs Table 1：`1124/2209` = 0.509，`1083/2435` = 0.445（“about half”）。
- Shazeer Table 6：`39.8/35.6` = 1.118；`17.80/1.14` = 15.6。若把 CV(Importance)=3.04 解释为“均匀落在 $k_{eff}$ 个专家上”，则 $CV^2=n/k_{eff}-1$ ⇒ $k_{eff}=256/(1+3.04^2)$ = 25.0（笔者的解释性换算，n=256）；完全坍缩到 k=4 个专家时 CV = √63 = 7.94。
- DeepSeekMoE 145B 与 GShard 137B：$4+128$ 个 0.125 专家 = 16.5 个标准 FFN；GShard 16 个。

**R12 共享专家“2 个”与“1 个双倍宽”的等价**
SwiGLU $\mathrm{FFN}(\mathbf u)=W_2\big(\mathrm{SiLU}(W_1\mathbf u)\odot W_3\mathbf u\big)$。两个专家相加 = 把 $W_1,W_3$ 按行、$W_2$ 按列拼接成中间维 2816 的一个 FFN（逐元素运算不跨中间单元）。代码正是这样实现（§3.2）。数值：`3*2048*2816` = 17,301,504 = `2*8650752` ✓。

**R13 例题数字（§6 用）**
- Switch 损失：$N=4$、$T=8$，argmax 计数 (5,1,1,1)，$P=(0.55,0.15,0.15,0.15)$ ⇒ $f=(0.625,0.125,0.125,0.125)$，loss/α = `4*(0.625*0.55+3*0.125*0.15)` = **1.6**（均匀为 1.0）。
- DeepSeekMoE 专家级：$N'=8,K'=2,T=4$，logits 行 1–4 = (2,1.5,0.5,0,0,0,0,−0.5)、(2,0.2,1.8,0,0,0,−0.3,0)、(1.6,1.4,0,0.2,0,0,0,0)、(0,0,0.1,1.2,1.1,0.3,0,0)。softmax 后 top-2（从 1 数）= {1,2},{1,3},{1,2},{4,5}；计数 (3,2,1,1,1,0,0,0)；$f=\frac{8}{2\cdot4}\cdot$计数 = (3,2,1,1,1,0,0,0)，$\sum f=8$ ✓；$P$ = (0.2980, 0.1637, 0.1387, 0.1117, 0.1019, 0.0695, 0.0593, 0.0572)；$\mathcal L/\alpha_1$ = **1.5736**。各 token 的路由门值之和（未归一化）= 0.6549, 0.6927, 0.5915, 0.4949。
- 设备级（$D=2$，专家 1–4 / 5–8）：$f'$ = (1.75, 0.25)，$P'$ = (0.7121, 0.2879)，$\mathcal L_{DevBal}/\alpha_2$ = **1.3181**。
- 门控顺序（8 专家 logits (2,1,0.5,0.2,0,−0.3,−0.5,−1)，k=2）：全 softmax = (0.4709, 0.1732, 0.1051, 0.0778, 0.0637, 0.0472, 0.0387, 0.0234)；softmax→top2 不归一化（Switch/DeepSeekMoE）门值 (0.4709, 0.1732)，和 0.6441；top2→softmax（Shazeer/Mixtral）= (0.7311, 0.2689)；GShard 重新归一化结果**与 top2→softmax 完全相同** (0.7311, 0.2689)。k=1 时：Switch 门值 0.4709（有梯度），KeepTopK→softmax 门值恒为 1（对路由器无梯度）。
- 容量：$T=1024$，8 专家，CF 1.0/1.25/2.0 ⇒ 容量 128/160/256；负载 (200,150,130,120,120,110,100,94)（和 1024）在 CF 1.25 下丢 40 个（3.9%），CF 1.0 下丢 96 个（9.4%）；GShard top-2 每组 $2S/E$（S=1024, E=8）= 256。
- Noisy top-k 概率：干净 logit 1.0、排除自身后的第 k 大 1.3、噪声 logit 0 ⇒ softplus(0) = 0.6931，$P=\Phi(-0.3/0.6931)$ = **0.3326**。
- CV²（总体标准差）：(1,1,1,1) → 0；(4,0,0,0) → 3.0（样本标准差则为 4.0）；(2,1,0.5,0.5) → 0.375（0.5）；(1.3,0.9,0.9,0.9) → 0.03（0.04）。
- Jacobs (1.4)/(1.5)：两个专家 $p=(0.5,0.5)$，误差范数 1.5 与 3.0：(1.4) 梯度大小 $2p\|d-o\|$ = (1.5, 3.0)（差的专家学得快）；(1.5) 责任 = (0.9669, 0.0331)，梯度大小 = (1.4504, 0.0993)（好的专家学得快）。

---

## 5. 图表清单与重绘建议

### 5.1 原论文图表清单

| 来源 | 图/表 | 标题要点 | 内容 | 本课用途 |
|---|---|---|---|---|
| jacobs1991 | 图 1 | “A system of expert and gating networks.” | 3 个专家 + 门控网络 + “Stochastic one-out-of-n selector” | 重绘为 MoE 起源图 |
| jacobs1991 | 图 2 | 元音数据与决策线 | F1/F2 散点、Net 0/1/2 决策线、Gate 0:2 | 可选（说明“专家分工”） |
| jacobs1991 | Table 1 | 元音任务收敛 epoch | 4/8 专家 vs BP 6/12 隐单元 | 算例 |
| shazeer2017 | 图 1 | MoE 嵌在循环语言模型中 | 门控选 2 个专家，输出乘门值 | 重绘（LSTM 之间） |
| shazeer2017 | Table 6（App. A） | 不同损失组合 | ppl、CV、max/mean load | 画均衡损失效果图 |
| gshard2020 | 图 3 | Transformer 编码器每隔一层换 MoE、分片到多设备 | (a)(b)(c) | 重绘“每隔一层” |
| gshard2020 | Algorithm 1 | 组级 top-2 门控 + aux loss | 伪代码 | 转成流程图 |
| gshard2020 | Table 1 | MoE Transformer 家族 | 专家数、层数、参数 | 时间线数据（注意两处印刷错误 §7-E9） |
| switch2021 | 图 2 | Switch 层示意（top-1） | — | 可合并入 top-1 讲解 |
| switch2021 | 图 3 | “Illustration of token routing dynamics.” | 容量因子 1.0 与 1.5 下的溢出与填充 | **重绘**（容量因子） |
| switch2021 | Table 1 | Switch vs MoE 不同 CF | 见 C24 | 表格 |
| switch2021 | Table 9 | 模型规格 | Switch-C 1571B | 时间线数据 |
| stmoe2022 | Table 4 | z-loss 稳定性 | 见 C27 | 选讲 |
| mixtral2024 | 图 1 | 8 选 2 的 MoE 层 | — | 对照图 |
| mixtral2024 | 图 7 | 各领域专家分配比例（第 0、15、31 层） | 分布相近 | 讨论“是否专精” |
| dsmoe2024 | 图 1（**ACL 图 7**，附录 E） | Open LLM Leaderboard：激活参数 vs 平均分 | 红色虚线为除 16B 外的线性拟合 | 可选 |
| dsmoe2024 | **图 2**（ACL 图 1） | “Illustration of DeepSeekMoE.” | (a) 1..N、K=2；(b) 1..2N、K=4；(c) 1 共享 + 2..2N 路由、K=3 | **核心重绘** |
| dsmoe2024 | 图 3（ACL 图 2） | 消融（归一化性能） | 6 个基准 × 4 配置 | 只知图例，不知数值（§8） |
| dsmoe2024 | 图 4（ACL 图 3） | 禁用 top 路由专家比例 vs Pile loss | x: 0, 1/16, …, 4/16；y 轴刻度 2–8 | 数值不可得，只能示意 |
| dsmoe2024 | 图 5（ACL 图 4） | 激活路由专家数 3–7 vs Pile loss | y 刻度 1.85–1.95，GShard 水平线 | 数值不可得 |
| dsmoe2024 | 图 6（ACL 图 5） | GShard vs 半激活 DeepSeekMoE | 6 个基准柱状（y 0–70） | 数值不可得 |
| dsmoe2024 | 图 7（App. C；ACL 图 6，附录 D） | 16B vs DeepSeek 7B 训练中基准曲线 | x: 0–2000B tokens | 可选 |
| dsmoe2024 | Table 1/2/3/4/6/7/8/9/10 | 见 §1、§3 | 数值齐全 | 柱状/散点图数据来源 |

注：arXiv v1 的 LaTeXML 渲染**没有**图片（构建失败），DeepSeekMoE 图的标签取自 ACL PDF 文本层；曲线/柱的数值无法从文本层取得。

### 5.2 建议本课绘制的图（13 张，每张给出可直接计算几何的数据）

**图 7.1 MoE 演进时间线**（横轴年份，纵轴 log₁₀ 总参数）
数据：1991 Jacobs（4–8 专家，稠密，参数量可忽略，画在底部不标数值）；2017 Shazeer（MoE 层 137B，131,072 专家，k=4 / 层级每级 2）；2020 GShard（600B，2048 专家/层，top-2）；2021 Switch-C（1571B，2048 专家，top-1）；2022 ST-MoE-32B（269B，32 专家，top-2）；2024-01 Mixtral（46.7B / 12.9B，8 选 2）；2024-01 DeepSeekMoE 16B（16.4B / 2.8B，2 + 64 选 6）。颜色：激活路由专家 `--fig-blue`，共享 `--fig-violet`。

**图 7.2 Jacobs 1991 结构图 + 两种误差的梯度权重**（据 Jacobs 图 1 重绘）
左：3 个 Expert Network + Gating Network + 随机选择器（$p_1,p_2,p_3$）。右：两专家 $p=(0.5,0.5)$、误差 1.5 / 3.0 的梯度大小柱：式 (1.4) (1.5, 3.0)；式 (1.5) (1.4504, 0.0993)。

**图 7.3 门控的三种顺序**（数值例：R13 的 8 个 logits，k=2）
三列柱状：①softmax 全体 → 取 top-2 不归一化（Switch/DeepSeekMoE）：(0.4709, 0.1732)，其余 0，和 0.6441；②top-2 → softmax（Shazeer/Mixtral）：(0.7311, 0.2689)；③softmax → top-2 → 重新归一化（GShard）：同②。底注：k=1 时①门值 0.4709、②恒为 1。

**图 7.4 容量因子与 token 丢弃**（据 Switch 图 3 重绘，改用计算数据）
8 个专家柱高 = (200,150,130,120,120,110,100,94)；水平线 CF 1.0 → 128、1.25 → 160、2.0 → 256；超出部分 `--fig-red`（CF 1.25 时共 40 个，全在专家 1；CF 1.0 时 96 个：72+22+2）；空槽（容量−负载，CF 2.0）用浅灰表示“浪费的计算”。

**图 7.5 DeepSeekMoE 四联结构图**（据 DeepSeekMoE 图 2 / ACL 图 1 重绘，加第 0 联）
(0) 稠密 FFN（1 个宽块）；(a) 传统 top-2：专家 1..N，K=2；(b) 细粒度：1..2N，K=4（每块宽度减半）；(c) + 共享专家：1 个共享（紫）+ 2N−1 路由，路由激活 K=3。论文原图用 m=2、K_s=1；课文可再给一版“2B 实例”：16 专家 top-2 → 64 专家 top-8 → 1 共享 + 63 路由 top-7。参数条注：每联专家总宽度相同、激活总宽度相同。

**图 7.6 组合数随切分粒度的爆炸**（log 纵轴）
$m=1..8$：log₁₀ C(16m,2m) = 2.079, 4.556, 7.089, 9.646, 12.217, 14.796, 17.381, 19.970；标注 m=4 点 4,426,165,368；另用散点标实际模型：Mixtral 28（log 1.447）、GShard-2048 2,096,128（6.321）、16B 74,974,368（7.875）、2B 553,270,671（8.743）。

**图 7.7 参数守恒条形图**（R2）
m=1,2,4,8：专家中间维 5632/2816/1408/704；单专家参数 34.6M/17.3M/8.65M/4.33M；专家数 16/32/64/128；总专家参数恒 16×34.6M = 553.6M；路由器参数 32,768→262,144（用第二纵轴或文字标注）。

**图 7.8 均衡损失的“账本”**（R13 例题数据）
上：8 个专家的 $f_i$ = (3,2,1,1,1,0,0,0) 与 $P_i$ = (0.2980,0.1637,0.1387,0.1117,0.1019,0.0695,0.0593,0.0572) 并排柱；下：梯度 $\partial\mathcal L/\partial P_i=\alpha_1 f_i$ 的箭头（向下，长度 ∝ f_i）；右侧设备级：两台设备 $f'$ = (1.75,0.25)，$P'$ = (0.7121,0.2879)。

**图 7.9 辅助损失随坍缩程度变化**（R10 插值曲线）
$N'=64,K'=6$：λ 与 $\mathcal L/\alpha_1$：(0,1.0000)(0.1,1.0967)(0.25,1.6042)(0.5,3.4167)(0.75,6.4375)(0.9,8.8300)(1,10.6667)；可在 Python 中用公式 $K'[(1-λ)+λN'/K'][(1-λ)/N'+λ/K']+(N'-K')(1-λ)^2/N'$ 连续画出。标注两端：均匀 = α₁，完全坍缩 = α₁N'/K'。

**图 7.10 2B 验证实验：计算量 vs Pile loss 散点**（Table 1、2、8）
(FLOPs/2K tokens, Pile loss)：Dense 0.2B (2.9T, 2.060)；Hash Layer (2.9T, 1.932)；Switch (2.9T, 1.881)；GShard (4.3T, 1.867)；DeepSeekMoE (4.3T, 1.808)；GShard×1.5 (5.8T, 1.808)；Dense×16 (24.6T, 1.806)。GShard×1.2 Pile 1.824（Table 8 未给 FLOPs；按 R6 公式约为 GShard 与 GShard×1.5 之间，若画需标“估算”或不画）。横轴 log。

**图 7.11 DeepSeekMoE 16B vs DeepSeek 7B vs LLaMA2 7B**（Table 3/4；分组柱状，选 10 项）
FLOPs/4K tokens：187.9T / 183.5T / 74.4T（LLaMA2 / DS7B / DSMoE16B）。基准（LLaMA2, DS7B, 16B）：HellaSwag 75.6/75.4/77.1；PIQA 78.0/79.2/80.2；ARC-c 49.0/48.1/49.8；GSM8K 15.5/17.4/18.8；HumanEval 14.6/26.2/26.8；MBPP 21.8/39.0/39.2；TriviaQA 63.8/59.7/64.8；NaturalQuestions 25.5/22.2/25.5；MMLU 45.8/48.2/45.0；CEval 33.9/45.0/40.6；Pile BPB 0.76/0.75/0.74。用 `--fig-green` 标 16B 胜出、`--fig-red` 标落后（MMLU、CEval 相对 DS7B）。

**图 7.12 145B 初步结果：计算量 vs Pile loss**（Table 6）
(FLOPs/4K tokens, Pile loss, 总/激活参数)：DeepSeek 67B (2057.5T, 1.905, 67.4B/67.4B)；GShard 137B (572.7T, 1.961, 136.5B/21.6B)；DeepSeekMoE 145B (585.6T, 1.876, 144.6B/22.2B)；DeepSeekMoE 142B half (374.6T, 1.888, 142.3B/12.2B)。附注：均只训练 245B tokens，“Ongoing”，ACL 版已删。

**图 7.13 16B 参数预算：总参数 vs 激活参数堆叠条**（R3）
总 16.376B：路由专家 14.9485、共享 0.4671、注意力 0.4698、embedding 0.2097、输出头 0.2097、首层 FFN 0.0672、路由器 0.0035；激活 2.829B：路由专家 1.4014，其余同上。视觉要点：总参数里 91% 是路由专家，激活参数里只一半是路由专家。

（可选）**图 7.14 Shazeer Table 6 均衡损失效果**：5 组 (w_imp, w_load) 的 ppl 与 max/mean load 双轴：(0,0) 39.8/17.80；(0.2,0) 35.6/1.47；(0,0.2) 35.7/1.15；(0.1,0.1) 35.6/1.14；(0.01,0.01) 35.7/1.37；(1,1) 35.7/1.07。
（可选）**图 7.15 Mixtral 层 vs DeepSeekMoE 16B 层（按 $d$ 归一的宽度）**：Mixtral 8 个宽 3.5d（14336/4096）专家选 2；DeepSeekMoE 2 个共享 + 64 个宽 0.6875d（1408/2048）专家选 6；激活 FFN 宽度：Mixtral 7.0d，DeepSeekMoE 8×0.6875 = 5.5d。

---

## 6. 算例与练习素材（附答案）

1. **组合数（计算）**：N=16、top-2 有多少种专家组合？切成 4 份后 top-8 呢？DeepSeekMoE 16B（64 路由选 6）呢？
   答：120；4,426,165,368；74,974,368。追问：为什么 16B 的组合数比 “64 选 8” 少得多？（共享专家拿走 2 个激活名额，且只从 64 个里选 6 个。）

2. **消融里的反例（概念）**：GShard（16 选 2，120 种）→ +1 共享（15 选 1，15 种），组合数下降 8 倍，论文图 3 却显示多数基准变好。这说明什么？
   答：组合灵活性不是唯一机制；共享专家的收益来自“公共知识不必在每个专家里重复存”（冗余论点），与组合数无关。组合数只是细粒度切分的动机之一。

3. **参数守恒（计算）**：$d=2048$，标准 SwiGLU FFN 中间维 5632。切成 m=4 份后每个专家多少参数？16 个标准专家切分后总参数变不变？路由器参数变多少？
   答：`3*2048*1408` = 8,650,752；总仍为 16×34,603,008 = 553,648,128；路由器从 16×2048 = 32,768 变为 64×2048 = 131,072（×4）。

4. **16B 参数（计算）**：用 config（28 层，d=2048，词表 102400，首层 FFN 10944，专家 1408，2+64 选 6，MHA，不绑权）复算总参数与激活参数。
   答：16,375,728,128 与 2,828,650,496（含 embedding 与输出头）；去掉两者的激活参数 2,409,220,096。

5. **均衡损失（计算）**：R13 的 4 个 token × 8 个专家例子，求 $f_i$、$P_i$、$\mathcal L_{ExpBal}/\alpha_1$；再按专家 1–4、5–8 分到两台设备求 $\mathcal L_{DevBal}/\alpha_2$。
   答：$f$ = (3,2,1,1,1,0,0,0)，$P$ 见 R13，1.5736；$f'$ = (1.75,0.25)，$P'$ = (0.7121,0.2879)，1.3181。

6. **损失的两端（推导）**：证明式 (13) 下 $\sum_i f_i=N'$；均匀路由时 $\mathcal L_{ExpBal}=\alpha_1$；全部 token 都选同一组 $K'$ 个专家、且这 $K'$ 个专家平分全部概率时 $\mathcal L=\alpha_1N'/K'$。16B 下是多少？
   答：$\sum_i\sum_t\mathbb 1=K'T$ ⇒ $\sum f_i=\frac{N'}{K'T}K'T=N'$；均匀：$\sum 1\cdot\frac1{N'}=1$；坍缩：$K'\cdot\frac{N'}{K'}\cdot\frac1{K'}=\frac{N'}{K'}$；16B：64/6 = 10.667，即 0.010667（α₁=0.001）。

7. **与 Switch 的关系（推导）**：证明 $K'=1$ 时式 (12)–(14) 退化为 Switch 式 (4)–(6)。并算 Switch 的例子：N=4，8 个 token 的 argmax 计数 (5,1,1,1)，$P=(0.55,0.15,0.15,0.15)$。
   答：$K'=1$ 时 $f_i^{DS}=N'\cdot\frac{\text{count}_i}{T}=N'\tilde f_i$，代入得 $\alpha_1N'\sum\tilde f_iP_i$；例子 loss/α = 1.6（均匀为 1）。

8. **门控顺序（计算 + 概念）**：logits (2,1,0.5,0.2,0,−0.3,−0.5,−1)，k=2。分别按 Switch/DeepSeekMoE、Mixtral、GShard 的规则算门值。k=1 时 Mixtral 式门控对路由器的梯度是多少？
   答：(0.4709, 0.1732)；(0.7311, 0.2689)；(0.7311, 0.2689)。k=1 时 Softmax(Top1) ≡ 1，对 $W_g$ 的梯度为 0，与 Shazeer “k>1 才有非零导数”的说法一致；而 Switch 式 (1)–(2) 的门值是对全部 N 个专家 softmax 后的 $p_i(x)$，k=1 时仍有梯度（Switch 原文：“permits differentiability of the router”）。注意：Switch **没有**说它是为了 top-1 的梯度才把 softmax 挪到 top-k 之前。它在 §2.1 转述 Shazeer 的 MoE 时就写成“先对 N 个 softmax，再选 top-k”，这个转述与 Shazeer 原文式 (3)（先 KeepTopK 再 Softmax）的顺序不同。课文不要把“调换顺序”说成 Switch 的设计动机。

9. **容量因子（计算）**：1024 个 token、8 个专家，负载 (200,150,130,120,120,110,100,94)。CF=1.0、1.25、2.0 时各丢多少 token？CF=2.0 时有多少“空槽”？
   答：96（9.4%）、40（3.9%）、0；CF 2.0 容量 256×8 = 2048 槽，空 1024 槽（一半计算被 padding 浪费）。

10. **共享专家等价（推导）**：证明两个 SwiGLU 共享专家之和等于一个中间维加倍的 SwiGLU；用 16B 的数字检查参数量。
    答：见 R12；`3*2048*2816` = 17,301,504 = 2×8,650,752。

11. **“40%”到底是什么（概念 + 计算）**：DeepSeekMoE 16B 相对 LLaMA2 7B 的“39.6%”是哪两个数之比？用激活参数比会得到多少？
    答：FLOPs per 4K tokens：74.4T / 187.9T = 0.396；激活参数比 2.8/6.7 = 0.418。两者不同，因为 FLOPs 还包括注意力的 $12Lds$ 项与输出头，且两个模型层数、词表不同。

12. **Jacobs 的梯度（计算）**：两个专家 $p=(0.5,0.5)$，误差范数 1.5 与 3.0。按式 (1.4) 与 (1.5) 计算两专家梯度大小，并说明哪个误差函数让“更合适的专家”学得更快。
    答：(1.4)：(1.5, 3.0)；(1.5)：责任 (0.9669, 0.0331)，梯度 (1.4504, 0.0993)。(1.5) 让好的专家学得快——这是“竞争/专精”的起点，与 DeepSeekMoE 追求的“专精”一脉相承。

---

## 7. 勘误与常见误解

**E1（门控顺序，最常见）。** “DeepSeekMoE 的门值是 top-K 上的 softmax，和为 1”——**错**。式 (4)(10) 是先对全部（路由）专家 softmax，再取 top-K，**不重新归一化**；config `norm_topk_prob: false`。R13 例子中每 token 路由门值之和 0.49–0.69。Shazeer（KeepTopK→Softmax）与 Mixtral（Softmax(TopK)）是和为 1 的；GShard 显式重新归一化（Algorithm 1 PDF 行 (6)(16)，数值与 Mixtral 式相同），但第二专家只以概率 $\min(1,2g_2)$ 派发、且受容量限制（行 (17)(19)），未派发时 $\mathcal G_s$ 里只剩 $g_1<1$，所以 GShard 的实际门值和**不一定**为 1；Switch 与 DeepSeekMoE 不归一化。DeepSeek-V3 改为 sigmoid + 归一化（C46）。

**E2（Switch 的“top-1 梯度”）。** 常见说法“top-1 路由不可导”。实际上 Switch 的门值 $p_i(x)$ 来自对全部 N 个专家的 softmax，所以 top-1 时路由器仍有梯度（C21 原文 “permits differentiability of the router”）；不可导的是 Shazeer 式“先 top-k 再 softmax”在 k=1 时的情形（门值恒为 1）。另注意“传话”问题：Switch §2.1 对 Shazeer MoE 的转述（式 (1)–(2)：softmax 全部 N 个后取 top-k）与 Shazeer 原文式 (3)（KeepTopK 后 softmax）顺序不同；讲门控顺序时应引 Shazeer 原文，不要引 Switch 的转述。

**E3（设备级损失的使用范围）。** “DeepSeekMoE 用专家级 + 设备级两个损失训练”——只有 **145B 初步实验**用了设备级（α₂=0.05，α₁=0.003）；2B（α₁=0.01）和 16B（α₁=0.001）都**没用**设备级损失，也不丢 token（C34、C38）。ACL 2024 版正文里**根本没有**设备级损失，只剩 $\mathcal L_{Bal}=\alpha\sum f_iP_i$。

**E4（版本漂移：arXiv v1 ↔ ACL 2024）。** ACL 版：删除 §6 对齐/Chat 与 §7 145B 两节；摘要末句改为 “comparable performance with DeepSeek 7B and LLaMA2 7B, with only about 40% of computations”，不再提 145B；Table 3/4 合并为 Table 2（三列）；共享专家原型致谢增加 Elbayad et al. (2023)（Findings ACL 2023, pp. 14237–14253）；新增 “Limitations and Future Work”（granularity scaling law、通信开销、16×/2× 参数比例的最优性）；图号：arXiv 图 2–7 = ACL 图 1–6（前移一位），但 **arXiv 图 1（Open LLM Leaderboard）移到 ACL 附录 E 成为图 7**，不是“整体前移”；arXiv 正文 Table 2（GShard×1.5 / Dense×16）移到 ACL 附录 C 成为 Table 3，arXiv Table 8/9 里的 GShard×1.2 与 Dense×4 在 ACL 版中不再出现；§5.2.1 “0.5B vs 2.5B attention parameters” 的括号数字在 ACL 版被删去。课文引用 145B 数字时必须写“arXiv v1 §7（初步、未进入正式发表版）”。

**E5（145B 的状态）。** 145B 只训练了 245B tokens，章节名就叫 “Ongoing”，摘要措辞 “preliminary efforts”。论文说 “upon the completion of the final version and full training of DeepSeekMoE 145B, we also plan to make it publicly available”——是否发布未核实（§8）。不能把 145B 当成已发布模型写进规格总表。

**E6（“40% 计算量”的含义）。** 39.6% / 40.5% 是论文表中 “FLOPs per 4K Tokens” 之比（74.4T/187.9T、74.4T/183.5T），不是激活参数之比（2.8/6.7 = 41.8%）。“2.5 倍激活参数”（图 1 标题）是 6.7/2.8 = 2.39 的约数。

**E7（“16B/2.8B”的口径）。** 2.8B 激活参数**包含**输入 embedding（0.21B）与输出头（0.21B）；非 embedding 激活参数约 2.41B。与 V2-Lite 的 “2.4B” 比较时口径不一定一致（§8）。

**E8（组合数的解读）。** 4,426,165,368 = C(64,8) 是论文举的**无共享专家**的示意；真实 2B 模型是 C(63,7) = 553,270,671，16B 是 C(64,6) = 74,974,368。组合数只数无序集合，不计门值的连续变化；并且消融中“加共享专家”让组合数从 120 降到 15 却提升了性能（练习 2）——组合数是动机，不是被证明的机制。

**E9（GShard Table 1 印刷错误）。** “Experts total” 列 MoE(2048E, 36L) 印为 **36684**、MoE(2048E, 12L) 印为 **12228**（arXiv PDF 文本层与 ar5iv 一致，确为原文）。按“每隔一层 MoE”应为 2048×18 = **36,864**、2048×6 = **12,288**；其余行（9216、3072、2304、768、61440）都符合 E×L/2。

**E10（GShard 算法行号）。** 正文说辅助损失在 “line (13) of algorithm 1”；PDF 中 (13) 正是 $\ell_{aux}$。ar5iv/LaTeXML 渲染把行号印成 (17)，引用时以 PDF 为准。

**E11（GShard 的 aux loss 与 Switch 的不同）。** GShard：$\ell_{aux}=\frac1E\sum_e\frac{c_e}{S}m_e$，前因子 $1/E$（均匀时值为 $1/E^2$），$c_e$ 只计**首选**专家；系数 $k$ 论文未给。Switch：乘 $N$ 使均匀时值为 1（α 与专家数解耦）。两者不能直接比较系数大小。

**E12（辅助损失的出处）。** ST-MoE Appendix A 写 “The auxiliary load balancing loss from Shazeer et al. [2017] is also used to here to balance tokens across experts.”（“used to here” 照原文印刷）然后给出 $\alpha N\sum f_iP_i$——但 Shazeer 2017 用的是 importance 与 load 两个 CV² 损失；$f\cdot P$ 形式由 GShard（$c_e/S\cdot m_e$）与 Switch 采用，Switch 原文明确说它 “simplifies the original design in Shazeer et al. (2017)”。课文应写“Shazeer 2017 首创均衡辅助损失（CV² 形式），GShard/Switch 简化为 $f\cdot P$ 形式”。

**E13（DeepSeek 7B 的注意力参数）。** DeepSeekMoE §5.2.1 称 DeepSeek 7B 有 “2.5B attention parameters”。按 DeepSeek LLM Table 2（30 层、d=4096、32 头、32 KV 头）复算为 30×4×4096² = **2.01B**。差 0.5B，原因未知（笔者推测可能把 embedding 0.42B 也算进去，2.43B ≈ 2.5B，未证实）。16B 的 “about 0.5B” 复算 0.47B，无误。课文引用时写“约 2.0B（按规格复算；arXiv v1 写 2.5B）”。ACL 2024 正式版已删去 “0.5B / 2.5B” 这两个数字，只保留 “limited attention parameters” 的定性说法——按“正式版优先”，课文最好也只作定性表述，复算值 0.47B / 2.01B 作为笔者核算给出。

**E14（词表）。** 论文 “100K”，config `vocab_size` 102,400。参数复算必须用 102,400。

**E15（初始化）。** 论文 init std 0.006；HF config `initializer_range` 0.02。后者只影响用 HF 代码从头初始化，与发布权重无关；但“DeepSeekMoE 用 0.02 初始化”是错误说法。

**E16（共享专家的实现）。** HF 代码把 2 个共享专家实现为**一个**中间维 2816 的 MLP。这与“2 个独立专家相加”数学等价（R12），不是“共享专家比路由专家大”。

**E17（seq_aux）。** HF 代码默认 `seq_aux=True`：每条序列单独算 $f_i,P_i$ 再对 batch 平均。这与论文 §2 把 $T$ 定义为“序列长度”相符，但与 Switch 的“batch 内 T 个 token”不同。DeepSeek-V3 的“序列级辅助损失”是另一个同名概念的后续发展（第 11 课），勿混淆。另注意：HF 代码只是推理/微调实现，论文训练用的是 HAI-LLM。

**E18（“细粒度切分完全免费”）。** 参数与 FLOPs 不变，但：路由器参数 ×m（R2）；每个专家的 GEMM 变小，硬件效率下降（论文 C38 “reduction in computational efficiency associated with excessively small expert sizes”）；每 token 派发到更多专家，跨设备时通信增加（ACL 版 Limitations 原文：“since DeepSeekMoE will select more experts, it has the potential to result in additional communication overhead when the experts are distributed across different devices”）。这正是 V2 设备受限路由（第 10 课）的动机。

**E19（“共享专家是 DeepSeek 发明的”）。** DeepSeekMoE 自己把原型归于 DeepSpeed-MoE 的 Residual-MoE（Rajbhandari et al., ICML 2022；ACL 版再加 Elbayad et al. 2023）。DeepSeekMoE 的贡献是从“知识冗余”的算法角度提出，并与细粒度切分组合、做了规模化验证。

**E20（Mixtral “专家按领域分工”）。** Mixtral 论文 §5 明确说没有观察到按主题的明显分配模式，更像句法层面的规律。也不要反过来把它当作 DeepSeekMoE “知识混杂”假说的直接证据——那是另一个模型、另一种分析。

**E21（“Mixtral 是 8×7B = 56B”）。** 总参数 46.7B、激活 12.9B（R9）；论文 §3 还有一处笔误 “47B tokens compared to 70B”，应为参数。

**E22（Dense×16 上界的适用范围）。** “DeepSeekMoE 接近 MoE 上界”只在 “about 2B parameters and 100B training tokens” 规模下验证（C35 原文限定语）；Dense×16 是用 16 个共享专家构造的（注意力 : FFN 参数比不是常见的 1:2）。

**E23（1991 年的 MoE 不稀疏）。** Jacobs 1991 所有专家都计算，门控给混合比例；“稀疏激活以省算力”是 Shazeer 2017 的条件计算。1991 年的目标是减少干扰、促使专精。

**E24（Shazeer 2017 的载体）。** MoE 层插在**堆叠 LSTM 之间**（“applied convolutionally between stacked LSTM layers”），不是 Transformer；进入 Transformer 是 GShard（2020）。

**E25（“GShard 是一个模型”）。** GShard 首先是 XLA 的分片注解 API 与编译器扩展（Abstract：“GShard is a module composed of a set of lightweight annotation APIs and an extension to the XLA compiler”）；“GShard 架构”在 DeepSeekMoE 里是“每层 top-2 的传统 MoE”的代称。

**E26（FLOPs 公式）。** 论文表中的 FLOPs 列没有给出计算公式。R6 的重建（$6(N_{act}+Vd)+12Lds$，注意力一律 $4d^2$）能吻合 2B 表全部数值与多数 4K 表数值，但 16B 仍差 2%。课文不能把重建公式说成论文公式。

---

## 8. 未能核实 / UNVERIFIED

1. **DeepSeekMoE 图 3–6 的数据值**：arXiv v1 LaTeXML 渲染无图片，ACL PDF 只取得坐标轴刻度与图例文本（本机无 pdftoppm，未渲染矢量图）。图 7.9 以外凡需这些曲线的，只能画示意并标“示意”。
2. **R6 FLOPs 公式**：是笔者重建；16B（+2.0%）与 142B-half（+1.1%）的残差未解释；145B 的专家中间维 1408 与首层 FFN 宽度是推断值（论文只说 0.125 倍、对齐 64 的倍数）。
3. **DeepSeek 7B “2.5B attention parameters”** 的计算口径（§7-E13）。
4. **V2-Lite “2.4B activated”** 的口径：同一复算方法得 2.661B（含 embedding 与输出头）、2.451B（去输入 embedding）、2.242B（两者都去），均不直接对上 2.4B；而 DeepSeekMoE 16B 的 2.8B 对上“含两者”的 2.829B。两篇论文可能口径不同，未证实。
5. **DeepSeekMoE 145B 是否最终完成训练/发布**：只见论文“plan to make it publicly available”，未查证后续。
6. **16B 首层稠密 FFN 为何是 10944**：论文未说明（10944 ≠ 5632、≠ 11264）。
7. **2B 验证模型的“标准 FFN”**：激活函数与中间维未说明（Dense×16 的 1.89B 与 2×1280×5120×16×9 = 1.887B 吻合，但与 SwiGLU 8/3 d 也吻合，无法区分）；8K 词表是 8000 还是 8192 未说明（对总参数影响 < 0.01B）。
8. **“nearly 2.5 times the inference speed”**：无测量设置，未核实。
9. **GShard 辅助损失系数 $k$**：论文未给数值。
10. **Mixtral 的训练负载均衡方式与系数**：论文未说明；HF `router_aux_loss_coef: 0.02` 的来源（是否即训练值）未核实。
11. **Shazeer CV 的标准差定义**（总体 vs 样本）：论文未说明（R13 两种都给了）；Appendix F “strictly-balanced gating” 未细读。
12. **ICLR 2017 / ICLR 2021 camera-ready 与 arXiv v1 的差异**：OpenReview 被人机验证拦截，未比对；venue 由 iclr.cc、dblp、Hinton 主页 PDF 页眉与检索结果交叉确认。
13. **Switch v1 与 v3（JMLR）的差异**：未比对（本笔记全部引自 v3）。
14. **ST-MoE 的正式发表状态**：只读了 arXiv v2，未查到会议/期刊版本。
15. **arXiv v1 图 2 与 ACL 图 1 是否完全相同**：只比对了标题与 ACL 文本层标签（(a) K=2、(b) K=4、(c) K=3、1..N / 1..2N、Shared/Routed Expert），未见到 arXiv 图像本身。
16. **Hash Layer、LLaMA 2 细节**：只读元数据；LLaMA2 7B 的参数复算用的是 LLaMA 式规格（32 层、4096、11008、词表 32000），未从 LLaMA 2 论文逐项核对。
17. **V2（160 路由选 6）、V3（256 路由选 8）的专家数**：只在 R1 中做算术，未在本课核对（留给第 10、11 课）。
18. **“knowledge hybridity / redundancy” 是否有直接度量**：论文给的是间接证据（禁用 top 专家的敏感度、禁用共享专家的损失上升、更少激活专家也够），没有对“专家存了什么知识”的直接探测；课文应标为“假说 + 间接证据”。
19. **ACL 2024 版与 arXiv v1 的逐表数值 diff**：复核时只对照了 ACL Table 1、Table 2、附录 Table 3 与 arXiv 对应表（一致），其余数值（如 Table 3 以外的附录、图中数值）未逐一比对。

---

## 9. 参考文献卡片草稿

```
id: jacobs1991
title: Adaptive Mixtures of Local Experts
a: Jacobs, Jordan, Nowlan, Hinton (1991), Neural Computation 3(1):79–87, doi 10.1162/neco.1991.3.1.79（无 arXiv）
url: https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf
finding: 提出由门控网络按 softmax 比例在多个专家间分配训练样本的“局部专家混合”，并用负对数混合高斯误差让最合适的专家学得最快，在元音任务上收敛约快一倍。
matters: 本课用它说明 MoE 的原始动机是“减少干扰、促成专精”而非省算力，并作为 DeepSeekMoE“专家专精”目标的思想源头。
```

```
id: shazeer2017
title: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
a: Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (2017), ICLR 2017, arXiv:1701.06538v1
url: https://arxiv.org/abs/1701.06538
finding: 用带可训练噪声的 top-k 稀疏门控把 MoE 层插在 LSTM 之间，以数千个专家把容量扩大上千倍，并用 importance 与 load 两个 CV² 辅助损失防止路由坍缩。
matters: 本课从这里引出稀疏门控公式 (3)–(5)、“路由坍缩自我强化”的现象和第一代均衡损失，也是“先 top-k 再 softmax”门控的原型。
```

```
id: gshard2020
title: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding
a: Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen (2020/2021), ICLR 2021, arXiv:2006.16668v1
url: https://arxiv.org/abs/2006.16668
finding: 在 Transformer 中每隔一层 FFN 换成 top-2 MoE，配合专家容量、分组调度、随机第二专家与 f·P 形式辅助损失，用 2048 个 TPU v3 在 4 天内训练 600B 参数翻译模型。
matters: DeepSeekMoE 把“GShard”当作传统 top-K MoE 的代表基线，本课用它讲容量、溢出 token 走残差和 aux loss 的原始形式。
```

```
id: switch2021
title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
a: Fedus, Zoph, Shazeer (2021/2022), JMLR 23(120):1–39, arXiv:2101.03961v3
url: https://arxiv.org/abs/2101.03961
finding: 证明每个 token 只路由到 1 个专家即可保持质量，给出容量因子公式与 α·N·Σf_iP_i（α=10⁻²）的可微均衡损失，并训练到 1.6T 参数。
matters: 本课的负载均衡损失、容量因子与 token 丢弃都以 Switch 的写法为标准形式，DeepSeekMoE 的专家级损失可由它推广到 K'>1 得到。
```

```
id: stmoe2022
title: ST-MoE: Designing Stable and Transferable Sparse Expert Models
a: Zoph, Bello, Kumar, Du, Huang, Dean, Shazeer, Fedus (2022), arXiv:2202.08906v2（预印本）
url: https://arxiv.org/abs/2202.08906
finding: 提出惩罚路由 logits 的 router z-loss（c_z=0.001），在不损质量的前提下稳定大规模稀疏模型训练，并系统比较容量因子与 top-n 路由。
matters: 作为选讲材料说明除负载均衡外 MoE 路由器还有数值稳定性问题，也是一处辅助损失出处被误记的例子。
```

```
id: dsmoe2024
title: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
a: Dai, Deng, Zhao, Xu, Gao, Chen, Li, Zeng, Yu, Wu, Xie, Li, Huang, Luo, Ruan, Sui, Liang (2024), ACL 2024 (Long) pp. 1280–1297, arXiv:2401.06066v1
url: https://arxiv.org/abs/2401.06066 ; https://aclanthology.org/2024.acl-long.70/
finding: 以细粒度专家切分（N→mN、激活 mK）和共享专家隔离两项策略缓解知识混杂与冗余，2B 规模追平 1.5 倍专家的 GShard，16B 以约 40% 计算量比肩 LLaMA2 7B 与 DeepSeek 7B。
matters: 这是本课主角，也是 DeepSeek-V2/V3/V4 全部 MoE 层的直接前身，本课的公式、规格与消融数字都出自此文。
```

```
id: mixtral2024
title: Mixtral of Experts
a: Jiang, Sablayrolles, Roux, Mensch, Savary, Bamford, et al. (2024), arXiv:2401.04088v1（预印本）
url: https://arxiv.org/abs/2401.04088
finding: 每层 8 个 SwiGLU 专家选 2、门值为 top-2 logits 上的 softmax，总参数 47B、激活 13B，路由分析未见按主题的明显分工。
matters: 与 DeepSeekMoE 同月发布，作为“少而大的专家”路线的对照组，帮助读者看清细粒度 + 共享专家到底改了什么。
```

```
id: dsv2
title: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
a: DeepSeek-AI (2024), arXiv:2405.04434v5
url: https://arxiv.org/abs/2405.04434
finding: 把 DeepSeekMoE 写成 N_s/N_r/K_r 记号（式 20–22）并在 Appendix B 给出沿用 16B FFN 配置的 V2-Lite（27 层、15.7B/2.4B、α₁=0.001）。
matters: 本课采用 V2 的记号体系，并用 V2-Lite 说明 DeepSeekMoE 16B 的 MoE 配置被原样延续，为第 9、10 课铺路。
```

```
id: dsv3
title: DeepSeek-V3 Technical Report
a: DeepSeek-AI (2024), arXiv:2412.19437
url: https://arxiv.org/abs/2412.19437
finding: V3 把亲和度从 softmax 改为 sigmoid，并在选中的亲和度之间归一化得到门值。
matters: 本课只用这一句做前瞻，提醒读者 DeepSeekMoE 的“不归一化 softmax 门值”在 V3 被改掉，详见第 11 课。
```

```
id: dsmoe_residual
title: DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale
a: Rajbhandari, Li, Yao, Zhang, Aminabadi, Awan, Rasley, He (2022), ICML 2022 (PMLR 162), arXiv:2201.05596v2
url: https://arxiv.org/abs/2201.05596
finding: 提出 Residual-MoE：每个 token 固定经过一个稠密 MLP 再加一个被选中的专家，把专家视为稠密 MLP 的误差修正项，质量与 top-2 相当而通信更少。
matters: DeepSeekMoE 明确把共享专家隔离的原型归于此文，本课据此纠正“共享专家由 DeepSeek 首创”的说法。
```

```
id: dsllm
title: DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
a: DeepSeek-AI (2024), arXiv:2401.02954v1
url: https://arxiv.org/abs/2401.02954
finding: Table 2 给出 DeepSeek LLM 7B（30 层、4096、MHA）与 67B（95 层、8192、GQA 8 组）规格，并采用 31.6%/10% 的多步学习率。
matters: 它是 DeepSeekMoE 16B 与 145B 的稠密对照基线，本课用其规格复算注意力参数并发现论文“2.5B”一处不符。
```

```
id: hashlayer
title: Hash Layers For Large Sparse Models
a: Roller, Sukhbaatar, Szlam, Weston (2021), arXiv:2106.04426v3
url: https://arxiv.org/abs/2106.04426
finding: 用固定哈希把 token 分配给专家的 top-1 路由方式（本笔记只读了元数据）。
matters: 作为 DeepSeekMoE Table 1 的基线之一出现，本课只在对比图中列名，不展开。
```

```
id: llama2
title: Llama 2: Open Foundation and Fine-Tuned Chat Models
a: Touvron et al. (2023), arXiv:2307.09288v2
url: https://arxiv.org/abs/2307.09288
finding: 开源 7B–70B 稠密模型，其中 7B 版本以 2T tokens 预训练（本笔记只读元数据）。
matters: DeepSeekMoE 16B 以它为主要外部对照（Table 4 的 39.6% 计算量），本课需要它的规格来解释 FLOPs 比。
```

```
id: hf_dsmoe16b   kind: artefact
title: deepseek-ai/deepseek-moe-16b-base（config.json, modeling_deepseek.py, configuration_deepseek.py）及 -chat/config.json
a: DeepSeek-AI, Hugging Face 模型仓库（2024）
url: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
finding: 配置为 28 层、d=2048、2 共享 + 64 路由选 6、专家中间维 1408、首层稠密 10944、softmax 打分且不归一化 top-k、默认 aux α=0.001 与按序列计算。
matters: 本课用它逐字段核对论文规格、复算 16.4B/2.8B，并指出词表、初始化、共享专家实现等论文与代码的差异。
```

```
id: gh_dsmoe   kind: artefact
title: deepseek-ai/DeepSeek-MoE（GitHub README）
a: DeepSeek-AI (2024)
url: https://github.com/deepseek-ai/DeepSeek-MoE
finding: README 称 16B 以 16.4B 参数、2T 中英 token 训练，可在单张 40GB GPU 上无量化部署，并发布 Base 与 Chat 两个 4096 上下文的检查点。
matters: 作为发布状态与部署说法的出处，本课据此只把 16B（而非 145B）列为已发布模型。
```

```
id: hf_mixtral   kind: artefact
title: mistralai/Mixtral-8x7B-v0.1 config.json
a: Mistral AI, Hugging Face（2023–2024）
url: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
finding: 配置与论文 Table 1 一致（4096、32 层、GQA 8、14336、8 选 2），另含 router_aux_loss_coef 0.02 与 rope_theta 1e6。
matters: 本课用它复算 Mixtral 46.7B/12.9B，并说明论文未公开的负载均衡系数只能在配置里看到。
```

```
id: hf_v2lite   kind: artefact
title: deepseek-ai/DeepSeek-V2-Lite config.json
a: DeepSeek-AI, Hugging Face（2024）
url: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
finding: 27 层、MLA（kv_lora_rank 512、rope 维 64）、2+64 选 6、专家 1408、首层稠密 10944、aux_loss_alpha 0.001、seq_aux true。
matters: 证明 V2-Lite 的 MoE 部分与 DeepSeekMoE 16B 完全相同，本课据此复算 15.7B 并标出 2.4B 激活参数的口径疑问。
```

```
id: hf_dsllm   kind: artefact
title: deepseek-ai/deepseek-llm-7b-base 与 deepseek-llm-67b-base config.json
a: DeepSeek-AI, Hugging Face（2023）
url: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
finding: 7B 为 30 层、4096、32 头 MHA、中间维 11008、词表 102400；67B 为 95 层、8192、64 头 8 KV 组、22016。
matters: 本课用它们重建论文 FLOPs 列与注意力参数，对比 MoE 与稠密模型的计算账。
```

---

## 10. 核查记录（对抗式复核，2026-09-24）

复核者对研究笔记中最容易被复述的 30 余条论断逐条回到一手来源重查，默认怀疑。手段：`arxiv.py text/grep`（LaTeXML 全文）；arXiv PDF 文本层（GShard v1，用于 Algorithm 1 行号与 Table 1）；ACL Anthology PDF 文本层（`2024.acl-long.70.pdf`，18 页）与条目页（DOI）；Hinton 主页 Jacobs 1991 扫描页逐页目读（p. 80、81、82、85）；Hugging Face `config.json`、`configuration_deepseek.py`、`modeling_deepseek.py` 原文；GitHub README 原文；所有派生数字在 Python 中重算。

**确认无误（逐字或逐符号核对）**
- DeepSeekMoE arXiv v1 式 (1)–(17)：逐符号一致，含 (4)(7)(10) 的 Topk 集合下标与个数、(13) 的 $N'/(K'T)$、(16) 取均值与 (17) 取和；ACL 版 (12)–(14) 为单一 $\mathcal L_{Bal}$、无设备级损失（全文无 “device-level”）。
- 2B / 16B / 145B 超参（§4.1.3、§5.1.2、§7.1、Table 7）：层数、隐藏维、头数、共享/路由/激活数、0.25/0.125、init 0.006、LR 1.08e-3 / 4.2e-4 / 3.0e-4、batch、步数 25,000 / 106,449 / 13,000、α₁ = 0.01 / 0.001 / 0.003、α₂ = 0.05、4 台设备、2B 与 16B 不丢 token 且不用设备级损失。
- Table 1/2/3/4/6 的 FLOPs 与 Pile 数（2.9T/4.3T/5.8T/24.6T；1.808/1.806/1.867 等；74.4T/183.5T/187.9T；585.6T/2057.5T/374.6T/572.7T）与标题中的 40.5%、39.6%、245%、28.5%、18.2%；1.808→2.414；1/2/4 共享 1.808/1.806/1.811；C(16,2)=120、C(64,8)=4,426,165,368 的原文印刷。
- ACL 版：页码 1280–1297、DOI 10.18653/v1/2024.acl-long.70、摘要末句、Elbayad et al. 2023（Findings ACL 2023 pp. 14237–14253）、Limitations 原文、Table 2 三列合并表数值、无 145B/Chat 章节。
- HF config（16B base/chat、V2-Lite、Mixtral、DeepSeek LLM 7B/67B）全部字段；`MoEGate` 先 softmax 再 topk、仅 `norm_topk_prob` 为真才归一化；`seq_aux`/`aux_loss_alpha` 默认 True/0.001；共享专家为一个 `moe_intermediate_size * n_shared_experts` 宽的 MLP。
- 复算：16B 总 16,375,728,128 / 激活 2,828,650,496 / 去两端 2,409,220,096；注意力 0.47B；DeepSeek 7B 注意力 2.013B、总 6.910B；LLaMA2 7B 6.738B；Mixtral 46.70B / 12.88B；V2-Lite 15.706B 与 2.661/2.451/2.242B；全部组合数与 log₁₀；R10 插值、R13 全部例题数值（含门值和 0.6549/0.6927/0.5915/0.4949、1.5736、1.3181、容量丢弃 96/40、Φ = 0.3326、CV²、Jacobs 责任 0.9669/0.0331）；显存与 KV cache 数字；训练 token 数。
- Switch v3：式 (1)–(6)、α = 10⁻² 与 10⁻¹…10⁻⁵ 扫描、“typically <1%”、“simplifies the original design”、Table 1 全部 CF/速度/Neg. Log Perp.、Table 9（Switch-C 1571B、2048 专家、Expert Freq. 1、15 层；XXL 395B、64 专家）。
- GShard v1：PDF 中 Algorithm 1 行 (2)(3)(5)(6)(8)–(11)(13)(16)(17)(19) 与笔记一致，(13) 为 $\ell_{aux}=\frac1E\sum_e\frac{c_e}{S}m_e$，LaTeXML 渲染为 (17)；Table 1 “36684”“12228” 确为 PDF 原文印刷，其余行符合 E×L/2；2N/(G·E)、溢出走残差、随机第二专家、600B / 2048 TPU v3 / 4 天、1T 未报告。
- Shazeer v1：式 (1)–(11) 编号与内容；k>1 导数原句；Table 6 全部六行；w = 0.1/0.1（App. C）、0.01/0.01（App. E）；137B、>1000x、65536 专家 39%。
- ST-MoE v2：z-loss 式 (5)、$L_{tot}$ 式 (6)、$c_z$ = 0.001、Table 4 三行、每 4 层一个 MoE、train/eval CF 1.25/2.0、Appendix A 用 $\alpha N\sum f_iP_i$ 且 α = 10⁻²。
- Mixtral v1：$G(x)=\mathrm{Softmax}(\mathrm{TopK}(x\cdot W_g))$、K = 2、Table 1、47B/13B、§3 “47B tokens compared to 70B” 笔误、§5 无按主题分工。
- DeepSeek-V2 式 (20)–(22) 与记号定义、Appendix B.1 V2-Lite（27 层、1408、2+64 选 6、15.7B/2.4B、5.7T、α₁ = 0.001）；V3 sigmoid + 归一化原句；DeepSpeed-MoE Residual-MoE 两句原文；DeepSeek LLM Table 2（7B：30/4096/32/32；67B：95/8192/64/8）与 31.6%/10% 学习率。
- Jacobs 1991：式 (1.1)–(1.5) 与 p. 82 “slowest/fastest”原句；图 1 标题的 softmax；Table 1（1124/23、1083/12、2209/83、2435/124，88%/90%）与 “p ≫ 0.999” 印刷。

**本次修改（均已在正文就地改正）**
1. §7-E4 图号映射：原写“图号整体前移一位”，不完整。ACL PDF 显示 arXiv 图 1（Open LLM Leaderboard）移到附录 E 成为 **ACL 图 7**，arXiv 图 7 成为 ACL 图 6。同时补记：arXiv Table 2 移到 ACL 附录 C（Table 3）；GShard×1.2 与 Dense×4 在 ACL 版中不再出现；“0.5B vs 2.5B attention parameters” 在 ACL 版被删。§5.1 图表清单的两行也补上了 ACL 编号。来源：ACL PDF pp. 1286, 1296–1297，附录 C/D/E 标题。
2. C40、§7-E13：补记 ACL 版删去了 “0.5B / 2.5B” 两个数字，并建议课文按“正式版优先”只作定性表述。来源：ACL PDF p. 1286 §5 观察 (3)。
3. §6 练习 8 答案与 §7-E2：原答案说“这就是 … Switch 把 softmax 放到 top-k 之前的原因”，属于相邻论断替换。Switch 原文没有这样的动机陈述；它在 §2.1 转述 Shazeer 的 MoE 时就写成“softmax 全部 N 个后取 top-k”，与 Shazeer 原文式 (3) 顺序不同（这本身是一处“传话”偏差）。已改为只陈述可核实的事实。来源：Switch v3 §2.1 式 (1)–(2) 与 “permits differentiability” 句；Shazeer v1 式 (3)(5)。
4. §7-E1：原写 GShard “显式重新归一化（结果与 Mixtral 式相同）”，暗示门值和为 1。实际上第二专家以 $2g_2$ 概率随机派发且受容量限制，未派发时只剩 $g_1<1$。已补充这一点。来源：GShard PDF Algorithm 1 行 (6)(16)(17)(19)。
5. C36：补换算说明，“1:3”对应 64 专家、激活 8 个里的 **2 共享 + 6 路由**（Pile 1.806），不是 1 共享的 2B 主模型（1:7）。来源：arXiv v1 §4.4 “keeping the number of total experts and activated experts constant”。
6. C3 页码：“Notice that …” 整句在 p. 80，不跨 p. 81。来源：扫描页 p. 80。
7. §7-E12 引文：ST-MoE 原文为 “is also used **to** here to balance tokens across experts”，已照原文补全并注明原文如此。来源：ST-MoE v2 Appendix A 首句。
8. §3.3 Shazeer 行：原写 “k=4（平坦）或每级 k=2”，只对 LM 与单语对 MT 实验成立。补上多语 MT 为平坦 n=512、k=2。来源：Shazeer v1 Appendix C、Appendix E（“The MoE layers in the encoder and decoder are non-hierarchical MoEs with n=512 experts, and k=2”）。
9. §4-R9：原写“天真的 8×7B = 57.9B”。字面 8×7 = 56B；57.9B 是“8 个同配置稠密模型”（按同一 config 只留 1 个专家复算为 7.24B）。已把两种算法都写明，并说明来源是复算。
10. §0 `dsv3` 版本：原写“版本未记录”。核查时 arXiv 最新版为 v2（v1 提交于 2024-12-27），grep 的是最新版全文。

**仍未能核实**：见 §8，本次未能关闭其中任何一项。另外，ACL 版除上述差异外是否还有其他数值改动，本次没有做逐表 diff，只核对了 Table 1、Table 2 和附录 Table 3。

---

## 11. 补充核实（写课时追加，2026-09-24）

写 `lesson-07-deepseekmoe.md` 时用到、但上文没有逐字收录的事实与复算。复算全部在 Python 中完成（生成脚本见 `deepseek/figures/gen/l07_*.py`，每个脚本都带 `assert` 核对关键数值）。

**S1（外部事实）DeepSeek LLM 7B / 67B 的训练数据量是 2T token。** 来源：DeepSeek LLM（arXiv:2401.02954）§1，原文 “We collect 2 trillion tokens for pre-training, primarily in Chinese and English.”（转引自 `lesson-06-doc.md` 第 6 条，已核对原句）。课文 §8 用它说明 DeepSeekMoE 16B 与 DeepSeek 7B 的训练 token 数相同（16B 为 2T，见 C38）。

**S2 DeepSeek LLM 7B 的参数构成（由 `hf_dsllm` 配置复算）。** 注意力 `30*4*4096**2` = 2,013,265,920；FFN `30*3*4096*11008` = 4,057,989,120；输入嵌入、输出头各 `102400*4096` = 419,430,400；归一化 `30*2*4096+4096` = 249,856；合计 6,910,365,696（与 R4 的 6.910B 一致）。FFN 占总参数 58.72%，占每层参数 66.84%；每层 FFN : 注意力 = `3*11008/(4*4096)` = 2.016。课文图 7.1、式 (7.1) 用。

**S3 Mixtral 的专家参数占比（由 `hf_mixtral` 配置复算）。** 每专家 `3*4096*14336` = 176,160,768；每层 8 个 = 1,409,286,144；32 层 = 45,097,156,608，占 46.70B 的 96.57%；激活比例 12.88/46.70 = 27.58%；路由器 `8*4096` = 32,768，约为一个专家的 1/5376。对照：DeepSeekMoE 16B 的路由器 `64*2048` = 131,072，只是一个专家（8,650,752）的 1/66——“路由器比专家小几个数量级”只对 Mixtral 这类大专家成立，课文据此没有写成普遍结论。

**S4 课文门控例题的 logit。** §2 与图 7.3/7.4 用的 logit `(0.2, 2, 0, -0.5, 1, 0.5, -0.3, -1)` 是 R13 门控例题 `(2, 1, 0.5, 0.2, 0, -0.3, -0.5, -1)` 的一个排列（让专家 2、5 胜出），门控值 0.4709 / 0.1732、0.7311 / 0.2689、截掉的概率和 0.3559 均不变。

**S5 专家级损失对路由 logit 的梯度（笔者推导，非论文公式）。** 由 (12)–(14)，把 $f_i$ 视为常数，$\partial\mathcal L_{ExpBal}/\partial z_{j,t}=\frac{\alpha_1}{T}s_{j,t}(f_j-\bar f_t)$，$\bar f_t=\sum_is_{i,t}f_i$；Switch 形式同理多一个因子 $N$。R13 例题 token 1：$\bar f_1$ = 1.9188，梯度/α₁ = (0.1102, 0.0050, −0.0209, −0.0127, −0.0127, −0.0265, −0.0265, −0.0161)。同一例题的中间量（5 位小数）：$3P_1$ = 0.89399，$2P_2$ = 0.32736，$P_3+P_4+P_5$ = 0.35227；设备级 1.75×0.71209 = 1.24616，0.25×0.28791 = 0.07198。

**S6 另一种专家放置（练习 8，笔者构造）。** 同一 R13 例题，专家 {1,6,7,8} 放设备 A、{2,3,4,5} 放设备 B：$f'$ = (0.75, 1.25)，$P'$ = (0.4840, 0.5160)，$\mathcal L_{DevBal}/\alpha_2$ = 1.0080（原放法 1.3181）。

**S7 设备级损失的两个性质（笔者推导）。** 各组大小相同（$|\mathcal E_i|=N'/D$）时 $f_i'=D\,F_i$，$F_i$ 为设备 $i$ 承接的选择比例，于是 $\mathcal L_{DevBal}=\alpha_2D\sum_iF_iP_i'$，与 Switch 式 (4) 同形；$D=1$ 时 $\mathcal L_{DevBal}\equiv\alpha_2$，梯度为零（与 16B“全部专家在同一设备、不用设备级损失”相容）。

**S8 组合数的熵界（标准结论，课文逐行推导）。** $2^{nH(p)}/(n+1)\le\binom nk\le2^{nH(p)}$，$p=k/n$。$H(1/8)$ = 0.543564 比特；$N=16,K=2$ 时每增加 1 个 $m$，上界上升 `16*H*log10(2)` = 2.618 个数量级；真实值相邻差 2.477, 2.533, 2.557, 2.571, 2.579, 2.585, 2.589。$\binom{64}{8}$ 的上界 2.967e10、下界 4.564e8，精确值为上界的 14.9%。图 7.8 的示意组合数：C(8,2) = 28，C(16,4) = 1,820，C(15,3) = 455。

**S9 细粒度切分不改变每个专家的平均 token 数与算术强度（笔者推导）。** 每专家平均 token 数 $T\cdot mK/(mN)=TK/N$；每读一遍专家权重（$3d\,d_{ff}/m$）的计算为 $2\cdot TK/N\cdot3d\,d_{ff}/m$，比值 $2TK/N$ 与 $m$ 无关。硬件效率的损失来自矩阵变窄变多与派发目的地增多，不来自算术强度——课文初稿曾误写“算术强度下降”，已在定稿前改正。

**S10 λ 坍缩曲线的其它配置。** 公式同 R10；$(N',K')=(63,7)$ 时 λ = 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1 → 1, 1.08, 1.5, 3.0, 5.5, 7.48, 9；$(16,2)$ → 1, 1.07, 1.4375, 2.75, 4.9375, 6.67, 8。

**S11 16B 的 KV cache（与 R8 一致，补一个单位换算）。** 229,376 B/token × 4096 = 939,524,096 B = 0.94 GB = 0.875 GiB 每条 4K 序列。

**S12 教学复核时的改正与补充推导（2026-09-24）。**
- 课文 §1 的 1991 例题原把“梯度大小 $2p_i\|\mathbf d-\mathbf o_i\|$”标成“先混合、再算平方误差”的结果；按 §2.1 的转录，这是式 (1.2)（随机选一个专家的期望误差）的导数 (1.4)，式 (1.1) 的导数依赖共同残差 $\mathbf d-\sum_jp_j\mathbf o_j$。课文已改为三种误差函数逐条列出，并引用 C4 的前提（初始等权、各专家误差 > 1）。
- 专家级损失的上界（笔者推导）：计数 ≤ $T$，故 $f_i\le N'/K'$，又 $\sum_iP_i=1$，于是 $\sum_if_iP_i\le\max_if_i\le N'/K'$；全坍缩值 $N'/K'$ 是严格上界，均匀值 1 不是下界。课文 (7.25) 后已按此改写。
- 路由 logit 的梯度（笔者推导，先 softmax、不归一化门控）：记 $a_{i,t}=(\partial\mathcal L/\partial\mathbf h_t')^\top\mathrm{FFN}_i(\mathbf u_t)$，$S_t=\sum_{i\in\mathcal T_t}s_{i,t}a_{i,t}$，则 $\partial\mathcal L/\partial z_{j,t}=s_{j,t}(a_{j,t}\mathbb 1[j\in\mathcal T_t]-S_t)$。演示数值（$a_2=-0.5$、$a_5=0.2$ 为假设值，logits 同 R13 的 8 专家例题）：$S_t=-0.20079$，梯度 = (0.01563, −0.14089, 0.01279, 0.00776, 0.06942, 0.02110, 0.00948, 0.00471)，与有限差分一致。
- “$DK/N$ 类知识”粗算对细粒度切分不变（$D\cdot mK/(mN)=DK/N$），课文 §4 改用“一个 token 同时用到 $r$ 个面、至少一个专家承担 $\lceil r/K\rceil$ 个面”的抽屉原理说法，并注明两者都是笔者的简化。

**S13 数学复核（2026-09-24，逐式重推、全部数字 Python 重算）。** 课文全部数值（参数账、路由例题、梯度与有限差分、容量、CV²、组合数与熵界、4 token 均衡损失例题、设备级、坍缩曲线、2B/16B/145B 比值、KV cache）复算无误；15 个生成脚本重跑后 SVG 逐字节不变。改动：
- 坍缩曲线（R10）可化简：记 $\rho=N'/K'$，坍缩组 $P=f/N'$，于是 $\mathcal L/\alpha_1=1+(\rho-1)\lambda^2$（抛物线，$\lambda=0$ 处斜率 0；而梯度 (7.12) 中 $f_j-\bar f$ 是 $\lambda$ 的一次量）。已核对与 R10 全部取值一致，写入课文练习 9 与图 7.12 说明，并在 `l07_collapse_curve.py` 加了断言。
- (7.25) 的上界是紧的但取不到：softmax 概率恒正，其余专家 $P_i>0$，只能逼近 $N'/K'$。均匀值 1 不是下界的具体例子（笔者构造）：$N'=2,K'=1$，3/4 token 以 $s\approx(0.5,0.5)$ 勉强选专家 1、1/4 以 $s\approx(0,1)$ 选专家 2 ⇒ $f=(1.5,0.5)$、$P\approx(0.375,0.625)$、$\sum fP\approx0.875$。
- (7.7) 原写“路由器比一个专家小三个数量级”，与 S3 矛盾（16B 只差 1/66）。改为比值 $N/(3d_{ff})$：Mixtral 1/5376，16B 1/66；相对全部专家参数恒为 $1/(3d_{ff})$。
- (7.29) 原只写激活参数比 2.8/6.7 = 41.8%，未说明对应 LLaMA2 7B；补上 DeepSeek 7B 的 2.8/6.9 = 40.6%（与 FLOPs 比 40.5% 只是碰巧接近）。
- GShard 第二专家的派发概率：Algorithm 1 为 $2g_2>\text{rnd}$，重新归一化后 $g_2\le\tfrac12$，所以概率恰为 $2g_2$（原写“与 $2g_2$ 成正比”）。
- Jacobs 前提“各专家误差 > 1”的来由：等权时 (1.5) 的梯度大小 ∝ $\epsilon e^{-\epsilon^2/2}$，导数 $(1-\epsilon^2)e^{-\epsilon^2/2}$ 在 $\epsilon>1$ 时为负。
- 16B/145B 的专家总量 66×0.25 = 132×0.125 = 16.5 个标准 FFN，所以 (7.20) 的 $N_s+N_r=mN$ 只在 2B 严格成立（相当于 $N=16.5$），激活量等式三个规模都严格成立。
- 记号：§4 粗算的“$D$ 类知识”改名 $Q$，避免与 §7 的设备数 $D$ 冲突；封面记号框补充 $T$（§3 为 batch token 数、§7 为序列长度）与 $D$（§7 为设备数）的本课含义。
- 练习 1 的 “$2\times2.8=5.7$” 改为 $2\times2.83\approx5.7$；图 7.2 说明里“空心圆与实心圆的距离 = 两者之差”改为“= 倍数”（对数轴）。
- (7.12) 后补一句：真正更新的是 $\mathbf e_j$，token $t$ 这一项使 $z_{j,t}$ 变化 $-\eta\,(\partial\mathcal L/\partial z_{j,t})\norm{\mathbf u_t}^2$（同号），一步更新中其他 token 通过 $\mathbf u_{t'}^\top\mathbf u_t$ 交叉影响；原文“路由器真正的参数是 logit”改为“参数 $\mathbf e_j$ 通过 logit 起作用”。

**S14 数学复核第二轮（2026-09-24）。** 全部数值再次用 Python 复算无误，15 个生成脚本重跑后 SVG 逐字节不变。改动：
- 封面“读完你能”第 3 条原写“推导 $f\cdot P$ 损失为什么在均匀路由时最小”，与课文 (7.25) 后的反例（$\sum fP\approx0.875<1$）矛盾。在均匀时最小的是理想目标 $\sum f_i^2$；$\sum f_iP_i$ 固定 $f$ 时是 $P$ 在单纯形上的线性函数，下确界 $\min_if_i$ 在顶点（softmax 的 $P_i>0$，取不到）。已改写封面第 3 条，并在 §3“容易忽略的性质”补上单纯形论证。
- 练习 9 原把“梯度正比于 $f_j-\bar f_t$”直接写成 $f_j-1$。补上理由：均匀路由的 token 有 $s_{i,t}=1/N'$，$\bar f_t=\frac1{N'}\sum_if_i=1$；已坍缩的 token（理想化的平分亲和度）梯度为零。
- §7 例题与练习 8(a)：“$3P_1=0.89399$”无法由课文列出的 4 位或 5 位 $P$ 复现（$3\times0.29800=0.89400$）。改为列出 5 位小数的 $P_{1..5}$ 逐项相加得 1.57363，并注明用 4 位小数得 1.5737。
- (7.27) “$P'$ 均匀时为 $1/D$”只在各组大小相同时成立；一般为 $|\mathcal E_i|/N'$，而均匀时 $\sum f'P'=1$ 总成立。
- 路由器梯度推导补上“把 $\mathcal T_t$ 当常数”为何精确：第 $K$ 与第 $K+1$ 大的 logit 不相等时 $\mathcal T_t$ 局部不变，仅在测度为零的并列点不可导。
- 记号：§8 末尾“每个 token 最多发往 $M$ 台设备”的 $M$ 与 §1 的每 token FLOPs $M$ 冲突，去掉符号；§3 补全 GShard 记号 $E,S,c_e,m_e$ 的含义；封面记号框注明 §5 与练习 12 的 $T$ 也是 batch token 数。
- §1 例题补一句：输入嵌入只是查表，严格按乘加计 16B 每 token 前向约 $2\times2{,}618{,}935{,}296\approx5.24$ GFLOPs（2.83B 口径为 5.66）。§1 历史段补“最大实验 131,072 个专家”（C14），与图 7.2 一致。
- 数学课链接补 `#pN` 页锚：M06 §5（softmax 雅可比）、M04 §2（方差）、M01 §2（矩阵乘向量）与 §7（练习 7）、M05 §1（二元熵）、M10 §3（算术强度）与 §6（all-to-all）。

**S15 事实与引用复核（2026-09-24，Facts 轮）。** 逐条列出课文中的数字、日期、规格、引文与归属，对照本笔记；笔记没有覆盖的，回到一手来源核对（`arxiv.py grep/text` 读 LaTeXML 全文；WebFetch 读 Mistral 公告与 GitHub README）。新核实的事实：

- **DeepSeekMoE arXiv v1 表 3 全表（DeepSeek 7B vs 16B，19 项基准，不含 Pile BPB）。** 课文图 7.15 的十项数值逐一吻合（HellaSwag 75.4/77.1、PIQA 79.2/80.2、ARC-c 48.1/49.8、GSM8K 17.4/18.8、HumanEval 26.2/26.8、MBPP 39.0/39.2、TriviaQA 59.7/64.8、NQ 22.2/25.5、MMLU 48.2/45.0、CEval 45.0/40.6）。其余九项：ARC-easy 67.9/68.1、RACE-middle 63.2/61.9、RACE-high 46.5/46.4、DROP 34.9/32.9、MATH 3.3/4.3、WinoGrande 70.5/70.2、CLUEWSC 73.1/72.1、CMMLU 47.2/42.5、CHID 89.3/89.4。**全表里 16B 落后 DeepSeek 7B 的共 8 项**（RACE-m −1.3、RACE-h −0.1、DROP −2.0、MMLU −3.2、WinoGrande −0.3、CLUEWSC −1.0、CEval −4.4、CMMLU −4.7），差距最大的是 CMMLU，不是 MMLU 或 CEval。表 4（vs LLaMA2 7B）：LLaMA2 列 ARC-easy 69.1、RACE-m 60.7、RACE-h 45.8、DROP 34.0、MATH 2.6、WinoGrande 69.6、CLUEWSC 64.0、CMMLU 32.6、CHID 37.9，其余十项与 §5.2 图 7.11 数据一致；16B 落后的是 ARC-easy、DROP、MMLU，NQ 持平。课文图 7.15 说明原只说“只有 MMLU 和 CEval 落后”，只对所选十项成立，已补上全表情况；§8 “弱项”补 CMMLU。
- **§5.2.1 观察 (2)** 原文：“DeepSeekMoE 16B exhibits notable strengths in language modeling and knowledge-intensive tasks such as Pile, HellaSwag, TriviaQA, and NaturalQuestions. Given that in an MoE model, FFN parameters are much heavier than attention parameters, these outcomes align with the proposition that FFNs in Transformers exhibit the capability for knowledge memorization (Dai et al., 2022a).” 课文 §8 据此加了一段“强项”，并标明是相关性观察。
- **消融的文字结论**（§4.4，arXiv 图 3）：“compared with GShard, the intentional isolation of a shared expert yields improved performance across a majority of benchmarks.” 支撑课文“组合数下降、多数基准变好”的说法（数值仍取不到，见 §8 第 1 条）。
- **组合灵活性的原话**（§1）：“we also activate more fine-grained experts to enable a more flexible and adaptable combination of activated experts. Fine-grained expert segmentation allows diverse knowledge to be decomposed more finely and be learned more precisely into different experts”；§3.1：“The surge in combinatorial flexibility enhances the potential for achieving more accurate and targeted knowledge acquisition.” 课文 (7.16) 的转述与之相符。
- **训练框架**（§4.1.2）：“We conduct experiments based on HAI-LLM (High-Flyer, 2023), an efficient and light-weight training framework …” 课文 §7 据此注明 HF 代码不是训练代码，代码默认值（如 `seq_aux`）只能在公式层面与论文对照（对应 §7-E17）。
- **ST-MoE z-loss 的 $B$**（§3.3，式 (5) 之后）：“where $B$ is the number of tokens, $N$ is the number of experts”。课文 §3 选讲框的说法正确。
- **Shazeer 2017 的 131,072 专家是层级 MoE**（Appendix D）：“using an ordinary MoE layer with 32 experts and hierarchical MoE layers with 256, 1024, 4096, 16384, 65536 and 131072 experts”；§5.2 同段 “This corresponds to up to 137 billion parameters in the MoE layer.”
- **Mixtral 权重早于论文公开。** Mistral AI 公告 “Mixtral of experts” 日期 2023-12-11，写明 “Licensed under Apache 2.0”（`mistral.ai/news/mixtral-of-experts`）。论文 arXiv:2401.04088 为 2024-01-08。课文原写“同一周里出现了两个开源 MoE”，已改为“两篇论文在同一周挂上 arXiv”，并注明 Mixtral 权重 2023-12-11 已公开；新增参考文献卡 `mistral2023blog`（kind: artefact）。
- **V2-Lite 发布日期** 2024-05-16（DeepSeek-V2 GitHub News “2024.05.16: We released the DeepSeek-V2-Lite.”，转引自 `lesson-00-doc.md` 与 `lesson-10-doc.md` 第 19 条，均已核对）；同日 arXiv v3 新增附录 B（`lesson-09-doc.md` 版本表）。课文与图 7.2 的 “2024-05” 正确。
- **GitHub README 复核**：“can be deployed on a single GPU with 40GB of memory without the need for quantization”；“16.4B parameters”；“trained from scratch on 2T English and Chinese tokens”；Base 与 Chat 的序列长度均为 4096。`gh_dsmoe` 卡片无误。
- **课文中的其他改正**（均无新事实，只是让表述与已核实事实一致）：`norm_topk_prob: false` 对应的是论文式 (4)/(10)，课文原写“式 (5)”（式 (5) 是 softmax）；“后来的 V2、V3、V4 的前馈层全部沿用”改为“MoE 层都沿用”（V2/V3 的前几层是稠密 FFN）；V3 的“取代辅助损失”补上“仍保留一个系数极小的序列级损失”（与 `lesson-11` 一致）；“本课引用的 Pile 损失都在 1.8–2.1 之间”限定为“训练完成的模型”（禁用共享专家的评测是 2.414）；“论文用‘个数’是为了……”改标为笔者的理解；封面记号框注明图号、表号按 arXiv v1；图 7.2 说明补上各行数据的出处引用；消融表的“图例原文如下”改为“按图例原文整理”；两处跨课链接补页锚（第 5 课 §5、第 12 课 §7）。

