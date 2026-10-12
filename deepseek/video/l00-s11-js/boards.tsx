// 第 0 课 §11 · 练习与答案 —— 10 段黑板。题目与答案全部来自 lesson-00 的练习页（研究笔记 lesson-00-doc.md），
// 数字都复算过。每道题：先出题，第一句之后停 2.5 秒，画面上转一圈"想一想"，再一步步给出答案。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Chip, F, Mono, Roadmap, Stage, Svg, Syllabus, Tex, Txt, camAt, ease, fit, lin, lit, project, rgba, syllabusLayout,
  up, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';

const Card: React.FC<{on?: boolean; dashed?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, dashed, style, children}) =>
  <div style={{padding: '14px 22px', borderRadius: 12, background: rgba(C.bg, 0.92),
    border: `1px ${dashed ? 'dashed' : 'solid'} ${on ? rgba(ACC, 0.7) : C.line}`, boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;

type Ans = {key: string; body: React.ReactNode};
/** One exercise: the question on top, a "think" ring while the narration pauses, then the answers step by step. */
const Ex: React.FC<{n: number; q: React.ReactNode; answers: Ans[]; side?: React.ReactNode}> = ({n, q, answers, side}) => {
  const {f, at, end, p} = useBoard();
  const t0 = end('l1'), t1 = at('l2');
  const think = f >= t0 - 4 && f < t1 + 6 ? Math.min(1, ease(f, t0 - 4, 8)) * (1 - ease(f, t1, 8)) : 0;
  const ring = lin(f, t0, Math.max(1, t1 - t0));
  return <AbsoluteFill>
    <Abs x={120} y={136} w={1680} style={up(p('l1'))}>
      <Card on={f < t1} style={{display: 'flex', gap: 22, alignItems: 'flex-start'}}>
        <Mono size={30} color={ACC} glow style={{flexShrink: 0}}>练习 {n}</Mono>
        <div style={{flexGrow: 1}}>{q}</div>
      </Card>
    </Abs>
    {think > 0 && <Svg opacity={think}>
      <circle cx={960} cy={560} r={70} fill="none" stroke={C.line} strokeWidth={6} />
      <circle cx={960} cy={560} r={70} fill="none" stroke={ACC} strokeWidth={6} strokeDasharray={`${2 * Math.PI * 70 * ring} 999`}
        transform="rotate(-90 960 560)" filter="url(#vglow)" />
      <text x={960} y={570} textAnchor="middle" fontFamily={F.zh} fontSize={30} fill={C.ink}>想一想</text>
    </Svg>}
    <Abs x={120} y={350} w={side ? 1060 : 1680} style={{display: 'flex', flexDirection: 'column', gap: 14}}>
      {answers.map((a) => <div key={a.key} style={up(p(a.key))}><Card on={f >= at(a.key) && f < at(a.key) + 90}>{a.body}</Card></div>)}
    </Abs>
    {side && <Abs x={1220} y={350} w={580} style={{opacity: ease(f, at('l2'), 12)}}>{side}</Abs>}
  </AbsoluteFill>;
};
const T: React.FC<{size?: number; c?: string; children?: React.ReactNode}> = ({size = 24, c = C.ink, children}) => <Txt size={size} color={c}>{children}</Txt>;

// ================================================================ 01 该写哪一天
export const B01: React.FC = () => <Ex n={1}
  q={<><T>（a）InstructGPT 该标 2022 年 1 月还是 3 月？</T><T>（b）V4 的论文编号是 2606.19348，为什么写成 2026-04-24？</T></>}
  answers={[
    {key: 'l2.a', body: <><T>（a）标 <span style={{color: ACC}}>1 月</span>：公告页与论文 PDF 在 2022-01-27 发出；arXiv v1 晚到 2022-03-04</T></>},
    {key: 'l2.b', body: <><T>（b）权重与技术报告 2026-04-24 已在 Hugging Face 公开；arXiv 的提交日是 04-26</T></>},
    {key: 'l3.a', body: <T c={ACC}>规则：模型写第一次公开的日子，再注明论文日期</T>},
    {key: 'l3.b', body: <T>DeepSeek-Coder 没有官方公告页：用 GitHub 最早一次提交（2023-11-02）作代理，并明确标出</T>},
  ]} />;

// ================================================================ 02 跟谁比
export const B02: React.FC = () => <Ex n={2}
  q={<><T>四句话，各写出比较的对象、单位和条件：</T><T size={22} c={C.ink2}>（a）V2 把 KV 降了 93.3%　（b）1M 下 V4-Pro 只要 V3.2 的 27% FLOPs、10% KV</T>
    <T size={22} c={C.ink2}>（c）DeepSeekMoE 16B 只用了 40% 的计算量　（d）V4.1-Flash 只用了 V4-Pro 1/4 的激活参数</T></>}
  answers={[
    {key: 'l2.a', body: <><T>（a）对象 67B · 单位：每 token 的 KV 字节 · 条件：部署态</T><div style={{marginTop: 6}}><Tex id="k933" size={26} /></div></>},
    {key: 'l2.b', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>（b）对象：外推到 1M 的 V3.2 · 等效 FP8 · 只比 1M 这一点；KV：</T><Tex id="k105" size={26} parts={{r: lit(1)}} /></div>},
    {key: 'l3.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>（c）对象 DeepSeek LLM 7B · 每 4K token 的前向 FLOPs：</T><Tex id="k40" size={26} /></div>},
    {key: 'l3.b', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>（d）解码、预填充、平均：</T><Tex id="k14" size={26} parts={{r: lit(1)}} /></div>},
  ]} />;

// ================================================================ 03 证据有多硬
const SIX: [string, string, string][] = [['两个双随机矩阵的乘积仍是双随机', '已证明', 'l2.a'], ['Engram 的最优分配在 75%–80%', '实测', 'l2.b'],
  ['RL 提升 Maj@K、不提升 Pass@K', '实测', 'l2.b'], ['辅助损失伤性能是因为干扰梯度', '推断', 'l3.a'], ['V4.1 取消了哈希路由', '推断', 'l3.a'],
  ['V4 / V4.1 的索引键每条 68 字节', '推算', 'l3.b']];
export const B03: React.FC = () => <Ex n={3}
    q={<><T>六句话，分成“已证明”“实测”“推算或推断”三类：</T>
      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 30px', marginTop: 6}}>
        {SIX.map(([s]) => <T key={s} size={21} c={C.ink2}>· {s}</T>)}</div></>}
    answers={[
      {key: 'l2.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><Chip on size={20}>已证明</Chip><Tex id="ds" size={26} /></div>},
      {key: 'l2.b', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><Chip size={20}>实测</Chip><T size={22}>Engram：只在两个算力点扫过；RL：K ≤ 64 下的观察，不是定理</T></div>},
      {key: 'l3.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><Chip size={20}>推断</Chip><T size={22}>“干扰梯度”是解释；取消哈希路由的证据是间接的（图注 + 配置）</T></div>},
      {key: 'l3.b', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><Chip size={20}>推算</Chip><Tex id="i68" size={26} parts={{r: lit(1)}} />
        <T size={20} c={C.ink2}>让 3,514 与 890 同时复现，但没有印在论文里</T></div>},
    ]} />;

// ================================================================ 04 两个 16B 的缓存
export const B04: React.FC = () => <Ex n={4}
  q={<><T>DeepSeekMoE 16B：MHA，28 层，16 头 × 128 维　V2-Lite：MLA，27 层，d_c = 512，d_h^R = 64　都按 BF16</T>
    <T size={22} c={C.ink2}>两者每 token 的 KV 各是多少？相差几倍？和 V2（69,120 字节）比呢？</T></>}
  answers={[
    {key: 'l2.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>16B：</T><Tex id="e4a" size={30} parts={{r: lit(1)}} /></div>},
    {key: 'l2.b', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>V2-Lite：</T><Tex id="e4b" size={30} parts={{r: lit(1)}} /></div>},
    {key: 'l2.c', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><Tex id="e4c" size={28} /><T size={22} c={C.ink2}>参数量几乎一样，差别几乎全在注意力</T></div>},
    {key: 'l3.b', body: <T c={ACC}>每 token 的 KV 由注意力结构、层数、字节决定：和总参数、激活参数没有直接关系</T>},
  ]} />;

// ================================================================ 05 一个请求的显存
export const B05: React.FC = () => <Ex n={5}
  q={<><T>只算全局 KV；1M = 2²⁰ token，128K = 2¹⁷ token，1 GB = 10⁹ 字节</T>
    <T size={22} c={C.ink2}>V4-Pro（约 5,031 字节/token）一个 1M 请求多少 GB，80 GB 的卡放几个？V3.2 一个 128K 请求呢？</T></>}
  answers={[
    {key: 'l2.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>V4-Pro：</T><Tex id="e5a" size={28} parts={{r: lit(1)}} /></div>},
    {key: 'l2.b', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>V3.2：</T><Tex id="e5b" size={28} parts={{r: lit(1)}} /></div>},
    {key: 'l2.c', body: <T c={ACC}>上下文长了 8 倍，缓存反而少了 16%</T>},
    {key: 'l3.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>67B 填满 80 GB：</T><Tex id="e5c" size={28} parts={{r: lit(1)}} />
      <T size={22} c={C.ink2}>token；但它实际只支持 4,096</T></div>},
  ]} />;

// ================================================================ 06 3.95 倍的拆法
export const B06: React.FC = () => <Ex n={6}
  q={<><T>V4-Flash 3,514.25 字节/token，V4.1-Flash 890：差 3.95 倍</T>
    <T size={22} c={C.ink2}>4 个 Full 层每 token 共摊 2.5 条缓存，每条再加 68 字节索引键。把 3.95 拆成“结构 × 精度”</T></>}
  answers={[
    {key: 'l2.a', body: <><T size={20} c={C.ink2}>中间模型 ①：条目仍用 V4 的 584 字节</T><div style={{marginTop: 6}}><Tex id="e6a" size={30} parts={{s: lit(1), p: lit(0.5)}} /></div></>},
    {key: 'l3.a', body: <><T size={20} c={C.ink2}>中间模型 ②：改用 V4.1 的 FP8 格式（528 字节）</T><div style={{marginTop: 6}}><Tex id="e6b" size={30} parts={{s: lit(1), p: lit(0.5)}} /></div></>},
    {key: 'l3.b', body: <T c={ACC}>中间模型不同，两种都对；共同结论：结构这一步比格式那一步省得多</T>},
  ]} />;

// ================================================================ 07 缩小的速度
export const B07: React.FC = () => {
  const {f, at} = useBoard();
  const X0 = 40, W = 520;
  const pts: [string, number, string][] = [['67B', 0, '389,120'], ['V3', 393, '70,272'], ['V4.1-Flash', 1016, '890']];
  const lx = (d: number) => X0 + (W * d) / 1016, ly = (v: number) => 300 - (Math.log10(v) - 2.9) / (5.7 - 2.9) * 260;
  const vals = [389120, 70272, 890];
  return <Ex n={7}
    q={<><T>67B（2023-11-29，389,120 字节）→ V3（2024-12-26，70,272）→ V4.1-Flash（2026-09-10，890）</T>
      <T size={22} c={C.ink2}>两段各隔多少天？按 r^(365/Δt) 年化，各是每年几倍？</T></>}
    answers={[
      {key: 'l2.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>67B → V3：393 天，÷5.54</T><Tex id="e7a" size={28} parts={{r: lit(1)}} /><T size={22}>倍 / 年</T></div>},
      {key: 'l2.b', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>V3 → V4.1：623 天，÷79.0</T><Tex id="e7b" size={28} parts={{r: lit(1)}} /><T size={22}>倍 / 年</T></div>},
      {key: 'l3.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>全程：</T><Tex id="e7c" size={28} /><T size={22} c={ACC}>V3 之后明显加快</T></div>},
      {key: 'l3.b', body: <T size={22} c={C.ink2}>按发布日做的年化（笔者口径）：不是规律，不能外推下一代</T>},
    ]}
    side={<Svg w={580} h={360}>
      <line x1={X0} x2={X0 + W} y1={320} y2={320} stroke={C.ink3} />
      <path d={pts.map(([, d], i) => `${i ? 'L' : 'M'}${lx(d)} ${ly(vals[i])}`).join(' ')} fill="none" stroke={ACC} strokeWidth={2.4}
        strokeDasharray={`${900 * lin(f, at('l2.a'), 40)} 900`} />
      {pts.map(([n, d, v], i) => <g key={n}>
        <circle cx={lx(d)} cy={ly(vals[i])} r={7} fill={ACC} />
        <text x={lx(d) + (i === 2 ? -10 : 10)} y={ly(vals[i]) - 12} textAnchor={i === 2 ? 'end' : 'start'} fontFamily={F.mono} fontSize={16} fill={C.ink}>{n} · {v}</text>
      </g>)}
      <text x={X0} y={350} fontFamily={F.zh} fontSize={16} fill={C.ink3}>纵轴对数：斜率越陡，缩得越快</text>
    </Svg>} />;
};

// ================================================================ 08 2.25 组 GQA
export const B08: React.FC = () => <Ex n={8}
  q={<><T>（a）从式 (0.1) 出发，写出 MHA、GQA、MQA、MLA 的 n_copy、m、n_dim</T><T>（b）证明：V2 的 MLA 缓存等于只有 2.25 组的 GQA</T></>}
  answers={[
    {key: 'l2.a', body: <T>四种都是每层一份、每个 token 一条：n_copy = l，m = 1</T>},
    {key: 'l2.b', body: <Tex id="e8" size={26} />},
    {key: 'l3.a', body: <div style={{display: 'flex', alignItems: 'center', gap: 20}}><T>V2：d_c = 4d_h，d_h^R = d_h / 2 →</T><Tex id="e8b" size={30} parts={{r: lit(1)}} /></div>},
    {key: 'l3.b', body: <T size={22} c={C.ink2}>GQA 只能按整组计数；MLA 的条目是一个潜向量，不是任何一个头的键或值：“2.25 组”只是等字节换算</T>},
  ]} />;

// ================================================================ 09 UE8M0 不溢出
export const B09: React.FC = () => <Ex n={9}
  q={<><T>UE8M0：s = 2^⌈y⌉，y = log₂(a / 448)，a 是组里的最大绝对值</T><T size={22} c={C.ink2}>证明缩放后每个数都不超过 448，且最大的数落在 (224, 448]；求利用率 u = a / (448 s) 的期望</T></>}
  answers={[
    {key: 'l2.a', body: <Tex id="e9a" size={30} />},
    {key: 'l2.b', body: <Tex id="e9b" size={30} />},
    {key: 'l3.a', body: <Tex id="e9c" size={30} parts={{r: lit(1)}} />},
    {key: 'l3.b', body: <Tex id="e9d" size={26} />},
  ]} />;

// ================================================================ 10 九道题之后
export const B10: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useEp().roadmap as Roadmap;
  const L = syllabusLayout(rm);
  const card1 = L.cards.find((c) => c.n === 1)!;
  const out = ease(f, at('l2.a') - 4, 18);
  const v = camAt(f, fit(L.bounds, 0.05), [{at: at('l2.a') + 10, dur: 70, to: fit(card1, 0.25)}]);
  const KINDS: [string, string, string][] = [['概念题', '认日期、问口径、分证据', 'l1.a'], ['计算题', '用式 (0.1) 复算缓存、显存与倍数', 'l1.b'],
    ['推导题', '把一个结论从头推出来', 'l1.c']];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - out}}>
      <Abs x={120} y={160} style={up(p('l1'))}><Txt size={30}>九道题做完了</Txt></Abs>
      <Abs x={120} y={250} w={1680} style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 22}}>
        {KINDS.map(([k, d, key]) => <div key={k} style={up(p(key))}><Card on style={{height: 220, boxSizing: 'border-box'}}>
          <Txt size={34} weight={400} color={ACC}>{k}</Txt><Txt size={24} style={{marginTop: 12}}>{d}</Txt></Card></div>)}
      </Abs>
    </AbsoluteFill>
    {out > 0 && <AbsoluteFill style={{opacity: out, background: C.bg}}>
      <Stage><Syllabus rm={rm} cam={project(v)} L={L} lit={(n) => (n === 1 ? ease(f, at('l2.a') + 50, 14) : 0)}
        focus={f >= at('l2.a') + 40 ? 1 : undefined} /></Stage>
      <Abs x={0} y={780} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l2.a') + 60, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={32} color={ACC} glow weight={400}>第 0 课结束 · 下一课：第 1 课，Transformer 解剖</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};
