// 第 0 课 §10 · 五个贯穿全课的问题 —— 6 段黑板。例子全部来自 lesson-00 §10 及其引用的本课各页
// （研究笔记 lesson-00-doc.md）。最后一段把镜头交回课程表，推到第 1 课。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Chip, F, Mono, Roadmap, Stage, Syllabus, Tex, Txt, camAt, ease, fit, lin, lit, project, rgba, syllabusLayout,
  up, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';

const Card: React.FC<{on?: boolean; dashed?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, dashed, style, children}) =>
  <div style={{padding: '16px 22px', borderRadius: 12, background: rgba(C.bg, 0.92),
    border: `1px ${dashed ? 'dashed' : 'solid'} ${on ? rgba(ACC, 0.7) : C.line}`, boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;
const Num: React.FC<{n: number}> = ({n}) => <Mono size={26} color={ACC} glow style={{marginRight: 14}}>{['一', '二', '三', '四', '五'][n - 1]}</Mono>;

// ================================================================ 01 账单在哪一行
const ROWS = ['显存容量', '每一步读的字节', '每个 token 的运算', '跨卡通信', '训练算力', '训练信号的质量'];
// design, rows it saves, rows it pays, mark
const LEDGER: [string, number[], number[], string, string][] = [
  ['MLA', [1], [2], 'l2.a', '付：每个缓存 token 约 3.4 倍运算'], ['DSA', [1, 2], [0], 'l2.b', '付：缓存多约 20%'],
  ['GRPO', [0], [], 'l3.a', '省：一个和策略一样大的价值网络'], ['V4.1 编码器–解码器', [2], [2], 'l3.b', '省预填充；解码多激活一些参数'],
];
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const X0 = 520, CW = 300, Y0 = 280, RH = 72;
  return <AbsoluteFill>
    <Abs x={120} y={140} style={{display: 'flex', alignItems: 'center', ...up(p('l1'))}}><Num n={1} /><Txt size={30}>账单在哪一行？</Txt></Abs>
    {ROWS.map((r, i) => <Abs key={r} x={120} y={Y0 + i * RH} w={380} style={{height: RH - 10, display: 'flex', alignItems: 'center',
      borderBottom: `1px solid ${C.rule}`, ...up(ease(f, at('l1.a') + i * 5, 10))}}><Txt size={24}>{r}</Txt></Abs>)}
    {LEDGER.map(([name, save, pay, key, note], j) => {
      const k = ease(f, at(key), 12);
      return <React.Fragment key={name}>
        <Abs x={X0 + j * CW} y={Y0 - 60} w={CW - 20} style={{textAlign: 'center', ...up(k)}}><Mono size={22} color={ACC}>{name}</Mono></Abs>
        {ROWS.map((_, i) => {
          const s = save.includes(i), py = pay.includes(i);
          if (!s && !py) return null;
          return <Abs key={i} x={X0 + j * CW + (CW - 20) / 2 - 60} y={Y0 + i * RH + 8} w={120} style={{height: RH - 26, borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: k,
            background: s && !py ? rgba(ACC, 0.25) : rgba(C.ink2, 0.2), border: `1px solid ${s && !py ? ACC : C.ink2}`}}>
            <Txt size={22} color={s && !py ? ACC : C.ink}>{s && py ? '省 / 付' : s ? '↓ 省' : '↑ 付'}</Txt></Abs>;
        })}
        <Abs x={X0 + j * CW} y={Y0 + 6 * RH + 6} w={CW - 20} style={{textAlign: 'center', ...up(ease(f, at(key) + 14, 12))}}>
          <Txt size={18} color={C.ink2}>{note}</Txt></Abs>
      </React.Fragment>;
    })}
    <Abs x={120} y={Y0 + 6 * RH + 70} style={up(p('l1.b'))}><Chip on size={22}>同一个“省”字，指的可能是完全不同的资源</Chip></Abs>
  </AbsoluteFill>;
};

// ================================================================ 02 跟谁比，按什么口径
const HEAD: [string, React.ReactNode, string][] = [
  ['KV 降了 93.3%', <><Txt size={21}>6 bit 部署的 V2 对 BF16 的 67B</Txt><div style={{marginTop: 8}}><Tex id="p822" size={28} parts={{r: lit(1)}} /></div>
    <Txt size={20} color={ACC}>两边同为 BF16：82.2%</Txt></>, 'l2.a'],
  ['27% FLOPs、10% KV', <><Txt size={21}>只在 1M 上下文</Txt><Txt size={21}>以外推的 V3.2 为基线</Txt><Txt size={21} color={ACC}>按等效 FP8 计</Txt></>, 'l2.b'],
  ['DeepSeekMoE 16B 只用 40% 计算', <Txt size={21} color={ACC}>是 FLOPs 之比，不是参数之比</Txt>, 'l3.a'],
  ['557.6 万美元 · 1/4 激活参数', <><Txt size={21}>不含研究与消融</Txt><Txt size={21} color={ACC}>“1/4”对不上任何一个口径</Txt></>, 'l3.b'],
];
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  return <AbsoluteFill>
    <Abs x={120} y={140} style={{display: 'flex', alignItems: 'center', ...up(p('l1'))}}><Num n={2} /><Txt size={30}>这个数在跟谁比，按什么口径？</Txt></Abs>
    <Abs x={120} y={200} style={up(p('l1.a'))}><Txt size={22} color={C.ink2}>头条数字几乎都有一个没写在标题里的分母</Txt></Abs>
    <Abs x={120} y={260} w={1680} style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20}}>
      {HEAD.map(([h, body, key], i) => {
        const on = f >= at(key) && (i === HEAD.length - 1 || f < at(HEAD[i + 1][2]));
        return <div key={h} style={up(p(key))}><Card on={on} style={{height: 270, boxSizing: 'border-box'}}>
          <Txt size={28} weight={400}>“{h}”</Txt>
          <div style={{marginTop: 12, opacity: ease(f, at(key) + 12, 12)}}>{body}</div>
        </Card></div>;
      })}
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 03 证据有多硬
const EVID: [string, string[], string][] = [
  ['证明', ['双随机矩阵的谱范数恰为 1：mHC 的复合映射不会放大', 'KV cache 的精确性由因果掩码保证'], 'l2.a'],
  ['实测', ['Engram 的 U 形曲线：只在两个算力点上测过', 'RL 提升 Maj@K、不提升 Pass@K：K ≤ 64 下的实测，不是定理'], 'l2.b'],
  ['推测或解释', ['“辅助损失伤性能是因为干扰梯度”', '“光学遗忘”：只是设想，没有多轮实验', '“aha moment”：要和长度设置的改动一起读'], 'l3.a'],
];
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  return <AbsoluteFill>
    <Abs x={120} y={140} style={{display: 'flex', alignItems: 'center', ...up(p('l1'))}}><Num n={3} /><Txt size={30}>证据有多硬：证明、实测，还是推测？</Txt></Abs>
    <Abs x={120} y={200} style={up(p('l1.a'))}><Txt size={22} color={C.ink2}>同一篇论文里，三种强度的说法常常并排出现</Txt></Abs>
    <Abs x={120} y={270} w={1680} style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 22}}>
      {EVID.map(([kind, items, key], i) => {
        const on = f >= at(key) && (i === EVID.length - 1 || f < at(EVID[i + 1][2]));
        return <div key={kind} style={up(p(key))}><Card on={on} dashed={i === 2} style={{height: 470, boxSizing: 'border-box'}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 12}}>
            {[0, 1, 2].map((b) => <div key={b} style={{width: 26, height: 14, borderRadius: 3, background: b < 3 - i ? ACC : rgba(C.ink2, 0.25)}} />)}
            <Txt size={30} weight={400} style={{marginLeft: 8}}>{kind}</Txt></div>
          {items.map((t, j) => <Txt key={t} size={22} color={C.ink} style={{marginTop: 16, opacity: ease(f, at(key) + 10 + j * 18, 12)}}>· {t}</Txt>)}
        </Card></div>;
      })}
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 04 代价转移到了哪里
const TRADE: [string, string, string, string][] = [
  ['MoE', '显存和通信', '计算', 'l2.a'], ['mHC', '访存', '参数和算术（几乎为零）', 'l2.a'],
  ['CSA / HCA', '逐 token 的细节', '缓存（1M 装得下 ≠ 用得好）', 'l2.b'],
  ['V4.1-Flash', '更大的模型、短上下文下更贵的解码、两处近似', '输入重负载下的存储与搬运', 'l3.a'],
  ['R1', '更长的思考：推理成本随输出长度线性甚至平方增长', '正确率', 'l3.b'],
];
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  return <AbsoluteFill>
    <Abs x={120} y={140} style={{display: 'flex', alignItems: 'center', ...up(p('l1'))}}><Num n={4} /><Txt size={30}>代价转移到了哪里？</Txt></Abs>
    <Abs x={120} y={200} style={up(p('l1.a'))}><Txt size={22} color={C.ink2}>几乎没有免费的改进：省下的东西，通常换成了另一样东西</Txt></Abs>
    <Abs x={120} y={262} w={1680}>
      <div style={{display: 'flex', fontSize: 18, color: C.ink3, paddingBottom: 6, borderBottom: `1px solid ${C.line}`}}>
        <div style={{width: 260}}>设计</div><div style={{width: 760}}>付出的</div><div style={{width: 80}} /><div>换来的</div></div>
      {TRADE.map(([d, pay, get, key], i) => {
        const k = ease(f, at(key) + (i === 1 ? 14 : 0), 12);
        return <div key={d} style={{display: 'flex', alignItems: 'center', minHeight: 96, borderBottom: `1px solid ${C.rule}`, ...up(k)}}>
          <Mono size={24} color={ACC} style={{width: 260}}>{d}</Mono>
          <Txt size={23} style={{width: 760}}>{pay}</Txt>
          <Mono size={28} color={C.ink2} style={{width: 80}}>⇄</Mono>
          <Txt size={23} color={ACC}>{get}</Txt>
        </div>;
      })}
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 下一个瓶颈在哪里
const INF: [string, string, string][] = [['参数和计算绑在一起', 'MoE', 'l2.a'], ['解码时读 KV 的字节', 'MLA', 'l2.b'], ['长上下文的注意力计算', 'DSA', 'l2.c'],
  ['百万 token 的缓存与计算', 'CSA / HCA', 'l2.d'], ['智能体负载下缓存的存储与搬运', 'V4.1', 'l2.e']];
const TRN: [string, string, string][] = [['价值网络的显存', 'GRPO', 'l3.a'], ['训练信号的来源', 'R1 的规则奖励', 'l3.b'], ['规模化 RL 的稳定性', 'V3.2', 'l3.c']];
const DROPPED = ['V2 的 token 丢弃', 'V3 的节点受限路由', 'V4 的哈希路由与 HCA', '预训练里的 MTP', '稠密注意力热身', 'R1 没用上的 PRM 与 MCTS'];
export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const row = (items: [string, string, string][], y: number, w: number) => <Abs x={120} y={y} style={{display: 'flex', alignItems: 'center', gap: 10}}>
    {items.map(([b, s, key], i) => <React.Fragment key={b}>
      {i > 0 && <Mono size={26} color={C.ink2} style={{opacity: ease(f, at(key), 10)}}>→</Mono>}
      <div style={{width: w, padding: '12px 14px', borderRadius: 10, border: `1px solid ${ACC}`, background: rgba(ACC, 0.06), ...up(ease(f, at(key), 12))}}>
        <Txt size={20} color={C.ink}>{b}</Txt><Mono size={20} color={ACC}>→ {s}</Mono></div>
    </React.Fragment>)}
  </Abs>;
  const gone = ease(f, at('l4.a'), 14);
  return <AbsoluteFill>
    <Abs x={120} y={140} style={{display: 'flex', alignItems: 'center', ...up(p('l1'))}}><Num n={5} /><Txt size={30}>下一个瓶颈在哪里？</Txt></Abs>
    <Abs x={120} y={200} style={up(p('l1.a'))}><Txt size={22} color={C.ink2}>每解决一个瓶颈，账单就落到下一行</Txt></Abs>
    <Abs x={120} y={262} style={up(p('l2'))}><Txt size={22} color={C.ink3}>推理一侧</Txt></Abs>
    {row(INF, 300, 300)}
    <Abs x={120} y={432} style={up(p('l3'))}><Txt size={22} color={C.ink3}>训练一侧</Txt></Abs>
    {row(TRN, 470, 300)}
    <Abs x={120} y={620} w={1680} style={{opacity: gone}}>
      <Txt size={22} color={C.ink3}>瓶颈移动之后被放弃的设计</Txt>
      <div style={{display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 10}}>
        {DROPPED.map((d, i) => <div key={d} style={{position: 'relative', padding: '6px 14px', border: `1px dashed ${C.ink3}`, borderRadius: 8,
          opacity: ease(f, at('l4.a') + 6 + i * 6, 10)}}>
          <Txt size={20} color={C.ink2}>{d}</Txt>
          <div style={{position: 'absolute', left: 6, right: 6, top: '52%', height: 1.5, background: C.ink2,
            transform: `scaleX(${lin(f, at('l4.a') + 20 + i * 6, 12)})`, transformOrigin: 'left'}} /></div>)}
      </div>
      <div style={{marginTop: 18, ...up(p('l4.b'))}}><Chip on size={24}>为什么被放弃：第 20 课</Chip></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 06 带着五个问题去读
const FIVE = ['账单在哪一行：省的是哪一种资源？', '这个数在跟谁比，按什么口径？', '证据有多硬：证明、实测，还是推测？', '代价转移到了哪里？', '下一个瓶颈在哪里？'];
export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useEp().roadmap as Roadmap;
  const L = syllabusLayout(rm);
  const card1 = L.cards.find((c) => c.n === 1)!;
  const out = ease(f, at('l3.a') - 4, 18);
  const v = camAt(f, fit(L.bounds, 0.05), [{at: at('l3.a') + 10, dur: 70, to: fit(card1, 0.25)}]);
  const lights: [number[], string][] = [[[0, 1], 'l1.a'], [[2, 3], 'l2.a'], [[4], 'l2.b']];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - out}}>
      <Abs x={120} y={150} style={up(p('l1'))}><Txt size={30}>带着这五个问题去读</Txt></Abs>
      <Abs x={120} y={230} w={1680} style={{display: 'flex', flexDirection: 'column', gap: 14}}>
        {FIVE.map((q, i) => {
          const g = lights.find(([ix]) => ix.includes(i))!;
          const k = ease(f, at(g[1]), 12);
          return <div key={q} style={{display: 'flex', alignItems: 'center', gap: 20, padding: '16px 24px', borderRadius: 12,
            border: `1px solid ${k > 0.5 ? rgba(ACC, 0.7) : C.line}`, background: rgba(C.bg, 0.9), boxShadow: k > 0.5 ? BOX_GLOW : undefined,
            ...up(ease(f, at('l1') + i * 6, 12))}}>
            <Mono size={30} color={ACC}>{i + 1}</Mono><Txt size={30} weight={400}>{q}</Txt>
            <Txt size={20} color={C.ink2} style={{marginLeft: 'auto', opacity: k}}>{i < 2 ? '对付头条数字' : i < 4 ? '对付论文里的结论' : '把整棵树连起来'}</Txt>
          </div>;
        })}
      </Abs>
    </AbsoluteFill>
    {out > 0 && <AbsoluteFill style={{opacity: out, background: C.bg}}>
      <Stage><Syllabus rm={rm} cam={project(v)} L={L} lit={(n) => (n === 1 ? ease(f, at('l3.a') + 50, 14) : 0)}
        focus={f >= at('l3.a') + 40 ? 1 : undefined} /></Stage>
      <Abs x={0} y={780} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.b'), 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={32} color={ACC} glow weight={400}>下一课 · 第 1 课：一个标准 GPT 的一次前向传播</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};

export {F};
