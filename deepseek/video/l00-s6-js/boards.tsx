// 第 0 课 §6 · 主线 E：残差流与记忆 —— 9 段黑板。数字全部来自 lesson-00 §6（研究笔记 lesson-00-doc.md）；
// 标"示意"的是：03 段的 4×4 双随机矩阵、04 段两条流的初值 (0.9, 0.1)、02 段的传送带。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Check, Chip, F, Mono, Roadmap, Svg, Tex, Txt, camAt, ease, fit, lin, lit, mix, project, rgba, swing, up,
  useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, TreeScene, laneRect} from '../l00-shared/scene';
import {Chain, Stop} from '../l00-shared/chain';

const useRm = () => useEp().roadmap as Roadmap;
const Card: React.FC<{on?: boolean; dashed?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, dashed, style, children}) =>
  <div style={{padding: '18px 24px', borderRadius: 12, background: rgba(C.bg, 0.92),
    border: `1px ${dashed ? 'dashed' : 'solid'} ${on ? rgba(ACC, 0.7) : C.line}`, boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;
const Note: React.FC<{x: number; y: number; p?: number; children?: React.ReactNode}> = ({x, y, p = 1, children}) =>
  <Abs x={x} y={y} style={{opacity: p}}><Mono size={16} color={C.ink3} style={{letterSpacing: '0.08em'}}>{children}</Mono></Abs>;

// ================================================================ 01 捷径上的系数
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const chart = ease(f, at('l2'), 14);
  // residual diagram
  const SY = 380, pulse = ((f % 70) / 70) * 1600 + 160;
  // gain chart: layers 0…60 against log10 gain from −2 to 4
  const X0 = 220, X1 = 1280, YB = 760, YT = 200;
  const gx = (n: number) => X0 + ((X1 - X0) * n) / 60, gy = (v: number) => YB - ((Math.log10(v) + 2) / 6) * (YB - YT);
  const curve = (lam: number, k: number) => Array.from({length: 61}, (_, n) => n).filter((n) => n <= 60 * k)
    .map((n, i) => `${i ? 'L' : 'M'}${gx(n)} ${gy(lam ** n)}`).join(' ');
  const LINES: [number, string, string, string | null][] = [[1, 'l2', '×1', null], [1.05, 'l2.a', '×1.05', 'g105'], [0.95, 'l2.b', '×0.95', 'g095'],
    [1.1428, 'l3.a', '×1.1428', 'g114']];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - chart}}>
      <Svg>
        <line x1={160} x2={1760} y1={SY} y2={SY} stroke={C.ink2} strokeWidth={3} opacity={ease(f, at('l1'), 12)} />
        <circle cx={pulse} cy={SY} r={8} fill={ACC} filter="url(#vglow)" opacity={ease(f, at('l1.b'), 12)} />
        {[0, 1, 2, 3].map((i) => {
          const x = 380 + i * 360, o = ease(f, at('l1') + 6 + i * 4, 12);
          return <g key={i} opacity={o}>
            <path d={`M${x - 60} ${SY} L${x - 60} 280`} stroke={C.ink3} strokeWidth={1.6} fill="none" />
            <rect x={x - 90} y={200} width={180} height={80} rx={10} fill={C.panel} stroke={f >= at('l1.a') ? ACC : C.line} />
            <text x={x} y={250} textAnchor="middle" fontFamily={F.mono} fontSize={24} fill={C.ink}>F{i + 1}</text>
            <path d={`M${x + 60} 280 L${x + 60} ${SY - 16}`} stroke={ACC} strokeWidth={1.6} fill="none" />
            <circle cx={x + 60} cy={SY} r={15} fill={C.bg} stroke={ACC} strokeWidth={1.6} />
            <text x={x + 60} y={SY + 7} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={ACC}>+</text>
          </g>;
        })}
        <text x={160} y={SY + 44} fontFamily={F.zh} fontSize={22} fill={C.ink2} opacity={ease(f, at('l1'), 12)}>残差流：每层只把增量加回去</text>
      </Svg>
      <Abs x={160} y={500} style={up(p('l1.a'))}><Tex id="res" size={48} parts={{f: f >= at('l1.a') && f < at('l1.b') ? lit(1) : {}}} /></Abs>
      <Abs x={160} y={610} style={up(p('l1.b'))}><Tex id="unroll" size={48} parts={{x: lit(1)}} /></Abs>
      <Abs x={1000} y={630} w={760} style={up(p('l1.b'))}><Txt size={24} color={C.ink2}>最初的输入以系数 1 出现在任意深处；梯度原样传回</Txt></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: chart}}>
      <Abs x={X0} y={130} style={up(chart)}><Txt size={26} color={C.ink2}>捷径上每层乘 λ：60 层以后（纵轴是对数）</Txt></Abs>
      <Svg>
        {[0.01, 0.1, 1, 10, 100, 1000, 10000].map((v) => <g key={v}>
          <line x1={X0} x2={X1} y1={gy(v)} y2={gy(v)} stroke={v === 1 ? C.ink3 : C.rule} />
          <text x={X0 - 12} y={gy(v) + 6} textAnchor="end" fontFamily={F.mono} fontSize={16} fill={C.ink3}>{v >= 1 ? v.toLocaleString('en-US') : v}</text>
        </g>)}
        {[0, 20, 40, 60].map((n) => <text key={n} x={gx(n)} y={YB + 28} textAnchor="middle" fontFamily={F.mono} fontSize={16} fill={C.ink3}>{n}</text>)}
        <text x={X1} y={YB + 54} textAnchor="end" fontFamily={F.zh} fontSize={18} fill={C.ink3}>层数 →</text>
        {LINES.map(([lam, key, label]) => {
          const k = lin(f, at(key) + 4, 34);
          if (k <= 0) return null;
          const safe = lam === 1;
          return <g key={key}>
            <path d={curve(lam, k)} fill="none" stroke={safe ? C.ink : ACC} strokeWidth={safe ? 2 : 2.6} filter={safe ? undefined : 'url(#vglow)'} />
            {k >= 1 && <circle cx={gx(60)} cy={gy(lam ** 60)} r={7} fill={safe ? C.ink : ACC} />}
            {k >= 1 && <text x={gx(60) + 14} y={gy(lam ** 60) + 6} fontFamily={F.mono} fontSize={20} fill={safe ? C.ink : ACC}>{label}</text>}
          </g>;
        })}
      </Svg>
      {LINES.filter((l) => l[3]).map(([lam, key, , fm]) => <Abs key={key} x={1440} y={gy(lam ** 60) - 22} style={up(ease(f, at(key) + 30, 12))}>
        <Tex id={fm as string} size={34} parts={{r: lit(1)}} /></Abs>)}
      <Abs x={1440} y={gy(1) - 22} style={up(ease(f, at('l2') + 30, 12))}><Txt size={24}>只有 1 是安全的</Txt></Abs>
      <Abs x={X0} y={836} style={{display: 'flex', alignItems: 'center', gap: 26, ...up(p('l3.b'))}}>
        <Tex id="lam" size={34} parts={{l: lit(1)}} /><Chip on size={24}>偏离 1 的代价，随深度指数增长</Chip></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 02 传送带
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const many = ease(f, at('l2'), 14), edge = ease(f, at('l3'), 14);
  const belt = (y: number, key: string, o = 1, hot = false) => <g key={key} opacity={o}>
    <rect x={160} y={y - 18} width={1600} height={36} rx={18} fill={C.panel} stroke={hot ? ACC : C.line} />
    <line x1={170} x2={1750} y1={y} y2={y} stroke={C.ink3} strokeDasharray="10 14" strokeDashoffset={-f * 2} />
  </g>;
  const parts = (y: number, n: number, seed: number) => Array.from({length: n}, (_, i) => {
    const x = 170 + ((i * 1580) / n + f * 2 + seed * 37) % 1580;
    return <rect key={i} x={x} y={y - 11} width={22} height={22} rx={3} fill={rgba(ACC, 0.7)} />;
  });
  const Y4 = [420, 500, 580, 660];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - many}}>
      <Abs x={160} y={160} style={up(p('l1'))}><Txt size={28} color={C.ink2}>残差流 = 一条传送带</Txt></Abs>
      <Svg>
        {belt(560, 'b0', ease(f, at('l1'), 12))}
        {parts(560, 9, 0)}
        {[0, 1, 2, 3].map((i) => {
          const x = 360 + i * 380;
          return <g key={i} opacity={ease(f, at('l1.a') + i * 4, 12)}>
            <rect x={x - 80} y={330} width={160} height={90} rx={10} fill={C.panel} stroke={ACC} />
            <text x={x} y={384} textAnchor="middle" fontFamily={F.zh} fontSize={24} fill={C.ink}>工位 {i + 1}</text>
            <path d={`M${x - 40} 540 L${x - 40} 424`} stroke={C.ink3} strokeWidth={1.6} fill="none" />
            <path d={`M${x + 40} 424 L${x + 40} 540`} stroke={ACC} strokeWidth={1.6} fill="none" />
            <text x={x + 52} y={490} fontFamily={F.mono} fontSize={20} fill={ACC}>+Δ</text>
          </g>;
        })}
      </Svg>
      <Abs x={160} y={640} style={up(p('l1.b'))}><Txt size={24} color={C.ink2}>没人动的件，原样往下走 = 恒等映射</Txt></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: many * (1 - edge)}}>
      <Abs x={160} y={160} style={up(p('l2.a'))}><Txt size={28} color={C.ink2}>Hyper-Connections：几条并行的传送带 + 带与带之间的调度</Txt></Abs>
      <Svg>
        {Y4.map((y, i) => belt(y, `m${i}`, ease(f, at('l2.a') + i * 4, 12)))}
        {Y4.map((y, i) => <g key={`p${i}`} opacity={ease(f, at('l2.a') + 10, 12)}>{parts(y, 6, i)}</g>)}
        {Y4.map((ya, i) => Y4.map((yb, j) => i !== j ? <path key={`${i}${j}`} d={`M900 ${ya} C960 ${ya} 960 ${yb} 1020 ${yb}`} fill="none"
          stroke={rgba(ACC, 0.35)} strokeWidth={1.2} opacity={ease(f, at('l2.a') + 20, 12)} /> : null))}
        <rect x={880} y={390} width={160} height={300} rx={10} fill="none" stroke={ACC} strokeDasharray="5 5" opacity={ease(f, at('l2.a') + 20, 12)} />
        <text x={960} y={720} textAnchor="middle" fontFamily={F.zh} fontSize={20} fill={ACC} opacity={ease(f, at('l2.a') + 20, 12)}>调度</text>
      </Svg>
      <Abs x={1100} y={760} style={up(p('l2.b'))}><Chip on size={24}>mHC：每条带送出 = 1，收进 = 1</Chip></Abs>
      <Abs x={160} y={760} style={up(p('l2.c'))}><Txt size={24}>件既不会越传越多，也不会凭空消失</Txt></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: edge}}>
      <Abs x={160} y={170} style={up(p('l3.a'))}><Txt size={28} color={C.ink2}>只管送出、不管收进：某条带仍可能被挤爆</Txt></Abs>
      <Svg opacity={ease(f, at('l3.a'), 12)}>
        {Y4.map((y, i) => belt(y - 60, `e${i}`, 1, i === 1))}
        {Array.from({length: Math.round(18 * lin(f, at('l3.a') + 10, 60))}, (_, k) => <rect key={k} x={980 + (k % 6) * 26} y={440 - 60 - 26 * Math.floor(k / 6) - 11}
          width={22} height={22} rx={3} fill={ACC} />)}
        <text x={1180} y={360} fontFamily={F.zh} fontSize={22} fill={ACC}>挤爆</text>
      </Svg>
      <Abs x={160} y={680} w={1600} style={up(p('l3.b'))}>
        <Card><Txt size={24}>比方的边界：真实的残差流是向量相加；“件”表达不了方向，也表达不了正负抵消</Txt></Card></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 03 一条流变四条
const DS = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2], [0.2, 0.1, 0.4, 0.3], [0.3, 0.2, 0.1, 0.4]];   // 示意: doubly stochastic
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const Y = [400, 460, 520, 580];
  const stage = f >= at('l1.c') ? 'res' : f >= at('l1.b') ? 'post' : f >= at('l1.a') ? 'pre' : '';
  const mat = ease(f, at('l2.b'), 14);
  return <AbsoluteFill>
    <Abs x={0} y={150} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}>
      <Tex id="hc" size={46} parts={{pre: stage === 'pre' ? lit(1) : {}, post: stage === 'post' ? lit(1) : {}, res: stage === 'res' || mat > 0 ? lit(1) : {}}} />
    </Abs>
    <Svg opacity={1 - 0.75 * mat}>
      {Y.map((y, i) => <g key={i} opacity={ease(f, at('l1') + i * 4, 12)}>
        <line x1={160} x2={1760} y1={y} y2={y} stroke={C.ink2} strokeWidth={2} />
        <text x={140} y={y + 6} textAnchor="end" fontFamily={F.mono} fontSize={18} fill={C.ink3}>流 {i + 1}</text>
      </g>)}
      {/* H_pre: read one input from the n streams */}
      {Y.map((y, i) => <path key={`r${i}`} d={`M420 ${y} C520 ${y} 560 300 640 300`} fill="none" stroke={stage === 'pre' ? ACC : C.ink3}
        strokeWidth={1.4} opacity={ease(f, at('l1.a'), 12)} />)}
      <rect x={640} y={270} width={180} height={60} rx={10} fill={C.panel} stroke={C.ink} opacity={ease(f, at('l1.a'), 12)} />
      <text x={730} y={308} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={C.ink} opacity={ease(f, at('l1.a'), 12)}>F_l</text>
      {/* H_post: write the output back to every stream */}
      {Y.map((y, i) => <path key={`w${i}`} d={`M820 300 C900 300 940 ${y} 1020 ${y}`} fill="none" stroke={stage === 'post' ? ACC : C.ink3}
        strokeWidth={1.4} opacity={ease(f, at('l1.b'), 12)} />)}
      {/* H_res: mix the streams */}
      {Y.map((ya, i) => Y.map((yb, j) => <path key={`m${i}${j}`} d={`M1180 ${ya} C1260 ${ya} 1300 ${yb} 1380 ${yb}`} fill="none"
        stroke={stage === 'res' ? rgba(ACC, 0.25 + 0.6 * DS[j][i]) : rgba(C.ink3, 0.6)} strokeWidth={1 + 3 * DS[j][i]}
        opacity={ease(f, at('l1.c'), 12)} />))}
      {[['H_pre', 520, 'pre', 'l1.a'], ['H_post', 920, 'post', 'l1.b'], ['H_res', 1280, 'res', 'l1.c']].map(([t, x, k, key]) =>
        <text key={t as string} x={x as number} y={650} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={stage === k ? ACC : C.ink2}
          opacity={ease(f, at(key as string), 12)}>{t as string}</text>)}
    </Svg>
    <Abs x={160} y={700} style={up(p('l2.a'))}><Chip size={22}>n = 1、三个映射都取 1 → 退回普通残差</Chip></Abs>
    <AbsoluteFill style={{opacity: mat}}>
      <Abs x={640} y={300}>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 96px) 110px', gap: 8, alignItems: 'center'}}>
          {DS.map((row, i) => <React.Fragment key={i}>
            {row.map((v, j) => <div key={j} style={{height: 70, borderRadius: 6, background: rgba(ACC, 0.1 + 0.8 * v), display: 'flex',
              alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: 24, color: v > 0.3 ? C.bg : C.ink}}>{v.toFixed(1)}</div>)}
            <Mono size={22} color={ACC} style={{opacity: ease(f, at('l2.b') + 20, 12)}}>行和 1</Mono>
          </React.Fragment>)}
          {[0, 1, 2, 3].map((j) => <Mono key={j} size={20} color={ACC} style={{textAlign: 'center', opacity: ease(f, at('l2.b') + 32, 12)}}>列和 1</Mono>)}
        </div>
      </Abs>
      <Note x={640} y={700} p={mat}>示意：一个 4×4 的双随机矩阵</Note>
      <Abs x={1180} y={330} w={620} style={{display: 'flex', flexDirection: 'column', gap: 18}}>
        <div style={up(p('l2.b'))}><Chip size={22}>元素非负；每行、每列的和都是 1</Chip></div>
        <div style={up(p('l3.a'))}><Chip on size={22}>谱范数恰好是 1</Chip></div>
        <div style={up(p('l3.b'))}><Chip on size={22}>双随机 × 双随机 = 双随机：连乘不放大</Chip></div>
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 04 两个 2×2
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const steps = [0, 1, 2, 3, 4].map((n) => 0.5 + 0.4 * 0.4 ** n);   // x1 after n layers of A, from (0.9, 0.1); x2 = 1 − x1
  const row = ease(f, at('l2'), 14), last = ease(f, at('l3'), 14);
  return <AbsoluteFill>
    <Abs x={160} y={150} style={{display: 'flex', alignItems: 'center', gap: 50}}>
      <div style={up(p('l1.a'))}><Tex id="A" size={40} /></div>
      <div style={up(p('l1.b'))}><Tex id="A2" size={40} /></div>
      <div style={{...up(row), display: 'flex', alignItems: 'center', gap: 16}}><Tex id="row" size={40} /></div>
      <div style={{...up(last), display: 'flex', alignItems: 'center', gap: 16}}><Txt size={22} color={C.ink2}>→ 全取平均</Txt><Tex id="avg" size={40} /></div>
    </Abs>
    <AbsoluteFill style={{opacity: 1 - row * (1 - last)}}>
      <Abs x={160} y={380} style={up(p('l1.c'))}><Txt size={24} color={C.ink2}>两条流过 A 若干层：总量一直是 1（初值 0.9 / 0.1，示意）</Txt></Abs>
      <Svg>
        {steps.map((x1, n) => {
          const k = ease(f, at('l1.c') + n * 8, 12);
          const X = 220 + n * 300, H = 300;
          return <g key={n} opacity={k}>
            <rect x={X} y={760 - H * x1} width={120} height={H * x1} fill={ACC} opacity={0.75} />
            <rect x={X} y={760 - H} width={120} height={H * (1 - x1)} fill={rgba(C.ink2, 0.5)} />
            <text x={X + 60} y={790} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink2}>{n === 0 ? '输入' : `A^${n}`}</text>
            <text x={X + 132} y={760 - H * x1 + 22} fontFamily={F.mono} fontSize={18} fill={ACC}>{x1.toFixed(2)}</text>
            <text x={X + 132} y={760 - H + 22} fontFamily={F.mono} fontSize={18} fill={C.ink2}>{(1 - x1).toFixed(2)}</text>
            <text x={X + 60} y={440} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink}>和 = 1.00</text>
          </g>;
        })}
      </Svg>
      <Abs x={160} y={830} style={up(last)}><Chip on size={24}>代价：流与流之间的差别只减不增（0.80 → 0.32 → 0.13 → …）</Chip></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: row * (1 - last)}}>
      <Abs x={160} y={380} w={760}>
        <Card on={f < at('l2.b')}><Txt size={26}>前向：R x = (x₁, x₁)</Txt><Txt size={22} color={C.ink2} style={{marginTop: 6}}>每行的和是 1：输出不会放大</Txt></Card></Abs>
      <Abs x={1000} y={380} w={760} style={up(p('l2.b'))}>
        <Card on><Txt size={26}>反向：Rᵀ g = (g₁ + g₂, 0)</Txt><Txt size={22} color={C.ink2} style={{marginTop: 6}}>第一列的和是 2：梯度可放大 2 倍</Txt></Card></Abs>
      <Abs x={160} y={600} style={up(p('l2.b'))}><Chip on size={26}>所以 mHC 要同时约束行与列</Chip></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 05 3000 倍与 1.6
export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const X0 = 420, W = 1200, lx = (v: number) => X0 + (W * Math.log10(v)) / 4;   // log10 from 0 to 4
  const BARS: [string, number, string, string, boolean][] = [['HC（不加约束）', 3000, '约 3000', 'l1.b', false], ['mHC', 1.6, '约 1.6', 'l1.c', true]];
  return <AbsoluteFill>
    <Abs x={160} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>mHC 论文 · 27B 模型 · 复合反向增益</Txt></Abs>
    <Svg>
      {[1, 10, 100, 1000, 10000].map((v) => <g key={v} opacity={ease(f, at('l1.a'), 12)}>
        <line x1={lx(v)} x2={lx(v)} y1={300} y2={560} stroke={v === 1 ? C.ink3 : C.rule} />
        <text x={lx(v)} y={590} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>{v.toLocaleString('en-US')}</text>
      </g>)}
      {BARS.map(([name, v, t, key, good], i) => {
        const k = lin(f, at(key), 24);
        return <g key={name} opacity={ease(f, at(key), 10)}>
          <text x={X0 - 20} y={370 + i * 120} textAnchor="end" fontFamily={F.zh} fontSize={24} fill={C.ink}>{name}</text>
          <rect x={X0} y={340 + i * 120} width={Math.max(4, (lx(v) - X0) * k)} height={50} rx={4} fill={good ? ACC : rgba(C.ink2, 0.5)}
            filter={good ? 'url(#vglow)' : undefined} />
          <text x={X0 + Math.max(4, (lx(v) - X0) * k) + 16} y={374 + i * 120} fontFamily={F.mono} fontSize={26} fill={good ? ACC : C.ink}>{t}</text>
        </g>;
      })}
    </Svg>
    <Note x={X0} y={612} p={ease(f, at('l1.c'), 12)}>第 16 课的数字化读数：2825 与 1.65</Note>
    <Abs x={160} y={680} w={1600} style={up(p('l2.a'))}>
      <Card dashed><Txt size={24}>注意：这是 4×4 矩阵的连乘，和前面的标量连乘不是同一个量；放在一起只为比较量级</Txt></Card></Abs>
    <Abs x={160} y={810} style={up(p('l2.c'))}><Chip on size={26}>代价不大：n = 4 时，训练只多 6.7% 的时间</Chip></Abs>
  </AbsoluteFill>;
};

// ================================================================ 06 记在哪里
export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const hash = lin(f, at('l2.a') + 10, 30), fetch = ease(f, at('l2.c'), 14);
  const pageToImg = swing(f, at('l3.a') + 10, 30);
  return <AbsoluteFill>
    <Abs x={160} y={150} style={up(p('l1'))}><Txt size={28} color={C.ink2}>第 16 课为止的两种记法，和第 17 课的两种新形态</Txt></Abs>
    <Abs x={160} y={220} w={1600} style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22}}>
      <div style={up(p('l1.b'))}><Card on={f >= at('l1.b') && f < at('l1.c')} style={{height: 170, boxSizing: 'border-box'}}>
        <Txt size={30}>权重</Txt><Txt size={22} color={C.ink2}>训练时写进去；靠一层层计算才能想起来</Txt>
        <Txt size={24} color={ACC} style={{marginTop: 8}}>付的是深度</Txt></Card></div>
      <div style={up(p('l1.c'))}><Card on={f >= at('l1.c') && f < at('l2')} style={{height: 170, boxSizing: 'border-box'}}>
        <Txt size={30}>KV cache</Txt><Txt size={22} color={C.ink2}>推理时写进去；每个 token 一份，留到对话结束</Txt>
        <Txt size={24} color={ACC} style={{marginTop: 8}}>付的是字节</Txt></Card></div>
      <div style={up(p('l2.a'))}><Card on={f >= at('l2') && f < at('l3')} style={{height: 330, boxSizing: 'border-box'}}>
        <Txt size={30}>Engram 条件记忆</Txt>
        <Txt size={22} color={C.ink2}>最近几个 token 的编号 → 哈希 → 查一张大表</Txt>
        <svg width={700} height={190} style={{marginTop: 10, overflow: 'visible'}}>
          {['t−2', 't−1', 't'].map((t, i) => <g key={t}>
            <rect x={i * 70} y={20} width={60} height={40} rx={6} fill={C.panel} stroke={C.line} />
            <text x={i * 70 + 30} y={46} textAnchor="middle" fontFamily={F.mono} fontSize={16} fill={C.ink}>{t}</text>
          </g>)}
          <path d={`M210 40 L${210 + 120 * hash} 40`} stroke={ACC} strokeWidth={2} />
          <text x={250} y={30} fontFamily={F.mono} fontSize={16} fill={ACC} opacity={hash}>hash</text>
          {Array.from({length: 8}, (_, r) => <rect key={r} x={360} y={r * 22} width={240} height={18} rx={2}
            fill={r === 5 && hash > 0.95 ? ACC : rgba(C.ink2, 0.18)} />)}
          <text x={610} y={124} fontFamily={F.mono} fontSize={14} fill={ACC} opacity={hash > 0.95 ? 1 : 0}>← 第 r 行</text>
          <text x={360} y={186} fontFamily={F.zh} fontSize={15} fill={C.ink3} opacity={fetch}>表放在主机内存，前向开始前就能预取</text>
        </svg>
      </Card></div>
      <div style={up(p('l3.a'))}><Card on={f >= at('l3')} style={{height: 330, boxSizing: 'border-box'}}>
        <Txt size={30}>DeepSeek-OCR 视觉压缩</Txt>
        <Txt size={22} color={C.ink2}>一页文字 → 图片 → 少得多的视觉 token</Txt>
        <svg width={700} height={190} style={{marginTop: 10, overflow: 'visible'}}>
          {Array.from({length: 7}, (_, r) => <rect key={r} x={0} y={10 + r * 20} width={mix(200, 120, pageToImg) - (r % 3) * 20} height={8} rx={2}
            fill={rgba(C.ink2, 0.4)} />)}
          <path d="M230 80 L300 80" stroke={C.ink2} strokeWidth={2} opacity={pageToImg} />
          {Array.from({length: 6}, (_, i) => <circle key={i} cx={330 + i * 34} cy={80} r={11} fill={ACC} opacity={pageToImg} />)}
          <text x={330} y={140} fontFamily={F.zh} fontSize={18} fill={C.ink} opacity={ease(f, at('l3.b'), 12)}>文本 token ≤ 视觉 token × 10 →</text>
          <text x={330} y={176} fontFamily={F.mono} fontSize={30} fill={ACC} opacity={ease(f, at('l3.b') + 10, 12)}>约 97%</text>
          <text x={480} y={176} fontFamily={F.zh} fontSize={16} fill={C.ink3} opacity={ease(f, at('l3.b') + 10, 12)}>逐字读回的精度</text>
        </svg>
      </Card></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 07 改进链
const STOPS: Stop[] = [
  {id: 'preln', title: '2019 · GPT-2 的 Pre-LN', gist: 'LayerNorm 挪进分支，主干只剩加法', key: 'l1.a', above: true},
  {id: 'xiong', title: '2020 · Pre-LN 的梯度分析', gist: 'Post-LN 顶层梯度更大：所以要 warm-up', key: 'l1.b', above: false},
  {id: 'hc', title: '2024-09 · Hyper-Connections', gist: 'ByteDance：残差流扩成 n 条，n = 4 最好', key: 'l2.a', above: true},
  {id: 'ocr', title: '2025-10 · DeepSeek-OCR', gist: '一页文字压成几十到几百个视觉 token', key: 'l2.b', above: true},
  {id: 'mhc', title: '2025-12 · mHC', gist: 'Sinkhorn–Knopp 投影成双随机矩阵', key: 'l2.c', above: false},
  {id: 'engram', title: '2026-01 · Engram', gist: 'N 元组哈希查表；MoE 与记忆的参数分配呈 U 形', key: 'l3.a', above: false},
  {id: 'ocr2', title: '2026-01 · OCR 2', gist: '按语义重排视觉 token', key: 'l3.b', above: true},
  {id: 'v4mhc', title: '2026-04 · V4', gist: 'mHC 进入生产：4 条流，Sinkhorn 迭代 20 轮', key: 'l4.a', above: true},
  {id: 'v41mem', title: '2026-09 · V4.1-Flash', gist: '单遍 mHC：读写 20d → 10d；196B Engram 表放主机内存', key: 'l4.b', above: false},
];
export const B07: React.FC = () => {
  const rm = useRm();
  const E = laneRect(rm, 'E');
  return <Chain lane="E" stops={STOPS} laneView={{x: 120, y: E.y - 150, w: 2430, h: E.h + 300}}
    hops={[['l1', 300], ['l2', 1600], ['l3', 1990], ['l4', 2250]]} end="l4.c" />;
};

// ================================================================ 08 代价
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  return <AbsoluteFill>
    <Abs x={160} y={160} style={up(p('l1'))}><Txt size={28} color={C.ink2}>按 V4-Flash 计，mHC 花在哪里</Txt></Abs>
    <Abs x={160} y={230} style={{display: 'flex', gap: 22, ...up(p('l1'))}}>
      <Chip size={24}>参数：约 0.012%</Chip><Chip size={24}>算术：约 0.33%</Chip>
      <div style={up(p('l1.a'))}><Chip on size={24}>瓶颈：读写</Chip></div></Abs>
    <Svg>
      {[['普通残差', 3, false, 'l1.a'], ['mHC（不做核融合）', 34, true, 'l1.a']].map(([n, v, hot, key], i) => {
        const w = (1100 * (v as number)) / 34 * lin(f, at(key as string) + 10 + i * 10, 24);
        return <g key={n as string} opacity={ease(f, at(key as string), 12)}>
          <text x={540} y={390 + i * 80} textAnchor="end" fontFamily={F.zh} fontSize={24} fill={C.ink}>{n as string}</text>
          <rect x={560} y={360 + i * 80} width={w} height={44} rx={4} fill={hot ? ACC : rgba(C.ink2, 0.5)} filter={hot ? 'url(#vglow)' : undefined} />
          <text x={576 + w} y={391 + i * 80} fontFamily={F.mono} fontSize={24} fill={hot ? ACC : C.ink}>{v as number}d</text>
        </g>;
      })}
      <text x={560} y={540} fontFamily={F.zh} fontSize={18} fill={C.ink3} opacity={ease(f, at('l1.a'), 12)}>每个 token、每个子层要读写的元素数</text>
    </Svg>
    <Abs x={1300} y={500} style={up(p('l1.b'))}><Tex id="io" size={40} parts={{r: lit(1)}} /></Abs>
    <Abs x={160} y={600} w={1600} style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22}}>
      <div style={up(p('l2.a'))}><Card on={f >= at('l2.a') && f < at('l2.b')}>
        <Txt size={26}>Engram 的 U 形曲线</Txt><Txt size={22} color={C.ink2}>只在两个算力点、固定稀疏度下测过</Txt></Card></div>
      <div style={up(p('l2.b'))}><Card on={f >= at('l2.b')}>
        <Txt size={26}>DeepSeek-OCR 的 97%</Txt><Txt size={22} color={C.ink2}>测的是逐字 OCR，不是在压缩后的上下文上推理</Txt>
        <Txt size={22} color={ACC} style={{opacity: ease(f, at('l2.c'), 12)}}>V4.1-Flash 也没有采用它的编码器</Txt></Card></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 09 三个问题的答案
export const B09: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const out = ease(f, at('l3.a') - 4, 16);
  const v = camAt(f, {...FULLV, w: FULLV.w * 0.85}, [{at: at('l3.a'), dur: 60, to: FULLV}]);
  const items: [number, string, string, string, number][] = [
    [1, '捷径上的系数必须是 1', '1.05⁶⁰ = 18.7，0.95⁶⁰ = 0.046：代价随深度指数增长', 'l1', at('l1.b')],
    [2, 'HC 变 n 条流，mHC 用双随机矩阵守住 1', '27B 上的复合反向增益：约 3000 → 约 1.6', 'l2', at('l2.b')],
    [3, 'Engram 查表、OCR 视觉压缩', 'Engram 的表放在主机内存；OCR 的证据限于逐字读回', 'l3', at('l3') + 30],
  ];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - out}}>
      <Abs x={120} y={190} w={1680} style={{display: 'flex', flexDirection: 'column', gap: 26}}>
        {items.map(([n, head, body, key, check]) => {
          const done = ease(f, check, 12);
          return <div key={n} style={{display: 'flex', alignItems: 'center', gap: 26, padding: '22px 28px', borderRadius: 14,
            background: rgba(C.bg, 0.9), border: `1px solid ${done > 0.5 ? rgba(ACC, 0.7) : C.line}`, boxShadow: done > 0.5 ? BOX_GLOW : undefined,
            ...up(p(key))}}>
            <Mono size={32} color={ACC} glow>Q{n}</Mono>
            <div style={{flexGrow: 1}}><Txt size={32} weight={400}>{head}</Txt><Txt size={24} color={C.ink2}>{body}</Txt></div>
            <div style={{position: 'relative', width: 48, height: 48}}>
              <Svg w={48} h={48}><Check x={2} y={2} size={42} p={lin(f, check, 14)} /></Svg></div>
          </div>;
        })}
      </Abs>
    </AbsoluteFill>
    {out > 0 && <AbsoluteFill style={{opacity: out}}>
      <TreeScene cam={project(v)} look={{laneLit: (k) => (k === 'F' ? ease(f, at('l3.a') + 30, 12) : 0)}} />
      <Abs x={0} y={120} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 主线 F：一次性的大训练，靠小实验下注</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};
