// 第 0 课 §7 · 主线 F：规模化规律与优化器 —— 9 段黑板。数字全部来自 lesson-00 §7（研究笔记 lesson-00-doc.md）；
// 标"示意"的是：01/03/04 段幂律图的位置（只有斜率是真的）、02 段两段条的比例、05 段奇异值的高度。
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

/** A log–log frame with no numbers on it: only slopes are real. One decade = `dec` px on both axes. */
const LogFrame: React.FC<{x0: number; y0: number; w: number; h: number; xl: string; yl: string; o?: number}> = ({x0, y0, w, h, xl, yl, o = 1}) =>
  <g opacity={o}>
    <line x1={x0} x2={x0 + w} y1={y0} y2={y0} stroke={C.ink3} />
    <line x1={x0} x2={x0} y1={y0} y2={y0 - h} stroke={C.ink3} />
    <text x={x0 + w} y={y0 + 30} textAnchor="end" fontFamily={F.zh} fontSize={18} fill={C.ink3}>{xl}</text>
    <text x={x0 + 8} y={y0 - h - 10} fontFamily={F.zh} fontSize={18} fill={C.ink3}>{yl}</text>
  </g>;

// ================================================================ 01 6ND
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const DEC = 140, X0 = 980, Y0 = 760, AX = X0 + 40, AY = Y0 - 60;   // the two fits start from a common point (示意)
  const line = (slope: number, k: number) => `M${AX} ${AY} L${AX + 4 * DEC * k} ${AY - slope * 4 * DEC * k}`;
  const kap = lin(f, at('l2.a') + 4, 30), chin = lin(f, at('l2.b') + 4, 30), gap = ease(f, at('l3.a'), 14);
  return <AbsoluteFill>
    <Abs x={160} y={170} style={up(p('l1'))}><Tex id="c6nd" size={64} parts={{six: f >= at('l1.a') && f < at('l2') ? lit(1) : {}}} /></Abs>
    <Abs x={160} y={290} w={700} style={up(p('l1.a'))}>
      <Txt size={22} color={C.ink2}>每个 token × 每个参数：</Txt>
      <div style={{display: 'flex', gap: 12, marginTop: 12}}>
        {[['前向', 2], ['反向', 4]].map(([n, k], i) => <div key={n as string} style={{display: 'flex', gap: 4, alignItems: 'center'}}>
          <Txt size={20} color={C.ink2} style={{width: 50}}>{n as string}</Txt>
          {Array.from({length: k as number}, (_, j) => <div key={j} style={{width: 34, height: 34, borderRadius: 4,
            background: rgba(ACC, 0.25 + 0.45 * ease(f, at('l1.a') + 6 + (i * 2 + j) * 4, 8)), border: `1px solid ${ACC}`}} />)}
        </div>)}
        <Mono size={26} color={ACC} style={{marginLeft: 10, opacity: ease(f, at('l1.a') + 34, 10)}}>= 6 次</Mono>
      </div>
    </Abs>
    <Abs x={160} y={440} style={up(p('l2'))}><Txt size={26}>给定算力 C：模型多大（N），数据多少（D）？</Txt></Abs>
    <Abs x={160} y={520} style={{display: 'flex', flexDirection: 'column', gap: 16}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 18, ...up(p('l2.a'))}}><Chip size={22}>Kaplan 2020：主要加大模型</Chip><Tex id="kap" size={34} parts={{e: lit(1)}} /></div>
      <div style={up(p('l2.b'))}><Chip size={22}>Chinchilla 2022：参数和数据同比例增长</Chip></div>
      <div style={up(p('l2.c'))}><Chip on size={22}>→ 经验法则：每个参数约 20 个 token</Chip></div>
    </Abs>
    <Svg>
      <LogFrame x0={X0} y0={Y0} w={700} h={540} xl="算力 C（对数）→" yl="最优参数量 N（对数）↑" o={ease(f, at('l2'), 12)} />
      {kap > 0 && <path d={line(0.73, kap)} stroke={ACC} strokeWidth={2.6} fill="none" filter="url(#vglow)" />}
      {chin > 0 && <path d={line(0.5, chin)} stroke={C.ink} strokeWidth={2.4} fill="none" strokeDasharray="8 6" />}
      {kap >= 1 && <text x={AX + 4 * DEC + 10} y={AY - 0.73 * 4 * DEC} fontFamily={F.zh} fontSize={20} fill={ACC}>Kaplan</text>}
      {chin >= 1 && <text x={AX + 4 * DEC + 10} y={AY - 0.5 * 4 * DEC + 6} fontFamily={F.zh} fontSize={20} fill={C.ink}>Chinchilla</text>}
      {gap > 0 && <g opacity={gap}>
        <line x1={AX + 4 * DEC - 20} x2={AX + 4 * DEC - 20} y1={AY - 0.73 * 4 * DEC} y2={AY - 0.5 * 4 * DEC} stroke={ACC} strokeWidth={2} />
        <text x={AX + 4 * DEC - 34} y={AY - 0.615 * 4 * DEC} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={ACC}>互相矛盾</text>
      </g>}
    </Svg>
    <Note x={X0} y={Y0 + 50} p={ease(f, at('l2'), 12)}>示意：只有斜率是真的（0.73 与同比例）</Note>
    <Abs x={160} y={800} style={up(p('l3.b'))}><Chip size={22}>超参数怎么随规模变：也没交代全</Chip></Abs>
  </AbsoluteFill>;
};

// ================================================================ 02 换一个记账方式
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const T = mix(0.3, 1, swing(f, at('l2.b') + 10, 50));
  return <AbsoluteFill>
    <Abs x={160} y={170} style={up(p('l1'))}><Txt size={26} color={C.ink2}>DeepSeek LLM 的第一步：换一个记账方式</Txt></Abs>
    <Abs x={160} y={250} style={{display: 'flex', alignItems: 'center', gap: 30, ...up(p('l1.a'))}}>
      <Mono size={40} color={C.ink3} style={{textDecoration: 'line-through'}}>6N</Mono><Mono size={34} color={C.ink2}>→</Mono>
      <Txt size={26}>每个 token 的非嵌入 FLOPs：<span style={{color: ACC}}>M</span></Txt></Abs>
    <Abs x={160} y={360} style={up(p('l1.b'))}><Tex id="cmd" size={54} parts={{m: f >= at('l1.b') && f < at('l2') ? lit(1) : {},
      a: f >= at('l2.a') && f < at('l2.b') ? lit(1) : {}, t: f >= at('l2.b') ? lit(1) : {}}} /></Abs>
    <Abs x={160} y={520} w={1600} style={up(p('l2.a'))}>
      <Txt size={22} color={C.ink2}>一个 token 的前向 FLOPs</Txt>
      <div style={{display: 'flex', gap: 3, marginTop: 12, height: 60}}>
        <div style={{width: 900, background: rgba(C.ink2, 0.45), borderRadius: '4px 0 0 4px', display: 'flex', alignItems: 'center', paddingLeft: 18}}>
          <Txt size={22}>72 l d²：权重的矩阵乘</Txt></div>
        <div style={{width: 600 * T * ease(f, at('l2.b'), 12), background: ACC, borderRadius: '0 4px 4px 0', display: 'flex', alignItems: 'center',
          paddingLeft: 14, overflow: 'hidden', whiteSpace: 'nowrap', boxShadow: BOX_GLOW}}><Txt size={22} color={C.bg}>12 l d T：注意力的打分与加权求和</Txt></div>
      </div>
      <Txt size={20} color={C.ink3} style={{marginTop: 12, opacity: ease(f, at('l2.b') + 20, 12)}}>T 是序列长度：序列越长，这一段越长（条的比例为示意）</Txt>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 03 学习率与 batch
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const DEC = 120, X0 = 220, Y0 = 720;
  const step = ease(f, at('l2'), 14), table = ease(f, at('l3'), 14);
  const pt = (slope: number, c: number, y0: number) => [X0 + 40 + c * DEC, y0 - slope * c * DEC] as const;
  return <AbsoluteFill>
    <Abs x={0} y={160} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}>
      <Tex id="hp" size={44} parts={{a: f >= at('l1.a') && f < at('l1.b') ? lit(1) : {}, b: f >= at('l1.b') && f < at('l2') ? lit(1) : {}}} /></Abs>
    <AbsoluteFill style={{opacity: 1 - table}}>
      <Svg>
        <LogFrame x0={X0} y0={Y0} w={620} h={420} xl="算力 C（对数）→" yl="batch B（对数）↑" o={ease(f, at('l1.b'), 12)} />
        <LogFrame x0={X0 + 760} y0={Y0} w={620} h={420} xl="算力 C（对数）→" yl="学习率 η（对数）↑" o={ease(f, at('l1.a'), 12)} />
        {(() => {
          const kb = lin(f, at('l1.b') + 4, 30), ka = lin(f, at('l1.a') + 4, 30);
          const [bx1, by1] = pt(0.3271, 4.5 * kb, Y0 - 60), [ax1, ay1] = pt(-0.125, 4.5 * ka, Y0 - 300);
          return <g>
            {kb > 0 && <line x1={X0 + 40} y1={Y0 - 60} x2={bx1} y2={by1} stroke={ACC} strokeWidth={2.6} filter="url(#vglow)" />}
            {ka > 0 && <line x1={X0 + 800} y1={Y0 - 300} x2={ax1 + 760} y2={ay1} stroke={ACC} strokeWidth={2.6} filter="url(#vglow)" />}
            {step > 0 && [1, 2, 3].map((c) => {
              const [x, y] = pt(0.3271, c, Y0 - 60), [xa, ya] = pt(-0.125, c, Y0 - 300);
              return <g key={c} opacity={step}>
                <circle cx={x} cy={y} r={6} fill={C.ink} />
                <circle cx={xa + 760} cy={ya} r={6} fill={C.ink} />
                {c < 3 && <text x={x + DEC / 2} y={y + 30} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={ACC}
                  opacity={ease(f, at('l2.a'), 12)}>×2.12</text>}
                {c < 3 && <text x={xa + 760 + DEC / 2} y={ya - 18} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={ACC}
                  opacity={ease(f, at('l2.b'), 12)}>×0.75</text>}
              </g>;
            })}
          </g>;
        })()}
      </Svg>
      <Abs x={X0} y={780} style={{display: 'flex', alignItems: 'center', gap: 24, ...up(step)}}>
        <Txt size={22} color={C.ink2}>算力每乘 10：</Txt><Tex id="x10" size={34} parts={{b: lit(1), a: lit(1)}} /></Abs>
      <Note x={X0} y={840} p={ease(f, at('l1'), 12)}>示意：位置是随手放的，斜率是真的；圆点之间算力差 10 倍</Note>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: table}}>
      <Abs x={160} y={320} w={1600}>
        <div style={{display: 'grid', gridTemplateColumns: '160px 1fr 1fr 1.3fr', rowGap: 6, alignItems: 'center'}}>
          {['', '规律预测', '报告图 3', '实际用的'].map((h) => <Txt key={h} size={22} color={C.ink3}>{h}</Txt>)}
          {[['7B', '4.24×10⁻⁴ · 9.24M token', '4.2e-04 · 9.2M', '4.2×10⁻⁴ · 2,304 条 ≈ 9.4M token', 'l3'],
            ['67B', '3.17×10⁻⁴ · 19.7M token', '3.2e-04 · 19.7M', '3.2×10⁻⁴ · 4,608 条 ≈ 18.9M token', 'l3.b']].map(([m, a, b, c, key]) =>
            <React.Fragment key={m}>
              {[m, a, b, c].map((t, i) => <div key={i} style={{padding: '18px 0', borderTop: `1px solid ${C.rule}`, ...up(ease(f, at(key) + i * 8, 12))}}>
                {i === 0 ? <Mono size={28} color={m === '7B' ? ACC : C.ink}>{t}</Mono> : <Txt size={24} color={i === 3 ? ACC : C.ink}>{t}</Txt>}
              </div>)}
            </React.Fragment>)}
        </div>
        <Txt size={20} color={C.ink3} style={{marginTop: 18, opacity: ease(f, at('l3.b'), 12)}}>每条序列 4,096 个 token</Txt>
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 04 最优分配
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const DEC = 120, X0 = 220, Y0 = 760, AX = X0 + 40;
  const km = lin(f, at('l1.a') + 4, 30), kd = lin(f, at('l1.b') + 4, 30), fan = ease(f, at('l2.a'), 14);
  const seg = (slope: number, k: number, y0: number) => `M${AX} ${y0} L${AX + 5 * DEC * k} ${y0 - slope * 5 * DEC * k}`;
  return <AbsoluteFill>
    <Abs x={160} y={160} style={up(p('l1'))}><Tex id="alloc" size={44} parts={{m: f >= at('l1.a') && f < at('l1.b') ? lit(1) : {},
      d: f >= at('l1.b') && f < at('l1.c') ? lit(1) : {}}} /></Abs>
    <Abs x={160} y={250} style={up(p('l1.c'))}><Tex id="sum1" size={36} /></Abs>
    <Svg>
      <LogFrame x0={X0} y0={Y0} w={700} h={420} xl="算力 C（对数）→" yl="最优分配（对数）↑" o={ease(f, at('l1'), 12)} />
      {km > 0 && <path d={seg(0.5243, km, Y0 - 120)} stroke={ACC} strokeWidth={2.6} fill="none" filter="url(#vglow)" />}
      {kd > 0 && <path d={seg(0.4757, kd, Y0 - 40)} stroke={C.ink} strokeWidth={2.4} fill="none" strokeDasharray="8 6" />}
      {km >= 1 && <text x={AX + 5 * DEC + 10} y={Y0 - 120 - 0.5243 * 5 * DEC + 6} fontFamily={F.mono} fontSize={20} fill={ACC}>M ∝ C^0.5243</text>}
      {kd >= 1 && <text x={AX + 5 * DEC + 10} y={Y0 - 40 - 0.4757 * 5 * DEC + 26} fontFamily={F.mono} fontSize={20} fill={C.ink}>D ∝ C^0.4757</text>}
      <LogFrame x0={X0 + 980} y0={Y0} w={480} h={420} xl="算力 C（对数）→" yl="模型一侧 M（对数）↑" o={fan} />
      {fan > 0 && [[0.450, '数据质量低'], [0.524, '中'], [0.578, '高']].map(([e, t], i) => {
        const k = lin(f, at('l2.a') + 6 + i * 10, 24);
        return <g key={t as string}>
          <path d={`M${X0 + 1020} ${Y0 - 60} L${X0 + 1020 + 400 * k} ${Y0 - 60 - (e as number) * 400 * k}`} stroke={i === 2 ? ACC : C.ink2}
            strokeWidth={2.2} fill="none" />
          {k >= 1 && <text x={X0 + 1430} y={Y0 - 60 - (e as number) * 400 + 6} fontFamily={F.mono} fontSize={18} fill={i === 2 ? ACC : C.ink2}>{(e as number).toFixed(3)} · {t as string}</text>}
        </g>;
      })}
    </Svg>
    <Note x={X0} y={Y0 + 50} p={ease(f, at('l1'), 12)}>示意：位置是随手放的，斜率是真的</Note>
    <Abs x={X0 + 980} y={Y0 + 40} style={up(p('l2.b'))}><Chip size={22}>数据质量越高，指数越大；原因被标为推测</Chip></Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 Muon
const SV0 = [3.2, 1.6, 0.8, 0.35, 0.12, 0.05], SV1 = [1, 1, 0.99, 0.97, 0.94, 0.9];   // 示意
export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const ns = swing(f, at('l2.b') + 10, 40);
  const stage = f >= at('l2.c') ? 'w' : f >= at('l2.b') ? 'o' : f >= at('l2.a') ? 'm' : '';
  return <AbsoluteFill>
    <Abs x={160} y={160} style={{display: 'flex', alignItems: 'center', gap: 24, ...up(p('l1'))}}>
      <Chip size={24}>DeepSeek LLM → V3：AdamW</Chip><Mono size={26} color={C.ink2}>→</Mono>
      <div style={up(p('l1.a'))}><Chip on size={24}>V4：大部分矩阵参数用 Muon</Chip></div></Abs>
    <Abs x={160} y={270} style={up(p('l2'))}><Tex id="muon" size={40} parts={{m: stage === 'm' ? lit(1) : {}, o: stage === 'o' ? lit(1) : {},
      s: stage === 'w' ? lit(1) : {}, w: stage === 'w' ? lit(1) : {}}} /></Abs>
    <Svg>
      <text x={1060} y={300} fontFamily={F.zh} fontSize={22} fill={C.ink2} opacity={ease(f, at('l2.b'), 12)}>更新矩阵的奇异值</text>
      <line x1={1060} x2={1760} y1={640} y2={640} stroke={C.ink3} opacity={ease(f, at('l2.b'), 12)} />
      <line x1={1060} x2={1760} y1={640 - 100} y2={640 - 100} stroke={ACC} strokeDasharray="6 6" opacity={ease(f, at('l2.b') + 30, 12)} />
      <text x={1770} y={640 - 94} fontFamily={F.mono} fontSize={18} fill={ACC} opacity={ease(f, at('l2.b') + 30, 12)}>1</text>
      {SV0.map((v, i) => {
        const h = 100 * mix(v, SV1[i], ns) * ease(f, at('l2.b'), 12);
        return <rect key={i} x={1080 + i * 110} y={640 - Math.min(h, 330)} width={70} height={Math.min(h, 330)} rx={3}
          fill={ns > 0.9 ? ACC : rgba(C.ink2, 0.5)} filter={ns > 0.9 ? 'url(#vglow)' : undefined} />;
      })}
      <text x={1060} y={680} fontFamily={F.zh} fontSize={18} fill={C.ink3} opacity={ease(f, at('l2.b'), 12)}>牛顿–舒尔茨：把所有奇异值推向 1（近似）</text>
    </Svg>
    <Note x={1060} y={700} p={ease(f, at('l2.b'), 12)}>示意：六个奇异值</Note>
    <Abs x={160} y={760} style={{display: 'flex', gap: 16, ...up(p('l3'))}}>
      <Chip size={22}>动量 μ = 0.95</Chip><Chip size={22}>权重衰减 λ = 0.1</Chip><Chip size={22}>更新的均方根 → 0.18</Chip>
      <div style={up(p('l3.b'))}><Chip on size={22}>NS：8 步快速 + 2 步精修</Chip></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 06 改进链
const STOPS: Stop[] = [
  {id: 'adamw', title: '2017 · AdamW', gist: '把权重衰减从梯度里解耦出来', key: 'l1.a', above: true},
  {id: 'kaplan', title: '2020 · Kaplan 规模化规律', gist: '损失随 N、D、C 呈幂律；主要加大模型', key: 'l1.b', above: false},
  {id: 'chinchilla', title: '2022 · Chinchilla', gist: '参数和数据同比例增长', key: 'l1.c', above: true, dx: 120},
  {id: 'hpscale', title: '2024-01 · DeepSeek LLM', gist: '超参数幂律；用 M 记账；分配随数据质量变；多步学习率调度', key: 'l2.a', above: true},
  {id: 'muon', title: '2024-12 · Muon（博客）', gist: '动量 + 牛顿–舒尔茨；嵌入与输出层仍用 AdamW', key: 'l2.b', above: false},
  {id: 'v4muon', title: '2026-04 · V4', gist: '大部分矩阵用 Muon', key: 'l3.a', above: true},
  {id: 'sinkupd', title: '2026-09 · V4.1-Flash', gist: '嵌入、预测头、Engram 表：动量 + Sinkhorn 平衡；省下二阶矩', key: 'l3.b', above: false},
];
export const B06: React.FC = () => {
  const rm = useRm();
  const Fl = laneRect(rm, 'F');
  return <Chain lane="F" stops={STOPS} laneView={{x: 30, y: Fl.y - 260, w: 2520, h: Fl.h + 330}}
    hops={[['l1', 330], ['l2', 1010], ['l3', 2250]]} end="l3.c" />;
};

// ================================================================ 07 每个参数分到多少 token
const TPP: [string, number, number | null, string, string][] = [
  ['DeepSeek LLM 67B', 29.9, 29.9, '29.9', 'l2.a'], ['V3', 22.1, 400, '22.1 / 400', 'l2.b'],
  ['V4.1-Flash', 81.5, 2812.5, '81.5 / 2,812.5', 'l2.c'], ['V2-Lite', 363, null, '363（按总参数）', 'l3.a'],
];
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const X0 = 480, W = 1000, lx = (v: number) => X0 + (W * (Math.log10(v) - 1)) / Math.log10(500);   // 10 … 5,000
  return <AbsoluteFill>
    <Abs x={160} y={160} style={up(p('l1'))}><Txt size={26} color={C.ink2}>每个参数分到多少训练 token</Txt></Abs>
    <Abs x={160} y={210} style={{display: 'flex', gap: 18, ...up(p('l1.a'))}}>
      <Chip size={22}><span style={{display: 'inline-block', width: 18, height: 14, background: rgba(C.ink2, 0.6)}} /> 按总参数</Chip>
      <Chip on size={22}><span style={{display: 'inline-block', width: 18, height: 14, background: ACC}} /> 按激活参数</Chip></Abs>
    <Svg>
      {[10, 100, 1000].map((v) => <g key={v} opacity={ease(f, at('l1'), 12)}>
        <line x1={lx(v)} x2={lx(v)} y1={300} y2={760} stroke={C.rule} />
        <text x={lx(v)} y={786} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>{v.toLocaleString('en-US')}</text>
      </g>)}
      <line x1={lx(20)} x2={lx(20)} y1={290} y2={770} stroke={C.ink} strokeDasharray="6 6" opacity={ease(f, at('l3.b'), 12)} />
      <text x={lx(20) + 8} y={290} fontFamily={F.zh} fontSize={18} fill={C.ink} opacity={ease(f, at('l3.b'), 12)}>Chinchilla 约 20（稠密模型的经验法则）</text>
      {TPP.map(([name, tot, act, text, key], i) => {
        const y = 330 + i * 110, k = lin(f, at(key) + 4, 20);
        return <g key={name} opacity={ease(f, at(key), 10)}>
          <text x={X0 - 20} y={y + 30} textAnchor="end" fontFamily={F.zh} fontSize={24} fill={C.ink}>{name}</text>
          <rect x={X0} y={y} width={(lx(tot) - X0) * k} height={22} rx={3} fill={rgba(C.ink2, 0.6)} />
          {act !== null && <rect x={X0} y={y + 28} width={(lx(act) - X0) * k} height={22} rx={3} fill={ACC} filter="url(#vglow)" />}
          <text x={Math.max(lx(tot), act ? lx(act) : 0) + 14} y={y + 34} fontFamily={F.mono} fontSize={22} fill={C.ink} opacity={k}>{text}</text>
        </g>;
      })}
    </Svg>
    <Abs x={160} y={820} style={{display: 'flex', gap: 30, alignItems: 'center', ...up(p('l2.b'))}}>
      <Tex id="tpp3" size={28} parts={{a: lit(0.5), b: lit(1)}} />
      <div style={up(p('l2.c'))}><Tex id="tpp41" size={28} parts={{a: lit(0.5), b: lit(1)}} /></div></Abs>
  </AbsoluteFill>;
};

// ================================================================ 08 代价
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  const lx = (v: number) => 220 + (680 * Math.log10(v)) / Math.log10(3000);
  return <AbsoluteFill>
    <Abs x={160} y={160} w={1600} style={{display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 22}}>
      <div style={up(p('l1'))}><Card on={f >= at('l1') && f < at('l2')} style={{height: 560, boxSizing: 'border-box'}}>
        <Txt size={30}>外推是下注</Txt>
        <Txt size={22} color={C.ink2} style={{marginTop: 8}}>幂律在小预算上拟合</Txt>
        <svg width={440} height={250} style={{marginTop: 18, overflow: 'visible'}}>
          {[['最大一组 IsoFLOP 实验', 1, C.ink2], ['7B', 283, ACC], ['67B', 2885, ACC]].map(([n, v, c], i) => <g key={n as string}
            opacity={ease(f, at('l1.b') + i * 10, 12)}>
            <rect x={0} y={i * 70} width={(lx(v as number) - 220) * 0.6 + 6} height={30} rx={3} fill={c as string} opacity={i ? 1 : 0.5} />
            <text x={(lx(v as number) - 220) * 0.6 + 16} y={i * 70 + 22} fontFamily={F.mono} fontSize={20} fill={C.ink}>{i ? `×${(v as number).toLocaleString('en-US')}` : '×1'}</text>
            <text x={0} y={i * 70 + 54} fontFamily={F.zh} fontSize={17} fill={C.ink2}>{n as string}</text>
          </g>)}
        </svg>
        <Txt size={20} color={C.ink2} style={{opacity: ease(f, at('l1.c'), 12)}}>拟合只看算力；同算力下形状不同，最优区域会略有移动</Txt>
      </Card></div>
      <div style={up(p('l2'))}><Card on={f >= at('l2') && f < at('l3')} style={{height: 560, boxSizing: 'border-box'}}>
        <Txt size={30}>口径</Txt>
        <Txt size={22} color={C.ink2} style={{marginTop: 8}}>最小的实验模型上，和 M 比</Txt>
        {[['M', 1, C.ink2, 'l2'], ['6N₁（只数非嵌入）', 0.43, ACC, 'l2.a'], ['6N₂（含输出头）', 1.32, ACC, 'l2.b']].map(([n, v, c, key]) =>
          <div key={n as string} style={{marginTop: 18, ...up(p(key as string))}}>
            <Txt size={20} color={C.ink2}>{n as string}</Txt>
            <div style={{display: 'flex', alignItems: 'center', gap: 12, marginTop: 6}}>
              <div style={{width: 300 * (v as number), height: 26, borderRadius: 3, background: c as string}} />
              <Mono size={20}>{v === 1 ? '1' : v === 0.43 ? '0.43' : '+32%'}</Mono></div>
          </div>)}
        <Txt size={20} color={C.ink2} style={{marginTop: 20, opacity: ease(f, at('l2.c'), 12)}}>MoE 又把 N 分成总参数、激活参数两个数</Txt>
      </Card></div>
      <div style={up(p('l3'))}><Card on={f >= at('l3')} style={{height: 560, boxSizing: 'border-box'}}>
        <Txt size={30}>Muon 不是全部都换</Txt>
        <Txt size={22} color={C.ink2} style={{marginTop: 8}}>V4 里仍用 AdamW 的：</Txt>
        {['嵌入', '预测头', 'RMSNorm 权重', 'mHC 的静态偏置和门控'].map((t, i) => <Txt key={t} size={22}
          style={{marginTop: 10, opacity: ease(f, at('l3.a') + i * 8, 10)}}>· {t}</Txt>)}
        <Txt size={22} color={ACC} style={{marginTop: 22, opacity: ease(f, at('l3.b'), 12)}}>牛顿–舒尔茨只把奇异值近似推到 1</Txt>
      </Card></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 09 三个问题的答案
export const B09: React.FC = () => {
  const {f, at, p} = useBoard();
  const out = ease(f, at('l3.a') - 4, 16);
  const v = camAt(f, {...FULLV, w: FULLV.w * 0.8}, [{at: at('l3.a'), dur: 60, to: FULLV}]);
  const items: [number, string, string, string, number][] = [
    [1, 'C ≈ 6ND；DeepSeek LLM 改用 C = MD', '最优分配：M ∝ C^0.5243，D ∝ C^0.4757，并随数据质量变', 'l1', at('l1.b')],
    [2, '学习率和 batch 也有幂律', '算力 ×10：batch ×2.12，学习率 ×0.75', 'l2', at('l2.a')],
    [3, 'AdamW → Muon；代价是外推和口径', '7B、67B 是最大实验的 283 倍与 2,885 倍', 'l3', at('l3') + 30],
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
      <TreeScene cam={project(v)} look={{laneLit: (k) => (k !== 'M' ? 0.25 + 0.35 * ease(f, at('l3.a') + 30, 20) : 0)}} />
      <Abs x={0} y={460} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '14px 32px', borderRadius: 12, background: rgba(C.bg, 0.94), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW, textAlign: 'center'}}>
          <Txt size={26} color={C.ink2}>六条主线讲完了</Txt>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 规格总表：规格、完整时间线、常见误读</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};
