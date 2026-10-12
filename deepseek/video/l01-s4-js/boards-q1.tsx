// Q1 为什么除以 √d_k —— 第 01–05 段。数字全部来自 lesson-01 §4（研究笔记 lesson-01-doc.md）；
// 随机游走与直方图是示意（固定种子的模拟），四分之三、万分之一等结论来自课文。
import React from 'react';
import {AbsoluteFill, random} from 'remotion';
import {
  ACC, Abs, Axes, BOX_GLOW, Bars, Brace, C, Chip, F, Mono, Slider, Sub, Svg, TEXT_GLOW, Tex, Txt, coinSum, curvePts, dim, drawOn,
  ease, fmt, focus, lin, lit, mix, normal, rgba, scale, softmax, swing, up, useBoard,
} from '../../../.cursor/skills/lesson-video/js/kit';

// ================================================================ 01 抛 64 次硬币
const N = 64, NW = 260;
const WALKS: number[][] = Array.from({length: NW}, (_, w) => {
  let s = 0;
  const pts = [0];
  for (let i = 0; i < N; i++) { s += random(`walk${w}.${i}`) < 0.5 ? -1 : 1; pts.push(s); }
  return pts;
});
const X1 = (i: number) => 380 + i * 17;           // step → px
const HX = 1600;                                  // the sideways histogram of end points starts here
const Y1 = (s: number) => 540 - s * 4.3;          // sum → px
const EXACT = coinSum(N);

/** A coin showing `face`; `phase` 0→1 is one flip (it squeezes to an edge at 0.5 and opens on the new face). */
const Coin: React.FC<{x: number; y: number; r: number; face: boolean; phase: number; label?: boolean; p: number}> = ({x, y, r, face, phase, label, p}) => {
  const sx = Math.abs(Math.cos(Math.min(1, phase) * Math.PI));
  return <Abs x={x - r} y={y - r} w={2 * r} style={{opacity: p}}>
    <div style={{width: 2 * r, height: 2 * r, borderRadius: r, border: `2px solid ${face ? ACC : C.ink2}`, transform: `scaleX(${Math.max(0.06, sx)})`,
      display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: face ? BOX_GLOW : undefined, background: C.panel}}>
      <Mono size={r * 0.62} color={face ? ACC : C.ink}>{face ? '+1' : '−1'}</Mono>
    </div>
    {label && <div style={{textAlign: 'center', marginTop: 14, fontSize: 26, color: C.ink2}}>{face ? '正面 +1' : '反面 −1'}</div>}
  </Abs>;
};

export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  // the first walk, one coin per step
  const w0 = at('l1.a') + 6, wDur = 100;
  const kf = lin(f, w0, wDur) * N;
  const k0 = Math.min(N, Math.floor(kf));
  const S0 = WALKS[0][k0];
  // before the walk the coin shows both faces in turn; during it, each step's real result
  const face = f < w0 ? Math.floor(f / 22) % 2 === 0 : k0 === 0 ? true : WALKS[0][k0] - WALKS[0][k0 - 1] > 0;
  const phase = f < w0 ? Math.min(1, ((f % 22) / 22) * 3) : k0 >= N ? 1 : Math.min(1, (kf - k0) * 2);
  // the crowd of walks
  const c0 = at('l3.a') + 4, spread = 66, each = 20;
  const wp = (w: number) => w === 0 ? (f < w0 ? 0 : lin(f, w0, wDur)) : lin(f, c0 + (w / NW) * spread, each);
  const done = WALKS.map((_, w) => wp(w) >= 1);
  const counts: Record<number, number> = {};
  WALKS.forEach((pts, w) => { if (done[w] && w > 0) counts[pts[N]] = (counts[pts[N]] ?? 0) + 1; });
  const maxCount = 30;
  const crowd = ease(f, c0, 12);
  const plotOut = ease(f, at('l5') - 4, 16);
  const guides = (1 - ease(f, at('l3'), 10) * 0.55);
  const band = p('l3.b', 14);
  const tails = p('l3.c', 14);
  const env = lin(f, at('l4.b') + 4, 40);
  const lin_n = lin(f, at('l4.b') + 30, 30);
  const sx = (i: number) => X1(i);
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - plotOut}}>
      <Coin x={190} y={420} r={74} face={face} phase={phase} label={f < w0} p={ease(f, 2, 14) * (1 - ease(f, at('l3.a'), 14))} />
      <Svg>
        <Axes x0={X1(0)} y0={Y1(66)} x1={X1(N) + 10} y1={Y1(-66)} zeroY={Y1(0)} p={ease(f, w0 - 14, 16)}
          xt={[[X1(16), '16'], [X1(32), '32'], [X1(48), '48'], [X1(64), '64']]} xlabel="抛了几次" ylabel="累计和 S" />
        {/* the guesses */}
        {[[64, 'l2.a'], [32, 'l2.b']].map(([v, key]) => {
          const g = p(key as string, 12) * guides;
          if (g <= 0) return null;
          return [1, -1].map((sg) => <g key={`${v}${sg}`} opacity={g}>
            <line x1={X1(0)} x2={X1(N)} y1={Y1(sg * (v as number))} y2={Y1(sg * (v as number))} stroke={C.ink2} strokeWidth={1.4} strokeDasharray="8 8" />
            <text x={X1(N) + 14} y={Y1(sg * (v as number)) + 8} fontFamily={F.mono} fontSize={24} fill={C.ink2}>{sg > 0 ? '+' : '−'}{v} ?</text>
          </g>);
        })}
        {/* ±8 band at the end */}
        {band > 0 && <rect x={X1(0)} y={Y1(8)} width={(X1(N) - X1(0)) * band} height={Y1(-8) - Y1(8)} fill={rgba(ACC, 0.07)} />}
        {/* walks: the crowd in grey, the first one in accent while it is drawn */}
        {WALKS.map((pts, w) => {
          const q = wp(w);
          if (q <= 0 || (w > 0 && crowd <= 0)) return null;
          const n = Math.max(1, Math.round(q * N));
          const xy = pts.slice(0, n + 1).map((s, i) => [sx(i), Y1(s)] as [number, number]);
          const first = w === 0;
          return <path key={w} d={drawOn(xy, 1)} fill="none" stroke={first && f < c0 ? ACC : rgba(C.ink2, first ? 0.5 : 0.16)}
            strokeWidth={first && f < c0 ? 2.4 : 1.2} filter={first && f < c0 ? 'url(#vglow)' : undefined} />;
        })}
        {/* √n envelope, and n for contrast */}
        {env > 0 && [1, -1].map((sg) => <path key={`env${sg}`} d={drawOn(curvePts((n) => sg * Math.sqrt(n), 0, N, sx, Y1, 64), env)}
          fill="none" stroke={ACC} strokeWidth={2.6} filter="url(#vglow)" />)}
        {lin_n > 0 && [1, -1].map((sg) => <path key={`lin${sg}`} d={drawOn([[sx(0), Y1(0)], [sx(N), Y1(sg * N)]], lin_n)}
          fill="none" stroke={C.ink2} strokeWidth={1.6} strokeDasharray="4 7" />)}
        {/* end-point histogram, sideways */}
        {Object.entries(counts).map(([k, c]) => {
          const v = Number(k);
          const inBand = Math.abs(v) <= 8;
          return <rect key={k} x={HX} y={Y1(v) - 3.6} width={(c / maxCount) * 210} height={7.2}
            fill={inBand && band > 0 ? rgba(ACC, 0.45 + 0.5 * band) : C.bar} />;
        })}
        {band > 0 && <path d={drawOn(EXACT.map(({k, p: q}) => [HX + q * NW * 2 * (210 / maxCount) * 0.5, Y1(k)] as [number, number]), band)}
          fill="none" stroke={rgba(ACC, 0.8)} strokeWidth={1.4} />}
      </Svg>
      {/* the running sum of the first walk */}
      {f >= w0 && f < c0 + 10 && <Abs x={X1(k0) + 14} y={Y1(S0) - 52} style={{opacity: 1 - ease(f, c0, 10)}}>
        <Mono size={30} color={ACC} glow>S = {S0 > 0 ? '+' : S0 < 0 ? '−' : ''}{Math.abs(S0)}</Mono></Abs>}
      <Abs x={HX} y={Y1(26)} w={300} style={{...up(band)}}>
        <Txt size={26} color={ACC} glow weight={400}>约 3/4 落在 ±8 以内</Txt></Abs>
      <Abs x={HX} y={Y1(-36)} w={300} style={{...up(tails)}}>
        <Txt size={24} color={C.ink2}>|S| ≥ 32：不到万分之一</Txt></Abs>
      {tails > 0 && <Svg>
        {[1, -1].map((sg) => <rect key={sg} x={X1(0)} y={sg > 0 ? Y1(64) : Y1(-32)} width={X1(N) - X1(0)} height={Y1(32) - Y1(64)}
          fill={rgba(C.ink2, 0.05 * tails)} />)}</Svg>}
      {env > 0 && <>
        <Abs x={X1(N) - 120} y={Y1(8) - 52} style={{opacity: env}}><Mono size={24} color={ACC} glow>+√n</Mono></Abs>
        <Abs x={X1(16) - 26} y={Y1(4) - 44} style={{opacity: env}}><Mono size={20} color={ACC}>√16 = 4</Mono></Abs>
        <Abs x={X1(40) + 8} y={Y1(46) - 30} style={{opacity: lin_n}}><Mono size={22} color={C.ink2}>n（不是这样）</Mono></Abs>
      </>}
      <Abs x={880} y={136} style={{display: 'flex', alignItems: 'center', gap: 40}}>
        <div style={up(p('l4.a'))}><Tex id="root8" size={60} glow /></div>
        <div style={up(p('l4.b'))}><Tex id="sigma" size={60} color={ACC} glow /></div>
      </Abs>
    </AbsoluteFill>

    {/* the bridge: q·k is a 64-term sum */}
    <AbsoluteFill style={{opacity: plotOut}}>
      <Abs x={0} y={210} w={1920} style={{display: 'flex', justifyContent: 'center'}}>
        <Tex id="dot64" size={70} parts={{t: f >= at('l5.a') ? lit(ease(f, at('l5.a'), 10)) : {}}} />
      </Abs>
      <DotStrips f={f} t0={at('l5') + 10} t1={at('l5.a')} />
      <Abs x={0} y={760} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l5.a'))}}>
        <Chip on size={30}>64 项 ±1 相加 = 抛 64 次硬币</Chip>
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

/** q, k and their products as three 64-cell strips; products light up left to right. */
const DotStrips: React.FC<{f: number; t0: number; t1: number}> = ({f, t0, t1}) => {
  const cell = 19, gap = 3, x0 = (1920 - 64 * (cell + gap)) / 2;
  const sign = (seed: string, i: number) => random(`${seed}${i}`) < 0.5 ? -1 : 1;
  const rows: [string, (i: number) => number, number][] = [
    ['q', (i) => sign('q', i), 480], ['k', (i) => sign('k', i), 516], ['q·k', (i) => sign('q', i) * sign('k', i), 590],
  ];
  const lit_n = Math.floor(lin(f, t1 - 30, 44) * 64);
  return <>{rows.map(([name, v, y], r) => <React.Fragment key={name}>
    <Abs x={x0 - 90} y={y - 4} w={70} style={{textAlign: 'right', opacity: ease(f, t0 + r * 6, 12)}}>
      <Mono size={22} color={r === 2 ? ACC : C.ink2}>{name}</Mono></Abs>
    {Array.from({length: 64}, (_, i) => {
      const pos = v(i) > 0;
      const on = r === 2 && i < lit_n;
      return <Abs key={i} x={x0 + i * (cell + gap)} y={y} w={cell} h={r === 2 ? 26 : cell} style={{borderRadius: 3,
        opacity: ease(f, t0 + r * 6 + i * 0.25, 8),
        background: r === 2 ? (on ? (pos ? rgba(ACC, 0.8) : rgba(C.ink2, 0.55)) : C.panel) : pos ? rgba(ACC, 0.35) : rgba(C.ink2, 0.25),
        border: `1px solid ${on ? ACC : C.line}`}} />;
    })}
  </React.Fragment>)}
    <Abs x={x0} y={642} w={64 * (cell + gap)} style={{display: 'flex', justifyContent: 'space-between', opacity: ease(f, t0 + 20, 12)}}>
      <Mono size={18} color={C.ink3}>i = 1</Mono><Mono size={18} color={C.ink3}>浅色 +1 · 深色 −1</Mono><Mono size={18} color={C.ink3}>i = 64</Mono>
    </Abs>
  </>;
};

// ================================================================ 02 4 维点积，手算一遍
const Q4 = [1, -1, 1, 1], K4 = [-1, -1, 1, -1];
const ROWS16 = Array.from({length: 16}, (_, r) => [0, 1, 2, 3].map((i) => ((r >> (3 - i)) & 1) ? -1 : 1));
const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
const pm = (v: number) => (v > 0 ? '+1' : '−1');
const COLX = (k: number) => 1010 + k * 75;                    // histogram column for a dot product k
const BASE = 770;

export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const sceneA = 1 - ease(f, at('l3'), 14);
  const groupKey: Record<number, string> = {[-4]: 'l3.b', [-2]: 'l3.c', 0: 'l3.d', 2: 'l3.e', 4: 'l3.f'};
  const squash = swing(f, at('l5.b') + 6, 30);              // ÷2: the columns move in
  const colX = (k: number) => mix(COLX(k), COLX(0) + (COLX(k) - COLX(0)) / 2, squash);
  const order: Record<number, number> = {};
  const termKey = ['l4.b', 'l4.c', 'l4.d', 'l4.e', 'l4.f'];
  const vparts: Record<string, any> = {};
  ['a', 'b', 'c', 'd', 'e'].forEach((name, i) => {
    const t = at(termKey[i]);
    vparts[name] = f < t ? dim(0.22) : lit(1 - ease(f, t + 24, 16));
  });
  vparts.f = f < at('l4.f') + 16 ? dim(0.22) : lit(1 - ease(f, at('l4.g'), 12));
  vparts.g = f < at('l4.g') ? dim(0.22) : lit(1);
  const hiCol = (k: number) => {
    const i = [-4, -2, 0, 2, 4].indexOf(k);
    const t = at(termKey[i]);
    return ease(f, t, 6) * (1 - ease(f, t + 26, 12));
  };
  return <AbsoluteFill>
    {/* q, k, products */}
    <AbsoluteFill style={{opacity: sceneA}}>
      <Abs x={120} y={170} style={{display: 'flex', gap: 40, alignItems: 'center'}}>
        <div style={up(p('l1.a'))}><Tex id="qk" size={46} /></div>
        <div style={up(p('l2'))}><Tex id="pm1" size={46} /></div>
      </Abs>
      {[['q', Q4, 330, 'l1.a'], ['k', K4, 430, 'l1.a'], ['q·k', Q4.map((v, i) => v * K4[i]), 560, 'l2']].map(([name, vals, y, key], r) =>
        <React.Fragment key={name as string}>
          <Abs x={500} y={(y as number) + 14} w={120} style={{textAlign: 'right', opacity: p(key as string)}}>
            <Mono size={34} color={r === 2 ? ACC : C.ink2}>{name as string}</Mono></Abs>
          {(vals as number[]).map((v, i) => {
            const shuffle = r < 2 && f < at('l1.a') + 30 && f >= at('l1.a') ? (random(`fl${r}${i}${Math.floor(f / 4)}`) < 0.5 ? -1 : 1) : v;
            return <Abs key={i} x={660 + i * 130} y={y as number} w={100} h={84} style={{borderRadius: 10, display: 'flex', alignItems: 'center',
              justifyContent: 'center', border: `1px solid ${r === 2 ? ACC : C.line}`, background: r === 2 ? rgba(ACC, 0.08) : C.panel,
              boxShadow: r === 2 ? BOX_GLOW : undefined, ...up(ease(f, at(key as string) + i * 3, 12))}}>
              <Mono size={38} color={r === 2 ? ACC : C.ink}>{pm(shuffle)}</Mono></Abs>;
          })}
        </React.Fragment>)}
      <Abs x={1200} y={574} style={{display: 'flex', alignItems: 'center', gap: 22, ...up(p('l2.a'))}}>
        <Mono size={40} color={C.ink2}>Σ =</Mono><Txt size={56} color={ACC} glow weight={400}>0</Txt>
      </Abs>
      <Abs x={660} y={700} style={up(p('l2.a'))}><Chip on>4 个 ±1 相加 = 抛 4 次硬币</Chip></Abs>
    </AbsoluteFill>

    {/* all 16 outcomes, then grouped */}
    <AbsoluteFill style={{opacity: ease(f, at('l3'), 14)}}>
      <Abs x={140} y={168} style={{...up(p('l3')), opacity: p('l3') * (1 - ease(f, at('l4') - 10, 12))}}>
        <Txt size={30} color={C.ink2}>2⁴ = 16 种等可能的结果（每行是 4 个乘积）</Txt></Abs>
      {ROWS16.map((row, r) => {
        const k = sum(row);
        const t = at(groupKey[k]);
        const fly = swing(f, t + (order[k] = (order[k] ?? -1) + 1) * 2, 16);
        const y0 = 236 + r * 34;
        const n = ROWS16.slice(0, r).filter((x) => sum(x) === k).length;
        const dx = colX(k), dy = BASE - 26 - n * 44;
        const enter = ease(f, at('l3.a') + r * 1.6, 10);
        return <React.Fragment key={r}>
          <div style={{position: 'absolute', left: 140, top: y0, display: 'flex', alignItems: 'center', gap: 4, opacity: enter * (1 - fly) * (fly > 0 ? 0.6 : 1)}}>
            {row.map((v, i) => <div key={i} style={{width: 46, height: 28, borderRadius: 4, fontFamily: F.mono, fontSize: 17, display: 'flex', alignItems: 'center',
              justifyContent: 'center', background: v > 0 ? rgba(ACC, 0.16) : C.panel, border: `1px solid ${C.line}`, color: v > 0 ? ACC : C.ink2}}>{pm(v)}</div>)}
            <Mono size={20} color={C.ink2} style={{marginLeft: 14, width: 60}}>= {k > 0 ? '+' : k < 0 ? '−' : ''}{Math.abs(k)}</Mono>
          </div>
          {fly > 0 && <div style={{position: 'absolute', left: mix(480, dx - 18, fly), top: mix(y0 - 4, dy - 18, fly), width: 36, height: 36, borderRadius: 18,
            background: rgba(ACC, 0.25 + 0.5 * hiColSafe(hiCol, k, f, at('l4.b'))), border: `1.5px solid ${ACC}`,
            boxShadow: hiColSafe(hiCol, k, f, at('l4.b')) > 0.3 ? BOX_GLOW : undefined, opacity: Math.min(1, fly * 2)}} />}
        </React.Fragment>;
      })}
      <Svg>
        <line x1={COLX(-4) - 80} x2={COLX(4) + 80} y1={BASE} y2={BASE} stroke={C.ink3} strokeWidth={1.2} opacity={p('l3.b')} />
        {f >= at('l4') && <line x1={colX(0)} x2={colX(0)} y1={BASE - 300} y2={BASE + 10} stroke={ACC} strokeDasharray="5 6" opacity={ease(f, at('l4'), 10) * (1 - ease(f, at('l4.a'), 10)) * 0.8} />}
        <Brace x0={colX(-2)} x1={colX(2)} y={BASE + 64} p={p('l5.a', 20)} up />
      </Svg>
      {[-4, -2, 0, 2, 4].map((k) => {
        const lab = squash > 0.5 ? k / 2 : k;
        return <React.Fragment key={k}>
          <Abs x={colX(k) - 50} y={BASE + 12} w={100} style={{textAlign: 'center', opacity: p(groupKey[k])}}>
            <Mono size={28} color={hiCol(k) > 0.3 ? ACC : C.ink}>{lab > 0 ? '+' : lab < 0 ? '−' : ''}{Math.abs(lab)}</Mono></Abs>
          <Abs x={colX(k) - 50} y={BASE + 52} w={100} style={{textAlign: 'center', opacity: p('l4.a') * (1 - p('l5.a'))}}>
            <Mono size={22} color={hiCol(k) > 0.3 ? ACC : C.ink3}>{k * k}</Mono></Abs>
          <Abs x={colX(k) - 50} y={BASE - 26 - [1, 4, 6, 4, 1][(k + 4) / 2] * 44 - 30} w={100} style={{textAlign: 'center', opacity: ease(f, at(groupKey[k]) + 10, 10)}}>
            <Mono size={22} color={C.ink2}>{[1, 4, 6, 4, 1][(k + 4) / 2]} 种</Mono></Abs>
        </React.Fragment>;
      })}
      <Abs x={COLX(-4) - 130} y={BASE + 52} style={{opacity: p('l4.a') * (1 - p('l5.a'))}}><Mono size={20} color={C.ink3}>平方</Mono></Abs>
      <Abs x={640} y={176} style={up(p('l4'))}><Txt size={30}>均值 = 0<span style={{color: C.ink3}}>（左右对称）</span></Txt></Abs>
      <Abs x={640} y={244} style={{opacity: p('l4.a') * (1 - p('l5.a'))}}><Tex id="var4" size={34} parts={vparts} /></Abs>
      <Abs x={COLX(4) + 90} y={BASE - 40} style={up(p('l5.a'))}><Tex id="sd2" size={40} color={ACC} glow /></Abs>
      <Abs x={640} y={250} style={{...up(p('l5.b'))}}><Chip on size={28}>每个分数 ÷ 2 → 方差回到 1</Chip></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};
const hiColSafe = (hi: (k: number) => number, k: number, f: number, t: number) => (f < t ? 0 : hi(k));

// ================================================================ 03 方差为什么是 d_k
const DER: [string, string, string, number][] = [
  ['d1', '期望可拆；独立；均值 0', 'l2', 300],
  ['d2', '乘积相互独立，方差相加', 'l3.a', 408],
  ['d3', '方差的定义', 'l3.b', 516],
  ['d4', 'd_k 个 1 相加', 'l3.c', 612],
  ['d5', '常数提出来要平方', 'l4', 700],
];
const SIGMAS: [number, string][] = [[2, 'd_k = 4'], [8, 'd_k = 64'], [Math.sqrt(128), 'd_k = 128']];

export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const later = ease(f, at('l5'), 16);                  // derivation steps back when the plot takes over
  const blocksIn = p('l3.a');
  const stack = swing(f, at('l3.c'), 26);
  const squash = swing(f, at('l4.a'), 24);
  const blocksOut = ease(f, at('l5'), 12);
  const collapse = swing(f, at('l5') + 40, 46);
  const cx = 1420, base = 700;
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - 0.75 * later}}>
      <Abs x={120} y={140} style={up(p('l1'))}><Tex id="dotdk" size={40} /></Abs>
      <Abs x={520} y={176} style={{display: 'flex', alignItems: 'center', gap: 18, height: 50, padding: '0 22px', border: `1px solid ${C.line}`,
        borderRadius: 25, ...up(p('l1.a'))}}>
        <Mono size={19} color={C.ink3} style={{letterSpacing: '0.12em'}}>前提</Mono>
        <Txt size={25} color={C.ink2}>各分量相互独立 · 均值 0 · 方差 1（原论文脚注）</Txt>
      </Abs>
      {DER.map(([id, why, key, y], i) => {
        const start = at(key);
        const next = i < DER.length - 1 ? at(DER[i + 1][2]) : at('l5');
        const o = focus(f, start, next);
        const lastOne = i === DER.length - 1;
        const parts: Record<string, any> = {};
        if (id === 'd1') { parts.e = f < at('l2.a') ? lit(ease(f, start, 10)) : {}; parts.z = f >= at('l2.a') ? lit(ease(f, at('l2.a'), 10)) : {}; }
        if (id === 'd2') parts.s = lit(ease(f, start, 10) * (1 - ease(f, at('l3.b'), 12)));
        if (id === 'd3') parts.o = lit(ease(f, start + 20, 10));
        if (id === 'd4') parts.d = lit(ease(f, start, 10));
        if (id === 'd5') { parts.c = lit(ease(f, start + 10, 10) * (1 - ease(f, at('l4.a'), 10))); parts.o = f >= at('l4.a') ? lit(1) : {}; }
        return <React.Fragment key={id}>
          <Abs x={120} y={y} style={{display: 'flex', alignItems: 'center', gap: 22, opacity: o}}>
            <Mono size={22} color={lastOne ? ACC : C.ink3} glow={lastOne} style={{width: 30}}>{i + 1}</Mono>
            <div style={{padding: lastOne ? '8px 16px' : 0, border: lastOne ? `1px solid ${rgba(ACC, ease(f, at('l4.a'), 12))}` : undefined,
              borderRadius: 10, boxShadow: lastOne && f > at('l4.a') ? BOX_GLOW : undefined}}>
              <Tex id={id} size={34} reveal={ease(f, start, 24)} parts={parts} />
            </div>
          </Abs>
          <Abs x={710} y={y + 14} style={{opacity: o * ease(f, start + 14, 12)}}>
            <Txt size={24} color={id === 'd4' ? ACC : C.ink3} glow={id === 'd4'}><Sub text={why} /></Txt>
          </Abs>
        </React.Fragment>;
      })}
    </AbsoluteFill>

    {/* variance as blocks: d_k blocks of height 1 stack into a tower of height d_k, then ÷ d_k */}
    {blocksIn > 0 && <AbsoluteFill style={{opacity: blocksIn * (1 - blocksOut)}}>
      <Abs x={1120} y={250} style={up(blocksIn)}><Txt size={26} color={C.ink2}>每个乘积 q<sub>i</sub>k<sub>i</sub> 的方差 = 一块</Txt></Abs>
      {Array.from({length: 8}, (_, i) => {
        const rowX = 1120 + i * 76, rowY = 400;
        const towX = cx - 32, towY = base - (i + 1) * 46 * mix(1, 1 / 8, squash);
        const x = mix(rowX, towX, stack), y = mix(rowY, towY, stack);
        const h = 64 * mix(1, 46 / 64, stack) * mix(1, 1 / 8, squash);
        return <Abs key={i} x={x} y={y} w={64} h={Math.max(2, h - 4 * (1 - squash))} style={{borderRadius: 4, border: `1px solid ${ACC}`,
          background: rgba(ACC, 0.18), boxShadow: i === 7 ? BOX_GLOW : undefined, opacity: ease(f, at('l3.a') + i * 2, 10),
          display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          {squash < 0.3 && <Mono size={22} color={ACC} style={{opacity: ease(f, at('l3.b') + i, 8) * (1 - squash * 3)}}>1</Mono>}
        </Abs>;
      })}
      <Abs x={1120 + 8 * 76} y={414} style={{opacity: blocksIn * (1 - stack)}}><Mono size={30} color={C.ink2}>…</Mono></Abs>
      <Svg>
        <line x1={cx - 70} x2={cx + 70} y1={base + 2} y2={base + 2} stroke={C.ink3} opacity={stack} />
        {stack > 0.5 && <path d={`M${cx + 50} ${base} L${cx + 62} ${base} L${cx + 62} ${base - 8 * 46 * mix(1, 1 / 8, squash)} L${cx + 50} ${base - 8 * 46 * mix(1, 1 / 8, squash)}`}
          fill="none" stroke={ACC} strokeWidth={1.6} filter="url(#vglow)" opacity={(stack - 0.5) * 2} />}
      </Svg>
      <Abs x={cx + 80} y={base - 4 * 46 * mix(1, 1 / 8, squash) - 20} style={{opacity: stack * (1 - squash)}}>
        <Mono size={34} color={ACC} glow>d<sub>k</sub></Mono></Abs>
      <Abs x={cx + 80} y={base - 40} style={{opacity: squash}}><Mono size={34} color={ACC} glow>1</Mono></Abs>
      <Abs x={cx - 160} y={base + 24} w={320} style={{textAlign: 'center', opacity: squash}}><Txt size={26} color={ACC} glow>÷ d<sub>k</sub> → 方差回到 1</Txt></Abs>
    </AbsoluteFill>}

    {/* three head widths, raw and scaled */}
    {later > 0 && <AbsoluteFill style={{opacity: later}}>
      <Svg>
        {[[1220, 'raw'], [1640, 'std']].map(([cx0, kind]) => <line key={kind as string} x1={(cx0 as number) - 190} x2={(cx0 as number) + 190} y1={700} y2={700}
          stroke={C.ink3} strokeWidth={1.2} />)}
        {SIGMAS.map(([sd, label], i) => {
          const raw = Array.from({length: 121}, (_, k) => {
            const z = -3.6 + (7.2 * k) / 120;
            return [1220 + z * sd * 4.6, 700 - normal(z * sd, sd) * 1150] as [number, number];
          });
          const std = Array.from({length: 121}, (_, k) => {
            const z = -3.6 + (7.2 * k) / 120;
            return [1640 + z * 50, 700 - normal(z, 1) * 520] as [number, number];
          });
          const hi = (i === 1 && f >= at('l5.a') && f < at('l5.b')) || (i === 2 && f >= at('l5.b'));
          return <g key={label}>
            <path d={drawOn(raw, ease(f, at('l5') + i * 6, 24))} fill="none" stroke={hi ? ACC : [C.ink, C.ink2, C.ink3][i]}
              strokeWidth={hi ? 3 : 2} filter={hi ? 'url(#vglow)' : undefined} />
            <path d={drawOn(std, ease(f, at('l5') + 30 + i * 8, 24))} fill="none" stroke={ACC} strokeWidth={2.2} opacity={0.5 + 0.25 * i}
              filter="url(#vglow)" />
          </g>;
        })}
        <path d="M1440 560 L1500 560" stroke={C.ink2} strokeWidth={1.5} opacity={ease(f, at('l5') + 26, 12)} />
        <path d="M1490 553 L1502 560 L1490 567" fill="none" stroke={C.ink2} strokeWidth={1.5} opacity={ease(f, at('l5') + 26, 12)} />
        {[-20, 0, 20].map((v) => <text key={v} x={1220 + v * 4.6} y={730} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>{v}</text>)}
        {[-2, 0, 2].map((v) => <text key={`z${v}`} x={1640 + v * 50} y={730} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>{v}</text>)}
      </Svg>
      {SIGMAS.map(([sd, label], i) => <Abs key={label} x={[1238, 1262, 1300][i]} y={[452, 622, 650][i]} style={{opacity: ease(f, at('l5') + 10 + i * 6, 12)}}>
        <Mono size={19} color={[C.ink, C.ink2, C.ink3][i]}><Sub text={label} /></Mono></Abs>)}
      <Abs x={1040} y={290} w={400} style={{...up(ease(f, at('l5'), 14))}}><Txt size={26} color={C.ink2}>不缩放：头越宽，分数越分散</Txt></Abs>
      <Abs x={1460} y={290} w={380} style={{...up(ease(f, at('l5') + 30, 14))}}><Txt size={26} color={ACC} glow weight={400}>÷ √d<sub>k</sub>：三条完全重合</Txt></Abs>
      <Abs x={1040} y={786} style={{display: 'flex', gap: 18}}>
        <div style={up(p('l5.a'))}><Chip size={24}><span style={{fontFamily: F.mono}}>d<sub>k</sub> = 64</span> → 不缩放 σ = 8</Chip></div>
        <div style={up(p('l5.b'))}><Chip size={24}><span style={{fontFamily: F.mono}}>d<sub>k</sub> = 128</span> → σ ≈ 11.31</Chip></div>
      </Abs>
    </AbsoluteFill>}

    {/* the empirical check */}
    <Abs x={120} y={640} w={760} style={{padding: '22px 28px', borderRadius: 12, border: `1px solid ${rgba(ACC, 0.6)}`, background: rgba(C.bg, 0.92),
      boxShadow: BOX_GLOW, ...up(p('l6'))}}>
      <Mono size={18} color={C.ink3} style={{letterSpacing: '0.14em'}}>实测 · 200,000 对 64 维随机向量（标准正态）</Mono>
      <div style={{display: 'flex', gap: 40, marginTop: 16, alignItems: 'baseline'}}>
        <div style={up(p('l6.a'))}><Txt size={26} color={C.ink2}>样本方差 <span style={{fontFamily: F.mono, color: C.ink, fontSize: 34}}>63.77</span> <span style={{fontSize: 20, color: C.ink3}}>理论 64</span></Txt></div>
        <div style={up(p('l6.b'))}><Txt size={26} color={C.ink2}>÷ 8 后 <span style={{fontFamily: F.mono, color: ACC, fontSize: 34, textShadow: TEXT_GLOW}}>0.996</span></Txt></div>
      </div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 04 温度：第一名通吃
const Z = [2, 1, 0.1];
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const steps: [number, number, number][] = [[at('l2.a'), 2, 22], [at('l2.b'), 4, 22], [at('l2.c'), 8, 22], [at('l4') + 4, 1, 26],
    [at('l4.a') + 4, Math.sqrt(128), 60]];
  let sv = 1, prev = 1;
  for (const [t, v, dur] of steps) {
    if (f < t) break;
    sv = Math.exp(mix(Math.log(prev), Math.log(v), swing(f, t, dur)));
    prev = v;
  }
  const pr = softmax(Z.map((z) => z * sv));
  const show = ease(f, at('l1.a'), 14);
  const readout = 1 - ease(f, at('l4'), 10);
  const lock = ease(f, at('l4') + 20, 12) * (1 - ease(f, at('l4.a'), 10));
  const LX = [330, 460, 590], RX = [1120, 1250, 1380], baseY = 650;
  const crush = p('l3', 12) * (1 - ease(f, at('l4'), 12));
  return <AbsoluteFill>
    <Abs x={0} y={140} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}>
      <Tex id="smx" size={50} parts={{s: f >= at('l2.a') - 4 ? lit(1) : {}}} />
    </Abs>
    <Abs x={290} y={262} style={{opacity: show}}><Txt size={26} color={C.ink2}>分数 × s</Txt></Abs>
    <Abs x={1080} y={262} style={{opacity: show}}><Txt size={26} color={C.ink2}>softmax 之后的概率</Txt></Abs>
    <Svg opacity={show}>
      <line x1={290} x2={700} y1={baseY} y2={baseY} stroke={C.ink3} />
      <line x1={1080} x2={1490} y1={baseY} y2={baseY} stroke={C.ink3} />
      <line x1={1080} x2={1490} y1={baseY - 300} y2={baseY - 300} stroke={C.line} strokeDasharray="4 6" />
      <text x={1500} y={baseY - 294} fontFamily={F.mono} fontSize={18} fill={C.ink3}>1</text>
      <path d={`M760 ${baseY - 150} L1010 ${baseY - 150}`} stroke={C.ink3} strokeWidth={1.5} />
      <path d={`M998 ${baseY - 158} L1012 ${baseY - 150} L998 ${baseY - 142}`} fill="none" stroke={C.ink3} strokeWidth={1.5} />
      <text x={885} y={baseY - 166} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink3}>softmax</text>
      {Z.map((z, i) => {
        const h = Math.min(300, z * sv * 18.75);
        return <g key={`l${i}`}>
          <rect x={LX[i]} y={baseY - h} width={90} height={Math.max(2, h)} rx={2} fill={i === 0 ? rgba(ACC, 0.55) : C.bar} />
          <text x={LX[i] + 45} y={baseY - h - 12} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={i === 0 ? ACC : C.ink2}>{(z * sv).toFixed(1)}</text>
          <text x={LX[i] + 45} y={baseY + 32} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink3}>z{['₁', '₂', '₃'][i]} = {z}</text>
        </g>;
      })}
      {pr.map((q, i) => {
        const h = q * 300;
        return <g key={`p${i}`}>
          <rect x={RX[i]} y={baseY - h} width={90} height={Math.max(2, h)} rx={2} fill={i === 0 ? ACC : C.bar}
            filter={i === 0 ? 'url(#vglow)' : undefined} opacity={i > 0 ? 1 - 0.5 * crush : 1} />
          <text x={RX[i] + 45} y={baseY - h - 12} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={i === 0 ? ACC : C.ink2}
            opacity={readout}>{q.toFixed(4)}</text>
        </g>;
      })}
      <Slider x={560} y={810} w={800} value={sv} min={1} max={Math.sqrt(128)} log label="放大倍数 s" p={ease(f, at('l1.a') + 10, 14)}
        ticks={[[1, '×1'], [2, '×2'], [4, '×4'], [8, '×8'], ...(f >= at('l4.a') ? [[Math.sqrt(128), '×11.31'] as [number, string]] : [])]} />
    </Svg>
    <Abs x={1090} y={760} w={420} style={{textAlign: 'right', ...up(p('l3.a'))}}>
      <Txt size={24} color={ACC} glow weight={400}>×8 = √64：d<sub>k</sub> = 64 不缩放</Txt></Abs>
    <Abs x={560 - 10} y={872} style={{display: 'flex', gap: 0, opacity: ease(f, at('l4.a') + 4, 12)}}>
      {[[2, '4'], [8, '64'], [Math.sqrt(128), '128']].map(([v, d]) => <div key={d as string} style={{position: 'absolute',
        left: (800 * Math.log(v as number)) / Math.log(Math.sqrt(128)) - 60, width: 120, textAlign: 'center'}}>
        <Mono size={18} color={ACC}>d<sub>k</sub>={d as string}</Mono></div>)}
    </Abs>
    <Abs x={480} y={736} w={160} style={{textAlign: 'center', ...up(lock)}}><Mono size={22} color={ACC} glow>固定 ×1</Mono></Abs>
    <Abs x={120} y={206} style={{...up(p('l4'))}}><Txt size={30} color={ACC} glow weight={400}>除以 √d<sub>k</sub> = 把 softmax 的温度固定下来</Txt></Abs>
    <Abs x={1240} y={206} style={{...up(p('l4.a'))}}><Txt size={28} color={C.ink2}>不除：头越宽，分布越尖</Txt></Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 饱和：梯度去哪了
const PX = (d: number) => 170 + (d + 4) * 50;          // lead of the first score over the second → px
const PY = (q: number) => 700 - q * 380;
const p1 = (d: number, c: number) => 1 / (1 + Math.exp(-d) + Math.exp(-c - d));

export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const move = swing(f, at('l2.c'), 40);
  const d0 = mix(1, 8, move), c = mix(0.9, 7.2, move);
  const q0 = p1(d0, c), m = q0 * (1 - q0);
  const plotIn = p('l2.a');
  const barsK = swing(f, at('l2.c'), 30);
  const diag1 = [0.2247, 0.1837, 0.0889], diag8 = [0.00034, 0.00034, 2.5e-7];
  const chainIn = p('l3');
  const cards = ease(f, at('l4'), 14);
  const pulse = lin(f, at('l3') + 20, 50);
  const tx = (d: number) => PX(d), ty = (d: number) => PY(q0 + m * (d - d0));
  const recall = p('l1', 14) * (1 - ease(f, at('l2.a') - 10, 14));
  return <AbsoluteFill>
    {recall > 0 && <Abs x={0} y={330} w={1920} style={{display: 'flex', justifyContent: 'center', gap: 160, opacity: recall}}>
      {[['×1', [0.659, 0.2424, 0.0986]], ['×8', [0.9997, 0.0003, 0.0]]].map(([cap, v]) =>
        <div key={cap as string} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14}}>
          <Bars values={v as number[]} texts={(v as number[]).map((x) => x.toFixed(4))} hi={[0]} maxH={260} barW={70} gap={30} size={20} />
          <Txt size={26} color={cap === '×8' ? ACC : C.ink2} glow={cap === '×8'}>{cap === '×8' ? '放大 8 倍：太陡' : '不放大'}</Txt>
        </div>)}
    </Abs>}
    <Abs x={0} y={140} w={1920} style={{display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 70}}>
      <div style={up(p('l1.a'))}><Tex id="grad" size={50} parts={{d: f >= at('l2') ? lit(ease(f, at('l2'), 10)) : {}}} /></div>
      <div style={up(p('l2'))}><Tex id="diag" size={46} color={ACC} glow /></div>
    </Abs>
    <Svg opacity={plotIn}>
      <Axes x0={PX(-4)} y0={PY(1.08)} x1={PX(10)} y1={PY(0)} p={1} xt={[[PX(0), '0'], [PX(1), '1'], [PX(4), '4'], [PX(8), '8']]}
        yt={[[PY(0.5), '0.5'], [PY(1), '1']]} xlabel="z₁ 比 z₂ 高多少" ylabel="p₁" />
      <path d={drawOn(curvePts((d) => p1(d, c), -4, 10, PX, PY, 200), ease(f, at('l2.a'), 24))} fill="none" stroke={C.ink2} strokeWidth={2.4} />
      <line x1={tx(d0 - 1.8)} y1={ty(d0 - 1.8)} x2={tx(d0 + 1.8)} y2={ty(d0 + 1.8)} stroke={ACC} strokeWidth={3} filter="url(#vglow)"
        opacity={ease(f, at('l2.a') + 16, 10)} />
      <circle cx={PX(d0)} cy={PY(q0)} r={9} fill={C.bg} stroke={ACC} strokeWidth={3} filter="url(#vglow)" opacity={ease(f, at('l2.a') + 10, 10)} />
    </Svg>
    <Abs x={move > 0.5 ? PX(d0) - 140 : PX(d0) - 360} y={move > 0.5 ? PY(q0) + 40 : PY(q0) - 10} w={280}
      style={{textAlign: move > 0.5 ? 'center' : 'right', opacity: ease(f, at('l2.a') + 20, 10)}}>
      <Mono size={24} color={ACC} glow>斜率 {m >= 0.01 ? m.toFixed(4) : m.toFixed(5)}</Mono>
      <div style={{fontSize: 20, color: C.ink3, marginTop: 4}}>{move < 0.5 ? '不放大（×1）' : '放大 8 倍'}</div>
    </Abs>
    <AbsoluteFill style={{opacity: 1 - cards * 0.85}}>
      <Abs x={1180} y={300} style={{...up(p('l2.b'))}}><Txt size={24} color={C.ink2}>对角线 p<sub>i</sub>(1 − p<sub>i</sub>)</Txt></Abs>
      <Svg opacity={p('l2.b')}>
        <line x1={1170} x2={1700} y1={700} y2={700} stroke={C.ink3} />
        {[0, 1, 2].map((i) => {
          const v = mix(diag1[i], diag8[i], barsK);
          const h = (v / 0.25) * 300;
          return <g key={i}>
            <rect x={1200 + i * 170} y={700 - h} width={100} height={Math.max(2, h)} fill={ACC} opacity={0.35 + 0.6 * (1 - barsK)} rx={2} />
            <text x={1250 + i * 170} y={700 - h - 12} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={barsK > 0.5 ? C.ink2 : ACC}>
              {barsK < 0.5 ? diag1[i].toFixed(4) : ['0.00034', '0.00034', '2.5e-7'][i]}</text>
            <text x={1250 + i * 170} y={732} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink3}>i = {i + 1}</text>
          </g>;
        })}
      </Svg>
    </AbsoluteFill>
    {/* backprop: the signal dies at a saturated softmax */}
    <Abs x={0} y={782} w={1920} style={{display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 22, opacity: chainIn * (1 - 0.6 * cards)}}>
      {[['q、k', 'qk'], ['分数 z', 'z'], ['概率 p', 'p'], ['损失 L', 'L']].map(([label, id], i) => {
        const blocked = id === 'qk' || id === 'z';
        const dead = blocked && f >= at('l3.a');
        return <React.Fragment key={id}>
          {i > 0 && <div style={{width: 110, height: 2, background: i <= 2 && f >= at('l3.a') ? C.line : rgba(ACC, 0.6), position: 'relative'}}>
            {i === 2 && f >= at('l3.a') && <Mono size={26} color={ACC} glow style={{position: 'absolute', left: 28, top: -40}}>× 0</Mono>}
          </div>}
          <div style={{padding: '12px 24px', borderRadius: 8, border: `1px solid ${dead ? C.line : rgba(ACC, 0.7)}`, fontFamily: F.zh,
            fontSize: 28, color: dead ? C.ink3 : C.ink, background: C.panel}}>{label}</div>
        </React.Fragment>;
      })}
    </Abs>
    {chainIn > 0 && f < at('l3.a') + 10 && <Abs x={mix(1390, 1040, pulse)} y={778} w={14} h={14} style={{borderRadius: 7, background: ACC, boxShadow: BOX_GLOW,
      opacity: 1 - lin(f, at('l3.a'), 10)}} />}
    <Abs x={0} y={846} w={1920} style={{textAlign: 'center', ...up(p('l3.b'))}}><Txt size={26} color={C.ink2} style={{opacity: 1 - cards}}>梯度传不回 q、k：这个头学不动</Txt></Abs>
    {/* honesty */}
    <Abs x={1090} y={300} w={720} style={{display: 'flex', flexDirection: 'column', gap: 18}}>
      {[['l4.a', '第一', '严格为 0，只发生在完全独热的极限'], ['l4.b', '', '中等饱和时，接近 0 的是流向非最大位置的梯度'],
        ['l5', '第二', '这是估算，不是证明。原文：We suspect（我们猜测）']].map(([key, n, text]) =>
        <div key={key} style={{display: 'flex', gap: 16, alignItems: 'flex-start', padding: '16px 20px', borderRadius: 10, background: rgba(C.bg, 0.94),
          border: `1px solid ${key === 'l5' ? rgba(ACC, 0.7) : C.line}`, ...up(p(key))}}>
          <Mono size={20} color={ACC} style={{width: 44, flexShrink: 0}}>{n}</Mono>
          <Txt size={26} color={C.ink}>{text}</Txt>
        </div>)}
    </Abs>
  </AbsoluteFill>;
};

export {Bars, fmt, scale};

