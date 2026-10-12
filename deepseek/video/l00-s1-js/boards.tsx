// 第 0 课 §1 · 一张科技树 —— 7 段黑板。日期与天数全部来自 lesson-00 §1 与完整时间线表（研究笔记 lesson-00-doc.md）。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Check, Chip, F, Mono, NodeTag, Roadmap, Svg, TEXT_GLOW, Tex, Txt, View, camAt, drawOn, ease, fit, lin, mix,
  project, rgba, swing, up, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, LANES, TreeScene, laneRect, nodeOf} from '../l00-shared/scene';

const useRm = () => useEp().roadmap as Roadmap;

// ================================================================ 01 3,377 天：一把按天数等比的尺子
const T0 = Date.UTC(2017, 5, 12);
const day = (s: string) => {
  const [y, m, d] = s.split('-').map(Number);
  return (Date.UTC(y, m - 1, d) - T0) / 864e5;
};
const RX0 = 150, RX1 = 1770, END = day('2026-10-01');
const rx = (s: string) => RX0 + ((RX1 - RX0) * day(s)) / END;
const RY = 500;
const MILES: [string, string, string, 1 | -1][] = [
  ['2017-06-12', 'Transformer', 'l1.a', -1], ['2018-06-11', 'GPT-1', 'l1.b', 1], ['2019-02-14', 'GPT-2', 'l1.b', -1],
  ['2020-05-28', 'GPT-3', 'l1.b', 1], ['2022-01-27', 'InstructGPT', 'l1.c', -1], ['2022-11-30', 'ChatGPT', 'l1.d', 1],
  ['2023-11-29', 'DeepSeek LLM', 'l2.a', -1], ['2026-09-10', 'V4.1-Flash', 'l2.b', -1],
];

const Span: React.FC<{a: string; b: string; y: number; p: number; label: string; big?: boolean}> = ({a, b, y, p, label, big}) => {
  if (p <= 0) return null;
  const x0 = rx(a), x1 = rx(b);
  return <g opacity={Math.min(1, p * 1.4)}>
    <path d={`M${x0} ${y + 10} L${x0} ${y} L${x0 + (x1 - x0) * p} ${y}${p > 0.98 ? ` L${x1} ${y + 10}` : ''}`} fill="none"
      stroke={big ? ACC : C.ink2} strokeWidth={big ? 2.2 : 1.4} filter={big ? 'url(#vglow)' : undefined} />
    <text x={(x0 + x1) / 2} y={y - 12} textAnchor="middle" fontFamily={F.mono} fontSize={big ? 34 : 24} fill={big ? ACC : C.ink}
      opacity={p}>{label}</text>
  </g>;
};

export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const rows = rm as any;
  const tl: {date: string; name: string; trunk: boolean; paper: boolean}[] = rows.timeline;
  const dots = ease(f, at('l3.a'), 20);
  // stack entries that fall within ~3 weeks of each other
  const ys: number[] = [];
  const lastX: number[] = [];
  tl.forEach((e) => {
    const x = rx(e.date);
    let k = 0;
    while (lastX[k] !== undefined && x - lastX[k] < 9) k++;
    lastX[k] = x;
    ys.push(k);
  });
  return <AbsoluteFill>
    <Svg>
      <line x1={RX0} x2={RX0 + (RX1 - RX0) * swing(f, 4, 40)} y1={RY} y2={RY} stroke={C.ink3} strokeWidth={2} />
      {[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026].map((yr) => <g key={yr} opacity={ease(f, 10 + (yr - 2018) * 2, 12)}>
        <line x1={rx(`${yr}-01-01`)} x2={rx(`${yr}-01-01`)} y1={RY - 6} y2={RY + 6} stroke={C.ink3} />
        <text x={rx(`${yr}-01-01`)} y={RY + 30} textAnchor="middle" fontFamily={F.mono} fontSize={17} fill={C.ink3}>{yr}</text>
      </g>)}
      {MILES.map(([d, name, key, side]) => {
        const q = ease(f, at(key) + (name === 'GPT-2' ? 8 : name === 'GPT-3' ? 16 : 0), 12);
        if (q <= 0) return null;
        const x = rx(d), ds = name === 'DeepSeek LLM' || name === 'V4.1-Flash';
        return <g key={name} opacity={q}>
          <line x1={x} x2={x} y1={RY} y2={RY + side * 62} stroke={ds ? ACC : C.ink2} strokeWidth={1.5} />
          <circle cx={x} cy={RY} r={7} fill={ds ? ACC : C.ink} filter={ds ? 'url(#vglow)' : undefined} />
          <text x={x} y={RY + side * 80 + (side > 0 ? 10 : 0)} textAnchor="middle" fontFamily={F.zh} fontSize={24}
            fill={ds ? ACC : C.ink}>{name}</text>
          <text x={x} y={RY + side * 80 + (side > 0 ? 36 : -26)} textAnchor="middle" fontFamily={F.mono} fontSize={16} fill={C.ink3}>{d}</text>
        </g>;
      })}
      <Span a="2017-06-12" b="2022-11-30" y={RY - 150} p={lin(f, at('l2'), 24)} label="1,997 天" />
      <Span a="2022-11-30" b="2023-11-29" y={RY - 150} p={lin(f, at('l2.a'), 14)} label="364" />
      <Span a="2023-11-29" b="2026-09-10" y={RY - 150} p={lin(f, at('l2.b'), 20)} label="1,016 天" />
      <Span a="2017-06-12" b="2026-09-10" y={RY - 236} p={lin(f, at('l3'), 30)} label="3,377 天" big />
      {dots > 0 && tl.map((e, i) => {
        const q = ease(f, at('l3.a') + i * 0.6, 8);
        const x = rx(e.date), y = RY + 150 + ys[i] * 15;
        return <circle key={i} cx={x} cy={y} r={e.trunk ? 6 : 4} fill={e.paper ? 'none' : e.trunk ? ACC : C.ink2}
          stroke={e.trunk ? ACC : C.ink2} strokeWidth={1.4} opacity={q} filter={e.trunk ? 'url(#vglow)' : undefined} />;
      })}
    </Svg>
    <Abs x={rx('2023-11-02') - 10} y={RY + 230} style={{...up(ease(f, at('l3.a') + 20, 12))}}>
      <Txt size={22} color={C.ink2}>DeepSeek-Coder（2023-11-02）→ DSec（2026-09-19）：50 个条目</Txt></Abs>
    <Abs x={120} y={RY + 300} style={{display: 'flex', gap: 34, ...up(p('l3.b'))}}>
      {[['主干模型（7 个）', ACC, true], ['旁支模型、开源制品', C.ink2, true], ['论文（按 arXiv 日期）', C.ink2, false]].map(([t, col, solid]) =>
        <div key={t as string} style={{display: 'flex', alignItems: 'center', gap: 10}}>
          <div style={{width: 12, height: 12, borderRadius: 6, background: solid ? (col as string) : 'transparent', border: `2px solid ${col as string}`}} />
          <Txt size={22} color={C.ink2}>{t as string}</Txt></div>)}
    </Abs>
    <Abs x={120} y={160} style={{...up(ease(f, at('l3') + 20, 14))}}><Tex id="days" size={44} color={C.ink2} /></Abs>
  </AbsoluteFill>;
};

// ================================================================ 02 从清单到树：50 个条目按账单分组
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const tl: {date: string; name: string; trunk: boolean; lanes: string[]; x: number}[] = (rm as any).timeline;
  const cam = project(FULLV);
  const fly = (i: number) => swing(f, at('l1.a') + 6 + (i % 17) * 1.2, 34);
  const tree = ease(f, at('l1.a') + 30, 24);
  const pulseOn = f >= at('l2.a') && f < at('l3');
  const xmax = rm.axis.x1 + 160;
  const pulseX = (k: number) => -60 + (xmax + 400) * lin(f, at('l2.a') + k * 5, 70);
  const laneY = (k: string) => {
    const ln = rm.lanes.find((l) => l.key === k);
    return ln ? ln.y + ln.h / 2 : rm.height + 40;
  };
  const count: Record<string, number> = {};
  const nodeLit = (n: {id: string; lane: string; x: number; w: number}) => {
    let g = 0;
    if (pulseOn) {
      const k = LANES.indexOf(n.lane);
      if (k >= 0) {
        const d = pulseX(k) - (n.x + n.w / 2);
        g = d > 0 ? Math.exp(-d / 220) * 0.9 : 0;
      }
    }
    if (n.id === 'mla') g = Math.max(g, ease(f, at('l3.a'), 10));
    if (n.id === 'gqa67') g = Math.max(g, ease(f, at('l3.b'), 10));
    return g;
  };
  return <AbsoluteFill>
    <TreeScene cam={cam} alpha={tree} look={{nodeLit: nodeLit as any, pulses: pulseOn ? LANES.map((l, k) => ({lane: l, x: pulseX(k), k: 1})) : []}}>
      <NodeTag rm={rm} cam={cam} id="mla" text="节点 = 一次改进" p={ease(f, at('l3.a') + 6, 10)} />
      <NodeTag rm={rm} cam={cam} id="gqa67" text="枝干 = 在它的基础上" above={false} p={ease(f, at('l3.b') + 6, 10)} />
    </TreeScene>
    {/* the list, flying into its lanes */}
    {tl.map((e, i) => {
      const q = fly(i);
      if (q >= 1 && tree >= 1) return null;
      const col = Math.floor(i / 17), row = i % 17;
      const lx = 150 + col * 560, ly = 170 + row * 36;
      const lane = e.lanes[0];
      const k = (count[lane] = (count[lane] ?? 0) + 1);
      const tx = cam.X(e.x), ty = cam.Y(laneY(lane)) + ((k % 3) - 1) * 8;
      const x = mix(lx, tx, q), y = mix(ly, ty, q);
      const show = ease(f, 4 + i * 0.5, 10);
      return <React.Fragment key={i}>
        <Abs x={x} y={y - 12} style={{opacity: show * (1 - q), display: 'flex', gap: 14, whiteSpace: 'nowrap'}}>
          <Mono size={16} color={C.ink3}>{e.date}</Mono>
          <span style={{fontSize: 19, color: e.trunk ? ACC : C.ink2}}>{e.name.length > 18 ? e.name.slice(0, 18) + '…' : e.name}</span>
        </Abs>
        {q > 0.05 && <Abs x={x - 5} y={y - 5} w={10} h={10} style={{borderRadius: 5, background: e.trunk ? ACC : C.ink2,
          opacity: Math.min(1, q * 2) * (1 - tree * 0.9)}} />}
      </React.Fragment>;
    })}
    <Abs x={cam.X(-300)} y={cam.Y(rm.height + 40) - 14} style={{opacity: fly(40) * (1 - tree * 0.9)}}>
      <Txt size={20} color={C.ink3}>旁支：多模态（M）· 形式证明（P）</Txt></Abs>
    <Abs x={0} y={100} w={1920} style={{textAlign: 'center', ...up(p('l2.a'))}}>
      <Txt size={30} color={C.ink2}>同一条线上：后一个节点 ≈ 在前一个节点的账单上<span style={{color: ACC}}> 再打一个折</span></Txt></Abs>
  </AbsoluteFill>;
};

// ================================================================ 03 六条主线，六张账单
const BILLS: [string, string, string][] = [
  ['A', '每个 token 缓存多少字节', 'l1.a'], ['B', '参数量与计算量绑在一起', 'l1.b'], ['C', '训练算力、显存与通信', 'l1.c'],
  ['D', '训练信号：从“像语料”到“做得对”', 'l2.a'], ['E', '深度方向的信号尺度；知识记在哪里', 'l2.b'], ['F', '一次性大训练的“下注”', 'l2.c'],
];
export const B03: React.FC = () => {
  const {f, at} = useBoard();
  const rm = useRm();
  const cam = project(FULLV);
  const cur = [...BILLS].reverse().find(([, , key]) => f >= at(key));
  const cross = ease(f, at('l3.a'), 14);
  const laneLit = (k: string) => (cur && cur[0] === k && cross <= 0 ? 1 : k === 'A' || k === 'C' ? cross : 0);
  const nodeLit = (n: {id: string}) => (['fp4kv', 'csa2'].includes(n.id) ? ease(f, at('l3.a') + 6, 10) : 0);
  const shown = (k: string) => {
    const b = BILLS.find((x) => x[0] === k)!;
    return ease(f, at(b[2]), 12);
  };
  const v4x = cam.X(nodeOf(rm, 'csa2').x + nodeOf(rm, 'csa2').w / 2);
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{laneLit, nodeLit: nodeLit as any, alpha: 1 - 0.4 * ease(f, at('l1'), 12) * (1 - cross)}}>
      {cross > 0 && <line x1={v4x} x2={v4x} y1={cam.Y(nodeOf(rm, 'csa2').y + nodeOf(rm, 'csa2').h)} y2={cam.Y(nodeOf(rm, 'fp4kv').y)}
        stroke={ACC} strokeWidth={2} strokeDasharray="5 5" filter="url(#glow)" opacity={cross} />}
    </TreeScene>
    {BILLS.map(([k, bill]) => {
      const ln = rm.lanes.find((l) => l.key === k)!;
      const q = shown(k) * (1 - cross * 0.8);
      const on = cur && cur[0] === k && cross <= 0;
      return <Abs key={k} x={cam.X(40)} y={cam.Y(ln.y + ln.h / 2) - 22} style={{...up(q, 8), display: 'flex', alignItems: 'center', gap: 14,
        padding: '6px 16px', borderRadius: 8, background: rgba(C.bg, 0.9), border: `1px solid ${on ? ACC : C.line}`,
        boxShadow: on ? BOX_GLOW : undefined}}>
        <Mono size={22} color={on ? ACC : C.ink2}>{k}</Mono><Txt size={24} color={on ? C.ink : C.ink2}>{bill}</Txt>
      </Abs>;
    })}
    <Abs x={v4x - 420} y={cam.Y(nodeOf(rm, 'fp4kv').y) + 60} w={400} style={{textAlign: 'right', ...up(ease(f, at('l3.b'), 12))}}>
      <Txt size={28} color={ACC} glow weight={400}>A ∩ C：V4.1 的 FP4 KV 缓存</Txt>
      <Txt size={22} color={C.ink2}>缓存更小（A）· 数值格式更短（C）</Txt>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 04 主干的节奏：两次发布之间隔了几天
const TRUNK: [string, string][] = [['llm', 'LLM'], ['v2', 'V2'], ['v3', 'V3'], ['r1', 'R1'], ['v32exp', 'V3.2-Exp'], ['v32', 'V3.2'],
  ['v4', 'V4 预览'], ['v4pro', 'V4-Pro'], ['v41', 'V4.1-Flash']];
const GAPS: [string, string, string, string][] = [
  ['llm', 'v2', '159', 'l1.a'], ['v2', 'v3', '234', 'l1.b'], ['v3', 'r1', '25', 'l1.c'], ['r1', 'v32exp', '252', 'l3'],
  ['v32exp', 'v32', '63', 'l3'], ['v32', 'v4', '144', 'l3.a'], ['v4', 'v4pro', '111', 'l3.a'], ['v4pro', 'v41', '28', 'l3.b'],
];
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const cam = project(FULLV);
  const X = (id: string) => cam.X(nodeOf(rm, id).x);
  const top = cam.Y(-30);
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{alpha: 0.75, nodeLit: ((n: {id: string}) => (n.id === 'r1' ? ease(f, at('l2'), 10) : 0)) as any}}>
      {TRUNK.map(([id], i) => <line key={id} x1={X(id)} x2={X(id)} y1={top} y2={cam.Y(rm.height)} stroke={rgba(ACC, 0.55)}
        strokeWidth={1.4} strokeDasharray="4 6" opacity={ease(f, at('l1') + i * 3, 10)} />)}
      {GAPS.map(([a, b, d, key], i) => {
        const q = lin(f, at(key) + (i === 4 || i === 6 ? 14 : 0), 14);
        if (q <= 0) return null;
        const x0 = X(a), x1 = X(b), y = top - 18 - (i % 2) * 38;
        return <g key={a} opacity={Math.min(1, q * 1.5)}>
          <path d={`M${x0} ${y + 8} L${x0} ${y} L${x0 + (x1 - x0) * q} ${y}${q > 0.98 ? ` L${x1} ${y + 8}` : ''}`} fill="none"
            stroke={d === '25' || d === '28' ? ACC : C.ink2} strokeWidth={1.6} />
          <text x={(x0 + x1) / 2} y={y - 9} textAnchor="middle" fontFamily={F.mono} fontSize={22}
            fill={d === '25' || d === '28' ? ACC : C.ink}>{d} 天</text>
        </g>;
      })}
    </TreeScene>
    {TRUNK.map(([id, name], i) => {
      // releases a few weeks apart share the same x: drop every other close one to a second row, with a short tick
      const row = TRUNK.slice(0, i + 1).reduce((r, [cur], j) => (j > 0 && X(cur) - X(TRUNK[j - 1][0]) < 110 ? 1 - r : 0), 0);
      return <Abs key={id} x={X(id) - 70} y={cam.Y(rm.height) + 10 + row * 26} w={140} style={{textAlign: 'center',
        opacity: ease(f, at('l1') + i * 3, 10)}}><Mono size={17} color={C.ink2}>{name}</Mono></Abs>;
    })}
    <Abs x={X('r1') + 20} y={cam.Y(rm.lanes[4].y) - 8} w={420} style={{...up(p('l2.a')), padding: '10px 16px', borderRadius: 8,
      background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
      <Txt size={24}>R1 没有训练新基座：</Txt><Txt size={24} color={ACC}>在 V3 的基座上做强化学习</Txt>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 读图的三种方法
const CHAIN_A: [string, string, string][] = [['mha', 'l1.b', 'MHA 2017'], ['mqa', 'l1.b', 'MQA'], ['gqa', 'l1.b', 'GQA'],
  ['gqa67', 'l1.b', '67B：GQA-8'], ['mla', 'l1.c', 'V2：MLA'], ['nsa', 'l1.c', 'NSA'], ['dsa', 'l1.c', 'V3.2：DSA'],
  ['csa', 'l1.c', 'V4：CSA / HCA'], ['csa2', 'l1.d', 'V4.1：CSA2']];
const COLS: [string, string[], string][] = [['v2', ['A', 'B', 'D'], 'l2.a'], ['v3', ['B', 'C', 'D'], 'l2.b'],
  ['v4', ['A', 'B', 'C', 'D', 'E', 'F'], 'l2.c'], ['r1', ['D'], 'l2.d']];
export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const cam = project(FULLV);
  const chainLit = (id: string) => {
    const i = CHAIN_A.findIndex(([n]) => n === id);
    if (i < 0) return 0;
    return ease(f, at(CHAIN_A[i][1]) + i * 7, 8) * (1 - ease(f, at('l2'), 12));
  };
  const col = [...COLS].reverse().find(([, , key]) => f >= at(key) && f < at('l3'));
  const laneLit = (k: string) => (f < at('l2') ? (k === 'A' ? ease(f, at('l1.a'), 10) : 0) : col && col[1].includes(k) ? 1 : 0);
  const gap = ease(f, at('l3.a'), 14);
  const E = rm.lanes.find((l) => l.key === 'E')!;
  const gx0 = nodeOf(rm, 'xiong').x + nodeOf(rm, 'xiong').w + 10, gx1 = nodeOf(rm, 'hc').x - 10;
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{laneLit, nodeLit: ((n: {id: string}) => Math.max(chainLit(n.id), n.id === 'hc' ? ease(f, at('l3.c'), 10) : 0)) as any,
      alpha: f >= at('l3') ? 0.6 : 1}}>
      {col && <rect x={cam.X(nodeOf(rm, col[0]).x) - 14} y={cam.Y(-20)} width={28} height={(rm.height + 30) * cam.s}
        fill={rgba(ACC, 0.12)} stroke={rgba(ACC, 0.6)} strokeDasharray="4 4" />}
      {gap > 0 && <rect x={cam.X(gx0)} y={cam.Y(E.y + 4)} width={(gx1 - gx0) * cam.s * gap} height={(E.h - 8) * cam.s}
        fill={rgba(C.ink2, 0.06)} stroke={C.ink2} strokeDasharray="6 6" />}
      {CHAIN_A.map(([id, , text]) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text}
        above={['mqa', 'gqa67', 'nsa', 'csa'].includes(id) ? false : true} p={chainLit(id)} />)}
      <NodeTag rm={rm} cam={cam} id="hc" text="Hyper-Connections（ByteDance）" p={ease(f, at('l3.c') + 6, 10)} />
    </TreeScene>
    <Abs x={120} y={110} style={{display: 'flex', gap: 24}}>
      {[['横着读：一条线的演进', 'l1.a', 'l2'], ['竖着读：一次发布动了哪几条线', 'l2', 'l3'], ['看空白', 'l3', 'end']].map(([t, a, b]) =>
        <div key={t} style={{...up(p(a)), opacity: p(a) * (b === 'end' || f < at(b) ? 1 : 0.35)}}>
          <Chip on={b === 'end' ? f >= at(a) : f >= at(a) && f < at(b)} size={24}>{t}</Chip></div>)}
    </Abs>
    {col && <Abs x={cam.X(nodeOf(rm, col[0]).x) - 80} y={cam.Y(rm.height) + 14} w={160} style={{textAlign: 'center'}}>
      <Mono size={20} color={ACC} glow>{nodeOf(rm, col[0]).label}</Mono></Abs>}
    <Abs x={cam.X((gx0 + gx1) / 2) - 260} y={cam.Y(E.y) - 56} w={520} style={{textAlign: 'center', ...up(gap)}}>
      <Txt size={24} color={C.ink}>空白：2020 年 – 2024 年 9 月</Txt>
      <Txt size={20} color={C.ink2} style={{opacity: ease(f, at('l3.b'), 10)}}>系数恒为 1 的捷径，几年没人动过</Txt>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 06 继承的，和自己提出的
const INHERITED = ['mqa', 'gqa', 'rope', 'yarn', 'gshard', 'switch', 'ppo', 'dpo', 'hc', 'muon', 'mtp_par'];
const OWN = ['dsmoe', 'grpo', 'mla', 'alf', 'v3moe', 'fp8train', 'nsa', 'dsa', 'mhc', 'engram', 'csa', 'csa2'];
export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const cam = project(FULLV);
  const inh = ease(f, at('l1.a'), 14), own = ease(f, at('l2'), 14), pair = ease(f, at('l3.a'), 14);
  const nodeLit = (n: {id: string}) => (OWN.includes(n.id) ? own * (pair > 0 && n.id !== 'mla' ? 0.4 : 1) : 0);
  const box = (id: string) => {
    const n = nodeOf(rm, id);
    return {x: cam.X(n.x), y: cam.Y(n.y), w: n.w * cam.s, h: n.h * cam.s};
  };
  const g = box('gqa67'), m = box('mla');
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{nodeLit: nodeLit as any, alpha: 0.85}}>
      {INHERITED.map((id) => {
        const b = box(id);
        const dimPair = pair > 0 && id !== 'gqa' ? 0.35 : 1;
        return <rect key={id} x={b.x - 3} y={b.y - 3} width={b.w + 6} height={b.h + 6} rx={6} fill="none" stroke={C.ink}
          strokeWidth={1.6} strokeDasharray="5 4" opacity={inh * dimPair} />;
      })}
      {pair > 0 && <path d={drawOn([[g.x + g.w, g.y + g.h / 2], [m.x, m.y + m.h / 2]], pair)} stroke={ACC} strokeWidth={2.5}
        fill="none" filter="url(#glow)" />}
    </TreeScene>
    <Abs x={120} y={110} style={{display: 'flex', gap: 30}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 12, ...up(inh)}}>
        <div style={{width: 28, height: 16, border: `1.6px dashed ${C.ink}`, borderRadius: 4}} /><Txt size={24}>继承：来自别的团队</Txt></div>
      <div style={{display: 'flex', alignItems: 'center', gap: 12, ...up(own)}}>
        <div style={{width: 28, height: 16, background: rgba(ACC, 0.3), border: `1.6px solid ${ACC}`, borderRadius: 4}} />
        <Txt size={24} color={ACC}>DeepSeek 自己提出</Txt></div>
    </Abs>
    <Abs x={(g.x + m.x) / 2 - 220} y={m.y - 120} w={440} style={{textAlign: 'center', ...up(ease(f, at('l3.b'), 12)), padding: '12px 16px',
      borderRadius: 10, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
      <Txt size={28} color={C.ink}>它嫌前者哪里不够？</Txt>
      <Txt size={20} color={C.ink2}>67B 的 GQA-8 → V2 的 MLA</Txt>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 07 三个问题的答案
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const out = ease(f, at('l3.a') - 4, 16);
  const A = laneRect(rm, 'A');
  const v = camAt(f, FULLV, [{at: at('l3.a'), dur: 60, to: fit({...A, x: -330, w: 1400}, 0.04)}]);
  const items: [number, string, string, string][] = [
    [1, '3,377 天', 'Transformer → ChatGPT 1,997 天 → DeepSeek LLM 364 天 → V4.1-Flash 1,016 天', 'l1.a'],
    [2, '六条主线', 'A 缓存字节 · B 参数与计算 · C 算力显存通信 · D 训练信号 · E 信号与记忆 · F 下注', 'l2'],
    [3, '三种读法', '横着读演进 · 竖着读发布 · 看空白；再分清继承的和自己提出的', 'l2.a'],
  ];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - out}}>
      <Abs x={120} y={190} w={1680} style={{display: 'flex', flexDirection: 'column', gap: 26}}>
        {items.map(([n, head, body, key]) => {
          const done = ease(f, at(key), 12);
          return <div key={n} style={{display: 'flex', alignItems: 'center', gap: 26, padding: '22px 28px', borderRadius: 14,
            background: rgba(C.bg, 0.9), border: `1px solid ${done > 0.5 ? rgba(ACC, 0.7) : C.line}`, boxShadow: done > 0.5 ? BOX_GLOW : undefined,
            ...up(p(n === 1 ? 'l1' : key))}}>
            <Mono size={32} color={ACC} glow>Q{n}</Mono>
            <div style={{flexGrow: 1}}><Txt size={32} weight={400}>{head}</Txt><Txt size={24} color={C.ink2}>{body}</Txt></div>
            <svg width={48} height={48} style={{overflow: 'visible'}}><Check x={2} y={2} size={42} p={lin(f, at(key), 14)} /></svg>
          </div>;
        })}
      </Abs>
    </AbsoluteFill>
    {out > 0 && <AbsoluteFill style={{opacity: out}}>
      <TreeScene cam={project(v)} look={{laneLit: (k) => (k === 'A' ? ease(f, at('l3.a') + 30, 12) : 0)}} />
      <Abs x={0} y={800} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 主线 A：每个 token 缓存多少字节</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};

export {TEXT_GLOW};
