// 第 0 课 §2 · 主线 A：注意力与 KV cache —— 8 段黑板。数字全部来自 lesson-00 §2（研究笔记 lesson-00-doc.md）。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Check, Chip, F, Mono, Roadmap, Svg, TEXT_GLOW, Tex, Txt, camAt, dim, ease, fit, lin, lit, mix,
  project, rgba, swing, up, useBoard, useEp, useTexBox,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, TreeScene, laneRect, nodeOf} from '../l00-shared/scene';
import {LogBars, Span} from '../l00-shared/charts';

const useRm = () => useEp().roadmap as Roadmap;
const Card: React.FC<{on?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, style, children}) =>
  <div style={{padding: '18px 24px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${on ? rgba(ACC, 0.7) : C.line}`,
    boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;

// ================================================================ 01 一个 1M token 的请求
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  // decode: one step per P frames while the first line plays; in each step a read head sweeps the whole cache
  const P = Math.max(8, Math.floor((at('l2') - 20) / 22));
  const t = Math.max(0, f - 10);
  const step = Math.min(21, Math.floor(t / P));
  const head = ((t % P) / P) * step;
  const partA = 1 - ease(f, at('l3'), 12);
  const gpu = ease(f, at('l3'), 16), cache = swing(f, at('l3') + 12, 40);
  const X0 = 160, W = 1500, GW = (W * 80) / 408, L = W * cache;
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: partA}}>
      <Abs x={160} y={170} style={up(p('l1'))}><Txt size={30} color={C.ink2}>解码的每一步：新 token 要读一遍前面全部的键和值</Txt></Abs>
      <Abs x={160} y={250} style={{display: 'flex', gap: 4}}>
        {Array.from({length: 22}, (_, i) => {
          const cached = i < step, now = i === step, read = cached && Math.abs(i + 0.5 - head) < 1.2;
          return <div key={i} style={{width: 56, height: 56, borderRadius: 6, opacity: cached || now ? 1 : 0.12,
            background: now ? rgba(ACC, 0.3) : read ? rgba(ACC, 0.5) : C.panel, border: `1px solid ${now || read ? ACC : C.line}`,
            boxShadow: read ? BOX_GLOW : undefined, display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: F.mono, fontSize: 15, color: now ? ACC : C.ink3}}>{now ? 'new' : cached ? 'KV' : ''}</div>;
        })}
      </Abs>
      <Abs x={160} y={318} style={{opacity: ease(f, 30, 14)}}><Mono size={18} color={C.ink3}>← KV cache：每一步从显存里整份读一遍</Mono></Abs>
      <Abs x={160} y={372} style={up(p('l1.a'))}><Chip on size={28}>解码速度 ≈ 每一步读多少字节，而不是算多少次乘法</Chip></Abs>
      <Abs x={160} y={500} style={{display: 'flex', alignItems: 'baseline', gap: 22, ...up(p('l2.a'))}}>
        <Txt size={30} color={C.ink2}>DeepSeek LLM 67B：每个 token</Txt>
        <Mono size={64} color={ACC} glow>389,120</Mono><Txt size={30} color={C.ink2}>字节</Txt>
      </Abs>
      <Abs x={160} y={620} style={up(p('l2.b'))}><Txt size={26} color={C.ink2}>一个 1M（2²⁰）token 的请求：</Txt></Abs>
      <Abs x={160} y={672} style={up(p('l2.b'))}><Tex id="req" size={50} /></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: gpu}}>
      <Abs x={X0} y={200} style={up(gpu)}><Txt size={30} color={C.ink2}>一个 1M token 请求的 KV cache，放进一张 80 GB 的显卡</Txt></Abs>
      <Svg>
        <rect x={X0} y={300} width={GW} height={150} rx={10} fill={rgba(C.ink, 0.03)} stroke={C.ink} strokeWidth={2} />
        <text x={X0 + 14} y={290} fontFamily={F.zh} fontSize={22} fill={C.ink}>80 GB</text>
        <rect x={X0} y={345} width={Math.min(L, GW)} height={60} fill={rgba(ACC, 0.45)} />
        {L > GW && <rect x={X0 + GW} y={345} width={L - GW} height={60} fill="url(#vhatch)" stroke={ACC} strokeWidth={1.5} />}
        {cache > 0 && <text x={X0 + L + 16} y={386} fontFamily={F.mono} fontSize={30} fill={ACC} filter="url(#vglow)">
          {Math.round(408 * cache)} GB</text>}
        {L > GW + 40 && <text x={X0 + GW + 16} y={436} fontFamily={F.zh} fontSize={20} fill={C.ink2} opacity={lin(f, at('l3') + 40, 12)}>
          放不下的部分</text>}
      </Svg>
      <Abs x={X0} y={520} style={{display: 'flex', alignItems: 'baseline', gap: 26, ...up(p('l3.a'))}}>
        <Txt size={34}>一张卡只放得下 <span style={{color: ACC, textShadow: TEXT_GLOW}}>0.2</span> 个这样的请求</Txt>
        <Mono size={26} color={C.ink2}>80 ÷ 408 ≈ 0.2</Mono>
      </Abs>
      <Abs x={X0} y={630} style={up(p('l3.b'))}><Chip on size={30}>主线 A 的全部工作：把每 token 的字节压下去</Chip></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 02 会议纪要本
// Each step of lane A as a small transformation of the notebook (drawn, 120 × 60 px).
const Icon: React.FC<{kind: string; k: number}> = ({kind, k}) => {
  const r = (key: string, x: number, y: number, w: number, h: number, o = 1, hot = false) =>
    <rect key={key} x={x} y={y} width={Math.max(0, w)} height={Math.max(0, h)} rx={2} fill={hot ? rgba(ACC, 0.4) : rgba(C.ink2, 0.2)}
      stroke={hot ? ACC : C.line} strokeWidth={1} opacity={o} />;
  const done = k > 0.92;
  let body: React.ReactNode = null;
  if (kind === 'gqa') body = [0, 1, 2, 3].map((i) => r(`g${i}`, mix(16 + i * 24, 51, k), 6, 18, 48, i === 0 ? 1 : 1 - k, done));
  if (kind === 'mla') body = r('m', 10, 18, mix(100, 30, k), 24, 1, done);
  if (kind === 'dsa') body = <>{r('toc', 6, 4, 10, 52, k, false)}{[0, 1, 2, 3, 4, 5].map((i) =>
    r(`d${i}`, 22, 4 + i * 9, 92, 7, i === 1 || i === 4 ? 1 : mix(1, 0.25, k), (i === 1 || i === 4) && done))}</>;
  if (kind === 'csa') body = [0, 1, 2, 3].map((i) => r(`c${i}`, 10, mix(6 + i * 13, 20, k), 100, mix(10, 18, k), i === 0 ? 1 : 1 - k, done));
  if (kind === 'csa2') body = Array.from({length: 10}, (_, i) => r(`b${i}`, mix(8 + i * 10.5, 56, k), 14, 8, 32, i === 0 ? 1 : 1 - k, done));
  if (kind === 'fp4') body = r('q', 10, 18, mix(100, 25, k), 24, 1, done);
  return <svg width={120} height={60} style={{overflow: 'visible', flexShrink: 0}}>{body}</svg>;
};
const STEPS: [string, string, string, string][] = [
  ['gqa', 'MQA / GQA', '几位发言者共用一栏', 'l2.a'], ['mla', 'MLA', '改记速记符号（潜向量）', 'l2.b'],
  ['dsa', 'DSA', '先查目录，只精读得分最高的 2048 条', 'l2.c'], ['csa', 'CSA / HCA', '每 4 句或每 128 句合写成一条', 'l3.a'],
  ['csa2', 'CSA2', '40 层共用 4 本', 'l3.b'], ['fp4', 'FP4', '用更短的字来写', 'l3.c'],
];
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const rows = 9;
  const scan = f >= at('l1.b') && f < at('l2') ? ((f - at('l1.b')) % 45) / 45 : -1;
  const cur = [...STEPS].reverse().find(([, , , key]) => f >= at(key));
  return <AbsoluteFill>
    <Abs x={150} y={160} style={up(p('l1'))}><Txt size={30} color={C.ink2}>KV cache = 会议纪要本</Txt></Abs>
    <Abs x={150} y={220} w={560} style={{padding: 20, borderRadius: 12, border: `1px solid ${C.line}`, background: C.panel, ...up(p('l1'))}}>
      {Array.from({length: rows}, (_, i) => {
        const q = ease(f, at('l1.a') + i * 4, 8);
        const reading = scan >= 0 && Math.abs(scan * rows - i - 0.5) < 0.8;
        return <div key={i} style={{display: 'flex', gap: 10, alignItems: 'center', height: 48, opacity: q}}>
          <Mono size={16} color={C.ink3} style={{width: 70}}>发言 {i + 1}</Mono>
          <div style={{width: 130, height: 28, borderRadius: 4, background: reading ? rgba(ACC, 0.4) : rgba(C.ink2, 0.2),
            border: `1px solid ${reading ? ACC : C.line}`, fontSize: 15, color: C.ink2, display: 'flex', alignItems: 'center',
            justifyContent: 'center'}}>标签 K</div>
          <div style={{width: 270, height: 28, borderRadius: 4, background: reading ? rgba(ACC, 0.25) : rgba(C.ink2, 0.12),
            border: `1px solid ${reading ? ACC : C.line}`, fontSize: 15, color: C.ink2, display: 'flex', alignItems: 'center',
            paddingLeft: 12}}>要点 V</div>
        </div>;
      })}
    </Abs>
    <Abs x={150} y={770} w={560} style={up(p('l1.b'))}><Txt size={24} color={C.ink2}>后来的人只翻本子，不必重听录音</Txt></Abs>
    <Abs x={780} y={160} style={up(p('l2'))}><Txt size={26} color={C.ink2}>每一步：让本子变薄，或者少翻几页</Txt></Abs>
    <Abs x={780} y={220} w={1020} style={{display: 'flex', flexDirection: 'column', gap: 10}}>
      {STEPS.map(([kind, name, what, key]) => {
        const on = !!cur && cur[0] === kind && f < at('l4');
        return <Card key={kind} on={on} style={{display: 'flex', gap: 24, alignItems: 'center', padding: '10px 22px', ...up(p(key))}}>
          <Icon kind={kind} k={swing(f, at(key) + 8, 26)} />
          <Mono size={24} color={ACC} style={{width: 150}}>{name}</Mono><Txt size={26}>{what}</Txt></Card>;
      })}
    </Abs>
    <Abs x={780} y={770} w={1020} style={up(p('l4.a'))}>
      <Card on><Txt size={25}>比方不对的地方：MLA 的速记、CSA 的合写是<span style={{color: ACC}}>训练出来的有损压缩</span>，
        <span style={{opacity: ease(f, at('l4.b'), 12)}}>不能事后套在现成模型上</span></Txt></Card>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 03 四个因子
const ASK: [string, string, string][] = [['c', '一共存几份', 'l1.a'], ['m', '每几个 token 存一条', 'l1.b'], ['d', '每条几个数', 'l1.c'],
  ['b', '每个数几个字节', 'l1.d']];
const DEFAULTS: [string, string, string][] = [['c', '普通模型：份数 = 层数 l', 'l2.a'], ['m', '不压缩：m = 1', 'l2.b'], ['b', 'BF16：每个数 2 字节', 'l2.c']];
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const box = useTexBox();
  const size = 96;
  const fb = box('bkv', size);
  const fx = (1920 - fb.w) / 2, fy = 210;
  let hot: string | undefined;
  if (f < at('l2')) hot = [...ASK].reverse().find(([, , key]) => f >= at(key))?.[0];
  else if (f < at('l3')) hot = [...DEFAULTS].reverse().find(([, , key]) => f >= at(key))?.[0];
  const parts: Record<string, any> = {};
  ASK.forEach(([name, , key]) => { parts[name] = f < at(key) ? dim(0.25) : hot === name ? lit(1) : {}; });
  const down = ease(f, at('l3'), 16);
  return <AbsoluteFill>
    <Abs x={fx} y={fy - 70 * down} style={{transform: `scale(${1 - 0.3 * down})`, transformOrigin: 'center top', opacity: ease(f, at('l1'), 14)}}>
      <Tex id="bkv" size={size} parts={parts} />
    </Abs>
    {ASK.map(([name, label, key]) => {
      const b = box('bkv', size, name);
      return <Abs key={name} x={fx + b.x + b.w / 2 - 130} y={fy + fb.h + 20} w={260} style={{textAlign: 'center',
        ...up(ease(f, at(key), 12) * (1 - down))}}>
        <Txt size={26} color={hot === name ? ACC : C.ink2} glow={hot === name}>{label}</Txt></Abs>;
    })}
    <Abs x={0} y={580} w={1920} style={{display: 'flex', justifyContent: 'center', gap: 22, opacity: 1 - down}}>
      {DEFAULTS.map(([name, t, key]) => <div key={t} style={up(p(key))}><Chip on={hot === name} size={26}>{t}</Chip></div>)}
    </Abs>
    <Abs x={0} y={400} w={1920} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 56, opacity: down}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 40, ...up(p('l3.a'))}}>
        <Tex id="mha" size={60} parts={{two: f >= at('l3.b') - 50 ? lit(1) : {}, g: f >= at('l3.a') + 20 && f < at('l3.b') - 50 ? lit(1) : {}}} />
        <Txt size={24} color={C.ink2}>2：键和值各一份</Txt></div>
      <div style={{display: 'flex', alignItems: 'center', gap: 40, ...up(p('l3.b'))}}>
        <Tex id="mla" size={60} parts={{d: f >= at('l3.b') + 20 ? lit(1) : {}}} />
        <Txt size={24} color={C.ink2}>每层一个潜向量 + 一个 RoPE 键，所有头共用</Txt></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 04 五个数
export const B04: React.FC = () => {
  const {f, at, end, p} = useBoard();
  const box = useTexBox();
  const BY = 290, ROW = 100, FS = 32;
  const rows = [
    {id: 'n1', label: '67B', v: 389120, text: '389,120', show: at('l1.a'), done: at('l1.b') - 24, paper: true},
    {id: 'n2', label: 'V2', v: 69120, text: '69,120', show: at('l1.b'), done: end('l1') - 24, paper: false},
    {id: 'n3', label: 'V3.2', v: 48068, text: '48,068', show: at('l2.a'), done: at('l2.b') - 24, paper: true},
    {id: 'n4', label: 'V4-Flash', v: 3514, text: '3,514', show: at('l2.b'), done: at('l2.c') + 6, paper: true},
    {id: 'n5', label: 'V4.1-Flash', v: 890, text: '890', show: at('l3.a'), done: end('l3') - 24, paper: true},
  ];
  const paper = ease(f, at('l4.a'), 14), fold = ease(f, at('l4'), 16);
  const sx = (v: number) => 1190 + (560 * (Math.log10(v) - Math.log10(500))) / 3;
  return <AbsoluteFill>
    <Abs x={100} y={160} style={up(p('l1'))}><Txt size={28} color={C.ink2}>代进五代模型（每 token 全局 KV，字节）</Txt></Abs>
    {rows.map((r, i) => {
      const h = box(r.id, FS).h;
      const glowK = 1 - 0.7 * ease(f, r.done + 50, 20);
      return <Abs key={r.id} x={100} y={BY + i * ROW + 6 + ROW * 0.25 - h / 2} style={up(ease(f, r.show, 14))}>
        <Tex id={r.id} size={FS} parts={{r: f >= r.done ? lit(glowK) : dim(0.35)}} /></Abs>;
    })}
    <Svg>
      <LogBars x={1000} y={BY} w={560} row={ROW} lo={500} hi={500000} ticks={[1000, 10000, 100000]} unit="字节 / token · 对数刻度"
        rows={rows.map((r) => ({label: r.label, value: r.v, text: r.text, p: lin(f, r.done, 22),
          lit: f >= r.done && f < r.done + 70 ? 1 : 0, mark: r.paper ? paper : 0}))} />
      <Span x0={sx(890)} x1={sx(389120)} y={BY + 5 * ROW + 6} p={lin(f, at('l4') + 6, 24)} label="÷437" size={30} accent down />
    </Svg>
    <Abs x={100} y={800} style={up(fold)}><Tex id="fold" size={36} color={ACC} /></Abs>
    <Abs x={1100} y={180} w={700} style={{textAlign: 'right', ...up(paper)}}>
      <Txt size={22} color={C.ink2}>○ 这四根柱子，V4.1 论文图 1(b) 里印着同样的数</Txt></Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 每一步动了哪个因子
const MOVES: [string, string[], string, string, string, string][] = [
  ['MHA → 67B 的 GQA-8', ['d'], '64 个 KV 头 → 8 个', '3,112,960 → 389,120', '÷8', 'l1.a'],
  ['67B → V2（MLA）', ['d'], '每层 2,048 个数 → 576；层数 95 → 60', '389,120 → 69,120', '÷5.63', 'l1.b'],
  ['V3 → V3.2', ['b'], 'BF16 → FP8，另加索引键', '70,272 → 48,068', '÷1.46', 'l2.a'],
  ['V3.2 → V4-Flash', ['m'], '每 4 或 128 个 token 一条', '48,068 → 3,514', '÷13.7', 'l2.b'],
  ['V4-Flash → V4.1-Flash', ['c', 'b'], '41 份 → 4 份；FP8 → FP4', '3,514 → 890', '÷3.95', 'l2.c'],
];
export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const cur = [...MOVES].reverse().find((m) => f >= at(m[5]));
  const parts: Record<string, any> = {};
  if (cur) cur[1].forEach((k) => { parts[k] = lit(1); });
  return <AbsoluteFill>
    <Abs x={0} y={140} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}><Tex id="bkv" size={64} parts={parts} /></Abs>
    <Abs x={120} y={330} w={1680} style={{display: 'flex', flexDirection: 'column', gap: 10}}>
      {MOVES.map(([step, , what, bytes, ratio, key]) => {
        const on = !!cur && cur[5] === key;
        return <div key={step} style={{display: 'flex', alignItems: 'center', gap: 24, height: 86, padding: '0 24px', borderRadius: 10,
          border: `1px solid ${on ? ACC : C.line}`, background: on ? rgba(ACC, 0.07) : C.panel, boxShadow: on ? BOX_GLOW : undefined,
          ...up(ease(f, at(key), 12))}}>
          <Txt size={26} style={{width: 360}}>{step}</Txt>
          <Txt size={24} color={C.ink2} style={{width: 560}}>{what}</Txt>
          <Mono size={24} color={C.ink2} style={{width: 380}}>{bytes}</Mono>
          <Mono size={34} color={on ? ACC : C.ink} glow={on}>{ratio}</Mono>
        </div>;
      })}
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 06 省下的字节谁来付
const COSTS: [string, string, string, string[], string][] = [
  ['gqa', 'GQA / MQA', '付质量', ['MQA：BLEU 26.7 → 26.5', 'GQA-8 几乎追平 MHA'], 'l1.a'],
  ['mla', 'MLA', '付计算', ['每个缓存 token', '多做约 3.4 倍运算'], 'l1.b'],
  ['dsa', 'DSA', '付存储', ['缓存反而多约 20%', '省的是读取和打分'], 'l2.a'],
  ['csa', 'CSA / HCA', '付细节', ['1M 装得下', '不等于用得好'], 'l2.b'],
  ['csa2', 'CSA2', '付近似', ['借用别人的 top-k', '可能选错'], 'l2.c'],
];
export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const cam = project({x: 1500, y: 592, w: 2250});
  const CX = (i: number) => 90 + i * 352, CY = 520, CW = 330;
  const cur = [...COSTS].reverse().find((c) => f >= at(c[4]));
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{nodeIn: (n) => (n.lane === 'A' ? 1 : 0), laneIn: (k) => (k === 'A' ? 1 : 0), axisIn: 0,
      laneLit: (k) => (k === 'A' ? 0.5 : 0), nodeLit: (n) => {
        const c = COSTS.find(([id]) => id === n.id);
        return c ? ease(f, at(c[4]), 10) : 0;
      }}}>
      {COSTS.map(([id, , , , key], i) => {
        const n = nodeOf(rm, id);
        const x0 = cam.X(n.x + n.w / 2), y0 = cam.Y(n.y + n.h) + 4, x1 = CX(i) + CW / 2, y1 = CY;
        const k = lin(f, at(key), 14);
        if (k <= 0) return null;
        return <path key={id} d={`M${x0} ${y0} C${x0} ${(y0 + y1) / 2} ${x1} ${(y0 + y1) / 2} ${x1} ${y1}`} fill="none" stroke={ACC}
          strokeWidth={1.4} pathLength={1} strokeDasharray={`${k} 1`} opacity={0.8} />;
      })}
    </TreeScene>
    <Abs x={120} y={140} style={up(p('l1'))}><Chip on size={28}>主线 A 上没有免费的一步</Chip></Abs>
    {COSTS.map(([id, name, pay, sub, key], i) => {
      const on = !!cur && cur[0] === id;
      return <Abs key={id} x={CX(i)} y={CY} w={CW} style={up(ease(f, at(key) + 8, 12))}>
        <div style={{padding: '18px 22px', borderRadius: 12, height: 230, boxSizing: 'border-box', background: rgba(C.bg, 0.94),
          border: `1px solid ${on ? ACC : C.line}`, boxShadow: on ? BOX_GLOW : undefined}}>
          <Mono size={22} color={C.ink2}>{name}</Mono>
          <Txt size={46} color={ACC} glow={on} weight={500} style={{margin: '6px 0 12px'}}>{pay}</Txt>
          {sub.map((s) => <Txt key={s} size={21} color={C.ink2}>{s}</Txt>)}
        </div>
      </Abs>;
    })}
  </AbsoluteFill>;
};

// ================================================================ 07 看清口径
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const bar = (v: number) => (v / 389120) * 560;
  const BARS: [string, number, string, string][] = [['67B（BF16）', 389120, '389,120', 'l1'], ['V2（部署，平均 6 bit）', 25920, '25,920', 'l1.a'],
    ['V2（BF16）', 69120, '69,120', 'l1.b']];
  return <AbsoluteFill>
    <Abs x={120} y={170} w={840} style={up(p('l1'))}>
      <Card on={f < at('l2')}>
        <Txt size={30} weight={400}>“KV cache 降了 93.3%”</Txt>
        <Txt size={22} color={C.ink2} style={{marginTop: 8}}>V2 论文：比的是部署态</Txt>
        <div style={{marginTop: 22, display: 'flex', flexDirection: 'column', gap: 14}}>
          {BARS.map(([t, v, s, key]) => <div key={t} style={{display: 'flex', alignItems: 'center', gap: 14, ...up(p(key))}}>
            <Txt size={20} color={C.ink2} style={{width: 210}}>{t}</Txt>
            <div style={{width: bar(v) * lin(f, at(key) + 4, 16), height: 22, borderRadius: 3, background: rgba(ACC, 0.35),
              border: `1px solid ${rgba(ACC, 0.7)}`}} />
            <Mono size={20} color={C.ink}>{s}</Mono></div>)}
        </div>
        <div style={{display: 'flex', flexDirection: 'column', gap: 16, marginTop: 26}}>
          <div style={up(p('l1.a'))}><Tex id="p933" size={30} color={C.ink2} /></div>
          <div style={up(p('l1.b'))}><Tex id="p822" size={30} color={ACC} /></div>
        </div>
      </Card>
    </Abs>
    <Abs x={1010} y={170} w={790} style={up(p('l2'))}>
      <Card on={f >= at('l2')}>
        <Txt size={30} weight={400}>“1M 上下文下，只要 V3.2 的 27% FLOPs、10% KV”</Txt>
        <Txt size={22} color={C.ink2} style={{marginTop: 8}}>V4 论文：成立的条件</Txt>
        <div style={{marginTop: 20, display: 'flex', flexDirection: 'column', gap: 14}}>
          {['只在 1M 这一个长度上比', '基线是外推到 1M 的 V3.2（官方只支持 128K）', '按“等效 FP8 FLOPs”计'].map((t, i) =>
            <div key={t} style={{display: 'flex', gap: 12, alignItems: 'center', ...up(ease(f, at('l2.b') + i * 14, 12))}}>
              <Mono size={22} color={ACC}>›</Mono><Txt size={24}>{t}</Txt></div>)}
        </div>
      </Card>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 08 三个问题的答案
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const out = ease(f, at('l3.a') - 4, 16);
  const B = laneRect(rm, 'B');
  const v = camAt(f, FULLV, [{at: at('l3.a'), dur: 60, to: fit({x: -330, y: B.y - 200, w: 2880, h: B.h + 400}, 0.03)}]);
  const items: [number, string, string, string, number][] = [
    [1, '每 token 的 KV = 每一步要读的字节', '67B：一个 1M token 请求 408 GB，一张 80 GB 卡只放得下 0.2 个', 'l1', at('l1.b')],
    [2, '四个因子相乘', '份数 × 1/m × 每条维数 × 每个数的字节；389,120 → 890，437 倍', 'l2', at('l2.b')],
    [3, '每一步都有代价', '质量 · 计算 · 存储 · 细节 · 近似', 'l3', at('l3') + 30],
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
      <TreeScene cam={project(v)} look={{laneLit: (k) => (k === 'B' ? ease(f, at('l3.a') + 30, 12) : 0)}} />
      <Abs x={0} y={800} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 主线 B：参数可以很多，每个 token 只用一小部分</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};
