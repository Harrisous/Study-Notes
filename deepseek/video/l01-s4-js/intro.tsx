// 开场：从整门课的科技树一路放大到这一页的三个问题（一个镜头，不切）。
// 分镜见 storyboard.md。树的数据来自第 0 课的图（l00_tech_tree.py），根部的 GPT 结构图与“后来被什么换掉”
// 来自第 1 课“这一课在科技树上的位置”那张表。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, Breadcrumb, C, ChapterMap, DotGrid, F, Minimap, Mono, NodeTag, PageTabs, Question, RNode, Root, RootBox, RootLink, Roadmap,
  Stage, SweepCursor, Tex, Tree, View, anchorOf, camAt, dateAt, ease, flash, lin, lit, project, rgba, swing, up, useBoard, useEp,
  useTexBox, dim, TEXT_GLOW, BOX_GLOW,
} from '../../../.cursor/skills/lesson-video/js/kit';

// ---------------------------------------------------------------- the root: lesson 1's GPT, top to bottom (world units)
export const ROOT_BOXES: RootBox[] = [
  {id: 'emb', label: '词嵌入 + 位置编码', x: -1000, y: 240, w: 380, h: 70},
  {id: 'ln1', label: 'LayerNorm', x: -930, y: 354, w: 310, h: 34, thin: true},
  {id: 'attn', label: '掩码多头注意力', x: -930, y: 398, w: 310, h: 112},
  {id: 'ln2', label: 'LayerNorm', x: -930, y: 552, w: 310, h: 34, thin: true},
  {id: 'ffn', label: '前馈网络 FFN', x: -930, y: 596, w: 310, h: 112},
  {id: 'out', label: 'LayerNorm · 反嵌入 · softmax', x: -1000, y: 846, w: 380, h: 70},
];
export const ROOT_FRAME = {x: -980, y: 334, w: 340, h: 404, label: '× 12 层'};
export const ROOT_RES = {x: -962, y0: 310, y1: 846, joins: [454, 652]};
// what lesson 1 says later replaces each box (lesson-01 §0 table), and the node that stands for it on the tree
const LINKS: (RootLink & {tags: [string, string][]})[] = [
  {from: 'emb', lane: 'A', tags: [['rope', 'RoPE · 第 5 课']]},
  {from: 'attn', lane: 'A', tags: [['mla', 'MLA · 第 9 课'], ['dsa', 'DSA · 第 14 课']]},
  {from: 'ffn', lane: 'B', tags: [['dsmoe', 'DeepSeekMoE · 第 7 课']]},
  {from: 'out', lane: 'D', tags: [['mtp', 'MTP · 第 11 课']]},
  {from: 'ffn', lane: 'E', fromY: 690, tags: [['hc', 'HC → mHC · 第 16 课']]},
];
const ATTN = ROOT_BOXES[2];

// ---------------------------------------------------------------- camera stops
const FULL0: View = {x: 760, y: 455, w: 3900};
const FULL: View = {x: 760, y: 455, w: 3660};
const ROOTV: View = {x: 470, y: 470, w: 3260};
const BOX: View = {x: ATTN.x + ATTN.w / 2, y: ATTN.y + ATTN.h / 2 + 14, w: 560};
const EQ: View = {x: ATTN.x + ATTN.w / 2, y: ATTN.y + ATTN.h / 2 + 22, w: 462};
const MAP: View = {x: ATTN.x + ATTN.w / 2, y: ATTN.y + ATTN.h / 2 + 118, w: 720};
export const WORLD = {x0: -1040, y0: 0, x1: 2520, y1: 928};

/** The page's formula sits inside the attention box: world width EQW, centred. */
const EQW = 236;

export const Intro: React.FC = () => {
  const {f, at, end} = useBoard();
  const {tl, roadmap} = useEp();
  const rm = roadmap as Roadmap;
  const box = useTexBox();

  // ------------------------------------------------ camera
  const v = camAt(f, FULL0, [
    {at: 0, dur: Math.max(30, at('i3') - 10), to: FULL},
    {at: at('i3') - 6, dur: 40, to: ROOTV},
    {at: at('i4') - 4, dur: Math.max(40, at('i4.a') - at('i4') + 14), to: BOX},
    {at: at('i4.f'), dur: 36, to: EQ},
    {at: at('i8') - 6, dur: 34, to: MAP},
  ]);
  const cam = project(v);

  // ------------------------------------------------ tree reveal (i1), pulses (i2), lesson-1 nodes (i3)
  const sweep0 = at('i1.a'), sweep1 = at('i1.b') + 10;
  const xmax = rm.axis.x1 + 160;
  const sweepX = -20 + (xmax + 40) * swing(f, sweep0, sweep1 - sweep0);
  const nodeIn = (n: RNode) => ease(f, sweep0 + ((n.x + 20) / (xmax + 40)) * (sweep1 - sweep0), 10);
  const p0 = at('i2'), p1 = end('i2');
  const pulseX = (k: number) => -60 + (xmax + 400) * lin(f, p0 + k * 5, p1 - p0 - 10);
  const lanes = ['A', 'B', 'C', 'D', 'E', 'F'];
  const pulses = f >= p0 && f < p1 + 20 ? lanes.map((l, k) => ({lane: l, x: pulseX(k), k: 1 - lin(f, p1 - 4, 20)})) : [];
  const tagged = Object.fromEntries(LINKS.flatMap((lk, i) => lk.tags.map(([id]) => [id, i])));
  const linkAt = (i: number) => at('i3.b') + 6 + i * 9;
  const nodeLit = (n: RNode) => {
    let g = 0;
    if (pulses.length) {
      const k = lanes.indexOf(n.lane);
      if (k >= 0) {
        const d = pulseX(k) - (n.x + n.w / 2);
        g = Math.max(g, d > 0 ? Math.exp(-d / 220) * 0.9 : 0);
      }
    }
    if (n.lessons.includes(rm.lesson)) g = Math.max(g, 0.85 * ease(f, at('i3'), 12) * (1 - ease(f, at('i3.b'), 20)));
    if (n.id in tagged) g = Math.max(g, ease(f, linkAt(tagged[n.id]) + 16, 10));
    return g;
  };
  const treeAlpha = 1 - 0.75 * ease(f, at('i4') - 4, 30) - 0.25 * ease(f, at('i4.a'), 20);
  const laneIn = (key: string) => ease(f, 2 + Math.max(0, 'MABCDEF'.indexOf(key)) * 3, 18);
  const laneLit = (key: string) => {
    let g = key === 'M' ? 0 : 0.9 * flash(f, at('i1.b') + 'ABCDEF'.indexOf(key) * 5, 5, 54);
    LINKS.forEach((lk, i) => { if (lk.lane === key) g = Math.max(g, ease(f, linkAt(i) + 12, 8)); });
    return g * (1 - ease(f, at('i4') + 10, 20));
  };

  // ------------------------------------------------ root (i3)
  const rootIn = (id: string) => {
    const k = ROOT_BOXES.findIndex((b) => b.id === id);
    return ease(f, at('i3.a') - 8 + Math.max(0, k) * 4, 14) * 0.95 + 0.05 * ease(f, 8, 20);
  };
  const boxLit = (id: string) => {
    if (id === 'attn') return ease(f, at('i4') - 6, 14);
    const k = LINKS.findIndex((lk) => lk.from === id || (id === 'residual' && lk.lane === 'E'));
    if (k < 0) return 0;
    return ease(f, linkAt(k), 8) * (1 - ease(f, at('i4') - 6, 14));
  };

  // ------------------------------------------------ the formula inside the attention box (i4 → i8)
  const size = (EQW * cam.s * 10) / 101.2;               // Tex size so that the formula spans EQW world units
  const fb = box('attn', size);
  const fx = cam.X(ATTN.x + ATTN.w / 2) - fb.w / 2, fy = cam.Y(ATTN.y + ATTN.h / 2) - fb.h / 2 + 4 * cam.s;
  const eqIn = f >= at('i4.a') ? 1 : Math.min(1, Math.max(0, (cam.s - 2.2) / 1.4));
  const verbs: [string, string, string][] = [['qk', 'i4.b', '比一比'], ['sq', 'i4.c', '调一调'], ['sm', 'i4.d', '分一分'], ['v', 'i4.e', '拿回来']];
  const part = (name: string) => box('attn', size, name);
  const parts: Record<string, any> = {};
  const qAt = {sq: at('i5'), m: at('i6')};
  for (const [name, key] of verbs) {
    const t = at(key);
    const on = ease(f, t, 8) * (1 - ease(f, t + 26, 18));
    parts[name] = f < t ? dim(0.28) : on > 0.02 ? lit(on) : {};
  }
  parts.m = f < at('i4.f') ? dim(0.28) : {};
  if (f >= qAt.sq) parts.sq = f < at('i6') ? lit(1) : {color: ACC};
  if (f >= qAt.m) parts.m = f < at('i7') ? lit(1) : {color: ACC};
  if (f >= at('i8')) { parts.sq = {color: ACC}; parts.m = {color: ACC}; }

  // ------------------------------------------------ chapter map (i8)
  const qs = tl.questions ?? [];
  const mapIn = ease(f, at('i8') + 6, 18);

  return <AbsoluteFill style={{background: C.bg, overflow: 'hidden'}}>
    <DotGrid v={v} />
    <Stage>
      <Tree rm={rm} cam={cam} nodeIn={nodeIn} nodeLit={nodeLit} laneIn={laneIn} laneLit={laneLit} axisIn={ease(f, 2, 26)}
        alpha={treeAlpha} here={rm.lesson} pulses={pulses} />
      <Root rm={rm} cam={cam} boxes={ROOT_BOXES} frame={ROOT_FRAME} residual={ROOT_RES} boxIn={rootIn} boxLit={boxLit}
        links={LINKS} linkP={(i) => lin(f, linkAt(i), 16) * (1 - ease(f, at('i4') - 4, 16))}
        alpha={1 - 0.75 * ease(f, at('i8'), 20)} open={{id: 'attn', k: eqIn}}
        title={{text: `第 ${rm.lesson} 课 · GPT 的一次前向传播`, p: ease(f, at('i3.a'), 14) * (1 - ease(f, at('i4'), 14))}} />
      {LINKS.flatMap((lk, i) => lk.tags.map(([id, text], j) =>
        <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={!(id === 'rope')}
          p={ease(f, linkAt(i) + 16 + j * 5, 10) * (1 - ease(f, at('i4') - 8, 12))} />))}
      {f >= sweep0 && f < sweep1 + 16 && <SweepCursor x={cam.X(sweepX)} y0={cam.Y(rm.lanes[0].y) - 30} y1={cam.Y(rm.height) + 8}
        p={1 - lin(f, sweep1, 16)} date={dateAt(rm, sweepX)} />}
    </Stage>

    {/* the attention box opens into the page's formula */}
    {eqIn > 0 && <>
      <PageTabs x={cam.X(ATTN.x)} y={cam.Y(ATTN.y) - 18 - Math.min(26, 9 * cam.s) * 1.6} s={cam.s}
        p={eqIn * ease(f, at('i4'), 20) * (1 - ease(f, at('i8') - 8, 10))}
        here={rm.page} pages={rm.pages} />
      <HeadStack x={fx} y={fy} w={fb.w} h={fb.h} n={12} p={ease(f, at('i7'), 30)} dim={ease(f, at('i8'), 20)} />
      <Abs x={fx} y={fy} style={{opacity: eqIn}}>
        <Tex id="attn" size={size} parts={parts} />
      </Abs>
      {verbs.map(([name, key, word], i) => {
        const b = part(name);
        const p = ease(f, at(key), 10) * (1 - ease(f, at('i4.f') + 8, 14));
        return <Abs key={name} x={fx + b.x + b.w / 2 - 100} y={fy + fb.h + 18} w={200} style={{textAlign: 'center', ...up(p)}}>
          <div style={{fontSize: 30, color: ACC, textShadow: TEXT_GLOW}}>{word}</div>
          <Mono size={16} color={C.ink3}>STEP {i + 1}</Mono>
        </Abs>;
      })}
      <Question n={1} q={qs[0]} anchor={anchorOf(fx, fy, part('sq'), 'below')} card={{x: 150, y: 700, w: 620}} p={ease(f, at('i5'), 14)}
        settle={ease(f, at('i8'), 16)} />
      <Question n={2} q={qs[1]} anchor={anchorOf(fx, fy, part('m'), 'above')} card={{x: 1130, y: 132, w: 650}} up p={ease(f, at('i6'), 14)}
        settle={ease(f, at('i8'), 16)} icon={<MaskIcon p={ease(f, at('i6') + 6, 18)} />} />
      <Question n={3} q={qs[2]} anchor={{x: fx + fb.w + 30, y: fy + fb.h * 0.5}} card={{x: 1180, y: 690, w: 600}} p={ease(f, at('i7'), 14)}
        settle={ease(f, at('i8'), 16)} icon={<HeadsIcon p={ease(f, at('i7') + 6, 20)} alt={ease(f, at('i7.a'), 14)} />} />
      {[['sq', 'i4.f'], ['m', 'i4.f']].map(([name, key], i) => {
        const b = part(name);
        const p = ease(f, at(key) + 6 + i * 6, 10) * (1 - ease(f, at(i === 0 ? 'i5' : 'i6'), 8));
        return <Abs key={`q${name}`} x={fx + b.x + b.w - 8} y={fy + b.y - 40} w={40} h={40} style={{...up(p, 10), borderRadius: 20,
          border: `1.5px solid ${ACC}`, background: rgba(C.bg, 0.9), boxShadow: BOX_GLOW, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          <Mono size={26} color={ACC} glow>?</Mono></Abs>;
      })}
    </>}

    {/* chapter map: which chapters answer which question */}
    {mapIn > 0 && <ChapterMap p={mapIn} f={f} zeroAt={at('i8.a')} engAt={at('i8.b')} />}

    {/* HUD: breadcrumb + minimap */}
    <Breadcrumb f={f} items={[
      ['DeepSeek 科技树', 6],
      [`第 ${rm.lesson} 课 · ${rm.lessons.find((l) => l.n === rm.lesson)?.title.split('：')[0] ?? ''}`, at('i3.a')],
      [`§${rm.page} ${rm.pages[rm.page - 1].title.split('：')[0]}`, at('i4.f')],
    ]} />
    <Stage><Minimap rm={rm} v={v} f={f} bounds={WORLD} root={{x: -1000, y: 240, w: 380, h: 676}} fullW={FULL.w}
      alpha={ease(f, 10, 20) * (1 - ease(f, at('i5'), 14))} /></Stage>
  </AbsoluteFill>;
};

// ---------------------------------------------------------------- pieces of the opening
/** Q2's picture: a 4×4 score table whose upper-right half is covered. */
const MaskIcon: React.FC<{p: number}> = ({p}) =>
  <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 26px)', gap: 4, opacity: p}}>
    {Array.from({length: 16}, (_, i) => {
      const r = Math.floor(i / 4), c = i % 4;
      const masked = c > r;
      return <div key={i} style={{width: 26, height: 26, borderRadius: 3, border: `1px solid ${masked ? C.ink3 : C.line}`,
        background: masked ? `repeating-linear-gradient(45deg, ${rgba(C.ink2, 0.2)} 0px, ${rgba(C.ink2, 0.2)} 3px, transparent 3px, transparent 7px)` : C.panel,
        display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: 12, color: C.ink3}}>{masked ? '−∞' : ''}</div>;
    })}
  </div>;

/** Q3's picture: 12 narrow heads of 64 against one wide head of 768. */
const HeadsIcon: React.FC<{p: number; alt: number}> = ({p, alt}) =>
  <div style={{display: 'flex', flexDirection: 'column', gap: 10, opacity: p}}>
    <div style={{display: 'flex', alignItems: 'center', gap: 14}}>
      <div style={{display: 'flex', gap: 3}}>{Array.from({length: 12}, (_, i) =>
        <div key={i} style={{width: 26, height: 18, borderRadius: 2, background: rgba(ACC, 0.25 + 0.05 * (i % 2)), border: `1px solid ${rgba(ACC, 0.6)}`}} />)}</div>
      <Mono size={18} color={ACC}>12 × 64</Mono>
    </div>
    <div style={{display: 'flex', alignItems: 'center', gap: 14, opacity: 0.35 + 0.65 * alt}}>
      <div style={{width: 12 * 26 + 11 * 3, height: 18, borderRadius: 2, background: C.panel, border: `1px solid ${C.ink3}`}} />
      <Mono size={18} color={C.ink2}>1 × 768 ?</Mono>
    </div>
  </div>;

/** Twelve copies of the formula card stacked behind it: "there are 12 of these". */
const HeadStack: React.FC<{x: number; y: number; w: number; h: number; n: number; p: number; dim: number}> = ({x, y, w, h, n, p, dim: d}) =>
  p <= 0 ? null : <>{Array.from({length: n - 1}, (_, i) => {
    const k = n - 1 - i;
    const q = Math.min(1, Math.max(0, p * (n - 1) - (k - 1)));
    return <Abs key={i} x={x - 30 + k * 6 * q} y={y - 22 - k * 6 * q} w={w + 60} h={h + 44} style={{borderRadius: 12,
      border: `1px solid ${rgba(ACC, (0.1 + 0.25 / k) * q * (1 - 0.7 * d))}`, background: rgba(C.bg, 0.92)}} />;
  })}
    <Abs x={x - 30} y={y - 22} w={w + 60} h={h + 44} style={{borderRadius: 12, border: `1px solid ${rgba(ACC, 0.55 * p * (1 - 0.6 * d))}`,
      background: rgba(C.bg, 0.7)}} />
  </>;

/** The closing shot of the last chapter: from inside the attention box back out to the map; the next box lights up. */
export const RoadmapView: React.FC<{f: number; t0: number; next: number}> = ({f, t0, next}) => {
  const {roadmap} = useEp();
  const rm = roadmap as Roadmap;
  const OUT: View = {x: -330, y: 560, w: 2150};
  const v = camAt(f, {...BOX, w: 700}, [{at: t0, dur: 70, to: OUT}]);
  const cam = project(v);
  const ffn = ROOT_BOXES.find((b) => b.id === 'ffn')!;
  const tag = (x: number, y: number, text: string, p: number, on: boolean) => p <= 0 ? null :
    <Abs x={x} y={y - 22} style={{display: 'flex', alignItems: 'center', gap: 12, opacity: p}}>
      <div style={{width: 44, height: 1.5, background: on ? ACC : C.ink3}} />
      <div style={{padding: '8px 16px', borderRadius: 8, border: `1px solid ${on ? ACC : C.line}`, background: rgba(C.bg, 0.92),
        boxShadow: on ? BOX_GLOW : undefined, fontSize: 26, color: on ? C.ink : C.ink2, whiteSpace: 'nowrap'}}>{text}</div>
    </Abs>;
  return <AbsoluteFill style={{background: C.bg}}>
    <DotGrid v={v} />
    <Stage>
      <Tree rm={rm} cam={cam} nodeIn={() => 1} laneIn={() => 1} alpha={0.8} here={rm.lesson} />
      <Root rm={rm} cam={cam} boxes={ROOT_BOXES} frame={ROOT_FRAME} residual={ROOT_RES} boxIn={() => 1}
        boxLit={(id) => (id === 'ffn' ? next : id === 'attn' ? 0.35 : 0)} />
    </Stage>
    {tag(cam.X(ATTN.x + ATTN.w) + 8, cam.Y(ATTN.y + ATTN.h / 2), `§${rm.page} ✓ 三个问题已回答`, ease(f, t0 + 50, 14), false)}
    {tag(cam.X(ffn.x + ffn.w) + 8, cam.Y(ffn.y + ffn.h / 2), `下一页 · §${rm.page + 1} ${rm.pages[rm.page]?.title.split('：')[0] ?? ''}`, next, true)}
  </AbsoluteFill>;
};
