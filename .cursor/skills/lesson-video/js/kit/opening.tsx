// The opening of an episode: pieces shared by every lesson's "from the course map down to this page" shot.
// An episode composes them in its intro.tsx with the Tree/Root/Minimap of roadmap.tsx and its own destination.
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {ACC, Abs, BOX_GLOW, C, F, LevelDot, Mono, Pill, Sub, TEXT_GLOW, ease, lin, rgba, up, useEp} from './core';
import {Roadmap, View} from './roadmap';

/** A world-anchored dot grid in two scales that cross-fade with the zoom: the sense of falling into the map. */
export const DotGrid: React.FC<{v: View}> = ({v}) => {
  const s = 1920 / v.w;
  const layer = (step: number, key: string) => {
    const px = step * s;
    const a = Math.min(1, Math.max(0, (px - 14) / 30)) * Math.min(1, Math.max(0, (420 - px) / 200));
    if (a <= 0) return null;
    const ox = ((960 - v.x * s) % px + px) % px, oy = ((540 - v.y * s) % px + px) % px;
    return <AbsoluteFill key={key} style={{opacity: a * 0.5, backgroundImage: `radial-gradient(circle, ${rgba(ACC, 0.35)} 1.2px, transparent 1.6px)`,
      backgroundSize: `${px}px ${px}px`, backgroundPosition: `${ox - px / 2}px ${oy - px / 2}px`}} />;
  };
  return <>{layer(100, 'g100')}{layer(10, 'g10')}{layer(1, 'g1')}</>;
};

/** The date cursor that writes the timeline left to right during the first line. */
export const SweepCursor: React.FC<{x: number; y0: number; y1: number; p: number; date: string}> = ({x, y0, y1, p, date}) =>
  <g opacity={p}>
    <line x1={x} x2={x} y1={y0} y2={y1} stroke={ACC} strokeWidth={1.5} filter="url(#glow)" />
    <rect x={x - 62} y={y0 - 40} width={124} height={30} rx={5} fill={C.bg} stroke={ACC} />
    <text x={x} y={y0 - 19} textAnchor="middle" fontFamily={F.mono} fontSize={17} fill={ACC}>{date}</text>
  </g>;


/** World x back to a year-month on the lesson-0 time scale. */
export function dateAt(rm: Roadmap, x: number): string {
  const ys = rm.years;
  for (let i = 0; i < ys.length - 1; i++) {
    if (x < ys[i + 1].x) {
      const t = Math.max(0, (x - ys[i].x) / (ys[i + 1].x - ys[i].x));
      const m = Math.min(12, Math.floor(t * 12) + 1);
      return `${ys[i].year}-${String(m).padStart(2, '0')}`;
    }
  }
  const lastY = ys[ys.length - 1];
  const m = Math.min(9, Math.max(1, Math.floor(((x - lastY.x) / (ys[1].x - ys[0].x + 1e-9)) * 12) + 1));
  return `${lastY.year}-${String(Math.min(9, m)).padStart(2, '0')}`;
}


export const PageTabs: React.FC<{x: number; y: number; s: number; p: number; here: number; pages: Roadmap['pages']}> = ({x, y, p, here, pages}) =>
  p <= 0 ? null : <Abs x={x} y={y} style={{display: 'flex', gap: 14, opacity: p}}>
    {pages.filter((pg) => pg.n === here - 1 || pg.n === here).map((pg) => {
      const on = pg.n === here;
      return <div key={pg.n} style={{display: 'flex', alignItems: 'center', gap: 12, padding: '8px 18px', borderRadius: 8,
        border: `1px solid ${on ? ACC : C.line}`, background: on ? rgba(ACC, 0.1) : C.panel, boxShadow: on ? BOX_GLOW : undefined}}>
        <Mono size={20} color={on ? ACC : C.ink3}>§{pg.n}</Mono>
        <div style={{fontSize: 22, color: on ? C.ink : C.ink3}}>{pg.title.split('：')[0]}</div>
        <Mono size={18} color={on ? ACC : C.ink3}>{on ? '● 这一页' : '✓ 上一页'}</Mono>
      </div>;
    })}
  </Abs>;


export type Anchor = {x: number; y: number};
export const anchorOf = (fx: number, fy: number, b: {x: number; y: number; w: number; h: number}, where: 'above' | 'below'): Anchor =>
  ({x: fx + b.x + b.w / 2, y: fy + (where === 'above' ? b.y - 6 : b.y + b.h + 6)});

/** A question pinned to a part of the formula: an elbow leader from the part to a card at a fixed place in the frame.
 *  It fades as the chapter map takes over (the map repeats the question as a column head). */
export const Question: React.FC<{n: number; q?: {short: string; text: string}; anchor: Anchor; card: {x: number; y: number; w: number};
  up?: boolean; p: number; settle: number; icon?: React.ReactNode}> = ({n, q, anchor, card, up: above, p, settle, icon}) => {
  if (!q || p <= 0) return null;
  const k = 1 - settle;
  const ex = Math.min(Math.max(anchor.x, card.x + 40), card.x + card.w - 40);      // where the leader meets the card
  const ey = above ? card.y + 150 : card.y;
  const midY = (anchor.y + ey) / 2;
  const pts = `${anchor.x},${anchor.y} ${anchor.x},${midY} ${ex},${midY} ${ex},${ey}`;
  return <>
    <svg width={1920} height={1080} style={{position: 'absolute', left: 0, top: 0, overflow: 'visible', opacity: p * k}}>
      <defs><filter id={`qg${n}`}><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
      <polyline points={pts} fill="none" stroke={ACC} strokeWidth={1.4} filter={`url(#qg${n})`} pathLength={1}
        strokeDasharray="1 1" strokeDashoffset={1 - Math.min(1, p * 1.3)} />
      <circle cx={anchor.x} cy={anchor.y} r={4.5} fill={ACC} filter={`url(#qg${n})`} />
    </svg>
    <Abs x={card.x} y={card.y} w={card.w} style={{...up(p * k, 12)}}>
      <div style={{display: 'flex', alignItems: 'flex-start', gap: 18, padding: '18px 24px', borderRadius: 12, background: rgba(C.bg, 0.94),
        border: `1px solid ${rgba(ACC, 0.7)}`, boxShadow: BOX_GLOW, minHeight: above ? 112 : undefined}}>
        <Mono size={30} color={ACC} glow style={{lineHeight: 1.25}}>Q{n}</Mono>
        <div style={{display: 'flex', flexDirection: 'column', gap: 14}}>
          <div style={{fontSize: 30, lineHeight: 1.35, color: C.ink}}>{q.text}</div>
          {icon}
        </div>
      </div>
    </Abs>
  </>;
};


export const ChapterMap: React.FC<{p: number; f: number; zeroAt: number; engAt: number}> = ({p, f, zeroAt, engAt}) => {
  const {tl} = useEp();
  const qs = tl.questions ?? [];
  const zero = ease(f, zeroAt, 10), eng = ease(f, engAt, 10);
  const cols = [...qs.map((q) => ({q, chs: tl.chapters.filter((c) => c.q === q.id)})), {q: null, chs: tl.chapters.filter((c) => !c.q)}];
  return <Abs x={100} y={402} w={1720} h={470} style={{display: 'flex', gap: 28, padding: '26px 24px 0', borderRadius: 14,
    background: rgba(C.bg, 0.94), border: `1px solid ${C.line}`, ...up(p, 24)}}>
    {cols.map(({q, chs}, ci) => <div key={ci} style={{flex: q ? (chs.length > 2 ? 1.25 : 1) : 0.8, display: 'flex', flexDirection: 'column', gap: 10,
      opacity: ease(f, zeroAt - 40 + ci * 5, 14)}}>
      <div style={{display: 'flex', alignItems: 'baseline', gap: 12, paddingBottom: 10, borderBottom: `1px solid ${q ? rgba(ACC, 0.5) : C.line}`}}>
        <Mono size={24} color={q ? ACC : C.ink3} glow={!!q}>{q ? `Q${q.id}` : '✓'}</Mono>
        <div style={{fontSize: 24, color: C.ink2}}>{q ? q.short : '回答'}</div>
      </div>
      {chs.map((c) => {
        const on = c.level === 'eng' ? eng : zero;
        return <div key={c.id} style={{display: 'flex', alignItems: 'center', gap: 14, height: 46}}>
          <Mono size={20} color={C.ink3}>{c.id}</Mono>
          <div style={{fontSize: 25, color: on > 0.5 ? C.ink : C.ink2, flexGrow: 1, whiteSpace: 'nowrap'}}><Sub text={c.title} /></div>
          <div style={{transform: `scale(${1 + 0.5 * on * (1 - lin(f, (c.level === 'eng' ? engAt : zeroAt) + 10, 20))})`}}><LevelDot level={c.level} /></div>
        </div>;
      })}
    </div>)}
    <Abs x={24} y={392} style={{display: 'flex', gap: 40}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 16, ...up(zero)}}><Pill level="zero" on={zero} />
        <div style={{fontSize: 24, fontWeight: 300, color: C.ink2}}>零基础也能听懂</div></div>
      {tl.chapters.some((c) => c.level === 'eng') && <div style={{display: 'flex', alignItems: 'center', gap: 16, ...up(eng)}}><Pill level="eng" on={eng} />
        <div style={{fontSize: 24, fontWeight: 300, color: C.ink2}}>要一点线性代数和概率，可跳过</div></div>}
    </Abs>
  </Abs>;
};


/** Top-left: where we are, growing one level per zoom. */
export const Breadcrumb: React.FC<{f: number; items: [string, number][]}> = ({f, items}) =>
  <Abs x={120} y={60} style={{display: 'flex', alignItems: 'baseline', gap: 14}}>
    {items.map(([text, t], i) => {
      const p = ease(f, t, 14);
      if (p <= 0) return null;
      const lastOne = i === items.length - 1 || ease(f, items[i + 1][1], 14) <= 0;
      return <React.Fragment key={i}>
        {i > 0 && <Mono size={20} color={C.ink3} style={{opacity: p}}>›</Mono>}
        <div style={{fontSize: 24, color: lastOne ? C.ink : C.ink3, opacity: p, textShadow: lastOne ? TEXT_GLOW : undefined}}>{text}</div>
      </React.Fragment>;
    })}
  </Abs>;

