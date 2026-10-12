// The syllabus as a world: parts in columns, lesson cards in each, the maths refresher underneath. Overview episodes
// open here and fly into a lesson card, which holds its page tabs, and into one tab, which holds the page's own
// world (the tech tree, a lane, a table). Nesting is a change of camera: an inner world is projected through the
// rectangle it lives in, so one continuous zoom crosses all levels.
import React from 'react';
import {ACC, C, F, rgba} from './core';
import {Cam, Roadmap, View} from './roadmap';

export type Rect = {x: number; y: number; w: number; h: number};
export type SylCard = Rect & {n: number; title: string; sub: string; part: number};
export type SylLayout = {cards: SylCard[]; parts: (Rect & {name: string})[]; maths: (Rect & {tag: string; title: string})[]; bounds: Rect};

const CW = 250, CH = 92, GAP = 16, COL = 286, HEAD = 64;

export function syllabusLayout(rm: Roadmap): SylLayout {
  const byN = Object.fromEntries(rm.lessons.map((l) => [l.n, l.title]));
  const cards: SylCard[] = [];
  const parts = (rm as any).parts as {name: string; lessons: number[]}[];
  parts.forEach((p, i) => {
    p.lessons.forEach((n, k) => {
      const [title, sub] = (byN[n] ?? '').split('：');
      cards.push({n, title: title.trim(), sub: (sub ?? '').trim(), part: i, x: i * COL, y: HEAD + k * (CH + GAP), w: CW, h: CH});
    });
  });
  const rows = Math.max(...parts.map((p) => p.lessons.length));
  const mY = HEAD + rows * (CH + GAP) + 70;
  const maths = ((rm as any).maths as {tag: string; title: string}[]).map((m, i) => ({...m, x: i * 184, y: mY, w: 170, h: 64}));
  const W = Math.max(parts.length * COL - (COL - CW), maths.length * 184);
  return {cards, maths, parts: parts.map((p, i) => ({name: p.name, x: i * COL, y: 0, w: CW, h: HEAD - 12})),
    bounds: {x: 0, y: 0, w: W, h: mY + 64}};
}

/** A view that fits `r` into the frame with `pad` (fraction of the frame left around it). */
export const fit = (r: Rect, pad = 0.12): View => {
  const w = Math.max(r.w / (1 - 2 * pad), (r.h / (1 - 2 * pad)) * (1920 / 1080));
  return {x: r.x + r.w / 2, y: r.y + r.h / 2, w};
};

/** The camera of a world that lives inside rectangle `r` (in the outer world) and spans `inner` in its own units. */
export const nest = (outer: Cam, r: Rect, inner: Rect): Cam => {
  const k = Math.min(r.w / inner.w, r.h / inner.h);
  const ox = r.x + (r.w - inner.w * k) / 2, oy = r.y + (r.h - inner.h * k) / 2;
  return {v: outer.v, s: outer.s * k, X: (x) => outer.X(ox + (x - inner.x) * k), Y: (y) => outer.Y(oy + (y - inner.y) * k)};
};

/** Where a lesson's page tabs sit inside its card (a card zoomed in shows them). */
export const pageTabs = (card: Rect, n: number): Rect[] => {
  const top = card.y + card.h * 0.34, h = (card.h * 0.6) / n;
  return Array.from({length: n}, (_, i) => ({x: card.x + card.w * 0.06, y: top + i * h, w: card.w * 0.88, h: h * 0.86}));
};

/** The small window at the right end of a page tab that holds the page's own world (aspect about 3:1). */
export const tabPreview = (tab: Rect): Rect => ({x: tab.x + tab.w - tab.h * 3.1, y: tab.y + tab.h * 0.08, w: tab.h * 2.95, h: tab.h * 0.84});

const zs = (world: number, s: number, min: number, max: number) => Math.min(max, Math.max(min, world * s));

export const Syllabus: React.FC<{rm: Roadmap; cam: Cam; L: SylLayout; lit?: (n: number) => number; partLit?: (i: number) => number;
  focus?: number; alpha?: number; open?: {n: number; k: number; here: number; tabLit: number; textFade?: number};
  preview?: (r: Rect) => React.ReactNode}> =
  ({rm, cam, L, lit = () => 0, partLit = () => 0, focus, alpha = 1, open, preview}) => {
    const {X, Y, s} = cam;
    const out: React.ReactNode[] = [];
    L.parts.forEach((p, i) => {
      const g = partLit(i);
      const fs = zs(15, s, 11, 40);
      out.push(<text key={`p${i}`} x={X(p.x)} y={Y(p.y + p.h - 14)} fontFamily={F.zh} fontSize={fs} fill={g > 0 ? ACC : C.ink3}
        opacity={alpha}>{p.name.replace('（', ' （')}</text>);
      out.push(<line key={`pl${i}`} x1={X(p.x)} x2={X(p.x + p.w)} y1={Y(p.y + p.h - 2)} y2={Y(p.y + p.h - 2)}
        stroke={g > 0 ? ACC : C.line} strokeWidth={1} opacity={alpha} />);
    });
    L.cards.forEach((c) => {
      const g = lit(c.n);
      const dim = focus !== undefined && focus !== c.n ? 0.35 : 1;
      const x = X(c.x), y = Y(c.y), w = c.w * s, h = c.h * s;
      if (x > 2200 || x + w < -200 || y > 1300 || y + h < -200) return;
      const tfs = zs(17, s, 8, 1e4), sfs = zs(12, s, 6, 1e4);
      const opening = open && open.n === c.n ? open.k : 0;
      out.push(<g key={`c${c.n}`} opacity={alpha * dim} filter={g > 0.4 ? 'url(#glow)' : undefined}>
        <rect x={x} y={y} width={w} height={h} rx={8 * s} fill={g > 0 ? rgba(ACC, 0.07 + 0.08 * g) : C.panel}
          stroke={g > 0 ? rgba(ACC, 0.5 + 0.5 * g) : C.line} strokeWidth={1.2} />
        <text x={x + 14 * s} y={opening > 0 ? y + 14 * s + Math.min(26, 13 * s) : y + 26 * s} fontFamily={F.mono}
          fontSize={opening > 0 ? Math.min(26, zs(13, s, 7, 1e4)) : zs(13, s, 7, 1e4)} fill={g > 0 ? ACC : C.ink3}>
          {c.n === 20 ? '第 20 课' : `第 ${c.n} 课`}</text>
        {tfs >= 7 && <text x={x + 14 * s} y={y + (opening > 0 ? 24 : 54) * s + (opening > 0 ? 18 * s : 0)} fontFamily={F.zh} fontSize={tfs}
          fill={g > 0 ? C.ink : C.ink2} opacity={1 - opening}>{c.title}</text>}
        {sfs >= 6 && <text x={x + 14 * s} y={y + 76 * s} fontFamily={F.zh} fontSize={sfs} fill={C.ink3} opacity={1 - opening}>
          {c.sub.length > 16 ? c.sub.slice(0, 16) + '…' : c.sub}</text>}
      </g>);
      if (opening > 0) {
        // the card opens: its page tabs
        const tabs = pageTabs(c, rm.pages.length);
        out.push(<text key={`t${c.n}`} x={x + 14 * s} y={y + 14 * s + Math.min(26, 13 * s) + 16 + Math.min(36, 14 * s)} fontFamily={F.zh}
          fontSize={Math.min(36, zs(14, s, 6, 1e4))} fill={C.ink} opacity={opening}>{c.title} · {rm.pages.length} 页</text>);
        tabs.forEach((r, i) => {
          const here = i + 1 === open!.here;
          const tl = here ? open!.tabLit : 0;
          const tf = here ? 1 - (open!.textFade ?? 0) : 1;
          out.push(<g key={`tab${c.n}-${i}`} opacity={opening}>
            <rect x={X(r.x)} y={Y(r.y)} width={r.w * s} height={r.h * s} rx={2 * s} fill={here ? rgba(ACC, 0.06 + 0.1 * tl) : 'none'}
              stroke={here ? rgba(ACC, (0.4 + 0.6 * tl) * tf) : C.line} strokeWidth={here ? 1.2 : 0.8} />
            <text x={X(r.x + 3)} y={Y(r.y + r.h * 0.72)} fontFamily={F.zh} fontSize={r.h * 0.62 * s} fill={here ? C.ink : C.ink3}
              opacity={tf}>§{rm.pages[i].n} {rm.pages[i].title}</text>
          </g>);
          if (here && preview) out.push(<g key={`pv${c.n}`} opacity={opening}>{preview(tabPreview(r))}</g>);
        });
      }
    });
    L.maths.forEach((m) => {
      const x = X(m.x), y = Y(m.y);
      out.push(<g key={m.tag} opacity={alpha * (focus !== undefined ? 0.35 : 1)}>
        <rect x={x} y={y} width={m.w * s} height={m.h * s} rx={6 * s} fill="none" stroke={C.line} strokeDasharray="4 4" />
        <text x={x + 10 * s} y={y + 24 * s} fontFamily={F.mono} fontSize={zs(13, s, 7, 1e4)} fill={C.ink3}>{m.tag}</text>
        {12 * s >= 6 && <text x={x + 10 * s} y={y + 48 * s} fontFamily={F.zh} fontSize={zs(11, s, 6, 1e4)} fill={C.ink3}>
          {m.title.length > 9 ? m.title.slice(0, 9) + '…' : m.title}</text>}
      </g>);
    });
    const mathTop = L.maths[0];
    if (mathTop) out.push(<text key="mh" x={X(mathTop.x)} y={Y(mathTop.y - 16)} fontFamily={F.zh} fontSize={zs(14, s, 10, 36)} fill={C.ink3}
      opacity={alpha}>数学补课 · 11 课（按需补）</text>);
    return <>{out}</>;
  };
