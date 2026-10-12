// Lesson 0 (the course map) shared scenes: the tech tree as a board, and the opening that flies from the syllabus
// into lesson 0, into one of its page tabs, and into that page's world. Every lesson-0 episode uses both.
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  Breadcrumb, C, Cam, ChapterMap, DotGrid, Move, Rect, Roadmap, Stage, Syllabus, SylLayout, Tree, TreeLook, View, camAt, ease, fit,
  nest, pageTabs, project, rgba, swing, syllabusLayout, tabPreview, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';

/** The tree's world, lane labels included. */
export const TREE: Rect = {x: -330, y: -34, w: 2880, h: 994};
export const FULLV: View = fit(TREE, 0.02);
export const laneRect = (rm: Roadmap, key: string): Rect => {
  const ln = rm.lanes.find((l) => l.key === key)!;
  return {x: -330, y: ln.y - 8, w: 2880, h: ln.h + 16};
};
export const nodeOf = (rm: Roadmap, id: string) => rm.nodes.find((n) => n.id === id)!;
/** The x of a date on the tree's time axis, interpolated between year ticks. Before 2023 and after 2024 the axis is
 *  linear within a year; inside 2023 it bends at October, so use the x of a node or timeline row for 2023 dates. */
export const xOfDate = (rm: Roadmap, date: string) => {
  const [y, m, d] = date.split('-').map(Number);
  const ys = rm.years;
  const i = Math.max(0, Math.min(ys.length - 1, y - ys[0].year));
  const frac = (m - 1 + (d - 1) / 30.4) / 12;
  const x1 = i + 1 < ys.length ? ys[i + 1].x : ys[i].x + (ys[i].x - ys[i - 1].x);
  return ys[i].x + (x1 - ys[i].x) * frac;
};

/** A full-frame tree with a camera and per-node / per-lane looks. */
export const TreeScene: React.FC<{cam: Cam; look?: Partial<TreeLook>; children?: React.ReactNode; alpha?: number}> =
  ({cam, look = {}, children, alpha = 1}) => {
    const {roadmap} = useEp();
    const rm = roadmap as Roadmap;
    return <Stage>
      <Tree rm={rm} cam={cam} nodeIn={look.nodeIn ?? (() => 1)} nodeLit={look.nodeLit} laneIn={look.laneIn} laneLit={look.laneLit}
        axisIn={look.axisIn ?? 1} alpha={(look.alpha ?? 1) * alpha} here={look.here} pulses={look.pulses} labelAlpha={look.labelAlpha} />
      {children}
    </Stage>;
  };

/** Outer (syllabus) view that shows the inner world's view `v` (inner world placed in `r` by `nest`). */
const toOuter = (v: View, r: Rect, inner: Rect): View => {
  const k = Math.min(r.w / inner.w, r.h / inner.h);
  const ox = r.x + (r.w - inner.w * k) / 2, oy = r.y + (r.h - inner.h * k) / 2;
  return {x: ox + (v.x - inner.x) * k, y: oy + (v.y - inner.y) * k, w: v.w * k};
};

/** The opening of a lesson-0 episode.
 *  i1: the whole syllabus (two lessons light at i1.a / i1.b) · i2: into the lesson-0 card, which opens into its pages ·
 *  i3: the page tab lights, at i3.a the camera dives into the tab's window, which holds the page's world ·
 *  then the episode's own beats (questions) over that world, and the chapter map on the last line. */
export const Opening: React.FC<{
  dest: Rect;                                            // the page world's bounds (its own units)
  destView?: (f: number) => View;                        // camera inside the page world after the dive (default: all of it)
  renderDest: (cam: Cam, f: number) => React.ReactNode;  // SVG content of the page world
  overlay?: (cam: Cam, f: number) => React.ReactNode;    // HTML on top (questions)
  lit?: [number, string][];                              // lessons to light in the syllabus: [lesson, mark]
}> = ({dest, destView, renderDest, overlay, lit = []}) => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const L: SylLayout = syllabusLayout(rm);
  const card = L.cards.find((c) => c.n === rm.lesson)!;
  const tab = pageTabs(card, rm.pages.length)[rm.page - 1];
  const win = tabPreview(tab);
  const lines = Object.keys(tl.intro.lines);
  const last = lines[lines.length - 1];
  const VFULL = fit(L.bounds, 0.05);
  const VCARD = fit(card, 0.1);
  const inner = destView ? destView(f) : fit(dest, 0.02);
  const VDEST = toOuter(inner, win, dest);
  const v = camAt(f, {...VFULL, w: VFULL.w * 1.06}, [
    {at: 0, dur: Math.max(30, at('i2') - 8), to: VFULL},
    {at: at('i2') - 4, dur: 44, to: VCARD},
    {at: at('i3.a') - 2, dur: Math.max(50, at('i4') - at('i3.a') - 4), to: VDEST},
  ] as Move[]);
  // after the dive the page world's own camera takes over (it may move with the beats)
  const dived = f >= at('i4') - 6;
  const cam = project(dived ? VDEST : v);
  const destCam = nest(cam, win, dest);
  const opening = ease(f, at('i2') + 30, 18);
  const tabLit = ease(f, at('i3'), 12);
  const textFade = Math.min(1, Math.max(0, (cam.s * tab.h - 160) / 200));
  const sylAlpha = 1 - Math.min(1, Math.max(0, (destCam.s - 0.12) / 0.2));
  const labelAlpha = Math.min(1, Math.max(0, (destCam.s - 0.3) / 0.15));
  const mapIn = ease(f, at(last) + 6, 18);
  return <AbsoluteFill style={{background: C.bg, overflow: 'hidden'}}>
    <DotGrid v={dived ? inner : v} />
    <Stage>
      {sylAlpha > 0.01 && <Syllabus rm={rm} cam={cam} L={L} alpha={sylAlpha}
        lit={(n) => Math.max(n === rm.lesson ? ease(f, at('i2'), 12) : 0,
          ...lit.map(([ln, mk]) => (ln === n ? ease(f, at(mk), 10) * (1 - ease(f, at('i2'), 14)) : 0)))}
        focus={f >= at('i2') ? rm.lesson : undefined}
        open={{n: rm.lesson, k: opening, here: rm.page, tabLit, textFade}}
        preview={() => <g opacity={Math.max(0.35, labelAlpha)}>{renderDest(destCam, f)}</g>} />}
      {sylAlpha <= 0.01 && renderDest(destCam, f)}
    </Stage>
    {overlay && labelAlpha > 0.5 && overlay(destCam, f)}
    {mapIn > 0 && <AbsoluteFill style={{background: rgba(C.bg, 0.55 * mapIn)}} />}
    {mapIn > 0 && <ChapterMap p={mapIn} f={f} zeroAt={at(`${last}.a`)} engAt={at(tl.intro.lines[last].marks.b !== undefined ? `${last}.b` : `${last}.a`)} />}
    <Breadcrumb f={f} items={[
      ['DeepSeek 科技树', 6],
      [`第 ${rm.lesson} 课 · ${rm.lessons.find((l) => l.n === rm.lesson)?.title.split('：')[0] ?? ''}`, at('i2')],
      [`§${rm.page} ${rm.pages[rm.page - 1].title.split('：')[0]}`, at('i3')],
    ]} />
    {void swing}
  </AbsoluteFill>;
};

/** Lane colours are not used (one accent); lanes are told apart by their letter. */
export const LANES = ['A', 'B', 'C', 'D', 'E', 'F'];
