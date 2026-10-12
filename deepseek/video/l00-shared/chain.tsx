// The "improvement chain" board of a lane episode: the camera walks along one lane in a few hops (one per narration
// line), each stop gets a card with its date and what it did, and at the end the camera backs out to the whole lane.
// Only the current hop's cards are shown. Stops that are not nodes of the lane get a dashed box at (x, y).
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {ACC, Abs, BOX_GLOW, C, Mono, Rect, Roadmap, Txt, View, camAt, ease, fit, lin, project, rgba, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {TreeScene, laneRect, nodeOf} from './scene';

export type Stop = {id?: string; x?: number; y?: number; title: string; gist: string; key: string; above: boolean; dx?: number};

export const Chain: React.FC<{lane: string; stops: Stop[]; laneView: Rect; hops: [string, number][]; end: string; width?: number}> =
  ({lane, stops, laneView, hops, end, width = 1450}) => {
    const {f, at} = useBoard();
    const rm = useEp().roadmap as Roadmap;
    const L = laneRect(rm, lane);
    const LANE = fit(laneView, 0.03);
    const Y = L.y + L.h / 2;
    const v: View = camAt(f, LANE, [
      ...hops.map(([line, x], i) => ({at: i === 0 ? at(line) + 4 : at(line) - 6, dur: i === 0 ? 34 : 42, to: {x, y: Y, w: width}})),
      {at: at(end) + 60, dur: 50, to: LANE},
    ]);
    const cam = project(v);
    const overview = ease(f, at(end) + 70, 20);
    const gone = (key: string) => {
      const next = `l${Number(key.split('.')[0].slice(1)) + 1}`;
      let t = Infinity;
      try { t = at(next); } catch { /* the last line: the overview takes the cards away */ }
      return Number.isFinite(t) ? ease(f, t - 8, 10) : overview;
    };
    const anchor = (s: Stop) => {
      if (s.id) {
        const n = nodeOf(rm, s.id);
        return {x: cam.X(n.x + n.w / 2), top: cam.Y(n.y), bottom: cam.Y(n.y + n.h)};
      }
      return {x: cam.X(s.x!), top: cam.Y(s.y ?? L.y + 18), bottom: cam.Y((s.y ?? L.y + 18) + 34)};
    };
    return <AbsoluteFill>
      <TreeScene cam={cam} look={{nodeIn: (n) => (n.lane === lane ? 1 : 0.12), laneIn: (k) => (k === lane ? 1 : 0.25), axisIn: 0.5,
        laneLit: (k) => (k === lane ? 0.6 : 0), nodeLit: (n) => {
          const c = stops.find((s) => s.id === n.id);
          return c ? ease(f, at(c.key), 10) : 0;
        }}}>
        {stops.map((s) => {
          const a = anchor(s), dx = s.dx ?? 0;
          const y0 = s.above ? a.top - 4 : a.bottom + 4, y1 = s.above ? y0 - 60 : y0 + 60;
          const k = lin(f, at(s.key), 10) * (1 - gone(s.key));
          return k > 0 ? <g key={s.title}>
            {!s.id && <rect x={a.x - 70} y={a.top} width={140} height={a.bottom - a.top} rx={6} fill="none" stroke={ACC} strokeDasharray="4 4" />}
            <path d={`M${a.x} ${y0} L${a.x} ${(y0 + y1) / 2} L${a.x + dx} ${(y0 + y1) / 2} L${a.x + dx} ${y1}`} fill="none" stroke={ACC}
              strokeWidth={1.4} pathLength={1} strokeDasharray={`${k} 1`} />
          </g> : null;
        })}
      </TreeScene>
      {stops.map((s) => {
        const a = anchor(s), dx = s.dx ?? 0;
        const y = s.above ? a.top - 64 : a.bottom + 64;
        const k = ease(f, at(s.key) + 6, 12) * (1 - gone(s.key));
        if (k <= 0 || a.x < -300 || a.x > 2220) return null;
        return <Abs key={s.title} x={a.x + dx - 210} y={y} w={420} style={{opacity: k, transform: s.above ? 'translateY(-100%)' : undefined}}>
          <div style={{padding: '12px 18px', borderRadius: 10, background: rgba(C.bg, 0.94), border: `1px ${s.id ? 'solid' : 'dashed'} ${ACC}`,
            boxShadow: s.id ? BOX_GLOW : undefined}}>
            <Mono size={20} color={ACC}>{s.title}</Mono>
            <Txt size={21} style={{marginTop: 4, lineHeight: 1.3}}>{s.gist}</Txt>
          </div>
        </Abs>;
      })}
    </AbsoluteFill>;
  };
