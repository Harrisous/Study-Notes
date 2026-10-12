// The course map: a world (the tech tree from lesson 0 plus a lesson's own "root" diagram) seen through a camera.
// Every episode opens by flying through it from the whole course down to its page, so viewers know where they are.
// Geometry comes from make_js_video.py roadmap (gen/<slug>/roadmap.json); everything is drawn in screen space each
// frame (no CSS scaling), so text stays crisp at any zoom and can switch level of detail.
import React from 'react';
import {ACC, C, EASE_INOUT, F, rgba} from './core';

// ---------------------------------------------------------------- camera
/** A view of the world: its centre and the world width that fills the 1920-px frame. */
export type View = {x: number; y: number; w: number};
const RHO = Math.SQRT2;

/** Smooth zoom-and-pan between two views (van Wijk & Nuij 2003, the path d3.interpolateZoom uses): the camera pulls
 *  back just enough to keep both ends in sight, so long flights read as one continuous move. */
export function zoomPath(a: View, b: View): (t: number) => View {
  const dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy;
  const w0 = a.w, w1 = b.w;
  if (d2 < 1e-6) {
    const S = Math.log(w1 / w0) / RHO;
    return (t) => ({x: a.x + t * dx, y: a.y + t * dy, w: w0 * Math.exp(RHO * t * S)});
  }
  const d1 = Math.sqrt(d2);
  const b0 = (w1 * w1 - w0 * w0 + RHO ** 4 * d2) / (2 * w0 * RHO ** 2 * d1);
  const b1 = (w1 * w1 - w0 * w0 - RHO ** 4 * d2) / (2 * w1 * RHO ** 2 * d1);
  const r0 = Math.log(Math.sqrt(b0 * b0 + 1) - b0);
  const r1 = Math.log(Math.sqrt(b1 * b1 + 1) - b1);
  const S = (r1 - r0) / RHO;
  return (t) => {
    const s = t * S;
    const u = (w0 / (RHO ** 2 * d1)) * (Math.cosh(r0) * Math.tanh(RHO * s + r0) - Math.sinh(r0));
    return {x: a.x + u * dx, y: a.y + u * dy, w: (w0 * Math.cosh(r0)) / Math.cosh(RHO * s + r0)};
  };
}

export type Move = {at: number; dur: number; to: View};
/** Where the camera is at frame f: holds between moves, flies (eased in and out) during them. */
export function camAt(f: number, start: View, moves: Move[]): View {
  let cur = start;
  for (const m of moves) {
    if (f < m.at) return cur;
    if (f < m.at + m.dur) return zoomPath(cur, m.to)(EASE_INOUT(Math.max(0, (f - m.at) / m.dur)));
    cur = m.to;
  }
  return cur;
}

export type Cam = {v: View; s: number; X: (x: number) => number; Y: (y: number) => number};
export const project = (v: View): Cam => {
  const s = 1920 / v.w;
  return {v, s, X: (x) => (x - v.x) * s + 960, Y: (y) => (y - v.y) * s + 540};
};

// ---------------------------------------------------------------- data
export type RNode = {id: string; lane: string; date: string; label: string; lessons: number[]; strong: boolean;
  x: number; y: number; w: number; h: number; row: number};
export type Roadmap = {lanes: {key: string; name: string; y: number; h: number}[]; nodes: RNode[]; edges: [string, string][];
  years: {year: number; x: number}[]; axis: {x0: number; break: number; gap: number; x1: number; y: number}; height: number;
  lessons: {n: number; title: string}[]; lesson: number; page: number; pages: {n: number; title: string; sub: string}[]};

const LABEL_X = -26;   // lane labels end here (they sit left of the 2017 edge)

/** One SVG for the whole frame, with the glow filter every layer uses. */
export const Stage: React.FC<{children?: React.ReactNode; opacity?: number}> = ({children, opacity = 1}) =>
  <svg width={1920} height={1080} style={{position: 'absolute', left: 0, top: 0, opacity, overflow: 'visible'}}>
    <defs>
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
      <filter id="glow2" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="9" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
      <pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="4" height="10" fill={rgba(C.ink2, 0.16)} />
      </pattern>
    </defs>
    {children}
  </svg>;

/** Text whose screen size follows the zoom but stays inside [min, max] px; fades out below `fade` px. */
const zsize = (world: number, s: number, min = 0, max = 999) => Math.min(max, Math.max(min, world * s));

// ---------------------------------------------------------------- the tree
export type TreeLook = {
  nodeIn: (n: RNode) => number;            // 0..1: has the node appeared
  nodeLit?: (n: RNode) => number;          // 0..1: accent highlight
  laneIn?: (key: string) => number;        // band and label
  laneLit?: (key: string) => number;
  axisIn?: number;
  alpha?: number;                          // whole layer
  here?: number;                           // lesson number drawn in accent (the current lesson)
  pulses?: {lane: string; x: number; k: number}[];
  labelAlpha?: number;                     // lane and year labels (billboards): fade them while the tree is a thumbnail
};

export const Tree: React.FC<{rm: Roadmap; cam: Cam} & TreeLook> = ({rm, cam, nodeIn, nodeLit, laneIn, laneLit, axisIn = 1,
  alpha = 1, here, pulses = [], labelAlpha = 1}) => {
  const {X, Y, s} = cam;
  const byId = Object.fromEntries(rm.nodes.map((n) => [n.id, n]));
  const right = rm.axis.x1 + 190;
  const lt = laneIn ?? (() => 1);
  const ll = laneLit ?? (() => 0);
  const nl = nodeLit ?? (() => 0);
  const out: React.ReactNode[] = [];
  // lane bands and labels
  rm.lanes.forEach((ln) => {
    const p = lt(ln.key) * alpha;
    if (p <= 0) return;
    const glow = ll(ln.key);
    out.push(<rect key={`band${ln.key}`} x={X(-12)} y={Y(ln.y)} width={(right + 12) * s} height={ln.h * s} rx={8 * s}
      fill={rgba(glow > 0 ? ACC : C.ink, 0.022 + 0.05 * glow)} stroke={rgba(ACC, 0.5 * glow)} strokeWidth={1} opacity={p} />);
  });
  // year grid and axis
  if (axisIn > 0) {
    const a = axisIn * alpha;
    rm.years.forEach((yr) => out.push(<line key={`yr${yr.year}`} x1={X(yr.x)} x2={X(yr.x)} y1={Y(0)} y2={Y(rm.height)}
      stroke={rgba(C.ink, yr.year >= 2024 ? 0.07 : 0.045)} strokeWidth={1} opacity={a} />));
    const ay = Y(rm.axis.y);
    out.push(<line key="ax1" x1={X(rm.axis.x0)} x2={X(rm.axis.x0) + (X(rm.axis.break) - X(rm.axis.x0)) * a} y1={ay} y2={ay} stroke={C.ink3} strokeWidth={1.2} />);
    out.push(<line key="ax2" x1={X(rm.axis.break + rm.axis.gap)} x2={X(rm.axis.break + rm.axis.gap) + (X(rm.axis.x1) - X(rm.axis.break + rm.axis.gap)) * a}
      y1={ay} y2={ay} stroke={C.ink3} strokeWidth={1.2} />);
    rm.years.forEach((yr) => {
      const big = yr.year >= 2024;
      out.push(<text key={`yl${yr.year}`} x={X(yr.x) + 5} y={ay - 7} fontFamily={F.mono} fontSize={zsize(13, s, 13, 22) * (big ? 1.08 : 1)}
        fill={big ? C.ink2 : C.ink3} opacity={labelAlpha * a * (X(yr.x) < X(rm.axis.x0) + (X(rm.axis.x1) - X(rm.axis.x0)) * a + 4 ? 1 : 0)}>{yr.year}</text>);
    });
  }
  // flagship release lines through the lanes
  rm.nodes.filter((n) => n.lane === 'M' && n.strong).forEach((n) => {
    const p = nodeIn(n) * alpha;
    if (p <= 0) return;
    out.push(<line key={`fl${n.id}`} x1={X(n.x)} x2={X(n.x)} y1={Y(n.y + n.h)} y2={Y(rm.height)} stroke={rgba(C.ink, 0.16)}
      strokeWidth={1} strokeDasharray="3 5" opacity={p} />);
  });
  // edges ("builds on")
  rm.edges.forEach(([a, b]) => {
    const na = byId[a], nb = byId[b];
    if (!na || !nb) return;
    const p = Math.min(nodeIn(na), nodeIn(nb)) * alpha;
    if (p <= 0) return;
    const sx = na.x + na.w, sy = na.y + na.h / 2, tx = nb.x, ty = nb.y + nb.h / 2;
    let d: string;
    if (tx - sx >= 14) {
      const dx = Math.min(46, (tx - sx) / 2);
      d = `M${X(sx)} ${Y(sy)} C${X(sx + dx)} ${Y(sy)} ${X(tx - dx)} ${Y(ty)} ${X(tx)} ${Y(ty)}`;
    } else {
      const down = nb.y > na.y;
      const x0 = Math.min(na.x + 14, tx - 10), y0 = down ? na.y + na.h : na.y;
      d = `M${X(x0)} ${Y(y0)} C${X(x0)} ${Y(ty)} ${X(x0)} ${Y(ty)} ${X(tx)} ${Y(ty)}`;
    }
    const glow = Math.max(nl(na), nl(nb));
    out.push(<path key={`e${a}${b}`} d={d} fill="none" stroke={glow > 0.05 ? rgba(ACC, 0.35 + 0.5 * glow) : C.bar}
      strokeWidth={Math.max(1, 1.3 * s)} opacity={p} />);
  });
  // lane pulses: a bright head running along the lane centre
  pulses.forEach((pu, i) => {
    const ln = rm.lanes.find((l) => l.key === pu.lane);
    if (!ln || pu.k <= 0) return;
    const yc = Y(ln.y + ln.h / 2);
    const x1 = X(pu.x), x0 = X(Math.max(-12, pu.x - 260));
    out.push(<line key={`pu${i}`} x1={x0} x2={x1} y1={yc} y2={yc} stroke={`url(#pg${i})`} strokeWidth={2} opacity={pu.k} />);
    out.push(<linearGradient key={`pg${i}`} id={`pg${i}`} gradientUnits="userSpaceOnUse" x1={x0} x2={x1} y1={yc} y2={yc}>
      <stop offset="0" stopColor={ACC} stopOpacity={0} /><stop offset="1" stopColor={ACC} stopOpacity={0.9} /></linearGradient>);
    out.push(<circle key={`ph${i}`} cx={x1} cy={yc} r={3.5} fill={ACC} filter="url(#glow)" opacity={pu.k} />);
  });
  // nodes
  const labelPx = 15 * s;
  rm.nodes.forEach((n) => {
    const p = nodeIn(n) * alpha;
    if (p <= 0) return;
    const g = nl(n);
    const x = X(n.x), y = Y(n.y), w = n.w * s, h = n.h * s;
    if (x > 1960 || x + w < -40 || y > 1120 || y + h < -40) return;
    const mine = here !== undefined && n.lessons.includes(here);
    out.push(<g key={`n${n.id}`} opacity={p} filter={g > 0.3 ? 'url(#glow)' : undefined}>
      <rect x={x} y={y} width={w} height={h} rx={6 * s} fill={g > 0 ? rgba(ACC, 0.08 + 0.12 * g) : C.panel}
        stroke={g > 0 ? rgba(ACC, 0.45 + 0.55 * g) : n.strong ? C.ink3 : C.line} strokeWidth={n.strong ? 1.6 : 1} />
      {n.lessons.map((les, i) => {
        const cx = x + (6 + 8 + i * 20) * s, cy = y + h / 2;
        const on = mine && les === here;
        return <g key={i}>
          <circle cx={cx} cy={cy} r={8 * s} fill={on || g > 0.5 ? ACC : les ? C.bar : 'none'} stroke={les ? 'none' : C.ink3} />
          {10 * s >= 6 && <text x={cx} y={cy + 3.6 * s} textAnchor="middle" fontFamily={F.mono} fontSize={10 * s} fontWeight={600}
            fill={on || g > 0.5 ? C.bg : C.ink2}>{les}</text>}
        </g>;
      })}
      {labelPx >= 6 && <text x={x + (8 + n.lessons.length * 20 + 6) * s} y={y + h / 2 + 5.2 * s} fontFamily={F.zh} fontSize={labelPx}
        fontWeight={n.strong ? 500 : 400} fill={g > 0.3 ? C.ink : n.strong ? C.ink2 : C.ink3}
        opacity={Math.min(1, (labelPx - 6) / 3)}>{n.label}</text>}
    </g>);
  });
  // lane labels (on a plate, so links can run in behind them)
  rm.lanes.forEach((ln) => {
    const p = lt(ln.key) * alpha * labelAlpha;
    if (p <= 0) return;
    const glow = ll(ln.key);
    const fs = zsize(16, s, 15, 30);
    const yc = Y(ln.y + ln.h / 2);
    const text = ln.key === 'M' ? ln.name : ln.name;
    const wTxt = [...text].reduce((a, c) => a + (c.charCodeAt(0) > 0x2e80 ? fs : fs * 0.56), 0);
    const x1 = X(LABEL_X);
    const wPlate = wTxt + (ln.key === 'M' ? 22 : fs * 1.6 + 26);
    out.push(<g key={`ll${ln.key}`} opacity={p}>
      <rect x={x1 - wPlate} y={yc - fs * 0.95} width={wPlate + 6} height={fs * 1.9} rx={fs * 0.5} fill={C.bg}
        stroke={glow > 0 ? rgba(ACC, 0.7 * glow) : 'none'} filter={glow > 0.5 ? 'url(#glow)' : undefined} />
      {ln.key !== 'M' && <text x={x1 - wTxt - fs * 0.9 - 8} y={yc + fs * 0.36} textAnchor="middle" fontFamily={F.mono} fontSize={fs}
        fontWeight={600} fill={glow > 0 ? ACC : C.ink2}>{ln.key}</text>}
      <text x={x1 - 6} y={yc + fs * 0.36} textAnchor="end" fontFamily={F.zh} fontSize={fs} fill={glow > 0 ? C.ink : C.ink2}>{text}</text>
    </g>);
  });
  return <>{out}</>;
};

/** A billboard tag next to a node (constant screen size): "MLA · 第 9 课". */
export const NodeTag: React.FC<{rm: Roadmap; cam: Cam; id: string; text: string; p: number; above?: boolean}> = ({rm, cam, id, text, p, above = true}) => {
  const n = rm.nodes.find((m) => m.id === id);
  if (!n || p <= 0) return null;
  const x = cam.X(n.x + n.w / 2), y = above ? cam.Y(n.y) - 10 : cam.Y(n.y + n.h) + 10;
  const fs = 17;
  const w = [...text].reduce((a, c) => a + (c.charCodeAt(0) > 0x2e80 ? fs : fs * 0.58), 0) + 24;
  const ty = above ? y - 34 : y + 4;
  return <g opacity={p}>
    <line x1={x} x2={x} y1={y} y2={above ? ty + 30 : ty} stroke={ACC} strokeWidth={1} />
    <rect x={x - w / 2} y={ty} width={w} height={30} rx={6} fill={C.bg} stroke={ACC} strokeWidth={1} filter="url(#glow)" />
    <text x={x} y={ty + 21} textAnchor="middle" fontFamily={F.zh} fontSize={fs} fill={C.ink}>{text}</text>
  </g>;
};

// ---------------------------------------------------------------- a lesson's root diagram (drawn in world units)
export type RootBox = {id: string; label: string; x: number; y: number; w: number; h: number; thin?: boolean};
export type RootLink = {from: string; lane: string; fromY?: number};

export const Root: React.FC<{rm: Roadmap; cam: Cam; boxes: RootBox[]; frame?: {x: number; y: number; w: number; h: number; label: string};
  residual?: {x: number; y0: number; y1: number; joins: number[]}; boxIn: (id: string) => number; boxLit?: (id: string) => number;
  links?: RootLink[]; linkP?: (i: number) => number; alpha?: number; title?: {text: string; p: number};
  open?: {id: string; k: number}}> =
  ({rm, cam, boxes, frame, residual, boxIn, boxLit, links = [], linkP, alpha = 1, title, open}) => {
    const {X, Y, s} = cam;
    const bl = boxLit ?? (() => 0);
    const out: React.ReactNode[] = [];
    if (frame) {
      const p = Math.min(...boxes.filter((b) => !b.thin).map((b) => boxIn(b.id))) * alpha * (1 - 0.95 * (open?.k ?? 0));
      out.push(<g key="frame" opacity={p}>
        <rect x={X(frame.x)} y={Y(frame.y)} width={frame.w * s} height={frame.h * s} rx={10 * s} fill="none" stroke={C.ink3}
          strokeWidth={1} strokeDasharray={`${6} ${5}`} />
        <text x={X(frame.x + frame.w) - 10} y={Y(frame.y) - 8} textAnchor="end" fontFamily={F.mono} fontSize={zsize(18, s, 12, 40)} fill={C.ink3}>{frame.label}</text>
      </g>);
    }
    if (residual) {
      const p = Math.min(...boxes.map((b) => boxIn(b.id))) * alpha * (1 - 0.95 * (open?.k ?? 0));
      const g = bl('residual');
      out.push(<g key="res" opacity={p} filter={g > 0.3 ? 'url(#glow)' : undefined}>
        <line x1={X(residual.x)} x2={X(residual.x)} y1={Y(residual.y0)} y2={Y(residual.y1)} stroke={g > 0 ? ACC : C.ink3} strokeWidth={Math.max(1.5, 3 * s)} />
        {residual.joins.map((jy, i) => <g key={i}>
          <circle cx={X(residual.x)} cy={Y(jy)} r={Math.max(5, 9 * s)} fill={C.bg} stroke={g > 0 ? ACC : C.ink3} strokeWidth={1.4} />
          <text x={X(residual.x)} y={Y(jy) + zsize(12, s, 7, 30) * 0.36} textAnchor="middle" fontFamily={F.mono} fontSize={zsize(12, s, 7, 30)}
            fill={g > 0 ? ACC : C.ink2}>+</text>
        </g>)}
      </g>);
    }
    const ok = open?.k ?? 0;
    boxes.forEach((b) => {
      const mine = open?.id === b.id;
      const p = boxIn(b.id) * alpha * (mine ? 1 : 1 - 0.92 * ok);
      if (p <= 0.01) return;
      const g = bl(b.id);
      const fs = zsize(b.thin ? 15 : 22, s, b.thin ? 10 : 13, b.thin ? 34 : 56);
      out.push(<g key={b.id} opacity={p} filter={g > 0.3 && !(mine && ok > 0.5) ? 'url(#glow)' : undefined}>
        <rect x={X(b.x)} y={Y(b.y)} width={b.w * s} height={b.h * s} rx={(b.thin ? 5 : 8) * s}
          fill={mine && ok > 0 ? rgba(C.bg, 0.6 + 0.35 * ok) : g > 0 ? rgba(ACC, 0.06 + 0.1 * g) : C.panel}
          stroke={g > 0 ? rgba(ACC, (0.5 + 0.5 * g) * (mine ? 1 - 0.85 * ok : 1)) : b.thin ? C.line : C.ink3} strokeWidth={b.thin ? 1 : 1.4} />
        <text x={X(b.x + b.w / 2)} y={Y(b.y + b.h / 2) + fs * 0.36} textAnchor="middle" fontFamily={F.zh} fontSize={fs}
          fontWeight={b.thin ? 300 : 400} fill={g > 0 ? C.ink : b.thin ? C.ink3 : C.ink2} opacity={mine ? Math.max(0, 1 - ok * 2.5) : 1}>{b.label}</text>
      </g>);
    });
    if (title && title.p > 0) {
      const b0 = boxes.reduce((a, b) => (b.y < a.y ? b : a), boxes[0]);
      out.push(<text key="title" x={X(b0.x + b0.w / 2)} y={Y(b0.y) - zsize(30, s, 16, 60) * 1.1} textAnchor="middle" fontFamily={F.zh}
        fontSize={zsize(26, s, 16, 60)} fill={ACC} filter="url(#glow)" opacity={title.p * alpha}>{title.text}</text>);
    }
    // links: from a box's right edge to its lane's label plate, drawn on like a signal
    links.forEach((lk, i) => {
      const p = (linkP ? linkP(i) : 0);
      if (p <= 0) return;
      const b = boxes.find((x) => x.id === lk.from);
      const ln = rm.lanes.find((l) => l.key === lk.lane);
      if (!b || !ln) return;
      const sx = b.x + b.w, sy = lk.fromY ?? b.y + b.h / 2, tx = LABEL_X - 200, ty = ln.y + ln.h / 2;
      const pts: [number, number][] = [];
      for (let k = 0; k <= 40; k++) {
        const t = k / 40, u = 1 - t, mx = (sx + tx) / 2;
        const x = u ** 3 * sx + 3 * u * u * t * mx + 3 * u * t * t * mx + t ** 3 * tx;
        const y = u ** 3 * sy + 3 * u * u * t * sy + 3 * u * t * t * ty + t ** 3 * ty;
        pts.push([X(x), Y(y)]);
      }
      let len = 0;
      for (let k = 1; k < pts.length; k++) len += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
      const drawn = Math.min(1, p);
      const head = pts[Math.round(drawn * 40)];
      out.push(<g key={`lk${i}`} opacity={alpha}>
        <polyline points={pts.map((q) => q.join(',')).join(' ')} fill="none" stroke={ACC} strokeWidth={2} strokeDasharray={`${len} ${len}`}
          strokeDashoffset={len * (1 - drawn)} filter="url(#glow)" opacity={0.85} />
        {drawn < 1 && <circle cx={head[0]} cy={head[1]} r={4} fill={ACC} filter="url(#glow2)" />}
      </g>);
    });
    return <>{out}</>;
  };

// ---------------------------------------------------------------- HUD pieces
/** The whole world in a corner, with the camera's view as a rectangle; a pulsing ring once the view is tiny. */
export const Minimap: React.FC<{rm: Roadmap; v: View; f: number; bounds: {x0: number; y0: number; x1: number; y1: number};
  root?: {x: number; y: number; w: number; h: number}; x?: number; y?: number; w?: number; alpha?: number; fullW?: number}> =
  ({rm, v, f, bounds, root, x = 1540, y = 112, w = 270, alpha = 1, fullW = 3640}) => {
    if (alpha <= 0) return null;
    const k = w / (bounds.x1 - bounds.x0);
    const h = (bounds.y1 - bounds.y0) * k;
    const mx = (wx: number) => x + (wx - bounds.x0) * k, my = (wy: number) => y + (wy - bounds.y0) * k;
    const vh = (v.w * 1080) / 1920;
    const rx0 = mx(v.x - v.w / 2), ry0 = my(v.y - vh / 2), rw = v.w * k, rh = vh * k;
    const tiny = rw < 10;
    const ring = 6 + 5 * ((f % 30) / 30);
    return <g opacity={alpha}>
      <rect x={x - 8} y={y - 8} width={w + 16} height={h + 16} rx={6} fill={rgba(C.bg, 0.85)} stroke={C.line} />
      {rm.lanes.map((ln) => <rect key={ln.key} x={mx(-12)} y={my(ln.y)} width={(rm.axis.x1 + 190) * k} height={Math.max(1, ln.h * k)}
        fill={rgba(C.ink, 0.06)} />)}
      {rm.nodes.map((n) => <rect key={n.id} x={mx(n.x)} y={my(n.y)} width={Math.max(1, n.w * k)} height={Math.max(1, n.h * k)}
        fill={n.strong ? C.ink3 : C.bar} />)}
      {root && <rect x={mx(root.x)} y={my(root.y)} width={root.w * k} height={root.h * k} fill="none" stroke={C.ink3} />}
      {!tiny && <rect x={Math.max(x - 8, rx0)} y={Math.max(y - 8, ry0)} width={Math.min(rw, w + 16)} height={Math.min(rh, h + 16)}
        fill={rgba(ACC, 0.08)} stroke={ACC} strokeWidth={1.2} />}
      {tiny && <>
        <circle cx={mx(v.x)} cy={my(v.y)} r={2.5} fill={ACC} />
        <circle cx={mx(v.x)} cy={my(v.y)} r={ring} fill="none" stroke={ACC} strokeWidth={1} opacity={1 - (f % 30) / 30} />
      </>}
      <text x={x - 8} y={y + h + 28} fontFamily={F.mono} fontSize={13} fill={C.ink3} letterSpacing="0.14em">MAP · ZOOM ×{(fullW / v.w).toFixed(1)}</text>
    </g>;
  };
