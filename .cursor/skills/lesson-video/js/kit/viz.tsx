// The drawing vocabulary of the boards. One visual grammar for every lesson:
//   a vector is a strip of square cells; a score is a number in mono; a probability is a bar on a baseline;
//   what matters now is accent + glow, everything else is grey; masked ("not allowed") is a grey hatch.
// Everything here is plain SVG/HTML in screen pixels, driven by numbers the board computes per frame.
import React from 'react';
import {ACC, C, F, rgba} from './core';

export const scale = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) * (r1 - r0)) / (d1 - d0);
export const normal = (x: number, sd: number) => Math.exp(-0.5 * (x / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
export const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const t = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / t);
};
/** Exact binomial distribution of a sum of n fair ±1 coins: {k: value of the sum, p}. */
export const coinSum = (n: number) => {
  const out: {k: number; p: number}[] = [];
  let c = 1;
  for (let i = 0; i <= n; i++) {
    out.push({k: 2 * i - n, p: c / 2 ** n});
    c = (c * (n - i)) / (i + 1);
  }
  return out;
};

/** A full-frame (or boxed) SVG layer with the shared filters and patterns. */
export const Svg: React.FC<{x?: number; y?: number; w?: number; h?: number; opacity?: number; children?: React.ReactNode}> =
  ({x = 0, y = 0, w = 1920, h = 1080, opacity = 1, children}) =>
    <svg width={w} height={h} style={{position: 'absolute', left: x, top: y, opacity, overflow: 'visible'}}>
      <defs>
        <filter id="vglow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <pattern id="vhatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="5" height="12" fill={rgba(C.ink2, 0.2)} />
        </pattern>
      </defs>
      {children}
    </svg>;

/** Points as an SVG path, drawn on to fraction p (by arc length). */
export function drawOn(pts: [number, number][], p = 1): string {
  if (pts.length < 2 || p <= 0) return '';
  const seg: number[] = [];
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    seg.push(d);
    total += d;
  }
  let left = total * Math.min(1, p);
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length; i++) {
    if (left >= seg[i - 1]) {
      d += `L${pts[i][0].toFixed(1)} ${pts[i][1].toFixed(1)}`;
      left -= seg[i - 1];
    } else {
      const t = seg[i - 1] ? left / seg[i - 1] : 0;
      d += `L${(pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t).toFixed(1)} ${(pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t).toFixed(1)}`;
      break;
    }
  }
  return d;
}

/** A curve y = fn(x) sampled over [x0, x1] and mapped through sx, sy. */
export const curvePts = (fn: (x: number) => number, x0: number, x1: number, sx: (v: number) => number, sy: (v: number) => number, n = 160) =>
  Array.from({length: n + 1}, (_, i) => {
    const x = x0 + ((x1 - x0) * i) / n;
    return [sx(x), sy(fn(x))] as [number, number];
  });

/** Plot frame: axes through (x0,y0)-(x1,y1) in px with ticks; labels are [value, text] pairs. */
export const Axes: React.FC<{x0: number; y0: number; x1: number; y1: number; xt?: [number, string][]; yt?: [number, string][];
  xlabel?: string; ylabel?: string; p?: number; zeroY?: number}> = ({x0, y0, x1, y1, xt = [], yt = [], xlabel, ylabel, p = 1, zeroY}) =>
  <g opacity={p}>
    <line x1={x0} x2={x0 + (x1 - x0) * p} y1={zeroY ?? y1} y2={zeroY ?? y1} stroke={C.ink3} strokeWidth={1.2} />
    <line x1={x0} x2={x0} y1={y1} y2={y1 + (y0 - y1) * p} stroke={C.ink3} strokeWidth={1.2} />
    {xt.map(([x, t]) => <g key={`x${t}`}>
      <line x1={x} x2={x} y1={(zeroY ?? y1) - 5} y2={(zeroY ?? y1) + 5} stroke={C.ink3} />
      <text x={x} y={y1 + 28} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>{t}</text>
    </g>)}
    {yt.map(([y, t]) => <g key={`y${t}`}>
      <line x1={x0 - 5} x2={x0 + 5} y1={y} y2={y} stroke={C.ink3} />
      <text x={x0 - 12} y={y + 6} textAnchor="end" fontFamily={F.mono} fontSize={18} fill={C.ink3}>{t}</text>
    </g>)}
    {xlabel && <text x={x1} y={y1 + 56} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={C.ink3}>{xlabel}</text>}
    {ylabel && <text x={x0} y={y0 - 18} textAnchor="start" fontFamily={F.zh} fontSize={20} fill={C.ink3}>{ylabel}</text>}
  </g>;

/** A leader line from an anchor to a label point, drawn on with p. */
export const Leader: React.FC<{x0: number; y0: number; x1: number; y1: number; p: number; color?: string; dot?: boolean}> =
  ({x0, y0, x1, y1, p, color = ACC, dot = true}) => p <= 0 ? null : <g>
    <line x1={x0} y1={y0} x2={x0 + (x1 - x0) * Math.min(1, p)} y2={y0 + (y1 - y0) * Math.min(1, p)} stroke={color} strokeWidth={1.4} filter="url(#vglow)" />
    {dot && <circle cx={x0} cy={y0} r={4} fill={color} filter="url(#vglow)" />}
  </g>;

/** A square bracket under (or over) a span, with an optional label. */
export const Brace: React.FC<{x0: number; x1: number; y: number; p: number; up?: boolean; color?: string; tick?: number}> =
  ({x0, x1, y, p, up: upward, color = ACC, tick = 12}) => p <= 0 ? null : <g opacity={Math.min(1, p * 1.5)} filter="url(#vglow)">
    <path d={`M${x0} ${y + (upward ? tick : -tick)} L${x0} ${y} L${x0 + (x1 - x0) * p} ${y}${p > 0.98 ? ` L${x1} ${y + (upward ? tick : -tick)}` : ''}`}
      fill="none" stroke={color} strokeWidth={1.6} />
  </g>;

/** A check mark drawn on with p. */
export const Check: React.FC<{x: number; y: number; size?: number; p: number; color?: string}> = ({x, y, size = 40, p, color = ACC}) =>
  p <= 0 ? null : <path d={drawOn([[x, y + size * 0.5], [x + size * 0.36, y + size * 0.86], [x + size, y + size * 0.1]], p)}
    fill="none" stroke={color} strokeWidth={size / 9} strokeLinecap="round" strokeLinejoin="round" filter="url(#vglow)" />;

/** An arc from x0 to x1 above (h < 0) or below a baseline y: weight sets thickness and brightness. */
export const Arc: React.FC<{x0: number; x1: number; y: number; h: number; w: number; p?: number; color?: string; label?: string}> =
  ({x0, x1, y, h, w, p = 1, color = ACC, label}) => {
    if (p <= 0) return null;
    const pts: [number, number][] = Array.from({length: 41}, (_, i) => {
      const t = i / 40;
      return [x0 + (x1 - x0) * t, y + h * 4 * t * (1 - t)];
    });
    const mid = pts[20];
    return <g>
      <path d={drawOn(pts, p)} fill="none" stroke={rgba(color, 0.25 + 0.75 * Math.min(1, w * 1.4))} strokeWidth={1.5 + 16 * w}
        strokeLinecap="round" filter={w > 0.4 ? 'url(#vglow)' : undefined} />
      {label && p > 0.9 && <text x={mid[0]} y={mid[1] + (h < 0 ? -14 : 30)} textAnchor="middle" fontFamily={F.mono} fontSize={24}
        fill={w > 0.4 ? color : C.ink2} opacity={(p - 0.9) * 10}>{label}</text>}
    </g>;
  };

/** A row of square cells (a vector), with optional values; `on` lights cells, `hatch` greys them out. */
export const Cells: React.FC<{x: number; y: number; n: number; size?: number; gap?: number; values?: (string | null)[];
  on?: (i: number) => number; hatch?: (i: number) => boolean; p?: (i: number) => number; fs?: number; vertical?: boolean;
  fill?: (i: number) => string | undefined}> =
  ({x, y, n, size = 56, gap = 6, values, on, hatch, p, fs, vertical, fill}) =>
    <>{Array.from({length: n}, (_, i) => {
      const a = p ? p(i) : 1;
      if (a <= 0) return null;
      const g = on ? on(i) : 0;
      const cx = vertical ? x : x + i * (size + gap), cy = vertical ? y + i * (size + gap) : y;
      const h = hatch?.(i);
      return <div key={i} style={{position: 'absolute', left: cx, top: cy, width: size, height: size, borderRadius: Math.max(3, size * 0.12),
        border: `1px solid ${g > 0 ? rgba(ACC, 0.5 + 0.5 * g) : h ? C.ink3 : C.line}`, opacity: a,
        background: h ? `repeating-linear-gradient(45deg, ${rgba(C.ink2, 0.16)} 0px, ${rgba(C.ink2, 0.16)} 5px, transparent 5px, transparent 12px)`
          : fill?.(i) ?? (g > 0 ? rgba(ACC, 0.06 + 0.14 * g) : C.panel),
        boxShadow: g > 0.5 ? `0 0 14px ${rgba(ACC, 0.5 * g)}` : undefined,
        display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: fs ?? size * 0.42,
        color: g > 0.5 ? ACC : C.ink}}>{values?.[i] ?? ''}</div>;
    })}</>;

/** A HUD slider: a track with ticks and a glowing handle at `value`. */
export const Slider: React.FC<{x: number; y: number; w: number; value: number; min: number; max: number; ticks: [number, string][];
  label?: string; p?: number; log?: boolean}> = ({x, y, w, value, min, max, ticks, label, p = 1, log}) => {
  const sx = log ? (v: number) => x + (w * Math.log(v / min)) / Math.log(max / min) : scale(min, max, x, x + w);
  const hx = sx(value);
  return <g opacity={p}>
    <line x1={x} x2={x + w} y1={y} y2={y} stroke={C.line} strokeWidth={4} strokeLinecap="round" />
    <line x1={x} x2={hx} y1={y} y2={y} stroke={ACC} strokeWidth={4} strokeLinecap="round" filter="url(#vglow)" />
    {ticks.map(([v, t]) => <g key={t}>
      <line x1={sx(v)} x2={sx(v)} y1={y + 10} y2={y + 18} stroke={C.ink3} />
      <text x={sx(v)} y={y + 44} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink3}>{t}</text>
    </g>)}
    <circle cx={hx} cy={y} r={11} fill={C.bg} stroke={ACC} strokeWidth={3} filter="url(#vglow)" />
    {label && <text x={x} y={y - 26} fontFamily={F.zh} fontSize={22} fill={C.ink2}>{label}</text>}
  </g>;
};
