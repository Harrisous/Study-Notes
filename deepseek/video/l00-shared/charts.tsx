// Charts for lesson 0: data-shaped pictures drawn from the lesson's own numbers.
import React from 'react';
import {ACC, C, F, rgba} from '../../../.cursor/skills/lesson-video/js/kit';

/** Horizontal bars on a log scale. Each row: label, value, text, progress (0..1, grows the bar), lit. */
export const LogBars: React.FC<{x: number; y: number; w: number; row: number; lo: number; hi: number;
  rows: {label: string; value: number; text: string; p: number; lit?: number; mark?: number}[]; ticks?: number[]; unit?: string}> =
  ({x, y, w, row, lo, hi, rows, ticks = [], unit}) => {
    const sx = (v: number) => (w * (Math.log10(v) - Math.log10(lo))) / (Math.log10(hi) - Math.log10(lo));
    const lw = 190;
    return <g>
      {ticks.map((t) => <g key={t}>
        <line x1={x + lw + sx(t)} x2={x + lw + sx(t)} y1={y - 16} y2={y + rows.length * row} stroke={C.rule} />
        <text x={x + lw + sx(t)} y={y - 24} textAnchor="middle" fontFamily={F.mono} fontSize={15} fill={C.ink3}>
          {t >= 1e6 ? `${t / 1e6}M` : t >= 1e3 ? `${t / 1e3}K` : t}</text>
      </g>)}
      {unit && <text x={x + lw + w} y={y - 48} textAnchor="end" fontFamily={F.zh} fontSize={17} fill={C.ink3}>{unit}</text>}
      {rows.map((r, i) => {
        if (r.p <= 0) return null;
        const yy = y + i * row, len = sx(r.value) * Math.min(1, r.p);
        const on = (r.lit ?? 0) > 0.5;
        return <g key={r.label} opacity={Math.min(1, r.p * 2)}>
          <text x={x + lw - 16} y={yy + row * 0.42} textAnchor="end" fontFamily={F.zh} fontSize={22} fill={on ? C.ink : C.ink2}>{r.label}</text>
          <rect x={x + lw} y={yy + 6} width={Math.max(2, len)} height={row * 0.5} rx={3} fill={on ? ACC : rgba(ACC, 0.32)}
            filter={on ? 'url(#vglow)' : undefined} />
          <text x={x + lw + len + 12} y={yy + row * 0.42} fontFamily={F.mono} fontSize={22} fill={on ? ACC : C.ink}
            opacity={Math.min(1, Math.max(0, r.p - 0.6) * 2.5)}>{r.text}</text>
          {(r.mark ?? 0) > 0 && <circle cx={x + lw + len} cy={yy + 6 + row * 0.25} r={8} fill="none" stroke={C.ink} strokeWidth={1.5}
            opacity={r.mark} />}
        </g>;
      })}
    </g>;
  };

/** A labelled bracket from (x0,y) to (x1,y), drawn on with p. */
export const Span: React.FC<{x0: number; x1: number; y: number; p: number; label: string; size?: number; accent?: boolean; down?: boolean}> =
  ({x0, x1, y, p, label, size = 24, accent, down}) => {
    if (p <= 0) return null;
    const t = down ? -10 : 10;
    return <g opacity={Math.min(1, p * 1.4)}>
      <path d={`M${x0} ${y + t} L${x0} ${y} L${x0 + (x1 - x0) * p} ${y}${p > 0.98 ? ` L${x1} ${y + t}` : ''}`} fill="none"
        stroke={accent ? ACC : C.ink2} strokeWidth={accent ? 2.2 : 1.5} filter={accent ? 'url(#vglow)' : undefined} />
      <text x={(x0 + x1) / 2} y={down ? y + size + 8 : y - 12} textAnchor="middle" fontFamily={F.mono} fontSize={size}
        fill={accent ? ACC : C.ink} opacity={p}>{label}</text>
    </g>;
  };
