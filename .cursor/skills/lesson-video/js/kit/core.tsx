// Kit core: design tokens, motion helpers, the timeline contexts and the small text/formula parts.
// The look ("tech-minimal + a little cyberpunk", approved 2026-09-29): one accent hue on a grey scale; glow and the
// RGB split are effects, never meaning. Formulas are Latin Modern, typeset by LaTeX (make_js_video.py math).
import React, {createContext, useContext} from 'react';
import {Easing, continueRender, delayRender, interpolate, staticFile, useCurrentFrame} from 'remotion';

// ---------------------------------------------------------------- tokens
export const C = {
  bg: '#06070B', panel: '#12151B', line: '#232830', rule: '#1B2027', bar: '#333A45', pend: '#2A3038',
  ink3: '#5F6874', ink2: '#9AA3AF', ink: '#E6EAF0', accent: '#00E5FF',
};
export const rgba = (hex: string, a: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};
export const ACC = C.accent;
export const GLOW = rgba(ACC, 0.6);
export const BOX_GLOW = `0 0 16px ${GLOW}, 0 0 3px ${GLOW}`;
export const TEXT_GLOW = `0 0 14px ${GLOW}`;
export const SPLIT = '-3px 0 rgba(255,43,214,0.65), 3px 0 rgba(0,229,255,0.55)';
export const F = {zh: "'Noto Sans SC', 'Microsoft YaHei', sans-serif", mono: "'Geist Mono', Consolas, monospace"};
export const LEVEL = {zero: '零基础', eng: '工程师向'} as const;

if (typeof document !== 'undefined') {
  const h = delayRender('fonts');
  const face = new FontFace('Geist Mono', `url(${staticFile('fonts/GeistMono-latin.woff2')}) format('woff2')`,
    {weight: '100 900'});
  face.load().then((f) => document.fonts.add(f)).catch((e) => console.error(e)).finally(() => continueRender(h));
}

// ---------------------------------------------------------------- motion
export const EASE_OUT = Easing.bezier(0.16, 1, 0.3, 1);
export const EASE_INOUT = Easing.bezier(0.65, 0, 0.35, 1);
export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
/** 0→1 with the kit's ease-out, starting at frame `start`. */
export const ease = (f: number, start: number, dur = 15) => interpolate(f, [start, start + dur], [0, 1], {easing: EASE_OUT, ...clamp});
/** 0→1 eased in and out (camera moves, morphs). */
export const swing = (f: number, start: number, dur: number) => interpolate(f, [start, start + dur], [0, 1], {easing: EASE_INOUT, ...clamp});
export const lin = (f: number, start: number, dur: number) => interpolate(f, [start, start + dur], [0, 1], clamp);
export const mix = (a: number, b: number, t: number) => a + (b - a) * t;
/** Enter: fade in while rising 16 px. */
export const up = (p: number, dy = 16): React.CSSProperties => ({opacity: p, transform: `translateY(${(1 - p) * dy}px)`});
/** Focus: full opacity from `from`, dimmed to 35% from `until` on. */
export const focus = (f: number, from: number, until = Infinity) =>
  ease(f, from, 12) * (1 - 0.65 * ease(f, until, 12));
/** A soft flash that peaks `rise` frames after `at` and decays over `fall` frames. */
export const flash = (f: number, at: number, rise = 4, fall = 30) =>
  f < at ? 0 : f < at + rise ? (f - at) / rise : Math.exp(-(f - at - rise) / (fall / 3));

// ---------------------------------------------------------------- timeline
export type Line = {start: number; end: number; marks: Record<string, number>};
export type Chapter = {id: string; title: string; level: 'zero' | 'eng'; tag: string; q?: number; start: number; card: number;
  board: number; dur: number; lines: Record<string, Line>};
export type Question = {id: number; short: string; text: string};
export type Timeline = {fps: number; total: number; title: string; questions?: Question[];
  intro: {start: number; dur: number; lines: Record<string, Line>};
  chapters: Chapter[]; cues: {from: number; to: number; text: string}[]; audio: {file: string; from: number}[]};
export type Formula = {svg: string; w: number; h: number; parts?: Record<string, [number, number, number, number]>};
export const EpCtx = createContext<{tl: Timeline; math: Record<string, Formula>; lesson: string; roadmap?: any}>(null as any);
export const LinesCtx = createContext<Record<string, Line>>({});
export const useEp = () => useContext(EpCtx);

/** Board hook: frame, `at('l3.a')` (frame of a line or bookmark, relative to the board) and `p(key)` (0→1 there). */
export function useBoard() {
  const f = useCurrentFrame();
  const lines = useContext(LinesCtx);
  const at = (key: string) => {
    const [id, m] = key.split('.');
    const ln = lines[id];
    if (!ln) throw new Error(`no line ${key}`);
    if (!m) return ln.start;
    if (!(m in ln.marks)) throw new Error(`no mark ${key}`);
    return ln.marks[m];
  };
  const end = (id: string) => lines[id].end;
  const p = (key: string, dur = 15) => ease(f, at(key), dur);
  /** 0→1 linearly across the span between two keys (for paced reveals). */
  const span = (a: string, b: string) => lin(f, at(a), Math.max(1, at(b) - at(a)));
  return {f, at, end, p, span};
}

// ---------------------------------------------------------------- small parts
export const Abs: React.FC<{x: number; y: number; w?: number; h?: number; style?: React.CSSProperties; children?: React.ReactNode}> =
  ({x, y, w, h, style, children}) => <div style={{position: 'absolute', left: x, top: y, width: w, height: h, ...style}}>{children}</div>;

/** d_k → d<sub>k</sub> in plain text titles. */
export const Sub: React.FC<{text: string}> = ({text}) => {
  const parts = text.split(/(\w)_(\w)/);
  const out: React.ReactNode[] = [];
  for (let i = 0; i < parts.length; i++) {
    if (i % 3 === 1) out.push(<span key={i}>{parts[i]}<sub style={{fontSize: '0.62em'}}>{parts[i + 1]}</sub></span>);
    else if (i % 3 === 0) out.push(parts[i]);
  }
  return <>{out}</>;
};

export type PartStyle = {color?: string; opacity?: number; glow?: number};
/** Style for a formula part: lit (accent + glow), dim, or plain. */
export const lit = (k = 1): PartStyle => ({color: ACC, glow: k});
export const dim = (o = 0.3): PartStyle => ({opacity: o});

/** A LaTeX formula (Latin Modern, pre-typeset). `size` is the text size in px; `reveal` wipes it in left→right;
 *  `parts` styles the \p{name}{...} pieces of the formula. */
export const Tex: React.FC<{id: string; size?: number; color?: string; reveal?: number; glow?: boolean;
  parts?: Record<string, PartStyle>; style?: React.CSSProperties}> =
  ({id, size = 48, color = C.ink, reveal = 1, glow, parts, style}) => {
    const {math} = useEp();
    const m = math[id];
    if (!m) throw new Error(`no formula ${id}`);
    const vars: Record<string, string | number> = {};
    for (const [k, v] of Object.entries(parts ?? {})) {
      if (v.color) vars[`--pc-${k}`] = v.color;
      if (v.opacity !== undefined) vars[`--po-${k}`] = v.opacity;
      if (v.glow) vars[`--pf-${k}`] = `drop-shadow(0 0 ${(6 * v.glow).toFixed(1)}px ${rgba(ACC, Math.min(0.85, 0.7 * v.glow))})`;
    }
    return <div style={{width: (m.w * size) / 10, height: (m.h * size) / 10, color, flexShrink: 0,
      clipPath: reveal < 1 ? `inset(-20% ${(1 - reveal) * 100}% -20% 0)` : undefined,
      filter: glow ? `drop-shadow(0 0 8px ${rgba(ACC, 0.45)})` : undefined, ...vars, ...style} as React.CSSProperties}
      dangerouslySetInnerHTML={{__html: m.svg}} />;
  };

/** Size of a formula at text size `size`, and the box of one of its parts, in px from the formula's top-left. */
export function useTexBox() {
  const {math} = useEp();
  return (id: string, size: number, part?: string) => {
    const m = math[id];
    const W = (m.w * size) / 10, H = (m.h * size) / 10;
    if (!part) return {x: 0, y: 0, w: W, h: H};
    const b = m.parts?.[part];
    if (!b) throw new Error(`no part ${id}.${part}`);
    return {x: b[0] * W, y: b[1] * H, w: b[2] * W, h: b[3] * H};
  };
}

export const Mono: React.FC<{size?: number; color?: string; glow?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> =
  ({size = 22, color = C.ink2, glow, style, children}) =>
    <span style={{fontFamily: F.mono, fontSize: size, color, textShadow: glow ? TEXT_GLOW : undefined, ...style}}>{children}</span>;

export const Txt: React.FC<{size?: number; color?: string; weight?: number; glow?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> =
  ({size = 36, color = C.ink, weight = 300, glow, style, children}) =>
    <div style={{fontSize: size, color, fontWeight: weight, textShadow: glow ? TEXT_GLOW : undefined, lineHeight: 1.35, ...style}}>{children}</div>;

export const Chip: React.FC<{on?: boolean; size?: number; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, size = 30, style, children}) =>
  <div style={{display: 'inline-flex', alignItems: 'center', gap: 14, padding: `${size * 0.4}px ${size * 0.8}px`, borderRadius: 10,
    border: `1px solid ${on ? ACC : C.line}`, background: on ? rgba(ACC, 0.1) : C.panel, color: on ? ACC : C.ink,
    boxShadow: on ? BOX_GLOW : undefined, fontSize: size, fontWeight: 300, whiteSpace: 'nowrap', ...style}}>{children}</div>;

export const LevelDot: React.FC<{level: 'zero' | 'eng'; k?: number}> = ({level, k = 1}) =>
  level === 'eng' ? <div style={{width: 10 * k, height: 10 * k, borderRadius: 5 * k, background: ACC, boxShadow: BOX_GLOW, flexShrink: 0}} />
    : <div style={{width: 6 * k, height: 6 * k, borderRadius: 5 * k, border: `${2 * k}px solid ${C.ink2}`, flexShrink: 0}} />;

export const Pill: React.FC<{level: 'zero' | 'eng'; size?: number; on?: number}> = ({level, size = 22, on = 1}) => {
  const eng = level === 'eng';
  return <div style={{display: 'flex', alignItems: 'center', gap: 10, height: size * 1.8, padding: `0 ${size * 0.8}px`,
    border: `1px solid ${eng ? rgba(ACC, 0.3 + 0.7 * on) : C.ink3}`, borderRadius: size, fontSize: size, color: eng ? ACC : C.ink2,
    boxShadow: eng && on > 0.5 ? BOX_GLOW : undefined, background: C.bg, whiteSpace: 'nowrap'}}>
    <LevelDot level={level} />
    {LEVEL[level]}
  </div>;
};

/** Vertical bars on a baseline. `hi` marks the one that matters (accent + glow); `p` grows them. */
export const Bars: React.FC<{values: number[]; labels?: string[]; texts?: string[]; hi?: number[]; vmax?: number;
  maxH?: number; barW?: number; gap?: number; p?: number; size?: number; color?: string}> =
  ({values, labels, texts, hi = [], vmax = 1, maxH = 300, barW = 64, gap = 34, p = 1, size = 22, color = ACC}) =>
    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12}}>
      <div style={{height: maxH + 40, display: 'flex', alignItems: 'flex-end', gap}}>
        {values.map((v, i) => {
          const on = hi.includes(i);
          return <div key={i} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, width: barW}}>
            <Mono size={size} color={on ? color : C.ink2} glow={on} style={{opacity: p}}>{texts ? texts[i] : v.toFixed(4)}</Mono>
            <div style={{width: barW, height: Math.max(2, (maxH * v * p) / vmax), background: on ? color : C.bar,
              borderRadius: '2px 2px 0 0', boxShadow: on ? `0 0 14px ${rgba(color, 0.6)}` : undefined}} />
          </div>;
        })}
      </div>
      <div style={{width: values.length * (barW + gap), height: 1, background: C.line}} />
      {labels && <div style={{display: 'flex', gap}}>{labels.map((l, i) =>
        <div key={i} style={{width: barW, textAlign: 'center', fontFamily: F.mono, fontSize: size - 2, color: C.ink3}}>{l}</div>)}</div>}
    </div>;

export const mmss = (frames: number, fps: number) => {
  const s = Math.floor(frames / fps);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
};

/** Format a number with a fixed number of decimals and a typographic minus. */
export const fmt = (v: number, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
