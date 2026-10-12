// Lesson-video kit (Remotion). Episodes import this file, their gen/<slug>/timeline.json (voice timing),
// math.json (LaTeX formulas as SVG) and roadmap.json (the course map), and call makeEpisode.
//   core.tsx     tokens, motion, timeline contexts, Tex (formula parts), text bits
//   viz.tsx      plots, grids, arcs, strips, callouts: the drawing vocabulary of the boards
//   roadmap.tsx  the course map and its camera: every episode opens by zooming from the whole course to its page
//   opening.tsx  the opening's shared pieces: dot grid, date cursor, breadcrumb, pinned questions, chapter map
//   syllabus.tsx the syllabus as a world (overview episodes fly from it into a lesson, a page, the page's world)
import React from 'react';
import {AbsoluteFill, Composition, Html5Audio, Sequence, getInputProps, random, staticFile, useCurrentFrame} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Chapter, EpCtx, F, Formula, GLOW, LevelDot, LinesCtx, Mono, Pill, SPLIT, Sub, Timeline, ease, lin, mmss,
  rgba, up, useBoard, useEp,
} from './core';

export * from './core';
export * from './viz';
export * from './roadmap';
export * from './opening';
export * from './syllabus';

const Grid: React.FC<{alpha?: number}> = ({alpha = 0.06}) =>
  <AbsoluteFill style={{backgroundImage: `linear-gradient(${rgba(ACC, alpha)} 1px, transparent 1px), linear-gradient(90deg, ${rgba(ACC, alpha)} 1px, transparent 1px)`,
    backgroundSize: '80px 80px'}} />;
export {Grid};

// ---------------------------------------------------------------- cover (episodes without a roadmap opening)
const Cover: React.FC = () => {
  const {tl, lesson} = useEp();
  const {f, p} = useBoard();
  return <AbsoluteFill style={{background: C.bg, overflow: 'hidden'}}>
    <Grid />
    <Abs x={-600} y={830} w={3120} h={900} style={{backgroundImage: `linear-gradient(${rgba(ACC, 0.35)} 1px, transparent 1px), linear-gradient(90deg, ${rgba(ACC, 0.35)} 1px, transparent 1px)`,
      backgroundSize: '120px 120px', backgroundPosition: `0 ${(f * 1.2) % 120}px`, transform: 'perspective(700px) rotateX(64deg)', transformOrigin: 'top center',
      WebkitMaskImage: 'linear-gradient(to bottom, black, transparent 75%)', opacity: ease(f, 0, 30)}} />
    <Abs x={0} y={829} w={1920} h={2} style={{background: ACC, boxShadow: BOX_GLOW, opacity: ease(f, 0, 30)}} />
    <Abs x={120} y={110}><Mono size={22} color={C.ink3} style={{letterSpacing: '0.18em'}}>{lesson}</Mono></Abs>
    <Abs x={120} y={300} style={{display: 'flex', flexDirection: 'column', gap: 32}}>
      <div style={{fontSize: 136, fontWeight: 500, lineHeight: 1, textShadow: SPLIT, ...up(ease(f, 4, 20))}}>{tl.title.split('：')[0].split('· ').pop()}</div>
      <div style={{fontSize: 56, fontWeight: 300, color: C.ink2, ...up(ease(f, 10, 20))}}>{tl.title.split('：')[1] ?? ''}</div>
    </Abs>
    <Abs x={1180} y={290} w={620} style={{display: 'flex', flexDirection: 'column'}}>
      {tl.chapters.map((c, i) => <div key={c.id} style={{display: 'flex', alignItems: 'center', gap: 24, height: 52,
        borderBottom: `1px solid ${C.rule}`, ...up(ease(f, 14 + i * 3, 16))}}>
        <Mono size={22} color={C.ink3} style={{width: 48}}>{c.id}</Mono>
        <div style={{flexGrow: 1, fontSize: 26}}><Sub text={c.title} /></div>
        <LevelDot level={c.level} />
      </div>)}
    </Abs>
    <Abs x={1560} y={904} w={240} style={{textAlign: 'right', opacity: ease(f, 20, 20)}}>
      <Mono size={28} color={ACC} glow>≈ {mmss(tl.total, tl.fps)}</Mono></Abs>
    {void p}
  </AbsoluteFill>;
};

// ---------------------------------------------------------------- chapter card
const Card: React.FC<{ch: Chapter; n: number; total: number}> = ({ch, n, total}) => {
  const {tl} = useEp();
  const f = useCurrentFrame();
  const out = 1 - lin(f, ch.card, 8);
  const glitch = f < 3;
  const dx = glitch ? (random(`gx${ch.id}${f}`) - 0.5) * 36 : 0;
  const clip = glitch ? `inset(${random(`ga${ch.id}${f}`) * 50}% 0 ${random(`gb${ch.id}${f}`) * 35}% 0)` : undefined;
  const id = String(n).padStart(2, '0');
  const q = ch.q ? tl.questions?.find((x) => x.id === ch.q) : undefined;
  return <AbsoluteFill style={{background: C.bg, opacity: out}}>
    <Grid />
    <Abs x={1920 - 100 - 640} y={120} w={640} style={{textAlign: 'right', fontFamily: F.zh, fontSize: 440, fontWeight: 200,
      lineHeight: 1, color: 'transparent', WebkitTextStroke: `2px ${rgba(ACC, 0.5)}`, textShadow: `0 0 30px ${rgba(ACC, 0.25)}`,
      ...up(ease(f, 0, 18), 30)}}>{id}</Abs>
    <Abs x={120} y={370} style={{display: 'flex', flexDirection: 'column', gap: 30}}>
      <Mono size={22} color={ACC} glow style={{letterSpacing: '0.18em', ...up(ease(f, 2, 16))}}>CHAPTER {id} / {String(total).padStart(2, '0')}</Mono>
      <div style={{fontSize: 104, fontWeight: 500, lineHeight: 1, textShadow: SPLIT, transform: `translateX(${dx}px)`, clipPath: clip,
        opacity: ease(f, 0, 6)}}><Sub text={ch.title} /></div>
      <div style={{display: 'flex', alignItems: 'center', gap: 24, paddingTop: 10, ...up(ease(f, 8, 16))}}>
        <Pill level={ch.level} /><Mono size={24} color={C.ink3}>{mmss(ch.dur - ch.card, tl.fps)}</Mono>
        {q && <div style={{display: 'flex', alignItems: 'center', gap: 14, marginLeft: 18, paddingLeft: 30, borderLeft: `1px solid ${C.line}`}}>
          <Mono size={22} color={ACC} glow>Q{q.id}</Mono><div style={{fontSize: 28, fontWeight: 300, color: C.ink2}}>{q.short}</div>
        </div>}
      </div>
    </Abs>
  </AbsoluteFill>;
};

// ---------------------------------------------------------------- chrome: header, HUD, progress, subtitles
const Chrome: React.FC = () => {
  const {tl, lesson, roadmap} = useEp();
  const f = useCurrentFrame();
  const i = tl.chapters.findIndex((c) => f >= c.start && f < c.start + c.dur);
  const ch = i >= 0 ? tl.chapters[i] : null;
  const head = ch ? ease(f, ch.board, 12) : 0;
  const cue = tl.cues.find((c) => f >= c.from && f < c.to);
  const corner = rgba(ACC, 0.7);
  const L = (x: number, y: number, w: number, h: number) => <Abs x={x} y={y} w={w} h={h} style={{background: corner}} />;
  const intro = tl.intro.dur;
  const span = 1680 - 6 * tl.chapters.length;
  const sum = intro + tl.chapters.reduce((a, c) => a + c.dur, 0);
  let x = 120;
  let headX = 120;
  const seg = (key: string, start: number, dur: number, kind: 'intro' | 'zero' | 'eng') => {
    const w = (span * dur) / sum;
    const k = Math.min(1, Math.max(0, (f - start) / dur));
    if (f >= start && f < start + dur) headX = x + k * w;
    if (f >= start + dur) headX = Math.max(headX, x + w);
    const eng = kind === 'eng';
    const el = <Abs key={key} x={x} y={1029} w={w} h={3} style={{background: eng ? rgba(ACC, 0.35) : kind === 'intro' ? C.rule : C.pend}}>
      <div style={{width: k * w, height: 3, background: eng ? ACC : kind === 'intro' ? C.ink3 : C.ink2, boxShadow: eng && k > 0 ? `0 0 10px ${GLOW}` : undefined}} />
    </Abs>;
    x += w + 6;
    return el;
  };
  const segs = [seg('intro', 0, intro, 'intro'), ...tl.chapters.map((c) => seg(c.id, c.start, c.dur, c.level))];
  const where = roadmap ? `第 ${roadmap.lesson} 课 §${roadmap.page}` : '';
  return <AbsoluteFill style={{pointerEvents: 'none', fontFamily: F.zh}}>
    <AbsoluteFill style={{background: 'radial-gradient(ellipse at center, transparent 58%, rgba(0,0,0,0.55) 100%)'}} />
    {ch && f >= ch.board && <>
      <Abs x={120} y={64} style={{display: 'flex', alignItems: 'baseline', gap: 16, opacity: head}}>
        {where && <><div style={{fontSize: 22, color: C.ink3}}>{where}</div><Mono size={18} color={C.ink3}>›</Mono></>}
        <Mono size={22} color={C.ink3}>{ch.id} / {String(tl.chapters.length).padStart(2, '0')}</Mono>
        <div style={{fontSize: 24, color: C.ink2}}><Sub text={ch.title} /></div>
      </Abs>
      <Abs x={1460} y={56} w={340} style={{display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 18, opacity: head}}>
        {ch.q ? <Mono size={18} color={rgba(ACC, 0.8)} style={{letterSpacing: '0.1em'}}>Q{ch.q}</Mono> : null}
        <Pill level={ch.level} />
      </Abs>
    </>}
    {L(60, 40, 28, 2)}{L(60, 40, 2, 28)}{L(1832, 40, 28, 2)}{L(1858, 40, 2, 28)}
    {L(60, 1048, 28, 2)}{L(60, 1022, 2, 28)}{L(1832, 1048, 28, 2)}{L(1858, 1022, 2, 28)}
    <Abs x={44} y={160} w={10} h={720} style={{background: `repeating-linear-gradient(to bottom, ${rgba(ACC, 0.3)} 0px, ${rgba(ACC, 0.3)} 1px, transparent 1px, transparent 40px)`}} />
    <Abs x={0} y={70} w={1920} style={{textAlign: 'center'}}>
      <Mono size={16} color={corner} style={{letterSpacing: '0.24em'}}>◢ {ch ? `SEG ${ch.id}/${String(tl.chapters.length).padStart(2, '0')} · ${ch.tag}` : 'ROADMAP'}</Mono>
    </Abs>
    <Abs x={120} y={994}><Mono size={16} color={C.ink3} style={{letterSpacing: '0.16em'}}>T+{mmss(f, tl.fps)}</Mono></Abs>
    <Abs x={1400} y={994} w={400} style={{textAlign: 'right'}}><Mono size={16} color={C.ink3} style={{letterSpacing: '0.16em'}}>{lesson.split(' · ').slice(-2).join(' · ')} // 1080P30</Mono></Abs>
    {segs}
    <Abs x={headX - 6} y={1024} w={12} h={12} style={{borderRadius: 6, background: C.ink, boxShadow: '0 0 8px rgba(230,234,240,0.6)'}} />
    {cue && <Abs x={0} y={918} w={1920} style={{display: 'flex', justifyContent: 'center'}}>
      <div style={{fontSize: 40, letterSpacing: '0.02em', color: C.ink, textShadow: '0 2px 12px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.9)'}}>{cue.text}</div>
    </Abs>}
  </AbsoluteFill>;
};

// ---------------------------------------------------------------- the episode
export type Sfx = {file: string; from: number; volume?: number; dur?: number; fadeOut?: number};

export function makeEpisode(opts: {tl: Timeline; math: Record<string, Formula>; roadmap?: any; lesson: string;
  boards: Record<string, React.FC>; sfx?: (tl: Timeline) => Sfx[]}) {
  // A render may pass another voice's timeline/formulas as input props (make_js_video.py --voice ...).
  const ip = getInputProps() as {timeline?: Timeline; math?: Record<string, Formula>; roadmap?: any};
  const tl = ip.timeline ?? opts.tl;
  const math = ip.math ?? opts.math;
  const roadmap = ip.roadmap ?? opts.roadmap;
  const {lesson, boards} = opts;
  const Intro = boards.intro ?? Cover;
  const Episode: React.FC = () => <EpCtx.Provider value={{tl, math, lesson, roadmap}}>
    <AbsoluteFill style={{background: C.bg, fontFamily: F.zh, color: C.ink}}>
      <Sequence durationInFrames={tl.intro.dur + 8} name="intro">
        <LinesCtx.Provider value={tl.intro.lines}><FadeOut at={tl.intro.dur} fadeIn={boards.intro ? 0 : 9}><Intro /></FadeOut></LinesCtx.Provider>
      </Sequence>
      {tl.chapters.map((c, i) => {
        const Board = boards[c.id] ?? (() => null);
        return <Sequence key={c.id} from={c.start} durationInFrames={c.dur + 8} name={`ch${c.id}`}>
          <Sequence durationInFrames={c.card + 8} name="card"><Card ch={c} n={i + 1} total={tl.chapters.length} /></Sequence>
          <Sequence from={c.card} durationInFrames={c.dur - c.card + 8} name="board">
            <LinesCtx.Provider value={c.lines}><FadeOut at={c.dur - c.card}><Board /></FadeOut></LinesCtx.Provider>
          </Sequence>
        </Sequence>;
      })}
      {tl.audio.map((a, i) => <Sequence key={i} from={a.from} name={`voice ${i}`}><Html5Audio src={staticFile(a.file)} /></Sequence>)}
      {(opts.sfx ? opts.sfx(tl) : []).map((a, i) => <Sequence key={`sfx${i}`} from={Math.max(0, a.from)} durationInFrames={a.dur} name={`sfx ${a.file}`}>
        <Html5Audio src={staticFile(a.file)} volume={(fr) => (a.volume ?? 1) * (a.dur && a.fadeOut ? Math.min(1, Math.max(0, (a.dur - fr) / a.fadeOut)) : 1)} />
      </Sequence>)}
      <Chrome />
    </AbsoluteFill>
  </EpCtx.Provider>;
  // The entry file must call registerRoot itself (Remotion checks for the literal call there).
  return () => <Composition id="episode" component={Episode} durationInFrames={tl.total} fps={tl.fps} width={1920} height={1080} />;
}

const FadeOut: React.FC<{at: number; fadeIn?: number; children: React.ReactNode}> = ({at, fadeIn = 9, children}) => {
  const f = useCurrentFrame();
  return <AbsoluteFill style={{opacity: (fadeIn ? ease(f, 0, fadeIn) : 1) * (1 - lin(f, at, 8))}}>{children}</AbsoluteFill>;
};

