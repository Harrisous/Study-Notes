// 第 1 课 §4 · 注意力（二）—— 开场 + 9 段（JavaScript / Remotion）。
// 口播在 script.json，公式在 formulas.txt，分镜在 storyboard.md；数字全部来自第 0、1 课的源文件。
// 构建：tts-venv python .cursor/skills/lesson-video/scripts/make_js_video.py all deepseek/video/l01-s4-js
import {registerRoot} from 'remotion';
import {Sfx, Timeline, makeEpisode} from '../../../.cursor/skills/lesson-video/js/kit';
import tl from '@gen/l01-s4-js/timeline.json';
import math from '@gen/l01-s4-js/math.json';
import roadmap from '@gen/l01-s4-js/roadmap.json';
import {Intro} from './intro';
import {BOARDS} from './boards';

/** Quiet sound design: about 20 dB under the voice before loudness normalisation (Kokoro speaks near -28 LUFS). */
const sfx = (t: Timeline): Sfx[] => {
  const L = t.intro.lines;
  const m = (k: string) => {
    const [id, mk] = k.split('.');
    return mk ? L[id].marks[mk] : L[id].start;
  };
  const ev: Sfx[] = [{file: 'sfx/drone.wav', from: 0, volume: 0.03, dur: t.intro.dur + 30, fadeOut: 60}];
  for (let k = 0; k < 6; k++) ev.push({file: 'sfx/tick.wav', from: m('i1.b') + k * 5, volume: 0.07});
  ev.push({file: 'sfx/whoosh.wav', from: m('i3') - 12, volume: 0.09});
  for (let i = 0; i < 5; i++) ev.push({file: 'sfx/tick.wav', from: m('i3.b') + 6 + i * 9, volume: 0.08});
  ev.push({file: 'sfx/whoosh.wav', from: m('i4') - 10, volume: 0.12});
  for (const k of ['i5', 'i6', 'i7']) ev.push({file: 'sfx/ping.wav', from: m(k), volume: 0.09});
  for (const c of t.chapters) ev.push({file: 'sfx/glitch.wav', from: c.start, volume: 0.09});
  const c9 = t.chapters[t.chapters.length - 1];
  for (const k of ['l1.b', 'l2.a', 'l3.a']) {
    const [id, mk] = k.split('.');
    ev.push({file: 'sfx/ping.wav', from: c9.board + c9.lines[id].marks[mk], volume: 0.08});
  }
  ev.push({file: 'sfx/whoosh.wav', from: c9.board + c9.lines.l4.start - 8, volume: 0.1});
  return ev;
};

registerRoot(makeEpisode({
  tl: tl as any, math: math as any, roadmap, lesson: 'DEEPSEEK 科技树 · LESSON 01 · §4',
  boards: {intro: Intro, ...BOARDS}, sfx,
}));
