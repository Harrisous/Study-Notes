// Sound cues shared by lesson-0 episodes (quiet: about 20 dB under the voice before loudness normalisation).
import {Sfx, Timeline} from '../../../.cursor/skills/lesson-video/js/kit';

export const sfx = (t: Timeline): Sfx[] => {
  const L = t.intro.lines;
  const m = (k: string) => {
    const [id, mk] = k.split('.');
    return mk ? L[id]?.marks[mk] ?? L[id]?.start : L[id]?.start;
  };
  const ev: Sfx[] = [{file: 'sfx/drone.wav', from: 0, volume: 0.03, dur: t.intro.dur + 30, fadeOut: 60}];
  for (const k of ['i1.a', 'i1.b']) if (m(k) !== undefined) ev.push({file: 'sfx/tick.wav', from: m(k)!, volume: 0.07});
  if (m('i2') !== undefined) ev.push({file: 'sfx/whoosh.wav', from: m('i2')! - 8, volume: 0.09});
  if (m('i3.a') !== undefined) ev.push({file: 'sfx/whoosh.wav', from: m('i3.a')! - 6, volume: 0.12});
  for (const k of ['i4', 'i5', 'i6']) if (m(k) !== undefined) ev.push({file: 'sfx/ping.wav', from: m(k)!, volume: 0.09});
  for (const c of t.chapters) ev.push({file: 'sfx/glitch.wav', from: c.start, volume: 0.09});
  return ev;
};
