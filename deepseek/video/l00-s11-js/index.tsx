// 第 0 课 §11 · 练习与答案 —— 开场 + 10 段（JavaScript / Remotion）。
// 口播在 script.json，分镜在 storyboard.md；答案全部来自第 0 课的练习页。
// 构建：tts-venv python .cursor/skills/lesson-video/scripts/make_js_video.py all deepseek/video/l00-s11-js --voice gsv:mambo
import {registerRoot} from 'remotion';
import {makeEpisode} from '../../../.cursor/skills/lesson-video/js/kit';
import tl from '@gen/l00-s11-js/timeline.json';
import math from '@gen/l00-s11-js/math.json';
import roadmap from '@gen/l00-s11-js/roadmap.json';
import {sfx} from '../l00-shared/sfx';
import {Intro} from './intro';
import {B01, B02, B03, B04, B05, B06, B07, B08, B09, B10} from './boards';

registerRoot(makeEpisode({
  tl: tl as any, math: math as any, roadmap, lesson: 'DEEPSEEK 科技树 · LESSON 00 · §11',
  boards: {intro: Intro, '01': B01, '02': B02, '03': B03, '04': B04, '05': B05, '06': B06, '07': B07, '08': B08, '09': B09, '10': B10}, sfx,
}));
