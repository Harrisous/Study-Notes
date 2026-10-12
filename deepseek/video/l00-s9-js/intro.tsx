// 开场：课程表（第 1–19 课一起亮起）→ 第 0 课 → §9 的小窗 → 整棵树；主线 A 与主线 D 两条跳读路线亮起，
// 三个问题钉在上面。
import React from 'react';
import {Question, Roadmap, Tree, ease, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {Opening, TREE, laneRect} from '../l00-shared/scene';

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const A = laneRect(rm, 'A'), D = laneRect(rm, 'D'), E = laneRect(rm, 'E');
  const route = ease(f, at('i4'), 14);
  return <Opening dest={TREE} lit={Array.from({length: 19}, (_, i) => [i + 1, 'i1.a'] as [number, string])}
    renderDest={(cam) => <Tree rm={rm} cam={cam} nodeIn={() => 1} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
      labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
      laneLit={(k) => (k === 'A' || k === 'D' ? route * (1 - 0.6 * ease(f, at('i5'), 12)) : 0)} />}
    overlay={(cam) => <>
      <Question n={1} q={qs[0]} anchor={{x: cam.X(1200), y: cam.Y(A.y) - 4}} card={{x: 640, y: 96, w: 640}} up
        p={ease(f, at('i4'), 14)} settle={settle} />
      <Question n={2} q={qs[1]} anchor={{x: cam.X(600), y: cam.Y(D.y + D.h) + 4}} card={{x: 120, y: 860, w: 560}}
        p={ease(f, at('i5'), 14)} settle={settle} />
      <Question n={3} q={qs[2]} anchor={{x: cam.X(1900), y: cam.Y(E.y + E.h) + 4}} card={{x: 1180, y: 860, w: 620}}
        p={ease(f, at('i6'), 14)} settle={settle} />
    </>} />;
};
