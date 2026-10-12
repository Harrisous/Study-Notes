// 开场：课程表（第 0 课亮起）→ 第 0 课 → §11 的小窗 → 整棵树；三组题各钉在它们练习的地方：
// 概念题钉在时间轴上（认日期），计算题钉在主线 A 上（复算 KV），推导题钉在主线 C 上（UE8M0）。
import React from 'react';
import {Question, Roadmap, Tree, ease, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {Opening, TREE, laneRect} from '../l00-shared/scene';

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const A = laneRect(rm, 'A'), Cl = laneRect(rm, 'C');
  return <Opening dest={TREE} lit={[[0, 'i1.a']]}
    renderDest={(cam) => <Tree rm={rm} cam={cam} nodeIn={() => 1} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
      labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
      laneLit={(k) => (k === 'A' ? ease(f, at('i5'), 12) : k === 'C' ? ease(f, at('i6'), 12) : 0)} />}
    overlay={(cam) => <>
      <Question n={1} q={qs[0]} anchor={{x: cam.X(700), y: cam.Y(rm.axis.y) - 4}} card={{x: 120, y: 96, w: 620}} up
        p={ease(f, at('i4'), 14)} settle={settle} />
      <Question n={2} q={qs[1]} anchor={{x: cam.X(1750), y: cam.Y(A.y) - 4}} card={{x: 1180, y: 96, w: 620}} up
        p={ease(f, at('i5'), 14)} settle={settle} />
      <Question n={3} q={qs[2]} anchor={{x: cam.X(1700), y: cam.Y(Cl.y + Cl.h) + 4}} card={{x: 640, y: 760, w: 700}}
        p={ease(f, at('i6'), 14)} settle={settle} />
    </>} />;
};
