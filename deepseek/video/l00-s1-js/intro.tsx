// 开场：从 21 课的课程表飞进第 0 课，再飞进 §1 的小窗，那里面就是整棵科技树；三个问题钉在树上。
import React from 'react';
import {Question, Roadmap, Tree, ease, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {Opening, TREE} from '../l00-shared/scene';

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  return <Opening dest={TREE} lit={[[1, 'i1.a'], [19, 'i1.b']]}
    renderDest={(cam) => <Tree rm={rm} cam={cam} nodeIn={() => 1} here={0} alpha={1 - 0.45 * ease(f, at('i4'), 16)}
      labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
      laneLit={(k) => (f >= at('i5') && f < at('i6') && k !== 'M' ? ease(f, at('i5') + 'ABCDEF'.indexOf(k) * 4, 8) : 0)} />}
    overlay={(cam) => <>
      <Question n={1} q={qs[0]} anchor={{x: cam.X((rm.axis.x0 + rm.axis.x1) / 2), y: cam.Y(rm.axis.y) - 4}}
        card={{x: 680, y: 96, w: 700}} up p={ease(f, at('i4'), 14)} settle={settle} />
      <Question n={2} q={qs[1]} anchor={{x: cam.X(-180), y: cam.Y(rm.lanes[6].y + rm.lanes[6].h) + 6}}
        card={{x: 120, y: 740, w: 620}} p={ease(f, at('i5'), 14)} settle={settle} />
      <Question n={3} q={qs[2]} anchor={{x: cam.X(1500), y: cam.Y(rm.lanes[6].y + rm.lanes[6].h) + 6}}
        card={{x: 1180, y: 740, w: 600}} p={ease(f, at('i6'), 14)} settle={settle} />
    </>} />;
};
