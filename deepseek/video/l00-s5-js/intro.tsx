// 开场：课程表（主线 D 所在的八课依次亮起）→ 第 0 课 → §5 的小窗 → 整棵树，再推到主线 D；
// DeepSeek 的五处改动挂在节点上，三个问题钉在上面。
import React from 'react';
import {Question, Roadmap, NodeTag, Tree, camAt, ease, fit, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, Opening, TREE, laneRect} from '../l00-shared/scene';

export const D_TAGS: [string, string][] = [['grpo', 'GRPO · 去掉价值网络'], ['r1rl', 'R1 · 只用规则奖励'], ['hybrid', 'V3.1 · 一模两式'],
  ['scalerl', 'V3.2 · 后训练 > 10%'], ['opd', 'V4 · 在线蒸馏']];

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const D = laneRect(rm, 'D');
  const LANEV = fit({x: 380, y: D.y - 150, w: 2170, h: D.h + 300}, 0.03);   // 2022 onwards; lane labels stay off-frame
  const destView = (fr: number) => camAt(fr, FULLV, [{at: at('i4') - 8, dur: 40, to: LANEV}]);
  return <Opening dest={TREE} lit={[[3, 'i1.a'], [8, 'i1.a'], [10, 'i1.a'], [11, 'i1.a'], [13, 'i1.b'], [15, 'i1.b'], [18, 'i1.b'], [19, 'i1.b']]}
    destView={destView}
    renderDest={(cam) => <>
      <Tree rm={rm} cam={cam} nodeIn={() => 1} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
        labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
        laneLit={(k) => (k === 'D' ? ease(f, at('i3.a') + 40, 14) : 0)}
        nodeLit={(n) => (D_TAGS.some(([id]) => id === n.id) ? ease(f, at('i5'), 12) : 0)} />
      {D_TAGS.map(([id, text], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
        p={ease(f, at('i5') + i * 5, 10) * (1 - settle)} />)}
    </>}
    overlay={(cam) => {
      const r = rm.nodes.find((n) => n.id === 'rlhf')!, s = rm.nodes.find((n) => n.id === 'scalerl')!;
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(r.x + r.w / 2), y: cam.Y(r.y)}} card={{x: 120, y: 96, w: 640}} up
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(1300), y: cam.Y(D.y + D.h) + 40}} card={{x: 640, y: 780, w: 640}}
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(s.x + s.w / 2), y: cam.Y(s.y)}} card={{x: 1180, y: 96, w: 620}} up
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
