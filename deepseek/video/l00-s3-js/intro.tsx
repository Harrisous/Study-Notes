// 开场：课程表（主线 B 所在的五课依次亮起）→ 第 0 课 → §3 的小窗 → 整棵树，再推到主线 B；
// 激活比例挂在各代的节点上，三个问题钉在上面。
import React from 'react';
import {Question, Roadmap, NodeTag, Tree, camAt, ease, fit, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, Opening, TREE, laneRect} from '../l00-shared/scene';

export const RATIO_TAGS: [string, string][] = [['dsmoe', '16B · 约 1/6'], ['devroute', 'V2 · 约 1/11'], ['v3moe', 'V3 · 约 1/18'],
  ['v4moe', 'V4-Pro · 约 1/33'], ['v41moe', 'V4.1 预填充 · 约 1/69']];

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const B = laneRect(rm, 'B');
  const LANEV = fit({x: 600, y: B.y - 150, w: 1950, h: B.h + 300}, 0.03);   // 2024 onwards; lane labels stay off-frame
  const destView = (fr: number) => camAt(fr, FULLV, [{at: at('i4') - 8, dur: 40, to: LANEV}]);
  return <Opening dest={TREE} lit={[[7, 'i1.a'], [10, 'i1.b'], [11, 'i1.b'], [18, 'i1.c'], [19, 'i1.c']]} destView={destView}
    renderDest={(cam) => <>
      <Tree rm={rm} cam={cam} nodeIn={() => 1} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
        labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
        laneLit={(k) => (k === 'B' ? ease(f, at('i3.a') + 40, 14) : 0)}
        nodeLit={(n) => (RATIO_TAGS.some(([id]) => id === n.id) ? ease(f, at('i6'), 12) : 0)} />
      {RATIO_TAGS.map(([id, text], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
        p={ease(f, at('i6') + i * 5, 10) * (1 - settle)} />)}
    </>}
    overlay={(cam) => {
      const d = rm.nodes.find((n) => n.id === 'dsmoe')!, v = rm.nodes.find((n) => n.id === 'v41moe')!;
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(d.x + d.w / 2), y: cam.Y(d.y + d.h) + 4}} card={{x: 120, y: 740, w: 640}}
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(1300), y: cam.Y(B.y) - 50}} card={{x: 640, y: 96, w: 680}} up
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(v.x + v.w / 2), y: cam.Y(v.y + v.h) + 4}} card={{x: 1120, y: 740, w: 680}}
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
