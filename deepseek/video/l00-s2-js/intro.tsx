// 开场：课程表（主线 A 所在的八课依次亮起）→ 第 0 课 → §2 的小窗 → 整棵树，再推到主线 A；五个字节数挂在主线 A 的节点上，三个问题钉在上面。
import React from 'react';
import {Question, Roadmap, NodeTag, Tree, camAt, ease, fit, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, Opening, TREE, laneRect} from '../l00-shared/scene';

export const KV_TAGS: [string, string][] = [['gqa67', '67B · 389,120'], ['mla', 'V2 · 69,120'], ['dsa', 'V3.2 · 48,068'],
  ['csa', 'V4-Flash · 3,514'], ['csa2', 'V4.1-Flash · 890']];

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const A = laneRect(rm, 'A');
  const LANEV = fit({x: 430, y: A.y - 150, w: 2120, h: A.h + 300}, 0.03);
  const destView = (fr: number) => camAt(fr, FULLV, [{at: at('i4') - 8, dur: 40, to: LANEV}]);
  return <Opening dest={TREE} lit={[[1, 'i1.a'], [4, 'i1.a'], [5, 'i1.a'], [6, 'i1.a'], [9, 'i1.b'], [14, 'i1.b'], [18, 'i1.b'], [19, 'i1.b']]} destView={destView}
    renderDest={(cam) => <>
      <Tree rm={rm} cam={cam} nodeIn={() => 1} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
        labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
        laneLit={(k) => (k === 'A' ? ease(f, at('i3.a') + 40, 14) : 0)}
        nodeLit={(n) => (KV_TAGS.some(([id]) => id === n.id) ? ease(f, at('i5'), 12) : 0)} />
      {KV_TAGS.map(([id, text], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
        p={ease(f, at('i5') + i * 5, 10) * (1 - settle)} />)}
    </>}
    overlay={(cam) => {
      const g = rm.nodes.find((n) => n.id === 'gqa67')!, d = rm.nodes.find((n) => n.id === 'dsa')!;
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(g.x + g.w / 2), y: cam.Y(g.y + g.h) + 4}} card={{x: 120, y: 740, w: 640}}
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(1500), y: cam.Y(A.y) - 50}} card={{x: 640, y: 96, w: 640}} up
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(d.x + d.w / 2), y: cam.Y(d.y + d.h) + 4}} card={{x: 1180, y: 740, w: 600}}
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
