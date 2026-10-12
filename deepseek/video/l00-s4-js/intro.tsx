// 开场：课程表（主线 C 所在的四课依次亮起）→ 第 0 课 → §4 的小窗 → 整棵树，再推到主线 C；
// 每一步的数值格式挂在节点上，三个问题钉在上面。
import React from 'react';
import {Question, Roadmap, NodeTag, Tree, camAt, ease, fit, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, Opening, TREE, laneRect} from '../l00-shared/scene';

export const FMT_TAGS: [string, string][] = [['fp8fmt', 'E4M3 · 最大 448'], ['fp8train', 'V3 · 557.6 万美元'], ['ue8m0', 'V3.1 · 2 的幂'],
  ['fp4', 'V4 · FP4 专家'], ['fp4kv', 'V4.1 · 每维 4.5 位']];

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const Cl = laneRect(rm, 'C');
  const LANEV = fit({x: 330, y: Cl.y - 150, w: 2220, h: Cl.h + 300}, 0.03);
  const destView = (fr: number) => camAt(fr, FULLV, [{at: at('i4') - 8, dur: 40, to: LANEV}]);
  return <Opening dest={TREE} lit={[[12, 'i1.a'], [15, 'i1.a'], [18, 'i1.b'], [19, 'i1.b']]} destView={destView}
    renderDest={(cam) => <>
      <Tree rm={rm} cam={cam} nodeIn={() => 1} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
        labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
        laneLit={(k) => (k === 'C' ? ease(f, at('i3.a') + 40, 14) : 0)}
        nodeLit={(n) => (FMT_TAGS.some(([id]) => id === n.id) ? ease(f, at('i5'), 12) : 0)} />
      {FMT_TAGS.map(([id, text], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
        p={ease(f, at('i5') + i * 5, 10) * (1 - settle)} />)}
    </>}
    overlay={(cam) => {
      const t = rm.nodes.find((n) => n.id === 'fp8train')!, k = rm.nodes.find((n) => n.id === 'fp4kv')!;
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(t.x + t.w / 2), y: cam.Y(t.y + t.h) + 4}} card={{x: 120, y: 740, w: 640}}
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(1700), y: cam.Y(Cl.y) - 50}} card={{x: 640, y: 96, w: 640}} up
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(k.x + k.w / 2), y: cam.Y(k.y + k.h) + 4}} card={{x: 1180, y: 740, w: 620}}
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
