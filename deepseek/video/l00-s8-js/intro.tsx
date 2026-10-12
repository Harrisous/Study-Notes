// 开场：课程表（第 6 课与第 19 课亮起）→ 第 0 课 → §8 的小窗 → 整棵树，再推到模型发布那一栏和时间轴；
// 七个主干挂上标签，三个问题钉在上面。
import React from 'react';
import {Question, Roadmap, NodeTag, Tree, camAt, ease, fit, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, Opening, TREE} from '../l00-shared/scene';

export const TRUNK_TAGS: [string, string][] = [['llm', 'DeepSeek LLM · 2023-11'], ['v2', 'V2 · 2024-05'], ['v3', 'V3 · 2024-12'], ['r1', 'R1 · 2025-01'],
  ['v32', 'V3.2 · 2025-12'], ['v4', 'V4 · 2026-04'], ['v41', 'V4.1-Flash · 2026-09']];

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const TOP = fit({x: 600, y: -34, w: 1950, h: 300}, 0.03);   // the release row and the time axis, 2023 onwards
  const destView = (fr: number) => camAt(fr, FULLV, [{at: at('i4') - 8, dur: 40, to: TOP}]);
  const zoom = ease(f, at('i4') - 8, 30);
  return <Opening dest={TREE} lit={[[6, 'i1.a'], [19, 'i1.b']]} destView={destView}
    renderDest={(cam) => <>
      <Tree rm={rm} cam={cam} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
        nodeIn={(n) => (n.lane === 'M' ? 1 : 1 - 0.65 * zoom)} laneIn={(k) => (k === 'M' ? 1 : 1 - 0.65 * zoom)}
        labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
        nodeLit={(n) => (TRUNK_TAGS.some(([id]) => id === n.id) ? ease(f, at('i5'), 12) : 0)} />
      {TRUNK_TAGS.map(([id, text], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
        p={ease(f, at('i5') + i * 4, 10) * (1 - settle)} />)}
    </>}
    overlay={(cam) => {
      const l = rm.nodes.find((n) => n.id === 'llm')!, v = rm.nodes.find((n) => n.id === 'v41')!;
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(l.x + l.w / 2), y: cam.Y(l.y)}} card={{x: 120, y: 120, w: 600}} up
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(1500), y: cam.Y(rm.axis.y) + 10}} card={{x: 640, y: 800, w: 640}}
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(v.x + v.w / 2), y: cam.Y(v.y)}} card={{x: 1200, y: 120, w: 600}} up
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
