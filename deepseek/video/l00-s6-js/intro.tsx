// 开场：课程表（主线 E 所在的五课依次亮起）→ 第 0 课 → §6 的小窗 → 整棵树，再推到主线 E；
// 五个节点的要点挂上去，三个问题钉在上面。
import React from 'react';
import {Question, Roadmap, NodeTag, Tree, camAt, ease, fit, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, Opening, TREE, laneRect} from '../l00-shared/scene';

export const E_TAGS: [string, string][] = [['preln', 'Pre-LN · 主干只剩加法'], ['hc', 'HC · 一条流变 n 条'], ['mhc', 'mHC · 双随机'],
  ['engram', 'Engram · 查表记忆'], ['v41mem', 'V4.1 · 读写减半']];

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const E = laneRect(rm, 'E');
  const LANEV = fit({x: 120, y: E.y - 150, w: 2430, h: E.h + 300}, 0.03);   // lane labels stay off-frame
  const destView = (fr: number) => camAt(fr, FULLV, [{at: at('i4') - 8, dur: 40, to: LANEV}]);
  return <Opening dest={TREE} lit={[[1, 'i1.a'], [16, 'i1.b'], [17, 'i1.b'], [18, 'i1.c'], [19, 'i1.c']]} destView={destView}
    renderDest={(cam) => <>
      <Tree rm={rm} cam={cam} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
        nodeIn={(n) => (n.lane === 'E' ? 1 : 1 - 0.6 * ease(f, at('i4') - 8, 30))}
        laneIn={(k) => (k === 'E' ? 1 : 1 - 0.6 * ease(f, at('i4') - 8, 30))}
        labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
        laneLit={(k) => (k === 'E' ? ease(f, at('i3.a') + 40, 14) : 0)}
        nodeLit={(n) => (E_TAGS.some(([id]) => id === n.id) ? ease(f, at('i5'), 12) : 0)} />
      {E_TAGS.map(([id, text], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
        p={ease(f, at('i5') + i * 5, 10) * (1 - settle)} />)}
    </>}
    overlay={(cam) => {
      const a = rm.nodes.find((n) => n.id === 'preln')!, h = rm.nodes.find((n) => n.id === 'hc')!, g = rm.nodes.find((n) => n.id === 'engram')!;
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(a.x + a.w / 2), y: cam.Y(a.y)}} card={{x: 120, y: 96, w: 600}} up
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(h.x + h.w / 2), y: cam.Y(h.y + h.h) + 4}} card={{x: 560, y: 800, w: 700}}
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(g.x + g.w / 2), y: cam.Y(g.y + g.h) + 4}} card={{x: 1280, y: 800, w: 520}}
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
