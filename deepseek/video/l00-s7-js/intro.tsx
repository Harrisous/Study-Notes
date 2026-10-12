// 开场：课程表（主线 F 所在的四课依次亮起）→ 第 0 课 → §7 的小窗 → 整棵树，再推到主线 F；
// 五个节点的要点挂上去，三个问题钉在上面（主线 F 在树的最下面，三张卡都放在上方）。
import React from 'react';
import {Question, Roadmap, NodeTag, Tree, camAt, ease, fit, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, Opening, TREE, laneRect} from '../l00-shared/scene';

export const F_TAGS: [string, string][] = [['kaplan', 'Kaplan · 主要加大模型'], ['chinchilla', 'Chinchilla · 约 20 token/参数'],
  ['hpscale', 'DeepSeek LLM · 超参数幂律'], ['muon', 'Muon · 正交化'], ['v4muon', 'V4 · 矩阵用 Muon']];

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const Fl = laneRect(rm, 'F');
  const LANEV = fit({x: 30, y: Fl.y - 260, w: 2520, h: Fl.h + 330}, 0.03);   // lane labels stay off-frame
  const destView = (fr: number) => camAt(fr, FULLV, [{at: at('i4') - 8, dur: 40, to: LANEV}]);
  return <Opening dest={TREE} lit={[[2, 'i1.a'], [6, 'i1.a'], [17, 'i1.b'], [18, 'i1.b']]} destView={destView}
    renderDest={(cam) => <>
      <Tree rm={rm} cam={cam} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
        nodeIn={(n) => (n.lane === 'F' ? 1 : 1 - 0.6 * ease(f, at('i4') - 8, 30))}
        laneIn={(k) => (k === 'F' ? 1 : 1 - 0.6 * ease(f, at('i4') - 8, 30))}
        labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
        laneLit={(k) => (k === 'F' ? ease(f, at('i3.a') + 40, 14) : 0)}
        nodeLit={(n) => (F_TAGS.some(([id]) => id === n.id) ? ease(f, at('i5'), 12) : 0)} />
      {F_TAGS.map(([id, text], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
        p={ease(f, at('i5') + i * 5, 10) * (1 - settle)} />)}
    </>}
    overlay={(cam) => {
      const k = rm.nodes.find((n) => n.id === 'kaplan')!, h = rm.nodes.find((n) => n.id === 'hpscale')!, m = rm.nodes.find((n) => n.id === 'v4muon')!;
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(k.x + k.w / 2), y: cam.Y(k.y)}} card={{x: 120, y: 96, w: 540}} up
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(h.x + h.w / 2), y: cam.Y(h.y)}} card={{x: 690, y: 96, w: 540}} up
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(m.x + m.w / 2), y: cam.Y(m.y)}} card={{x: 1260, y: 96, w: 540}} up
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
