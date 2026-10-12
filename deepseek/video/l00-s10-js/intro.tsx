// 开场：课程表（第 1 课，再到第 2–19 课一起亮起）→ 第 0 课 → §10 的小窗 → 整棵树；
// i4 挂上"账单"的两个例子，i5 挂上"证据"的两个例子，i6 画出推理一侧瓶颈的移动路线。
import React from 'react';
import {ACC, NodeTag, Question, Roadmap, Tree, ease, lin, useBoard, useEp} from '../../../.cursor/skills/lesson-video/js/kit';
import {Opening, TREE} from '../l00-shared/scene';

const TAGS: [string, string, string][] = [['mla', 'MLA · 省字节，付运算', 'i4'], ['dsa', 'DSA · 省读取，存储多 20%', 'i4'],
  ['mhc', 'mHC · 证明：谱范数 = 1', 'i5'], ['engram', 'Engram · 实测：两个算力点', 'i5']];
const PATH = ['dsmoe', 'mla', 'dsa', 'csa', 'csa2'];   // the inference-side bottleneck, solved in this order

export const Intro: React.FC = () => {
  const {f, at} = useBoard();
  const {roadmap, tl} = useEp();
  const rm = roadmap as Roadmap;
  const qs = tl.questions ?? [];
  const settle = ease(f, at('i7'), 16);
  const node = (id: string) => rm.nodes.find((n) => n.id === id)!;
  const path = lin(f, at('i6'), 50);
  return <Opening dest={TREE} lit={[[1, 'i1.a'], ...Array.from({length: 18}, (_, i) => [i + 2, 'i1.b'] as [number, string])]}
    renderDest={(cam) => {
      const pts = PATH.map((id) => {
        const n = node(id);
        return [cam.X(n.x + n.w / 2), cam.Y(n.y + n.h / 2)] as const;
      });
      const seg = path * (pts.length - 1);
      return <>
        <Tree rm={rm} cam={cam} nodeIn={() => 1} alpha={1 - 0.35 * ease(f, at('i6'), 16)}
          labelAlpha={Math.min(1, Math.max(0, (cam.s - 0.3) / 0.15))}
          nodeLit={(n) => {
            const t = TAGS.find(([id]) => id === n.id);
            const onPath = PATH.indexOf(n.id);
            return Math.max(t ? ease(f, at(t[2]), 12) : 0, onPath >= 0 && seg >= onPath - 0.05 ? 1 : 0);
          }} />
        {TAGS.map(([id, text, key], i) => <NodeTag key={id} rm={rm} cam={cam} id={id} text={text} above={i % 2 === 0}
          p={ease(f, at(key) + (i % 2) * 6, 10) * (1 - ease(f, at('i6'), 12))} />)}
        {path > 0 && <path d={pts.slice(0, Math.floor(seg) + 1).map(([x, y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' ') +
          (seg < pts.length - 1 ? ` L${pts[Math.floor(seg)][0] + (pts[Math.floor(seg) + 1][0] - pts[Math.floor(seg)][0]) * (seg % 1)} ${pts[Math.floor(seg)][1] + (pts[Math.floor(seg) + 1][1] - pts[Math.floor(seg)][1]) * (seg % 1)}` : '')}
          fill="none" stroke={ACC} strokeWidth={3} filter="url(#glow)" opacity={1 - settle * 0.5} />}
      </>;
    }}
    overlay={(cam) => {
      const m = node('mla'), h = node('mhc'), c = node('csa2');
      return <>
        <Question n={1} q={qs[0]} anchor={{x: cam.X(m.x + m.w / 2), y: cam.Y(m.y)}} card={{x: 120, y: 120, w: 620}} up
          p={ease(f, at('i4'), 14)} settle={settle} />
        <Question n={2} q={qs[1]} anchor={{x: cam.X(h.x + h.w / 2), y: cam.Y(h.y + h.h) + 4}} card={{x: 860, y: 760, w: 560}}
          p={ease(f, at('i5'), 14)} settle={settle} />
        <Question n={3} q={qs[2]} anchor={{x: cam.X(c.x + c.w / 2), y: cam.Y(c.y)}} card={{x: 1300, y: 120, w: 500}} up
          p={ease(f, at('i6'), 14)} settle={settle} />
      </>;
    }} />;
};
