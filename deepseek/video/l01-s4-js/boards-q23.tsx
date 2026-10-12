// Q2 为什么盖住未来（06）、Q3 为什么拆成多头（07–08）、三个问题的答案（09）。
// 数字全部来自 lesson-01 §3–§4：The/cat/sat 的 q、k、v，掩码那一行，两个头，W^O 的 2×2 例子，
// 原论文表 3 (A) 的 BLEU，GPT-2 small 一层注意力的参数。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, Arc, Axes, BOX_GLOW, C, Check, Chip, F, Mono, Svg, TEXT_GLOW, Tex, Txt, drawOn, ease, lin, lit, mix, rgba, swing, up, useBoard,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {RoadmapView} from './intro';

const PAPER = rgba('#C9CED6', 0.92);
const hatchBg = `repeating-linear-gradient(45deg, ${rgba(C.ink2, 0.18)} 0px, ${rgba(C.ink2, 0.18)} 5px, transparent 5px, transparent 12px)`;

// ================================================================ 06 因果掩码：盖住未来
const TOK = ['The', 'cat', 'sat'];
const W3 = [['1.0000', null, null], ['0.8044', '0.1956', null], ['0.0743', '0.6200', '0.3057']];
const GX = 330, GY = 300, CW = 150, CH = 112, PX = 158, PY = 122;

export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const sA = 1 - ease(f, at('l2'), 12);
  const sB = ease(f, at('l2'), 14) * (1 - ease(f, at('l3'), 12));
  const sC = ease(f, at('l3'), 14);
  // scene A: one sheet of paper moving down a word list
  const slide = at('l1.a');
  const k = f < slide ? 4 : f < slide + 34 ? 1 : f < slide + 68 ? 2 : 3;
  const sheetY = mix(330 + (k - 1) * 80, 330 + k * 80, ease(f, k === 1 ? slide : slide + (k - 1) * 34, 12)) - 14;
  // scene C–F: the score table
  const papers = (r: number) => swing(f, at('l3') + 8 + r * 4, 22);
  const toHatch = ease(f, at('l3.b'), 18);
  const catFocus = f >= at('l4') && f < at('l5');
  const grow = swing(f, at('l6'), 30);
  const n = grow > 0 ? 4 : 3;
  const cellVal = (r: number, c: number): string | null => {
    if (c > r) return null;
    if (r === 3) return '…';
    if (r === 1) {
      if (f >= at('l4.d')) return W3[1][c];
      if (f >= at('l4.c')) return ['4.1133', '1'][c];
      if (f >= at('l4.a')) return ['1.4142', '0'][c];
      return null;
    }
    return f >= at('l5') ? W3[r][c] : null;
  };
  const steps: [string, string, React.ReactNode][] = [
    ['l4.a', '缩放后', <>(1.4142, 0, <span style={{color: C.ink3}}>1.4142</span>)</>],
    ['l4.b', '加掩码', <>(1.4142, 0, <span style={{color: ACC}}>−∞</span>)</>],
    ['l4.c', '取指数', <>(4.1133, 1, 0) <span style={{fontSize: 22, color: C.ink3}}>Σ = 5.1133</span></>],
    ['l4.d', '归一化', <>(0.8044, 0.1956, 0)</>],
  ];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: sA}}>
      {[['apple', '苹果'], ['book', '书'], ['cat', '猫'], ['dog', '狗']].map(([e, z], i) =>
        <Abs key={e} x={620} y={330 + i * 80} style={{display: 'flex', gap: 120, ...up(ease(f, 4 + i * 3, 14))}}>
          <Mono size={40} color={C.ink} style={{width: 180}}>{e}</Mono><Txt size={40}>{z}</Txt>
        </Abs>)}
      <Abs x={580} y={sheetY} w={520} h={330 + 4 * 80 - sheetY + 10} style={{background: PAPER, borderRadius: 4,
        boxShadow: '0 -6px 18px rgba(0,0,0,0.5)', opacity: ease(f, slide, 10)}} />
      <Abs x={1200} y={430} style={up(ease(f, slide + 80, 16))}><Txt size={36} color={ACC} glow weight={400}>那张纸 = 因果掩码</Txt></Abs>
    </AbsoluteFill>

    <AbsoluteFill style={{opacity: sB}}>
      {['The', 'cat', 'sat', 'on'].map((w, i) => <React.Fragment key={w}>
        <Abs x={520 + i * 240} y={300} w={180} h={80} style={{border: `1px solid ${C.ink2}`, borderRadius: 10, display: 'flex', alignItems: 'center',
          justifyContent: 'center', ...up(ease(f, at('l2') + i * 3, 12))}}><Mono size={36} color={C.ink}>{w}</Mono></Abs>
        <Abs x={609 + i * 240} y={392} w={2} h={100 * p('l2.a')} style={{background: C.ink3}} />
        <Abs x={520 + i * 240} y={506} w={180} style={{textAlign: 'center', ...up(ease(f, at('l2.a') + i * 4, 12))}}>
          <Mono size={20} color={C.ink3}>位置 {i + 1} 预测</Mono>
          <div style={{fontFamily: F.mono, fontSize: 34, color: C.ink2, marginTop: 6}}>{['cat', 'sat', 'on', '…'][i]}</div>
        </Abs>
      </React.Fragment>)}
      <Svg opacity={p('l2.b')}>
        <Arc x0={610} x1={850} y={292} h={-90} w={0.5} p={lin(f, at('l2.b'), 16)} color={C.ink2} />
        <text x={730} y={186} textAnchor="middle" fontFamily={F.zh} fontSize={24} fill={C.ink}>偷看位置 2</text>
      </Svg>
      <Abs x={520} y={640} w={940} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l2.b') + 20, 14))}}>
        <Chip>位置 1 要预测的正是 cat：看得到，就是抄答案</Chip>
      </Abs>
    </AbsoluteFill>

    <AbsoluteFill style={{opacity: sC}}>
      {/* headers */}
      {Array.from({length: n}, (_, c) => <Abs key={`h${c}`} x={GX + c * PX} y={GY - 44} w={CW} style={{textAlign: 'center',
        opacity: c < 3 ? 1 : grow}}><Mono size={24} color={C.ink2}>{[...TOK, 'on'][c]}</Mono></Abs>)}
      <Abs x={GX} y={GY - 82} w={3 * PX} style={{textAlign: 'center', opacity: 0.8}}><Mono size={18} color={C.ink3}>被看的位置 j →</Mono></Abs>
      {Array.from({length: n}, (_, r) => <Abs key={`r${r}`} x={GX - 160} y={GY + r * PY} w={140} h={CH} style={{display: 'flex',
        alignItems: 'center', justifyContent: 'flex-end', gap: 10, opacity: r < 3 ? 1 : grow}}>
        <Mono size={18} color={C.ink3}>t={r + 1}</Mono>
        <Mono size={26} color={r === 1 && catFocus ? ACC : C.ink2} glow={r === 1 && catFocus}>{[...TOK, 'on'][r]}</Mono></Abs>)}
      {Array.from({length: n * n}, (_, i) => {
        const r = Math.floor(i / n), c = i % n;
        const masked = c > r;
        const isNew = r === 3 || c === 3;
        const dimRow = catFocus && r !== 1 ? 0.35 : 1;
        const v = cellVal(r, c);
        const old = grow > 0 && r < 3 && !masked;
        const lockGlow = old ? ease(f, at('l6.a'), 12) : 0;
        return <Abs key={i} x={GX + c * PX} y={GY + r * PY} w={CW} h={CH} style={{borderRadius: 8, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: 4, fontFamily: F.mono, opacity: dimRow * (isNew ? grow : 1),
          border: `1px solid ${r === 1 && catFocus && !masked ? ACC : lockGlow > 0 ? rgba(ACC, 0.4 + 0.5 * lockGlow) : masked && toHatch > 0 ? C.ink3 : C.line}`,
          boxShadow: (r === 1 && catFocus && !masked) || lockGlow > 0.5 ? BOX_GLOW : undefined,
          background: masked ? (toHatch > 0 || isNew ? hatchBg : C.panel) : r === 1 && catFocus ? rgba(ACC, 0.08) : C.panel}}>
          {masked && (toHatch > 0 || isNew) && <span style={{fontSize: 22, color: C.ink2, opacity: isNew ? 1 : toHatch}}>−∞</span>}
          {masked && f >= at('l5') && !isNew && <span style={{fontSize: 26, color: C.ink3}}>0</span>}
          {!masked && v && <span style={{fontSize: 28, color: r === 1 && catFocus ? ACC : C.ink}}>{v}</span>}
        </Abs>;
      })}
      {/* one sheet per row (scene C), turning into the hatch */}
      {[0, 1].map((r) => {
        const q = papers(r) * (1 - toHatch);
        if (q <= 0) return null;
        const x0 = GX + (r + 1) * PX - 4;
        return <Abs key={`p${r}`} x={x0 + (1 - papers(r)) * 700} y={GY + r * PY - 3} w={GX + 3 * PX - x0} h={CH + 6} style={{background: PAPER,
          borderRadius: 4, boxShadow: '-6px 0 16px rgba(0,0,0,0.55)', opacity: Math.min(1, q * 1.5),
          border: r === 1 && f >= at('l3.a') && f < at('l3.b') ? `2px solid ${ACC}` : undefined}} />;
      })}
      <Abs x={GX + 3 * PX + 20} y={GY + PY + 34} style={{...up(ease(f, at('l3.a'), 12) * (1 - toHatch))}}>
        <Txt size={24} color={ACC} glow>第 t 行：盖住 j &gt; t</Txt></Abs>
      <Svg opacity={toHatch * (1 - ease(f, at('l4'), 14))}>
        <path d={`M${GX + PX - 4} ${GY - 4} L${GX + 3 * PX - 4} ${GY - 4} L${GX + 3 * PX - 4} ${GY + 2 * PY - 4} L${GX + 2 * PX - 4} ${GY + 2 * PY - 4}
          L${GX + 2 * PX - 4} ${GY + PY - 4} L${GX + PX - 4} ${GY + PY - 4} Z`} fill="none" stroke={ACC} strokeWidth={2} filter="url(#vglow)" />
      </Svg>
      <Abs x={GX + 3 * PX + 20} y={GY + 8} style={{...up(toHatch * (1 - ease(f, at('l4'), 14)))}}><Txt size={28} color={ACC} glow weight={400}>右上角 = 未来</Txt></Abs>
      {/* row sums */}
      {[0, 1, 2].map((r) => <Abs key={`s${r}`} x={GX + 3 * PX + 12} y={GY + r * PY} h={CH} style={{display: 'flex', alignItems: 'center',
        opacity: (r === 1 ? ease(f, at('l4.d'), 10) : ease(f, at('l5'), 10)) * (1 - grow)}}><Mono size={22} color={C.ink2}>Σ = 1</Mono></Abs>)}
      {/* the cat row, step by step */}
      <Abs x={1060} y={292} w={760} style={{display: 'flex', flexDirection: 'column', opacity: 1 - ease(f, at('l5'), 14)}}>
        <div style={{...up(p('l4')), marginBottom: 18}}><Txt size={28} color={C.ink2}>softmax <b style={{color: ACC, fontWeight: 400}}>之前</b>，被盖住的格子加 −∞</Txt></div>
        {steps.map(([key, label, v], i) => {
          const lastOne = i === 3;
          return <div key={label} style={{display: 'flex', alignItems: 'center', gap: 26, height: 92, borderLeft: `1px solid ${lastOne ? ACC : C.line}`,
            paddingLeft: 32, ...up(p(key))}}>
            <Txt size={25} color={lastOne ? ACC : C.ink3} style={{width: 100}}>{label}</Txt>
            <Mono size={34} color={lastOne ? ACC : C.ink2} glow={lastOne}>{v}</Mono>
          </div>;
        })}
        <div style={{...up(ease(f, at('l4.c') + 16, 12)), marginTop: 8, paddingLeft: 32}}><Tex id="einf" size={44} color={C.ink2} /></div>
      </Abs>
      {/* consequences */}
      <Abs x={1060} y={318} w={760} style={{display: 'flex', flexDirection: 'column', gap: 22, opacity: ease(f, at('l5'), 14) * (1 - 0.5 * grow)}}>
        <Txt size={30}>被盖住的格子：权重正好是 0</Txt>
        <Txt size={30} color={C.ink2}>每行剩下的仍然加起来是 1</Txt>
        <div style={up(p('l5.a'))}><Chip on size={28}>一次前向 → 所有位置并行预测，不作弊</Chip></div>
      </Abs>
      <Abs x={1060} y={640} w={760} style={{display: 'flex', flexDirection: 'column', gap: 18}}>
        <div style={up(p('l6'))}><Txt size={28} color={C.ink2}>末尾接上 on：只多出一行一列</Txt></div>
        <div style={up(p('l6.a'))}><Txt size={28} color={ACC} glow weight={400}>旧位置的 K、V 一点都不变</Txt></div>
        <div style={up(p('l6.b'))}><Chip on size={26}>→ 第 4 课：KV cache</Chip></div>
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 07 多头：几位评委
const TX = [640, 960, 1280], BY = 540;
const ONE = [0.0743, 0.62, 0.3057], H1 = [0.0159, 0.8668, 0.1173], H2 = [0.4223, 0.1554, 0.4223];
const QK: [string, [number, number], [number, number]][] = [['The', [1, 1], [0, 1]], ['cat', [0, 2], [2, 0]], ['sat', [2, 1], [1, 1]]];

const Arcs: React.FC<{w: number[]; below?: boolean; p: number; label: string; hi?: boolean}> = ({w, below, p, label, hi}) => {
  if (p <= 0) return null;
  const sg = below ? 1 : -1;
  return <g opacity={Math.min(1, p * 1.5)}>
    {[0, 1].map((j) => <Arc key={j} x0={TX[2]} x1={TX[j]} y={BY + sg * 34} h={sg * (TX[2] - TX[j]) * (below ? 0.36 : 0.42)} w={w[j]} p={p}
      color={hi ? ACC : ACC} label={w[j].toFixed(4)} />)}
    <circle cx={TX[2]} cy={BY + sg * (34 + 34)} r={30} fill="none" stroke={rgba(ACC, 0.25 + 0.75 * Math.min(1, w[2] * 1.4))}
      strokeWidth={1.5 + 16 * w[2]} opacity={p} />
    <text x={TX[2] + 60} y={BY + sg * 76 + 8} fontFamily={F.mono} fontSize={24} fill={w[2] > 0.4 ? ACC : C.ink2} opacity={p}>{w[2].toFixed(4)}</text>
    <text x={TX[2] + 70} y={BY + sg * 190} fontFamily={F.zh} fontSize={28} fill={ACC} opacity={p}>{label}</text>
  </g>;
};

export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const sA = 1 - ease(f, at('l3'), 12);
  const sB = ease(f, at('l3'), 14);
  const single = p('l3.a') * (1 - ease(f, at('l4') - 6, 12));
  const split = swing(f, at('l3.c'), 20);
  const Judge: React.FC<{name: string; on?: boolean}> = ({name, on}) =>
    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12}}>
      <div style={{width: 90, height: 90, borderRadius: 45, border: `2px solid ${on ? ACC : C.ink2}`, boxShadow: on ? BOX_GLOW : undefined}} />
      <Txt size={28} color={on ? ACC : C.ink2}>{name}</Txt>
    </div>;
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: sA}}>
      <Abs x={280} y={250} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22, ...up(p('l1'))}}>
        <Judge name="总分" /><Txt size={28} color={C.ink3}>1 位评委：一个总分</Txt>
      </Abs>
      <Abs x={880} y={250} style={{display: 'flex', gap: 70, ...up(p('l1.a'))}}>
        <Judge name="内容" on /><Judge name="结构" on /><Judge name="文笔" on />
      </Abs>
      <Abs x={1000} y={452} style={up(p('l1.a'))}><Txt size={28} color={C.ink3}>3 位评委：三个角度</Txt></Abs>
      <Abs x={280} y={610} style={up(p('l2'))}><Txt size={34}>一个头 = 一套 softmax 权重 = 一种标准</Txt></Abs>
      <Abs x={280} y={690} style={up(p('l2.a'))}><Txt size={34} color={ACC} glow weight={400}>多头 = 拆成几个窄头，各自打分、各自加权平均</Txt></Abs>
    </AbsoluteFill>

    <AbsoluteFill style={{opacity: sB}}>
      {/* tokens on a line */}
      {TOK.map((t, i) => <Abs key={t} x={TX[i] - 70} y={BY - 30} w={140} h={60} style={{borderRadius: 10, border: `1px solid ${i === 2 ? ACC : C.ink2}`,
        background: C.panel, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: i === 2 ? BOX_GLOW : undefined,
        ...up(ease(f, at('l3') + i * 3, 12))}}><Mono size={30} color={i === 2 ? ACC : C.ink}>{t}</Mono></Abs>)}
      <Svg>
        <Arcs w={ONE} p={single} label="单头" />
        <Arcs w={H1} p={lin(f, at('l4'), 22)} label="头 1" />
        <Arcs w={H2} below p={lin(f, at('l4.a'), 22)} label="头 2" />
      </Svg>
      {/* q, k of §3, split into two heads by coordinate */}
      <Abs x={110} y={150} style={{opacity: ease(f, at('l3.b'), 14)}}>
        <div style={{display: 'flex', gap: 34, marginLeft: 86, marginBottom: 10}}>
          {['q', 'k'].map((n) => <Mono key={n} size={22} color={C.ink3} style={{width: 2 * 48 + 26 * split}}>{n}</Mono>)}
        </div>
        {QK.map(([t, q, k]) => <div key={t} style={{display: 'flex', alignItems: 'center', gap: 34, marginBottom: 8}}>
          <Mono size={22} color={t === 'sat' ? ACC : C.ink2} style={{width: 52}}>{t}</Mono>
          {[q, k].map((v, j) => <div key={j} style={{display: 'flex', gap: 4 + 22 * split}}>
            {v.map((x, c) => <div key={c} style={{width: 44, height: 40, borderRadius: 5, display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: F.mono, fontSize: 22, color: split > 0.5 && c === 0 ? ACC : C.ink,
              border: `1px solid ${split > 0.5 ? (c === 0 ? ACC : C.ink2) : C.line}`, background: split > 0.5 && c === 0 ? rgba(ACC, 0.12) : C.panel}}>{x}</div>)}
          </div>)}
        </div>)}
        <div style={{display: 'flex', gap: 22, marginTop: 10, marginLeft: 86, opacity: split}}>
          <Mono size={20} color={ACC}>■ 头 1：第 1 个坐标</Mono><Mono size={20} color={C.ink}>□ 头 2：第 2 个坐标</Mono>
        </div>
      </Abs>
      <Abs x={1440} y={170} w={420} style={{display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'flex-start'}}>
        <div style={up(p('l5'))}><Chip size={24}>一个头：一套折中的权重</Chip></div>
        <div style={up(p('l5.a'))}><Chip on size={24}>两个头：两套看法</Chip></div>
        <div style={{...up(p('l5.b')), maxWidth: 400, fontSize: 24, color: C.ink2, lineHeight: 1.45, marginTop: 10}}>
          比方不对的地方：评委的分工事先定好，头的分工是训练出来的</div>
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 08 合头与算账
const ORG = {x: 1300, y: 640}, U = 560;     // the 2-D plane for the W^O example: 1 unit = 560 px
const P2 = (x: number, y: number): [number, number] => [ORG.x + x * U, ORG.y - y * U];

export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  const sA = 1 - ease(f, at('l3'), 12);
  const sC = ease(f, at('l3'), 14) * (1 - ease(f, at('l4'), 12));
  const sD = ease(f, at('l4'), 14) * (1 - ease(f, at('l5'), 12));
  const sE = ease(f, at('l5'), 14) * (1 - ease(f, at('l6'), 12));
  const sF = ease(f, at('l6'), 14);
  const cut = swing(f, at('l1.b'), 22), stack = swing(f, at('l1.b') + 36, 30);
  const merge = ease(f, at('l2'), 14);
  const blocks = swing(f, at('l2.a'), 20);
  const write = lin(f, at('l2.b') + 6, 60);
  const a1 = lin(f, at('l3.b'), 24), a2 = lin(f, at('l3.b') + 50, 24), su = lin(f, at('l3.c'), 24);
  const v1: [number, number] = [0.1332, -0.1332], v2: [number, number] = [0.5777 * 0.5, 0.5777];
  const bleu: [number, number, string, string][] = [[1, 24.9, '1 头 × 512 维', 'l5.c'], [8, 25.8, '8 头 × 64 维', 'l5.b'], [32, 25.4, '32 头 × 16 维', 'l5.d']];
  const bx = (h: number) => 520 + (Math.log2(h) / 5) * 860, by = (b: number) => 800 - (b - 24.5) * 290;
  return <AbsoluteFill>
    {/* split: one 768-wide product, cut into 12 heads of 64 */}
    <AbsoluteFill style={{opacity: sA * (1 - merge)}}>
      <Abs x={160} y={240} style={{display: 'flex', alignItems: 'center', gap: 26, ...up(p('l1.a'))}}>
        <div style={{width: 220, height: 220, border: `1px solid ${C.ink2}`, borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: C.panel}}><Mono size={24} color={C.ink}>W<sup>Q</sup> 768×768</Mono></div>
        <Mono size={36} color={C.ink3}>×</Mono>
        <div style={{width: 36, height: 220, border: `1px solid ${C.ink2}`, borderRadius: 4, background: C.panel}} />
        <Mono size={36} color={C.ink3}>=</Mono>
      </Abs>
      <Abs x={640} y={330} style={{display: 'flex', gap: 0}}>
        {Array.from({length: 12}, (_, i) => <div key={i} style={{width: 92, height: 40, marginRight: 6 * cut,
          transform: `translate(${-i * 98 * stack + (i % 6) * 0 * stack}px, ${stack * (60 + i * 34)}px)`,
          border: `1px solid ${rgba(ACC, 0.5 + 0.5 * cut)}`, background: rgba(ACC, i % 2 ? 0.1 : 0.2), opacity: ease(f, at('l1.a') + 6, 14),
          display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          <Mono size={18} color={ACC} style={{opacity: cut}}>头 {i + 1}</Mono></div>)}
      </Abs>
      <Abs x={640} y={290} style={{opacity: ease(f, at('l1.a') + 6, 14) * (1 - cut)}}><Mono size={20} color={C.ink3}>q：768 个数</Mono></Abs>
      <Abs x={790} y={420} style={{...up(stack)}}><Txt size={26} color={ACC} glow weight={400}>每 64 个一段 → [12, 64]：12 个头的查询，不多算一次乘法</Txt></Abs>
    </AbsoluteFill>
    {/* merge: W^O in 12 column blocks */}
    <AbsoluteFill style={{opacity: sA * merge}}>
      <Abs x={160} y={200} style={up(merge)}><Txt size={30} color={C.ink2}>合头：W<sup>O</sup>（768 × 768）按列切成 12 块</Txt></Abs>
      <Abs x={160} y={270} style={{display: 'flex', gap: 14 * blocks}}>
        {Array.from({length: 12}, (_, i) => {
          const w = ease(f, at('l2.b') + 6 + i * 5, 10);
          return <div key={i} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10}}>
            <div style={{width: 84, height: 300, border: `1px solid ${w > 0.5 ? ACC : C.ink2}`, background: w > 0.5 ? rgba(ACC, 0.12) : C.panel,
              boxShadow: w > 0.5 ? BOX_GLOW : undefined, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
              <Mono size={17} color={C.ink2} style={{opacity: blocks}}>W<sup>O</sup><sub>{i + 1}</sub></Mono></div>
            <div style={{width: 30, height: 64, borderRadius: 3, border: `1px solid ${rgba(ACC, 0.6)}`, background: rgba(ACC, 0.25), opacity: ease(f, at('l2.b'), 12)}} />
            <Mono size={16} color={C.ink3} style={{opacity: ease(f, at('l2.b'), 12)}}>o<sub>{i + 1}</sub></Mono>
          </div>;
        })}
      </Abs>
      <Svg opacity={ease(f, at('l2.b') + 20, 14)}>
        <path d={drawOn([[1470, 420], [1600, 420]], write)} stroke={ACC} strokeWidth={2} fill="none" filter="url(#vglow)" />
        <circle cx={1640} cy={420} r={34} fill={C.bg} stroke={ACC} strokeWidth={2} filter="url(#vglow)" />
        <text x={1640} y={432} textAnchor="middle" fontFamily={F.mono} fontSize={34} fill={ACC}>Σ</text>
        <text x={1640} y={500} textAnchor="middle" fontFamily={F.zh} fontSize={22} fill={C.ink2}>写回 768 维</text>
      </Svg>
      <Abs x={160} y={720} style={{...up(p('l2.b'))}}><Tex id="wo" size={40} parts={{w: lit(write), o: lit(write)}} /></Abs>
    </AbsoluteFill>

    {/* the 2-D example */}
    <AbsoluteFill style={{opacity: sC}}>
      <Abs x={140} y={170} style={up(p('l3'))}><Tex id="wo2" size={34}
        parts={{a: f >= at('l3.b') ? lit(1 - ease(f, at('l3.c'), 10) * 0.6) : {}, b: f >= at('l3.b') + 50 ? lit(1 - ease(f, at('l3.c'), 10) * 0.6) : {},
          u: f >= at('l3.c') ? lit(1) : {}}} /></Abs>
      <Abs x={140} y={330} style={{display: 'flex', alignItems: 'center', gap: 18, ...up(p('l3.a'))}}>
        <Txt size={26} color={C.ink2}>拼接</Txt>
        {[['0.1332', '头 1'], ['0.5777', '头 2']].map(([v, h]) => <div key={h} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6}}>
          <div style={{padding: '10px 18px', borderRadius: 8, border: `1px solid ${rgba(ACC, 0.6)}`, background: rgba(ACC, 0.08)}}><Mono size={30} color={ACC}>{v}</Mono></div>
          <Mono size={18} color={C.ink3}>{h}</Mono></div>)}
      </Abs>
      <Svg>
        <Axes x0={P2(-0.15, 0)[0]} y0={P2(0, 0.62)[1]} x1={P2(0.62, 0)[0]} y1={P2(0, -0.28)[1]} zeroY={ORG.y} p={ease(f, at('l3.a'), 14)}
          xt={[[P2(0.25, 0)[0], '0.25'], [P2(0.5, 0)[0], '0.5']]} yt={[[P2(0, 0.25)[1], '0.25'], [P2(0, 0.5)[1], '0.5']]} />
        <line x1={ORG.x} x2={ORG.x} y1={P2(0, 0.62)[1]} y2={P2(0, -0.28)[1]} stroke={C.ink3} strokeWidth={1.2} opacity={ease(f, at('l3.a'), 14)} />
        <path d={drawOn([P2(0, 0), P2(...v1)], a1)} stroke={C.ink} strokeWidth={4} fill="none" />
        <path d={drawOn([P2(...v1), P2(v1[0] + v2[0], v1[1] + v2[1])], a2)} stroke={C.ink2} strokeWidth={4} fill="none" />
        <path d={drawOn([P2(0, 0), P2(0.422, 0.4445)], su)} stroke={ACC} strokeWidth={4} fill="none" filter="url(#vglow)" />
        {su > 0.95 && <circle cx={P2(0.422, 0.4445)[0]} cy={P2(0.422, 0.4445)[1]} r={7} fill={ACC} filter="url(#vglow)" />}
      </Svg>
      <Abs x={P2(...v1)[0] + 16} y={P2(...v1)[1] - 4} style={{opacity: a1}}><Mono size={20} color={C.ink}>头 1：0.1332 · (1, −1)</Mono></Abs>
      <Abs x={P2(0.28, 0.2)[0] + 70} y={P2(0.28, 0.2)[1] + 30} style={{opacity: a2}}><Mono size={20} color={C.ink2}>头 2：0.5777 · (0.5, 1)</Mono></Abs>
      <Abs x={P2(0.422, 0.4445)[0] + 18} y={P2(0.422, 0.4445)[1] - 40} style={{opacity: su}}><Mono size={26} color={ACC} glow>(0.4220, 0.4445)</Mono></Abs>
      <Abs x={140} y={760} style={{...up(ease(f, at('l3.c') + 30, 14))}}><Chip on size={26}>和直接乘 W<sup>O</sup> 一样</Chip></Abs>
    </AbsoluteFill>

    {/* cost: the same total width */}
    <AbsoluteFill style={{opacity: sD}}>
      <Abs x={160} y={250} style={up(p('l4'))}><Txt size={30} color={C.ink2}>打分的运算量 ∝ T² × 每个头的维度，再乘头数</Txt></Abs>
      {[['l4', '1 个 768 维的头', 1, 'cost1'], ['l4.a', '12 个 64 维的头', 12, 'cost2']].map(([key, label, k, tex], r) =>
        <Abs key={label as string} x={160} y={380 + r * 190} style={{...up(p(key as string))}}>
          <Txt size={28} color={C.ink}>{label as string}</Txt>
          <div style={{display: 'flex', gap: (k as number) > 1 ? 6 : 0, marginTop: 16}}>
            {Array.from({length: k as number}, (_, i) => <div key={i} style={{width: (1152 - ((k as number) - 1) * 6) / (k as number), height: 46,
              background: rgba(ACC, (k as number) > 1 ? (i % 2 ? 0.2 : 0.32) : 0.25), border: `1px solid ${rgba(ACC, 0.6)}`}} />)}
          </div>
          <div style={{marginTop: 14}}><Tex id={tex as string} size={36} color={C.ink2} /></div>
        </Abs>)}
      <Abs x={1360} y={590} style={{...up(p('l4.b'))}}><Chip on size={34}>一样多</Chip></Abs>
    </AbsoluteFill>

    {/* head count vs BLEU (the paper's table 3, row A) */}
    <AbsoluteFill style={{opacity: sE}}>
      <Abs x={160} y={176} style={up(p('l5'))}><Txt size={32}>头是不是越多越好？</Txt></Abs>
      <Abs x={160} y={228} style={up(p('l5.a'))}><Txt size={24} color={C.ink2}>原论文 · 机器翻译 · 总计算量不变（表 3 的 A 行）</Txt></Abs>
      <Svg opacity={ease(f, at('l5.a'), 14)}>
        <Axes x0={480} y0={by(26.05)} x1={1500} y1={by(24.5)} p={1} xt={[[bx(1), '1'], [bx(8), '8'], [bx(32), '32']]}
          yt={[[by(24.5), '24.5'], [by(25), '25.0'], [by(25.5), '25.5'], [by(26), '26.0']]} xlabel="头数（每头维度 = 512 ÷ 头数）" ylabel="BLEU" />
        <path d={drawOn([[bx(1), by(24.9)], [bx(8), by(25.8)], [bx(32), by(25.4)]], ease(f, at('l5.d') + 16, 20))} fill="none"
          stroke={rgba(ACC, 0.6)} strokeWidth={2} />
        {bleu.map(([h, b, , key]) => <circle key={h} cx={bx(h)} cy={by(b)} r={10 * ease(f, at(key), 10)} fill={h === 8 ? ACC : C.bg}
          stroke={ACC} strokeWidth={3} filter="url(#vglow)" />)}
      </Svg>
      {bleu.map(([h, b, label, key]) => <React.Fragment key={h}>
        <Abs x={bx(h) - 120} y={by(b) - 58} w={240} style={{textAlign: 'center', ...up(p(key))}}>
          <Mono size={30} color={h === 8 ? ACC : C.ink} glow={h === 8}>{b.toFixed(1)}</Mono></Abs>
        <Abs x={bx(h) - 120} y={by(b) + 18} w={240} style={{textAlign: 'center', ...up(p(key))}}>
          <div style={{fontSize: 20, color: C.ink3}}>{label}</div></Abs>
      </React.Fragment>)}
      <Abs x={1560} y={620} w={360} style={{...up(ease(f, at('l5.d') + 24, 14)), display: 'flex', flexDirection: 'column', gap: 8}}>
        <Txt size={22} color={C.ink2}>头太少：只能按一种标准取信息</Txt>
        <Txt size={22} color={C.ink2}>头太多：每个头的窗口太窄</Txt></Abs>
    </AbsoluteFill>

    {/* GPT-2 small: one attention layer */}
    <AbsoluteFill style={{opacity: sF}}>
      <Abs x={160} y={190} style={up(p('l6'))}><Txt size={30} color={C.ink2}>GPT-2 small 一层注意力的参数</Txt></Abs>
      <Abs x={160} y={270} style={{display: 'flex', alignItems: 'flex-end', gap: 28}}>
        {['W^Q', 'W^K', 'W^V', 'W^O'].map((name, i) => <div key={name} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
          ...up(ease(f, at('l6') + i * 5, 12))}}>
          <div style={{width: 190, height: 190, border: `1px solid ${rgba(ACC, 0.6)}`, background: rgba(ACC, 0.12), display: 'flex', alignItems: 'center',
            justifyContent: 'center'}}><Mono size={26} color={C.ink}>W<sup>{name.split('^')[1]}</sup></Mono></div>
          <div style={{width: 190, height: 4, background: C.ink2}} />
          <Mono size={18} color={C.ink3}>768 × 768 + 偏置</Mono>
        </div>)}
        <Mono size={18} color={C.ink3} style={{marginLeft: 10, marginBottom: 40}}>（前三块在代码里拼成 c_attn）</Mono>
      </Abs>
      <Abs x={160} y={600} style={{display: 'flex', flexDirection: 'column', gap: 20}}>
        <Tex id="params" size={36} reveal={ease(f, at('l6'), 30)} />
        <div style={up(p('l6.a'))}><Tex id="params2" size={36} color={ACC} glow /></div>
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 09 三个问题的答案
export const B09: React.FC = () => {
  const {f, at, p} = useBoard();
  const out = ease(f, at('l4') - 6, 18);
  const items: [number, string, string, string, string, string][] = [
    [1, 'l1', 'l1.b', '让分数的方差回到 1：softmax 的温度不再随头宽变化', 'recap1', 'sq'],
    [2, 'l2', 'l2.a', 'softmax 之前加 −∞：训练并行又诚实；旧位置的 K、V 能精确缓存', 'recap2', 'm'],
    [3, 'l3', 'l3.a', '几套独立的 softmax，同时按几种标准取信息；总运算量不变', 'recap3', 'qk'],
  ];
  const parts: Record<string, any> = {};
  if (f >= at('l1') && f < at('l2')) parts.sq = lit(1);
  if (f >= at('l2') && f < at('l3')) parts.m = lit(1);
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - out}}>
      <Abs x={0} y={150} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}>
        <Tex id="attn" size={92} parts={parts} />
      </Abs>
      <Abs x={120} y={420} w={1680} style={{display: 'flex', gap: 30}}>
        {items.map(([n, key, ans, text, tex]) => {
          const done = ease(f, at(ans), 12);
          return <div key={n} style={{flex: 1, padding: '26px 28px', borderRadius: 14, background: rgba(C.bg, 0.9),
            border: `1px solid ${done > 0.5 ? rgba(ACC, 0.7) : C.line}`, boxShadow: done > 0.5 ? BOX_GLOW : undefined,
            display: 'flex', flexDirection: 'column', gap: 18, ...up(p(key))}}>
            <div style={{display: 'flex', alignItems: 'center', gap: 16}}>
              <Mono size={30} color={ACC} glow>Q{n}</Mono>
              <div style={{fontSize: 28, color: C.ink}}>{['为什么除以 √dₖ', '为什么盖住未来', '为什么拆成多头'][n - 1]}</div>
              <svg width={44} height={44} style={{marginLeft: 'auto', overflow: 'visible'}}><Check x={2} y={2} size={38} p={lin(f, at(ans), 14)} /></svg>
            </div>
            <div style={{fontSize: 26, color: C.ink2, lineHeight: 1.45, opacity: done}}>{text}</div>
            <div style={{opacity: done}}><Tex id={tex} size={30} color={C.ink2} /></div>
          </div>;
        })}
      </Abs>
    </AbsoluteFill>
    {/* bookend: back out to the map; the next box is lit */}
    {out > 0 && <AbsoluteFill style={{opacity: out}}>
      <RoadmapView f={f} t0={at('l4') - 6} next={ease(f, at('l4.a'), 14)} />
    </AbsoluteFill>}
  </AbsoluteFill>;
};

export {TEXT_GLOW};
