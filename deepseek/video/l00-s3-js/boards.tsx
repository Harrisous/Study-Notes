// 第 0 课 §3 · 主线 B：前馈层的稀疏化 —— 9 段黑板。数字全部来自 lesson-00 §3 与规格总表（研究笔记 lesson-00-doc.md）；
// 标"示意"的数值（专家打分、负载、科室名）是为了演示机制编的，画面上也标了"示意"。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Check, Chip, F, Mono, Roadmap, Svg, Tex, Txt, View, camAt, dim, ease, fit, lin, lit, mix,
  project, rgba, swing, up, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, TreeScene, laneRect, nodeOf} from '../l00-shared/scene';

const useRm = () => useEp().roadmap as Roadmap;
const Card: React.FC<{on?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, style, children}) =>
  <div style={{padding: '18px 24px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${on ? rgba(ACC, 0.7) : C.line}`,
    boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;
const Note: React.FC<{x: number; y: number; p?: number; children?: React.ReactNode}> = ({x, y, p = 1, children}) =>
  <Abs x={x} y={y} style={{opacity: p}}><Mono size={16} color={C.ink3} style={{letterSpacing: '0.08em'}}>{children}</Mono></Abs>;

// ================================================================ 01 绑在一起的账单
const XP = [3, 12, 21, 30];   // 示意: the experts one token visits
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const phaseA = 1 - ease(f, at('l3'), 14), phaseB = ease(f, at('l3'), 14);
  // dense: a token sweeps through every column of the parameter grid
  const grow = swing(f, at('l1.b') + 10, 36);
  const cols = Math.round(mix(12, 18, grow));
  const sweep = ((Math.max(0, f - 16)) % 54) / 54;
  const GX = 160, GY = 250, CS = 32;
  const barW = (k: number) => 700 * k;
  // sparse: one block splits into 32 experts in 4 rows × 8 columns; 4 GPUs hold 2 columns each
  const split = swing(f, at('l3.a'), 26);
  const EX = 230, EY = 300, EW = 56, EH = 40, EG = 10;
  const ex = (i: number) => EX + (i % 8) * (EW + EG), ey = (i: number) => EY + Math.floor(i / 8) * (EH + EG);
  const route = lin(f, at('l3.b'), 18), gpu = ease(f, at('l4.a'), 14), hop = lin(f, at('l4.b'), 60);
  const TX = 110, TY = EY + 2 * (EH + EG) - EG / 2;
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: phaseA}}>
      <Abs x={GX} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>稠密模型：每个参数都参与一次乘加</Txt></Abs>
      <Svg>
        {Array.from({length: 8 * cols}, (_, i) => {
          const c = i % cols, r = Math.floor(i / cols);
          const hot = Math.abs(sweep * cols - c - 0.5) < 0.9;
          const born = c >= 12 ? ease(f, at('l1.b') + 10 + (c - 12) * 4, 10) : ease(f, 4 + c * 2, 10);
          return <rect key={i} x={GX + c * CS} y={GY + r * CS} width={CS - 6} height={CS - 6} rx={4} opacity={born}
            fill={hot ? rgba(ACC, 0.5) : C.panel} stroke={hot ? ACC : C.line} strokeWidth={1} />;
        })}
        <circle cx={GX + sweep * cols * CS} cy={GY - 22} r={9} fill={ACC} filter="url(#vglow)" opacity={ease(f, 16, 10)} />
        <text x={GX + sweep * cols * CS + 16} y={GY - 16} fontFamily={F.mono} fontSize={18} fill={ACC} opacity={ease(f, 16, 10)}>token</text>
      </Svg>
      <Abs x={900} y={250} style={up(p('l1.a'))}>
        <Txt size={24} color={C.ink2}>参数 N</Txt>
        <div style={{width: barW(cols / 18), height: 30, marginTop: 8, borderRadius: 4, background: rgba(C.ink2, 0.35)}} />
        <Txt size={24} color={C.ink2} style={{marginTop: 22}}>每个 token 的计算</Txt>
        <div style={{width: barW(cols / 18), height: 30, marginTop: 8, borderRadius: 4, background: rgba(ACC, 0.45), boxShadow: BOX_GLOW}} />
        <div style={{marginTop: 26}}><Tex id="dense" size={46} parts={{n: f >= at('l1.b') ? lit(1) : {}}} /></div>
      </Abs>
      <Abs x={1640} y={300} style={up(p('l1.b'))}><Txt size={26} color={ACC} glow>绑在一起</Txt></Abs>
      <Abs x={GX} y={620} w={1600} style={up(p('l2'))}>
        <Txt size={24} color={C.ink2}>DeepSeek LLM 7B 的参数</Txt>
        <div style={{display: 'flex', marginTop: 10, height: 56, borderRadius: 6, overflow: 'hidden', border: `1px solid ${C.line}`}}>
          <div style={{width: `${58.7 * lin(f, at('l2.a'), 20)}%`, background: rgba(ACC, 0.5), display: 'flex', alignItems: 'center',
            paddingLeft: 18, whiteSpace: 'nowrap', boxShadow: BOX_GLOW}}><Txt size={26}>前馈网络 58.7%</Txt></div>
          <div style={{flexGrow: 1, background: C.panel, display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 18}}>
            <Txt size={22} color={C.ink3}>其余 41.3%</Txt></div>
        </div>
      </Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: phaseB}}>
      <Abs x={EX} y={210} style={up(phaseB)}><Txt size={28} color={C.ink2}>条件计算：一个大前馈网络 → 许多个小的“专家”</Txt></Abs>
      <Svg>
        <rect x={EX} y={EY} width={8 * (EW + EG) - EG} height={4 * (EH + EG) - EG} rx={10} fill={rgba(C.ink2, 0.18)} stroke={C.ink3}
          opacity={1 - split} />
        {Array.from({length: 32}, (_, i) => {
          const on = XP.includes(i) && route > 0;
          return <rect key={i} x={mix(EX + 4 * (EW + EG), ex(i), split)} y={mix(EY + 1.5 * (EH + EG), ey(i), split)} width={EW} height={EH} rx={5}
            opacity={split} fill={on ? rgba(ACC, 0.25 + 0.3 * route) : C.panel} stroke={on ? ACC : C.line} strokeWidth={on ? 1.6 : 1}
            filter={on && route > 0.9 ? 'url(#vglow)' : undefined} />;
        })}
        {route > 0 && XP.map((i) => <line key={i} x1={TX + 10} y1={TY} x2={TX + 10 + (ex(i) - TX - 10) * route} y2={TY + (ey(i) + EH / 2 - TY) * route}
          stroke={ACC} strokeWidth={1.6} opacity={0.85} />)}
        {route > 0 && <circle cx={TX} cy={TY} r={10} fill={ACC} filter="url(#vglow)" />}
        {[0, 1, 2, 3].map((g) => <g key={g} opacity={gpu}>
          <rect x={EX + g * 2 * (EW + EG) - 8} y={EY - 10} width={2 * (EW + EG) - EG + 16} height={4 * (EH + EG) - EG + 20} rx={10}
            fill="none" stroke={C.ink} strokeWidth={1.4} strokeDasharray="5 5" />
          <text x={EX + g * 2 * (EW + EG) + (EW + EG) - EG / 2} y={EY + 4 * (EH + EG) + 28} textAnchor="middle" fontFamily={F.mono}
            fontSize={18} fill={C.ink2}>GPU {g}</text>
        </g>)}
        {hop > 0 && [[0, 2], [1, 3], [3, 0], [2, 1]].map(([a, b], k) => {
          const xa = EX + a * 2 * (EW + EG) + (EW + EG), xb = EX + b * 2 * (EW + EG) + (EW + EG), y = EY + 4 * (EH + EG) + 44;
          const t = Math.min(1, Math.max(0, hop * 1.6 - k * 0.18));
          const mx = (xa + xb) / 2, my = y + 40 + Math.abs(xb - xa) * 0.12;
          const px = (1 - t) * (1 - t) * xa + 2 * (1 - t) * t * mx + t * t * xb, py = (1 - t) * (1 - t) * y + 2 * (1 - t) * t * my + t * t * y;
          return <g key={k}>
            <path d={`M${xa} ${y} Q${mx} ${my} ${xb} ${y}`} fill="none" stroke={rgba(ACC, 0.5)} strokeWidth={1.4} strokeDasharray="4 6" />
            {t > 0 && t < 1 && <circle cx={px} cy={py} r={6} fill={ACC} filter="url(#vglow)" />}
          </g>;
        })}
      </Svg>
      <Abs x={1000} y={300} w={800} style={up(p('l3.c'))}>
        <Txt size={24} color={C.ink2}>总参数（全部 32 个专家）</Txt>
        <div style={{width: 640, height: 30, marginTop: 8, borderRadius: 4, background: rgba(C.ink2, 0.35)}} />
        <Txt size={24} color={C.ink2} style={{marginTop: 22}}>每个 token 的激活参数（4 个）</Txt>
        <div style={{width: 80, height: 30, marginTop: 8, borderRadius: 4, background: rgba(ACC, 0.5), boxShadow: BOX_GLOW}} />
        <Txt size={30} color={ACC} glow style={{marginTop: 24}}>两件事拆开了</Txt>
      </Abs>
      <Abs x={1000} y={640} w={800} style={up(p('l4'))}>
        <Card on={f >= at('l4.a')}>
          <Txt size={26}>另一笔账：<span style={{opacity: ease(f, at('l4.a'), 12)}}>全部专家都放在显存里；</span></Txt>
          <Txt size={26} style={{opacity: ease(f, at('l4.b'), 12)}}>token 要在卡与卡之间来回搬</Txt>
        </Card>
      </Abs>
      <Note x={EX} y={EY + 4 * (EH + EG) + 90} p={route}>示意：32 个专家，一个 token 经过 4 个</Note>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 02 医院
const ROOMS = ['内科', '外科', '眼科', '皮肤科', '骨科', '耳鼻喉', '神经科', '口腔科'];   // 示意
const QUEUE = [1, 7, 1, 2, 1, 2, 1, 1];                                                   // 示意: patients waiting
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const RX = 620, RY = 330, RW = 250, RH = 104, RG = 22;
  const rx = (i: number) => RX + (i % 4) * (RW + RG), ry = (i: number) => RY + Math.floor(i / 4) * (RH + RG);
  const tri = ease(f, at('l1.a'), 12), spec = ease(f, at('l1.b'), 12), gp = ease(f, at('l1.c'), 12);
  const fine = swing(f, at('l2.a') + 6, 24), bal = swing(f, at('l2.b') + 30, 40), bias = ease(f, at('l2.c'), 14);
  const cur = f >= at('l3') ? '' : f >= at('l2.c') ? 'bias' : f >= at('l2.b') ? 'bal' : f >= at('l2.a') ? 'fine' : f >= at('l1.c') ? 'gp'
    : f >= at('l1.b') ? 'spec' : f >= at('l1.a') ? 'tri' : '';
  const box = (on: boolean): React.CSSProperties => ({position: 'absolute', borderRadius: 12, border: `1px solid ${on ? ACC : C.line}`,
    background: on ? rgba(ACC, 0.08) : C.panel, boxShadow: on ? BOX_GLOW : undefined, display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center'});
  const queued = (i: number) => Math.round(mix(QUEUE[i], 2, bal));
  return <AbsoluteFill>
    <Abs x={150} y={160} style={up(p('l1'))}><Txt size={30} color={C.ink2}>MoE = 一家医院</Txt></Abs>
    {/* triage desk */}
    <div style={{...box(cur === 'tri' || cur === 'bias'), left: 150, top: 380, width: 300, height: 150, opacity: ease(f, at('l1'), 12)}}>
      <Txt size={34} weight={400}>分诊台</Txt><Txt size={24} color={ACC} style={{opacity: tri}}>= 路由器</Txt>
    </div>
    {/* bias: the triage desk's recommendation scores, the crowded room's pulled down */}
    <Abs x={150} y={560} w={300} style={{opacity: bias}}>
      <Mono size={16} color={C.ink3}>推荐分（示意）</Mono>
      {[['外科', 0.82, -0.3], ['眼科', 0.55, 0], ['骨科', 0.6, 0]].map(([n, v, b]) => <div key={n as string} style={{display: 'flex',
        alignItems: 'center', gap: 10, marginTop: 8}}>
        <Txt size={18} color={C.ink2} style={{width: 56}}>{n as string}</Txt>
        <div style={{width: 200 * ((v as number) + (b as number) * lin(f, at('l2.c') + 20, 20)), height: 16, borderRadius: 3,
          background: (b as number) < 0 ? rgba(ACC, 0.6) : rgba(C.ink2, 0.35)}} />
        {(b as number) < 0 && <Mono size={16} color={ACC} style={{opacity: lin(f, at('l2.c') + 20, 20)}}>−b</Mono>}
      </div>)}
    </Abs>
    {/* the general practitioner everyone sees */}
    <div style={{...box(cur === 'gp'), left: RX, top: 190, width: 4 * RW + 3 * RG, height: 100, opacity: ease(f, at('l1'), 12)}}>
      <Txt size={28}>全科医生</Txt><Txt size={22} color={ACC} style={{opacity: gp}}>= 共享专家：每个病人都看</Txt>
    </div>
    {/* specialists */}
    {ROOMS.map((name, i) => {
      const splitting = i === 0 ? fine : 0;
      return <React.Fragment key={name}>
        <div style={{...box(cur === 'spec' || (cur === 'bal' && i === 1)), left: rx(i), top: ry(i), width: RW, height: RH,
          opacity: ease(f, at('l1') + 4 + i * 2, 12) * (1 - splitting)}}>
          <Txt size={26}>{name}</Txt>
          <div style={{display: 'flex', gap: 5, marginTop: 6, height: 12}}>
            {Array.from({length: queued(i)}, (_, k) => <div key={k} style={{width: 10, height: 10, borderRadius: 5,
              background: i === 1 && bal < 0.5 && QUEUE[1] > 3 ? ACC : C.ink2, opacity: ease(f, at('l2.b'), 10)}} />)}
          </div>
        </div>
        {i === 0 && fine > 0 && ['心内', '消化', '呼吸'].map((sub, k) => <div key={sub} style={{...box(cur === 'fine'),
          left: rx(0) + k * ((RW - 2 * 8) / 3 + 8), top: ry(0), width: (RW - 2 * 8) / 3, height: RH, opacity: fine}}>
          <Txt size={22}>{sub}</Txt></div>)}
      </React.Fragment>;
    })}
    <Abs x={rx(4)} y={ry(4) + RH + 14} style={{opacity: spec}}><Txt size={22} color={ACC}>专科医生 = 路由专家</Txt></Abs>
    <Note x={rx(7) + 60} y={ry(4) + RH + 18} p={spec}>科室名为示意</Note>
    {/* the three mechanisms */}
    <Abs x={150} y={740} w={1650} style={{display: 'flex', gap: 18}}>
      {[['细粒度切分：内科拆成心内、消化、呼吸', 'l2.a', 'fine'], ['负载均衡：别都挤到一个科室', 'l2.b', 'bal'],
        ['V3 的偏置：排队太长，推荐分暂时调低', 'l2.c', 'bias']].map(([t, key, id]) =>
        <div key={key} style={up(p(key))}><Chip on={cur === id} size={22}>{t}</Chip></div>)}
    </Abs>
    {/* where the analogy breaks */}
    <AbsoluteFill style={{opacity: ease(f, at('l3'), 14), background: rgba(C.bg, 0.86 * ease(f, at('l3'), 14))}}>
      <Abs x={150} y={200} style={up(p('l3'))}><Txt size={30} color={C.ink2}>比方不对的地方</Txt></Abs>
      <Abs x={150} y={290} w={800} style={up(p('l3.a'))}>
        <Card on={f < at('l3.b')}>
          <Txt size={28}>分诊按 token 进行</Txt>
          <Txt size={22} color={C.ink2} style={{marginTop: 6}}>同一句话里的词，会去不同的科室</Txt>
          <div style={{display: 'flex', gap: 12, marginTop: 20}}>
            {[['同', 2], ['一', 5], ['句', 1], ['话', 7], ['里', 4]].map(([w, r], k) => <div key={k} style={{display: 'flex', flexDirection: 'column',
              alignItems: 'center', gap: 6, opacity: ease(f, at('l3.a') + 10 + k * 5, 10)}}>
              <div style={{padding: '6px 14px', border: `1px solid ${C.line}`, borderRadius: 8, fontSize: 26}}>{w}</div>
              <Mono size={16} color={ACC}>→ 专家 {r}</Mono></div>)}
          </div>
        </Card>
      </Abs>
      <Abs x={1010} y={290} w={790} style={up(p('l3.b'))}>
        <Card on={f >= at('l3.b')}>
          <Txt size={28}>专科是训练中自己形成的</Txt>
          <Txt size={22} color={C.ink2} style={{marginTop: 6}}>没有人指定哪个专家管什么</Txt>
          <div style={{marginTop: 22, display: 'inline-block', position: 'relative', padding: '6px 18px', border: `1px dashed ${C.ink3}`,
            borderRadius: 8}}>
            <Txt size={26} color={C.ink3}>数学专家</Txt>
            <div style={{position: 'absolute', left: -6, right: -6, top: '50%', height: 2, background: ACC,
              transform: `scaleX(${lin(f, at('l3.b') + 30, 14)})`, transformOrigin: 'left', boxShadow: BOX_GLOW}} />
          </div>
        </Card>
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 03 一层 DeepSeekMoE
const ROUTED = [17, 40, 86, 101, 139, 170, 205, 233];   // 示意: 8 of 256
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const stage = f >= at('l2') ? (f >= at('l2.b') ? 'g' : 's2') : f >= at('l1.d') ? 'r' : f >= at('l1.c') ? 's' : f >= at('l1.b') ? 'u' : '';
  const parts: Record<string, any> = {u: stage === 'u' ? lit(1) : {}, s: stage === 's' || stage === 's2' ? lit(1) : {},
    r: stage === 'r' ? lit(1) : {}, g: stage === 'g' ? lit(1) : {}};
  const UX = 200, UY = 470, GX = 900, GY = 420, CELL = 15, GAP = 3;
  const gx = (i: number) => GX + (i % 16) * (CELL + GAP), gy = (i: number) => GY + Math.floor(i / 16) * (CELL + GAP);
  const SX = 1640, SY = 560;
  const pu = ease(f, at('l1.b'), 12), ps = ease(f, at('l1.c'), 12), pr = ease(f, at('l1.d'), 12), pg = ease(f, at('l2.b'), 12);
  const hot = (k: string) => (stage === k ? ACC : C.ink3);
  return <AbsoluteFill>
    <Abs x={0} y={170} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}><Tex id="moe" size={50} parts={parts} /></Abs>
    <Svg>
      {/* u_t */}
      {Array.from({length: 6}, (_, k) => <rect key={k} x={UX} y={UY + k * 30} width={40} height={24} rx={4} opacity={ease(f, at('l1.a'), 12)}
        fill={stage === 'u' ? rgba(ACC, 0.4) : C.panel} stroke={stage === 'u' ? ACC : C.line} />)}
      <text x={UX + 20} y={UY - 16} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={C.ink} opacity={ease(f, at('l1.a'), 12)}>u_t</text>
      {/* skip path */}
      <path d={`M${UX + 40} ${UY + 10} L${UX + 140} ${UY + 10} L${UX + 140} 380 L${SX} 380 L${SX} ${SY - 22}`} fill="none" stroke={hot('u')}
        strokeWidth={stage === 'u' ? 2.4 : 1.4} opacity={pu} filter={stage === 'u' ? 'url(#vglow)' : undefined} />
      {/* shared expert */}
      <rect x={520} y={470} width={260} height={80} rx={10} fill={stage === 's' || stage === 's2' ? rgba(ACC, 0.18) : C.panel}
        stroke={stage === 's' || stage === 's2' ? ACC : C.line} opacity={ps} />
      <text x={650} y={518} textAnchor="middle" fontFamily={F.zh} fontSize={24} fill={C.ink} opacity={ps}>共享专家 × N_s</text>
      <path d={`M${UX + 40} ${UY + 75} L520 510`} fill="none" stroke={hot('s') === ACC || stage === 's2' ? ACC : C.ink3} strokeWidth={1.4} opacity={ps} />
      <path d={`M780 510 L${SX - 22} ${SY}`} fill="none" stroke={stage === 's' || stage === 's2' ? ACC : C.ink3} strokeWidth={1.4} opacity={ps} />
      {/* routed experts: 16 × 16 */}
      {Array.from({length: 256}, (_, i) => {
        const on = ROUTED.includes(i) && pg > 0;
        return <rect key={i} x={gx(i)} y={gy(i)} width={CELL} height={CELL} rx={2} opacity={pr * (on ? 1 : stage === 'g' ? 0.5 : 1)}
          fill={on ? rgba(ACC, 0.7) : stage === 'r' ? rgba(ACC, 0.12) : C.panel} stroke={on ? ACC : C.line} strokeWidth={0.8} />;
      })}
      <text x={GX + 8 * (CELL + GAP)} y={GY - 16} textAnchor="middle" fontFamily={F.zh} fontSize={22} fill={C.ink} opacity={pr}>
        路由专家 × N_r</text>
      <path d={`M${UX + 40} ${UY + 140} L${GX - 10} ${GY + 8 * (CELL + GAP)}`} fill="none" stroke={stage === 'r' || stage === 'g' ? ACC : C.ink3}
        strokeWidth={1.4} opacity={pr} />
      {pg > 0 && ROUTED.map((i) => <line key={i} x1={gx(i) + CELL / 2} y1={gy(i) + CELL / 2} x2={SX - 22} y2={SY + 8} stroke={ACC}
        strokeWidth={1} opacity={0.55 * pg} />)}
      {/* sum */}
      <circle cx={SX} cy={SY} r={22} fill={C.bg} stroke={C.ink} strokeWidth={1.6} opacity={pu} />
      <text x={SX} y={SY + 9} textAnchor="middle" fontFamily={F.mono} fontSize={28} fill={C.ink} opacity={pu}>+</text>
      <line x1={SX + 22} y1={SY} x2={SX + 90} y2={SY} stroke={C.ink} strokeWidth={1.6} opacity={pu} />
      <text x={SX + 100} y={SY + 8} fontFamily={F.mono} fontSize={24} fill={C.ink} opacity={pu}>h′_t</text>
    </Svg>
    <Abs x={520} y={580} w={300} style={up(ease(f, at('l2.a'), 12))}><Txt size={22} color={ACC}>每个 token 必经</Txt></Abs>
    <Abs x={GX} y={GY + 16 * (CELL + GAP) + 14} w={420} style={up(pg)}>
      <Txt size={22} color={ACC}>只有 K_r 个门控值 g 不为 0</Txt></Abs>
    <Abs x={0} y={800} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l2.c'))}}>
      <Chip on size={28}>V3：N_s = 1，N_r = 256，K_r = 8</Chip></Abs>
    <Note x={GX} y={GY + 16 * (CELL + GAP) + 50} p={pg}>示意：亮起的 8 个位置是随手选的</Note>
  </AbsoluteFill>;
};

// ================================================================ 04 选择用 s+b，加权用 s
// 示意 values: 12 experts choose 4. By s alone the top 4 would be e3, e7, e9, e1; e3 is overloaded (b < 0), so s+b
// picks e9, e7, e1, e5. Weights use s only: 0.65, 0.70, 0.60, 0.55 over 2.50.
const S = [0.60, 0.35, 0.81, 0.44, 0.55, 0.29, 0.70, 0.40, 0.65, 0.52, 0.33, 0.47];
const Bv = [0, 0.05, -0.30, 0, 0.02, 0, -0.03, 0.08, 0.10, 0, 0, 0.06];
const PICK = [8, 6, 0, 4];
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const BX = 170, BASE = 760, SC = 520, BW = 52, BG = 20;
  const bx = (i: number) => BX + i * (BW + BG);
  const grow = ease(f, at('l1.a'), 20), bias = ease(f, at('l2.a'), 16), pick = ease(f, at('l2.a') + 30, 12), wt = ease(f, at('l2.b'), 14);
  const sum = PICK.reduce((a, i) => a + S[i], 0);
  return <AbsoluteFill>
    <Svg>
      <line x1={BX - 20} x2={bx(11) + BW + 20} y1={BASE} y2={BASE} stroke={C.ink3} />
      {S.map((s, i) => {
        const h = SC * s * grow, hb = SC * Bv[i] * bias;
        const chosen = PICK.includes(i) && pick > 0;
        return <g key={i}>
          <rect x={bx(i)} y={BASE - h} width={BW} height={h} rx={3} fill={chosen ? rgba(ACC, 0.55) : rgba(C.ink2, 0.35)}
            stroke={chosen ? ACC : 'none'} strokeWidth={1.6} filter={chosen ? 'url(#vglow)' : undefined} />
          {Bv[i] > 0 && <rect x={bx(i)} y={BASE - h - hb} width={BW} height={hb} fill="url(#vhatch)" stroke={rgba(ACC, 0.8)} strokeWidth={1} />}
          {Bv[i] < 0 && <rect x={bx(i)} y={BASE - h} width={BW} height={-hb} fill={rgba(C.bg, 0.7)} stroke={rgba(ACC, 0.8)} strokeWidth={1}
            strokeDasharray="4 3" />}
          <text x={bx(i) + BW / 2} y={BASE + 30} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={chosen ? ACC : C.ink3}>e{i + 1}</text>
          {chosen && wt > 0 && <text x={bx(i) + BW / 2} y={BASE - h - Math.max(0, hb) - 14} textAnchor="middle" fontFamily={F.mono}
            fontSize={20} fill={ACC} opacity={wt}>{(s / sum).toFixed(2)}</text>}
        </g>;
      })}
      {pick > 0 && <g opacity={pick}>
        <text x={bx(2) + BW / 2} y={BASE - SC * S[2] - 40} textAnchor="middle" fontFamily={F.zh} fontSize={20} fill={C.ink2}>过载：b &lt; 0</text>
        <text x={bx(2) + BW / 2} y={BASE - SC * S[2] - 16} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={C.ink2}>✕</text>
      </g>}
    </Svg>
    <Abs x={BX} y={170} style={up(p('l1.a'))}><Txt size={24} color={C.ink2}>亲和度 s（灰柱）<span style={{opacity: bias}}>，偏置 b（斜纹 / 虚框）</span></Txt></Abs>
    <Note x={BX} y={BASE + 56} p={grow}>示意：12 个专家选 4 个，数值是编的</Note>
    <Abs x={1100} y={200} w={720} style={{display: 'flex', flexDirection: 'column', gap: 26}}>
      <div style={up(p('l1.a'))}><Tex id="aff" size={40} parts={{d: f >= at('l1.a') + 20 && f < at('l2') ? lit(1) : {}}} /></div>
      <div style={up(p('l2.a'))}><Tex id="sel" size={40} parts={{sb: f >= at('l2.a') && f < at('l2.b') ? lit(1) : {}}} />
        <Txt size={22} color={C.ink2} style={{marginTop: 6}}>选谁：看 s + b 的前 K_r 名</Txt></div>
      <div style={up(p('l2.b'))}><Tex id="gate" size={40} parts={{s: f >= at('l2.b') && f < at('l3') ? lit(1) : {}}} />
        <Txt size={22} color={C.ink2} style={{marginTop: 6}}>加权：只用 s，再归一化</Txt></div>
      <div style={{display: 'flex', gap: 30, alignItems: 'center', ...up(p('l3.a'))}}>
        <div><Txt size={20} color={C.ink2}>过载</Txt><Tex id="over" size={34} /></div>
        <div><Txt size={20} color={C.ink2}>欠载</Txt><Tex id="under" size={34} /></div>
        <Mono size={24} color={ACC}>γ = 0.001</Mono>
      </div>
      <div style={up(p('l3.b'))}><Chip on size={24}>b 只管选谁：不进门控值，不产生梯度</Chip></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 改进链
const CHAIN: [string, string, string, string, boolean, number?][] = [   // node, title, gist, mark, above, card shift
  ['sgmoe', '2017 · 稀疏门控 MoE', 'top-k 门控，专家放大到几千个；辅助损失防坍缩', 'l1.a', true, -130],
  ['gshard', '2020 · GShard', '每个 token 去 2 个专家', 'l1.b', true, 110],
  ['switch', '2021 · Switch', '每个 token 去 1 个专家', 'l1.b', false],
  ['dsmoe', '2024-01 · DeepSeekMoE', '细粒度切分 + 共享专家隔离', 'l2.a', true],
  ['devroute', '2024-05 · V2', '2 + 160；至多 3 台设备；三种均衡损失', 'l2.b', false],
  ['alf', '2024-08 · 无辅助损失均衡', '用偏置代替损失', 'l3.a', true],
  ['v3moe', '2024-12 · V3', '1 + 256，激活 8；至多 4 个节点；不丢 token', 'l3.b', false],
  ['v4moe', '2026-04 · V4', '取消节点限制；前 3 个 MoE 层哈希路由', 'l4.a', true],
  ['v41moe', '2026-09 · V4.1-Flash', '1 + 384；按图注和配置，没有哈希路由；另加 196B Engram 表', 'l4.b', false],
];
export const B05: React.FC = () => {
  const {f, at} = useBoard();
  const rm = useRm();
  const B = laneRect(rm, 'B');
  const LANE = fit({x: -60, y: B.y - 150, w: 2620, h: B.h + 300}, 0.03);
  const W = 1450, Y = B.y + B.h / 2;
  const v: View = camAt(f, LANE, [
    {at: at('l1') + 4, dur: 34, to: {x: 230, y: Y, w: W}}, {at: at('l2') - 6, dur: 40, to: {x: 900, y: Y, w: W}},
    {at: at('l3') - 6, dur: 40, to: {x: 1310, y: Y, w: W}}, {at: at('l4') - 6, dur: 48, to: {x: 2290, y: Y, w: W}},
    {at: at('l4.c') + 40, dur: 50, to: LANE},
  ]);
  const cam = project(v);
  const overview = ease(f, at('l4.c') + 50, 20);
  // a stop's card leaves when the narration moves on to the next line (the camera moves with it)
  const gone = (key: string) => {
    const next = `l${Number(key.split('.')[0].slice(1)) + 1}`;
    let t = Infinity;
    try { t = at(next); } catch { /* the last line: the overview takes the cards away */ }
    return Number.isFinite(t) ? ease(f, t - 8, 10) : overview;
  };
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{nodeIn: (n) => (n.lane === 'B' ? 1 : 0.12), laneIn: (k) => (k === 'B' ? 1 : 0.25), axisIn: 0.5,
      laneLit: (k) => (k === 'B' ? 0.6 : 0),
      nodeLit: (n) => {
        const c = CHAIN.find(([id]) => id === n.id);
        return c ? ease(f, at(c[3]), 10) : 0;
      }}}>
      {CHAIN.map(([id, , , key, above, dx = 0]) => {
        const n = nodeOf(rm, id);
        const x = cam.X(n.x + n.w / 2), y0 = above ? cam.Y(n.y) - 4 : cam.Y(n.y + n.h) + 4, y1 = above ? y0 - 60 : y0 + 60;
        const k = lin(f, at(key), 10) * (1 - gone(key));
        return k > 0 ? <path key={id} d={`M${x} ${y0} L${x} ${(y0 + y1) / 2} L${x + dx} ${(y0 + y1) / 2} L${x + dx} ${y1}`} fill="none"
          stroke={ACC} strokeWidth={1.4} pathLength={1} strokeDasharray={`${k} 1`} /> : null;
      })}
    </TreeScene>
    {CHAIN.map(([id, title, gist, key, above, dx = 0]) => {
      const n = nodeOf(rm, id);
      const x = cam.X(n.x + n.w / 2) + dx, y = above ? cam.Y(n.y) - 64 : cam.Y(n.y + n.h) + 64;
      const k = ease(f, at(key) + 6, 12) * (1 - gone(key));
      if (k <= 0 || x < -300 || x > 2220) return null;
      return <Abs key={id} x={x - 200} y={y} w={400} style={{opacity: k, transform: above ? 'translateY(-100%)' : undefined}}>
        <div style={{padding: '12px 18px', borderRadius: 10, background: rgba(C.bg, 0.94), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Mono size={20} color={ACC}>{title}</Mono>
          <Txt size={21} style={{marginTop: 4, lineHeight: 1.3}}>{gist}</Txt>
        </div>
      </Abs>;
    })}
  </AbsoluteFill>;
};

// ================================================================ 06 组合数与激活比例
const PTS: [string, number, number, string, string, string | null][] = [
  // label, total (B), activated (B), ratio, mark, formula
  ['DeepSeek LLM 7B', 7, 7, '稠密', 'l2', null], ['DeepSeek LLM 67B', 67, 67, '稠密', 'l2', null],
  ['DeepSeekMoE 16B', 16.4, 2.8, '约 1/6', 'l2.a', 'r16'], ['V2', 236, 21, '约 1/11', 'l2.b', 'rv2'],
  ['V3', 671, 37, '约 1/18', 'l2.c', 'rv3'], ['V4-Pro', 1600, 49, '约 1/33', 'l2.d', 'rpro'],
  ['V4.1-Flash 解码', 552, 16, '2.9%', 'l3.a', 'rdec'], ['V4.1-Flash 预填充', 552, 8, '约 1/69', 'l3.b', 'rpre'],
];
export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const one = 1 - ease(f, at('l2'), 14), two = ease(f, at('l2'), 14);
  // combinatorics: 16 experts choose 2, then each split in 4: 64 choose 8
  const cut = swing(f, at('l1.b'), 24);
  const c16 = [2, 9], c64 = [3, 12, 22, 29, 38, 45, 51, 60];
  const count = Math.round(4426165368 * Math.pow(lin(f, at('l1.b') + 30, 40), 3));
  // log-log chart
  const X0 = 240, X1 = 1300, YB = 820, YT = 230, LX0 = 0.5, LX1 = 3.4, LY0 = 0, LY1 = 2;
  const X = (v: number) => X0 + ((Math.log10(v) - LX0) / (LX1 - LX0)) * (X1 - X0);
  const Y = (v: number) => YB - ((Math.log10(v) - LY0) / (LY1 - LY0)) * (YB - YT);
  const cur = [...PTS].reverse().find((q) => q[5] && f >= at(q[4]));
  const line = (r: number) => {
    const xa = Math.max(LX0, LY0 - Math.log10(r)), xb = Math.min(LX1, LY1 - Math.log10(r));
    return {x0: X(10 ** xa), y0: Y(10 ** xa * r), x1: X(10 ** xb), y1: Y(10 ** xb * r)};
  };
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: one}}>
      <Abs x={160} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>细粒度切分换来了什么？</Txt></Abs>
      <Abs x={160} y={250} style={{display: 'grid', gridTemplateColumns: 'repeat(4, 64px)', gap: 10, opacity: (1 - cut) * ease(f, at('l1.a'), 12)}}>
        {Array.from({length: 16}, (_, i) => <div key={i} style={{width: 64, height: 64, borderRadius: 6,
          background: c16.includes(i) ? rgba(ACC, 0.45) : C.panel, border: `1px solid ${c16.includes(i) ? ACC : C.line}`}} />)}
      </Abs>
      <Abs x={160} y={250} style={{display: 'grid', gridTemplateColumns: 'repeat(8, 30px)', gap: 7, opacity: cut}}>
        {Array.from({length: 64}, (_, i) => <div key={i} style={{width: 30, height: 30, borderRadius: 4,
          background: c64.includes(i) && f >= at('l1.b') + 26 ? rgba(ACC, 0.5) : C.panel,
          border: `1px solid ${c64.includes(i) && f >= at('l1.b') + 26 ? ACC : C.line}`}} />)}
      </Abs>
      <Abs x={640} y={260} style={{display: 'flex', flexDirection: 'column', gap: 40}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 30, ...up(p('l1.a'))}}>
          <Txt size={24} color={C.ink2} style={{width: 260}}>16 个专家选 2 个</Txt><Tex id="c16" size={44} parts={{r: lit(1)}} /></div>
        <div style={{display: 'flex', alignItems: 'center', gap: 30, ...up(p('l1.b'))}}>
          <Txt size={24} color={C.ink2} style={{width: 260}}>每个切成 4 份：64 选 8</Txt>
          <Tex id="c64" size={44} parts={{r: f >= at('l1.b') + 70 ? lit(1) : dim(0)}} />
        </div>
        <Mono size={56} color={ACC} glow style={{opacity: f >= at('l1.b') + 30 && f < at('l1.b') + 72 ? 1 : 0, marginTop: -110,
          marginLeft: 330}}>{count.toLocaleString('en-US')}</Mono>
      </Abs>
      <Abs x={160} y={640} style={up(p('l1.c'))}><Chip on size={28}>总参数、每个 token 的计算量：都没变</Chip></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: two}}>
      <Abs x={X0} y={160} style={up(two)}><Txt size={26} color={C.ink2}>激活比例：每个 token 用到总参数的几分之一</Txt></Abs>
      <Svg>
        {[10, 100, 1000].map((t) => <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={YT} y2={YB} stroke={C.rule} />
          <text x={X(t)} y={YB + 28} textAnchor="middle" fontFamily={F.mono} fontSize={17} fill={C.ink3}>{t >= 1000 ? '1T' : `${t}B`}</text>
        </g>)}
        {[1, 10, 100].map((t) => <g key={t}>
          <line x1={X0} x2={X1} y1={Y(t)} y2={Y(t)} stroke={C.rule} />
          <text x={X0 - 12} y={Y(t) + 6} textAnchor="end" fontFamily={F.mono} fontSize={17} fill={C.ink3}>{t}B</text>
        </g>)}
        <text x={X1} y={YB + 58} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={C.ink3}>总参数 →</text>
        <text x={X0} y={YT - 18} fontFamily={F.zh} fontSize={20} fill={C.ink3}>每个 token 的激活参数 ↑</text>
        {[[1, '1'], [0.1, '1/10'], [0.01, '1/100']].map(([r, t]) => {
          const L = line(r as number);
          return <g key={t as string}>
            <line x1={L.x0} y1={L.y0} x2={L.x1} y2={L.y1} stroke={rgba(C.ink2, 0.35)} strokeDasharray="6 6" />
            <text x={L.x1 + 8} y={L.y1 + 6} fontFamily={F.mono} fontSize={17} fill={C.ink3}>{t as string}</text>
          </g>;
        })}
        {(() => {
          const k = ease(f, at('l3.b'), 12);
          return k > 0 ? <line x1={X(552)} x2={X(552)} y1={Y(16)} y2={Y(16) + (Y(8) - Y(16)) * k} stroke={ACC} strokeWidth={2.2} /> : null;
        })()}
        {PTS.map(([label, tot, act, ratio, key, fm]) => {
          const k = ease(f, at(key) + 4, 12);
          if (k <= 0) return null;
          const on = !!cur && cur[0] === label;
          const dense = fm === null;
          const right = !label.includes('解码');
          return <g key={label} opacity={k}>
            <circle cx={X(tot)} cy={Y(act)} r={on ? 10 : 8} fill={dense ? 'none' : on ? ACC : rgba(ACC, 0.55)} stroke={dense ? C.ink2 : ACC}
              strokeWidth={1.6} filter={on ? 'url(#vglow)' : undefined} />
            <text x={X(tot) + (right ? 16 : -16)} y={Y(act) + (label.includes('预填充') ? 22 : -10)} textAnchor={right ? 'start' : 'end'}
              fontFamily={F.zh} fontSize={19} fill={on ? C.ink : C.ink2}>{label} · {ratio}</text>
          </g>;
        })}
      </Svg>
      <Abs x={1500} y={300} w={360} style={{display: 'flex', flexDirection: 'column', gap: 24}}>
        {PTS.filter((q) => q[5]).map(([label, , , , key, fm]) => {
          const on = !!cur && cur[0] === label;
          return <div key={label} style={{opacity: ease(f, at(key) + 6, 12) * (on ? 1 : 0.4)}}>
            <Tex id={fm as string} size={on ? 34 : 28} color={on ? ACC : C.ink2} /></div>;
        })}
      </Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 07 代价
const LOADS = [0.4, 0.7, 0.2, 0.5, 0.3, 0.9, 0.1, 0.6, 0.4, 0.2, 0.8, 0.3, 0.5, 0.1, 0.6, 17.8, 0.3, 0.4, 0.7, 0.2, 0.5, 0.6, 0.1, 0.3,
  0.8, 0.4, 0.2, 0.6, 0.3, 0.5, 0.4, 0.7];                                     // 示意; index 15 is replaced by 17.8 × the mean
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const ph = (a: string, b?: string) => ease(f, at(a), 14) * (b ? 1 - ease(f, at(b), 14) : 1);
  // the busiest load M solves M = 17.8 × (S + M) / 32, where S is the sum of the other 31
  const S = LOADS.reduce((a, b, i) => a + (i === 15 ? 0 : b), 0);
  const M = (17.8 * S) / (32 - 17.8), mean = (S + M) / 32;
  const L = LOADS.map((v, i) => (i === 15 ? M : v));
  const unit = 380 / M;
  const GPUL = [0.55, 0.62, 0.48, 0.95, 0.58, 0.51, 0.66, 0.6];                   // 示意
  return <AbsoluteFill>
    {/* memory: storage is the total, compute is the activated part */}
    <AbsoluteFill style={{opacity: ph('l1', 'l2')}}>
      <Abs x={160} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>省的是计算，显存一点没省</Txt></Abs>
      {[['V3', 671, 37, '671B', '37B'], ['V4-Pro', 1600, 49, '1.6T', '49B']].map(([name, tot, act, ts, as], k) =>
        <Abs key={name as string} x={160} y={290 + k * 230} w={1640} style={up(ease(f, at('l1.a') + k * 10, 14))}>
          <Mono size={26} color={C.ink}>{name as string}</Mono>
          <div style={{display: 'flex', alignItems: 'center', gap: 16, marginTop: 14}}>
            <Txt size={20} color={C.ink2} style={{width: 170}}>每个 token 计算</Txt>
            <div style={{width: (1300 * (act as number)) / 1600, height: 30, borderRadius: 3, background: ACC, boxShadow: BOX_GLOW}} />
            <Mono size={22} color={ACC}>{as as string}</Mono></div>
          <div style={{display: 'flex', alignItems: 'center', gap: 16, marginTop: 10}}>
            <Txt size={20} color={C.ink2} style={{width: 170}}>显存里要放</Txt>
            <div style={{width: (1300 * (tot as number)) / 1600 * lin(f, at('l1.a') + 10 + k * 10, 24), height: 30, borderRadius: 3,
              background: `repeating-linear-gradient(45deg, ${rgba(C.ink2, 0.3)} 0px, ${rgba(C.ink2, 0.3)} 6px, transparent 6px, transparent 13px)`,
              border: `1px solid ${C.ink2}`}} />
            <Mono size={22} color={C.ink}>{ts as string}</Mono></div>
        </Abs>)}
    </AbsoluteFill>
    {/* communication: all-to-all and the slowest card */}
    <AbsoluteFill style={{opacity: ph('l2', 'l3')}}>
      <Abs x={160} y={170} style={up(p('l2'))}><Txt size={28} color={C.ink2}>专家分散在许多卡上</Txt></Abs>
      <Svg>
        {Array.from({length: 8}, (_, i) => Array.from({length: 8}, (_, j) => j > i ? <path key={`a${i}-${j}`}
          d={`M${240 + i * 205} 330 Q${240 + (i + j) * 102.5} ${330 - 24 * (j - i)} ${240 + j * 205} 330`} fill="none"
          stroke={rgba(ACC, 0.3)} strokeWidth={1} opacity={lin(f, at('l2.a') + (i + j) * 2, 12)} /> : null))}
        {Array.from({length: 8}, (_, i) => <g key={i}>
          <rect x={160 + i * 205} y={330} width={160} height={90} rx={10} fill={C.panel} stroke={i === 3 && f >= at('l2.b') ? ACC : C.line}
            strokeWidth={i === 3 && f >= at('l2.b') ? 2 : 1} />
          <text x={240 + i * 205} y={384} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink2}>GPU {i}</text>
          <rect x={200 + i * 205} y={760 - 300 * GPUL[i] * ease(f, at('l2.b'), 16)} width={80} height={300 * GPUL[i] * ease(f, at('l2.b'), 16)} rx={3}
            fill={i === 3 ? ACC : rgba(C.ink2, 0.35)} filter={i === 3 ? 'url(#vglow)' : undefined} />
        </g>)}
        {f >= at('l2.b') && <g opacity={lin(f, at('l2.b') + 16, 12)}>
          <line x1={160} x2={1760} y1={760 - 300 * 0.95} y2={760 - 300 * 0.95} stroke={ACC} strokeDasharray="6 6" />
          <text x={1760} y={760 - 300 * 0.95 - 12} textAnchor="end" fontFamily={F.zh} fontSize={22} fill={ACC}>整一步的时间 = 最忙的那张卡</text>
        </g>}
        <line x1={160} x2={1760} y1={760} y2={760} stroke={C.ink3} />
      </Svg>
      <Abs x={160} y={440} style={up(p('l2.a'))}><Txt size={24}>每个 token 要去的设备越多，all-to-all 通信越贵</Txt></Abs>
      <Note x={160} y={780} p={ease(f, at('l2.b'), 12)}>示意：每张卡这一步的负载</Note>
    </AbsoluteFill>
    {/* routing collapse */}
    <AbsoluteFill style={{opacity: ph('l3', 'l4')}}>
      <Abs x={160} y={170} style={up(p('l3'))}><Txt size={28} color={C.ink2}>路由坍缩：被偏爱的专家学得更快，被选得更多</Txt></Abs>
      <Svg>
        {L.map((v, i) => {
          const h = v * unit * lin(f, at('l3') + 10 + (i === 15 ? 20 : 0), i === 15 ? 50 : 20);
          return <rect key={i} x={200 + i * 48} y={800 - h} width={34} height={h} rx={3} fill={i === 15 ? ACC : rgba(C.ink2, 0.35)}
            filter={i === 15 ? 'url(#vglow)' : undefined} />;
        })}
        <line x1={180} x2={1760} y1={800 - mean * unit} y2={800 - mean * unit} stroke={C.ink} strokeDasharray="5 5" opacity={ease(f, at('l3.b'), 12)} />
        <text x={1760} y={800 - mean * unit - 10} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={C.ink} opacity={ease(f, at('l3.b'), 12)}>平均</text>
        <text x={200 + 15 * 48 + 17} y={800 - M * unit - 16} textAnchor="middle" fontFamily={F.mono} fontSize={30} fill={ACC}
          opacity={ease(f, at('l3.b'), 12)}>17.8×</text>
        <line x1={180} x2={1760} y1={800} y2={800} stroke={C.ink3} />
      </Svg>
      <Note x={200} y={820} p={ease(f, at('l3'), 12)}>不加约束时；其余各柱为示意</Note>
    </AbsoluteFill>
    {/* the auxiliary-loss dilemma */}
    <AbsoluteFill style={{opacity: ph('l4')}}>
      <Abs x={160} y={170} style={up(p('l4'))}><Txt size={28} color={C.ink2}>辅助损失的两难</Txt></Abs>
      <Svg>
        <line x1={360} x2={1560} y1={380} y2={380} stroke={C.ink2} strokeWidth={2} opacity={ease(f, at('l4.a'), 12)} />
        <circle cx={960 + 520 * Math.sin((f - at('l4.a')) / 22)} cy={380} r={16} fill={ACC} filter="url(#vglow)" opacity={ease(f, at('l4.a'), 12)} />
        <text x={360} y={440} fontFamily={F.zh} fontSize={26} fill={C.ink} opacity={ease(f, at('l4.a'), 12)}>系数小 → 不均衡</text>
        <text x={1560} y={440} textAnchor="end" fontFamily={F.zh} fontSize={26} fill={C.ink} opacity={ease(f, at('l4.a'), 12)}>系数大 → 伤性能</text>
      </Svg>
      <Abs x={360} y={530} style={up(p('l4.b'))}><Chip on size={28}>1B 实验：偏置法均衡更好，困惑度也更低</Chip></Abs>
      <Abs x={360} y={630} style={up(p('l4.c'))}><Chip size={26}>“伤性能是因为干扰梯度”：是解释，不是证明</Chip></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 08 看清口径
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  const bar = (v: number) => (v / 183.5) * 900;
  return <AbsoluteFill>
    <Abs x={160} y={170} w={1600} style={up(p('l1'))}>
      <Card on={f < at('l2')}><Txt size={34} weight={400}>“DeepSeekMoE 16B 只用了 40% 的计算量”</Txt></Card>
    </Abs>
    <Abs x={160} y={330} w={1600} style={up(p('l2.a'))}>
      <Txt size={24} color={C.ink2}>每 4K token 的前向 FLOPs</Txt>
      {[['DeepSeekMoE 16B', 74.4, '74.4T', true], ['DeepSeek LLM 7B（稠密）', 183.5, '183.5T', f >= at('l2.b')]].map(([n, v, s, on]) =>
        <div key={n as string} style={{display: 'flex', alignItems: 'center', gap: 16, marginTop: 14}}>
          <Txt size={22} color={on ? C.ink : C.ink2} style={{width: 300}}>{n as string}</Txt>
          <div style={{width: bar(v as number) * lin(f, at('l2.a') + 6, 20), height: 30, borderRadius: 3,
            background: on ? rgba(ACC, 0.5) : rgba(C.ink2, 0.35)}} />
          <Mono size={24} color={C.ink}>{s as string}</Mono></div>)}
      <div style={{marginTop: 22}}><Tex id="f40" size={40} color={ACC} /></div>
    </Abs>
    <Abs x={160} y={680} w={1600} style={{display: 'flex', gap: 18, flexWrap: 'wrap'}}>
      <div style={up(p('l2.b'))}><Chip size={24}>分母是 DeepSeek LLM 7B，不是激活参数之比</Chip></div>
      <div style={up(p('l2.c'))}><Chip size={24}>激活参数 2.8B 还包含嵌入和输出头</Chip></div>
    </Abs>
    <Abs x={160} y={790} style={up(p('l3'))}><Chip on size={30}>先问：分子是什么？分母是什么？</Chip></Abs>
  </AbsoluteFill>;
};

// ================================================================ 09 三个问题的答案
export const B09: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const out = ease(f, at('l3.a') - 4, 16);
  const Cl = laneRect(rm, 'C');
  const v = camAt(f, FULLV, [{at: at('l3.a'), dur: 60, to: fit({x: -330, y: Cl.y - 200, w: 2880, h: Cl.h + 400}, 0.03)}]);
  const items: [number, string, string, string, number][] = [
    [1, '换成许多专家，每个 token 只经过几个', '总参数和每个 token 的计算，就此拆开', 'l1', at('l1.b')],
    [2, '共享专家 + 路由专家', '三年里改的：门控怎么算、负载怎么均衡、路由受什么限制', 'l2', at('l2.b')],
    [3, '显存和通信', '全部参数都要放下；token 在卡与卡之间来回搬', 'l3', at('l3') + 30],
  ];
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - out}}>
      <Abs x={120} y={190} w={1680} style={{display: 'flex', flexDirection: 'column', gap: 26}}>
        {items.map(([n, head, body, key, check]) => {
          const done = ease(f, check, 12);
          return <div key={n} style={{display: 'flex', alignItems: 'center', gap: 26, padding: '22px 28px', borderRadius: 14,
            background: rgba(C.bg, 0.9), border: `1px solid ${done > 0.5 ? rgba(ACC, 0.7) : C.line}`, boxShadow: done > 0.5 ? BOX_GLOW : undefined,
            ...up(p(key))}}>
            <Mono size={32} color={ACC} glow>Q{n}</Mono>
            <div style={{flexGrow: 1}}><Txt size={32} weight={400}>{head}</Txt><Txt size={24} color={C.ink2}>{body}</Txt></div>
            <div style={{position: 'relative', width: 48, height: 48}}>
              <Svg w={48} h={48}><Check x={2} y={2} size={42} p={lin(f, check, 14)} /></Svg></div>
          </div>;
        })}
      </Abs>
    </AbsoluteFill>
    {out > 0 && <AbsoluteFill style={{opacity: out}}>
      <TreeScene cam={project(v)} look={{laneLit: (k) => (k === 'C' ? ease(f, at('l3.a') + 30, 12) : 0)}} />
      <Abs x={0} y={800} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 主线 C：用更短的数、更聪明的调度，把算力和显存省下来</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};

