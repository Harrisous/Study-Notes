// 第 0 课 §5 · 主线 D：训练目标与后训练 —— 9 段黑板。数字全部来自 lesson-00 §5（研究笔记 lesson-00-doc.md）；
// 标"示意"的是：01 段的句子和概率、03 段的六份答卷分数、08 段的两根趋势条。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Check, Chip, F, Mono, Roadmap, Svg, Tex, Txt, View, camAt, ease, fit, lin, lit, mix, project, rgba,
  swing, up, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, TreeScene, laneRect, nodeOf} from '../l00-shared/scene';

const useRm = () => useEp().roadmap as Roadmap;
const Card: React.FC<{on?: boolean; dashed?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, dashed, style, children}) =>
  <div style={{padding: '18px 24px', borderRadius: 12, background: rgba(C.bg, 0.92),
    border: `1px ${dashed ? 'dashed' : 'solid'} ${on ? rgba(ACC, 0.7) : C.line}`, boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;
const Note: React.FC<{x: number; y: number; p?: number; children?: React.ReactNode}> = ({x, y, p = 1, children}) =>
  <Abs x={x} y={y} style={{opacity: p}}><Mono size={16} color={C.ink3} style={{letterSpacing: '0.08em'}}>{children}</Mono></Abs>;
const Box: React.FC<{x: number; y: number; w: number; h: number; on?: boolean; p?: number; children?: React.ReactNode}> = ({x, y, w, h, on, p = 1, children}) =>
  <Abs x={x} y={y} w={w} h={h} style={{borderRadius: 12, border: `1px solid ${on ? ACC : C.line}`, background: on ? rgba(ACC, 0.08) : C.panel,
    boxShadow: on ? BOX_GLOW : undefined, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', ...up(p)}}>
    {children}</Abs>;

// ================================================================ 01 像语料，不等于做得对
const CTX = ['猫', '坐', '在', '垫'];                     // 示意
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const rlPhase = ease(f, at('l3'), 14);
  const train = swing(f, at('l1.a') + 10, 60);
  const prob = mix(0.12, 0.81, train);
  const loop = (f - at('l3.a')) / 50;
  const leash = ease(f, at('l3.b'), 14);
  const drift = Math.sin(f / 18) * 60 * leash;
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - rlPhase}}>
      <Abs x={0} y={150} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}>
        <Tex id="ntp" size={48} parts={{p: f >= at('l1.a') && f < at('l2') ? lit(1) : {}}} /></Abs>
      <Abs x={260} y={320} style={{display: 'flex', gap: 14, alignItems: 'center', ...up(p('l1.a'))}}>
        {CTX.map((t) => <div key={t} style={{width: 76, height: 76, borderRadius: 10, border: `1px solid ${C.line}`, background: C.panel,
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 38}}>{t}</div>)}
        <Mono size={34} color={C.ink2}>→</Mono>
        <div style={{width: 76, height: 76, borderRadius: 10, border: `1px solid ${ACC}`, background: rgba(ACC, 0.12), display: 'flex',
          alignItems: 'center', justifyContent: 'center', fontSize: 38, color: ACC, boxShadow: BOX_GLOW}}>子</div>
      </Abs>
      <Abs x={1000} y={320} w={700} style={up(p('l1.a'))}>
        <Txt size={22} color={C.ink2}>真实的下一个 token“子”的概率</Txt>
        <div style={{display: 'flex', alignItems: 'center', gap: 16, marginTop: 10}}>
          <div style={{width: 520 * prob, height: 34, borderRadius: 4, background: ACC, boxShadow: BOX_GLOW}} />
          <Mono size={26} color={ACC}>{prob.toFixed(2)}</Mono>
        </div>
        <Txt size={20} color={C.ink3} style={{marginTop: 8}}>训练：在语料的每个位置上，把它推高</Txt>
      </Abs>
      <Note x={260} y={420} p={ease(f, at('l1.a'), 12)}>示意：一句话和一个概率</Note>
      <Abs x={260} y={500} style={{display: 'flex', gap: 22, ...up(p('l1.b'))}}>
        <Chip on size={26}>像语料 ✓</Chip><Chip size={26}>对？</Chip><Chip size={26}>有用？</Chip></Abs>
      <Abs x={260} y={620} w={1400} style={{display: 'flex', gap: 30, ...up(p('l2'))}}>
        {[['回答 A', '帮上忙了', true], ['回答 B', '题做错了', false]].map(([n, v, ok], i) => <Card key={n as string} style={{width: 420}}>
          <Mono size={20} color={C.ink3}>{n as string}</Mono>
          <div style={{height: 8, width: 300, background: rgba(C.ink2, 0.25), borderRadius: 4, marginTop: 12}} />
          <div style={{height: 8, width: 240, background: rgba(C.ink2, 0.25), borderRadius: 4, marginTop: 10}} />
          <Txt size={24} color={ok ? ACC : C.ink} style={{marginTop: 12, opacity: ease(f, at('l2') + 20 + i * 10, 10)}}>
            {ok ? '✓' : '✗'} {v as string}</Txt>
        </Card>)}
        <div style={{alignSelf: 'center', ...up(p('l2.a'))}}><Chip on size={24}>只能事后打分：对参数不可导</Chip></div>
      </Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: rlPhase}}>
      <Abs x={0} y={150} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(rlPhase)}}>
        <Tex id="rl" size={44} parts={{r: f >= at('l3.a') && f < at('l3.b') ? lit(1) : {}, k: f >= at('l3.b') ? lit(1) : {}}} /></Abs>
      <Svg>
        {/* the loop: write → score → push up */}
        {[[420, '模型写回答'], [960, '奖励打分 r'], [1500, '把高分回答的概率推高']].map(([x, t], i) => {
          const on = f >= at('l3.a') && Math.floor(Math.max(0, loop)) % 3 === i;
          return <g key={t as string} opacity={ease(f, at('l3.a') + i * 6, 12)}>
            <rect x={(x as number) - 190} y={360} width={380} height={90} rx={12} fill={on ? rgba(ACC, 0.18) : C.panel} stroke={on ? ACC : C.line} />
            <text x={x as number} y={414} textAnchor="middle" fontFamily={F.zh} fontSize={28} fill={C.ink}>{t as string}</text>
          </g>;
        })}
        {[[610, 770], [1150, 1310]].map(([a, b]) => <line key={a} x1={a} x2={b} y1={405} y2={405} stroke={C.ink2} strokeWidth={2}
          opacity={ease(f, at('l3.a') + 12, 12)} markerEnd="" />)}
        <path d="M1500 450 Q960 560 420 450" fill="none" stroke={C.ink2} strokeWidth={1.6} strokeDasharray="6 6" opacity={ease(f, at('l3.a') + 20, 12)} />
        {/* the KL leash */}
        <g opacity={leash}>
          <circle cx={700} cy={680} r={44} fill={rgba(C.ink2, 0.2)} stroke={C.ink2} />
          <text x={700} y={690} textAnchor="middle" fontFamily={F.mono} fontSize={24} fill={C.ink}>π_ref</text>
          <circle cx={1060 + drift} cy={680} r={44} fill={rgba(ACC, 0.2)} stroke={ACC} />
          <text x={1060 + drift} y={690} textAnchor="middle" fontFamily={F.mono} fontSize={24} fill={ACC}>π_θ</text>
          <path d={`M744 680 ${Array.from({length: 12}, (_, i) => `L${744 + ((1016 + drift - 744) * (i + 0.5)) / 12} ${680 + (i % 2 ? -14 : 14)}`).join(' ')} L${1016 + drift} 680`}
            fill="none" stroke={ACC} strokeWidth={2} />
          <text x={880 + drift / 2} y={640} textAnchor="middle" fontFamily={F.mono} fontSize={22} fill={ACC}>KL</text>
        </g>
      </Svg>
      <Abs x={1200} y={650} w={600} style={up(p('l3.c'))}><Txt size={24} color={C.ink2}>拴住它：免得跑到奖励打分不可靠的地方</Txt></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 02 四个模型
const MODELS: [string, string, boolean, string][] = [['策略 π_θ', '训练', true, 'l1.a'], ['参考模型 π_ref', '冻结', false, 'l1.b'],
  ['奖励模型', '冻结', false, 'l1.c'], ['价值网络', '训练', true, 'l1.d']];
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const GB = 4.4;    // px per GB
  return <AbsoluteFill>
    <Abs x={160} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>InstructGPT 的 PPO：同时放四个模型</Txt></Abs>
    {MODELS.map(([name, kind, train, key], i) => <Box key={name} x={160 + i * 410} y={250} w={380} h={150} on={train} p={p(key)}>
      <Txt size={30}>{name}</Txt><Mono size={20} color={train ? ACC : C.ink3}>{kind} · 每参数 {train ? 16 : 2} 字节</Mono>
    </Box>)}
    <Abs x={160} y={480} w={1600} style={up(p('l2'))}>
      <Txt size={24} color={C.ink2}>7B 规模的显存估算</Txt>
      {[['SFT', [[16, true]], 112, 'l2.a'], ['PPO', [[16, true], [2, false], [2, false], [16, true]], 252, 'l2.b']].map(([n, segs, tot, key]) =>
        <div key={n as string} style={{display: 'flex', alignItems: 'center', gap: 20, marginTop: 20, ...up(p(key as string))}}>
          <Mono size={26} style={{width: 90}}>{n as string}</Mono>
          <div style={{display: 'flex', gap: 3}}>
            {(segs as [number, boolean][]).map(([b, t], k) => <div key={k} style={{width: 7 * b * GB * lin(f, at(key as string) + 4 + k * 5, 14),
              height: 44, background: t ? rgba(ACC, 0.5) : rgba(C.ink2, 0.4), borderRadius: 3}} />)}
          </div>
          <Mono size={28} color={ACC}>≈ {tot as number} GB</Mono>
        </div>)}
      <Txt size={20} color={C.ink3} style={{marginTop: 18, ...up(p('l2.b'))}}>7B × (16 + 2 + 2 + 16) 字节 = 252 GB；条的顺序同上面四个模型</Txt>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 03 同题多答
const SCORES = [62, 85, 70, 91, 55, 77];                 // 示意
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const mean = SCORES.reduce((a, b) => a + b, 0) / SCORES.length;
  const sd = Math.sqrt(SCORES.reduce((a, b) => a + (b - mean) ** 2, 0) / SCORES.length);
  const BASE = 700, K = 4.2;
  const allSame = ease(f, at('l2.b'), 14);
  return <AbsoluteFill>
    <Abs x={160} y={160} style={up(p('l1'))}><Card style={{width: 640}}><Txt size={26}>同一道题 · 写 G 份答卷</Txt></Card></Abs>
    <Svg opacity={1 - allSame}>
      {SCORES.map((s, i) => {
        const x = 220 + i * 190, h = s * K * ease(f, at('l1.a') + i * 4, 14);
        const z = (s - mean) / sd;
        return <g key={i}>
          <rect x={x} y={BASE - h} width={110} height={h} rx={4} fill={z > 0 ? rgba(ACC, 0.5) : rgba(C.ink2, 0.35)} />
          <text x={x + 55} y={BASE + 30} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink2}>答卷 {i + 1}</text>
          <text x={x + 55} y={BASE - h - 12} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink}>{s}</text>
          <text x={x + 55} y={BASE + 64} textAnchor="middle" fontFamily={F.mono} fontSize={24} fill={z > 0 ? ACC : C.ink}
            opacity={ease(f, at('l1.b') + i * 4, 12)}>{z > 0 ? '+' : ''}{z.toFixed(2)}σ</text>
        </g>;
      })}
      <line x1={200} x2={1360} y1={BASE - mean * K} y2={BASE - mean * K} stroke={ACC} strokeDasharray="8 6" strokeWidth={2}
        opacity={ease(f, at('l1.b'), 12)} />
      <text x={1370} y={BASE - mean * K + 7} fontFamily={F.zh} fontSize={22} fill={ACC} opacity={ease(f, at('l1.b'), 12)}>
        这组的平均分 {mean.toFixed(1)}</text>
    </Svg>
    <Abs x={1440} y={200} w={380} style={up(p('l1.c'))}>
      <Card dashed><Txt size={24} color={C.ink3} style={{textDecoration: 'line-through'}}>预测分数的老师</Txt>
        <Txt size={20} color={C.ink2}>（价值网络）：不用请</Txt></Card></Abs>
    <Note x={220} y={BASE + 96} p={ease(f, at('l1.a'), 12) * (1 - allSame)}>示意：六份答卷的分数</Note>
    <Abs x={160} y={810} style={up(p('l2.a'))}><Chip size={24}>只在同一道题里比较才有意义</Chip></Abs>
    <AbsoluteFill style={{opacity: allSame}}>
      <Abs x={220} y={330}><Txt size={26} color={C.ink2}>如果一组全对：</Txt></Abs>
      <Abs x={220} y={400} style={{display: 'flex', gap: 30}}>
        {Array.from({length: 6}, (_, i) => <div key={i} style={{width: 150, padding: '18px 0', textAlign: 'center', borderRadius: 10,
          border: `1px solid ${C.line}`, background: C.panel}}><Txt size={30} color={ACC}>✓</Txt><Mono size={22}>Â = 0</Mono></div>)}
      </Abs>
      <Abs x={220} y={580}><Chip on size={28}>优势全是 0：这道题不提供学习信号</Chip></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 04 组内标准化的优势
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const R = [1, 0, 0, 0], A = [1.732, -0.577, -0.577, -0.577];
  const ZERO = 640, U = 120;
  const curve = ease(f, at('l3'), 14);
  const sx = (q: number) => 1040 + q * 700, sy = (v: number) => 760 - v * 110;
  const pts = Array.from({length: 60}, (_, i) => {
    const q = 0.08 + (i / 59) * 0.92;
    return `${i === 0 ? 'M' : 'L'}${sx(q)} ${sy(Math.sqrt((1 - q) / q))}`;
  }).join(' ');
  return <AbsoluteFill>
    <Abs x={160} y={150} style={up(p('l1'))}><Tex id="adv" size={46} parts={{m: f >= at('l1.a') && f < at('l1.b') ? lit(1) : {},
      s: f >= at('l1.a') + 30 && f < at('l1.b') ? lit(1) : {}}} /></Abs>
    <Abs x={900} y={170} w={900} style={up(p('l1.b'))}><Txt size={22} color={C.ink2}>第 i 个回答的每个 token，共用同一个 Â_i</Txt></Abs>
    <Svg>
      <line x1={160} x2={860} y1={ZERO} y2={ZERO} stroke={C.ink3} opacity={ease(f, at('l2'), 12)} />
      {R.map((r, i) => {
        const x = 200 + i * 160;
        const hr = r * U * ease(f, at('l2') + i * 4, 12);
        const ha = A[i] * U * swing(f, at('l2.b') + i * 4, 18);
        return <g key={i}>
          <rect x={x} y={ZERO - hr} width={50} height={hr} rx={3} fill={rgba(C.ink2, 0.4)} />
          <rect x={x + 56} y={ha > 0 ? ZERO - ha : ZERO} width={50} height={Math.abs(ha)} rx={3} fill={A[i] > 0 ? ACC : rgba(ACC, 0.35)}
            filter={A[i] > 0 ? 'url(#vglow)' : undefined} />
          <text x={x + 25} y={ZERO + 26} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink2}
            opacity={ease(f, at('l2'), 12)}>r={r}</text>
          <text x={x + 81} y={A[i] > 0 ? ZERO - ha - 12 : ZERO + Math.abs(ha) + 26} textAnchor="middle" fontFamily={F.mono} fontSize={20}
            fill={ACC} opacity={ease(f, at('l2.b') + 20, 12)}>{A[i].toFixed(3)}</text>
        </g>;
      })}
      <line x1={160} x2={860} y1={ZERO - 0.25 * U} y2={ZERO - 0.25 * U} stroke={C.ink} strokeDasharray="6 5" opacity={ease(f, at('l2.a'), 12)} />
      <text x={870} y={ZERO - 0.25 * U + 6} fontFamily={F.mono} fontSize={18} fill={C.ink} opacity={ease(f, at('l2.a'), 12)}>均值 0.25</text>
      {/* rarer success, larger push */}
      <g opacity={curve}>
        <line x1={sx(0)} x2={sx(1)} y1={sy(0)} y2={sy(0)} stroke={C.ink3} />
        <path d={pts} fill="none" stroke={ACC} strokeWidth={2.4} strokeDasharray={`${3000 * lin(f, at('l3') + 6, 40)} 3000`} />
        <circle cx={sx(0.25)} cy={sy(1.732)} r={9} fill={ACC} filter="url(#vglow)" opacity={ease(f, at('l3') + 40, 10)} />
        <text x={sx(1)} y={sy(0) + 30} textAnchor="end" fontFamily={F.zh} fontSize={18} fill={C.ink3}>这组答对的比例 p →</text>
        <text x={sx(0) + 10} y={sy(3.4)} fontFamily={F.zh} fontSize={18} fill={C.ink3}>答对那份的优势</text>
      </g>
    </Svg>
    <Abs x={160} y={290} style={{display: 'flex', flexDirection: 'column', gap: 12}}>
      <div style={up(p('l2.a'))}><Tex id="std" size={30} /></div>
      <div style={up(p('l2.b'))}><Tex id="exA" size={30} parts={{a: lit(1)}} /></div>
    </Abs>
    <Abs x={1180} y={250} style={up(ease(f, at('l3') + 40, 12))}><Tex id="rare" size={36} parts={{r: lit(1)}} /></Abs>
    <Abs x={160} y={830} style={{display: 'flex', alignItems: 'center', gap: 24, ...up(p('l3.b'))}}>
      <Chip size={22}>V3.2：只减均值，不除标准差</Chip><Tex id="v32" size={32} parts={{r: lit(1)}} /></Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 多看一步
export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const LX0 = 1000, LX1 = 1760, LY = 700, TT = 14.8;     // λ over the 14.8T training tokens
  const tx = (t: number) => LX0 + ((LX1 - LX0) * t) / TT;
  const draw = lin(f, at('l2') + 6, 40);
  const v41 = ease(f, at('l3'), 14);
  return <AbsoluteFill>
    <Abs x={160} y={150} style={up(p('l1'))}><Txt size={26} color={C.ink2}>V3：下一词预测之外，多预测一个 token</Txt></Abs>
    <Abs x={160} y={240} style={{display: 'flex', gap: 10, alignItems: 'center', ...up(p('l1.a'))}}>
      {['t−2', 't−1', 't'].map((t) => <div key={t} style={{width: 90, height: 64, borderRadius: 8, border: `1px solid ${C.line}`,
        background: C.panel, display: 'flex', alignItems: 'center', justifyContent: 'center'}}><Mono size={22}>{t}</Mono></div>)}
      <Mono size={28} color={C.ink2} style={{margin: '0 10px'}}>→</Mono>
      <div style={{width: 110, height: 64, borderRadius: 8, border: `1px solid ${C.ink}`, display: 'flex', alignItems: 'center',
        justifyContent: 'center'}}><Mono size={22}>t+1</Mono></div>
      <div style={{width: 110, height: 64, borderRadius: 8, border: `1px solid ${ACC}`, background: rgba(ACC, 0.12), display: 'flex',
        alignItems: 'center', justifyContent: 'center', boxShadow: BOX_GLOW, opacity: 1 - 0.7 * v41}}><Mono size={22} color={ACC}>t+2</Mono></div>
    </Abs>
    <Abs x={640} y={320} w={500} style={up(p('l1.a'))}><Txt size={20} color={C.ink2}>主模型预测 t+1；MTP 模块再往后看一个</Txt></Abs>
    <Abs x={160} y={400} style={up(p('l1.b'))}><Tex id="mtp" size={50} parts={{l: f >= at('l2') ? lit(1) : {}, d: f >= at('l1.b') && f < at('l2') ? lit(1) : {}}} /></Abs>
    <Abs x={160} y={560} style={{display: 'flex', flexDirection: 'column', gap: 14}}>
      <div style={up(p('l2'))}><Chip size={24}>V3：D = 1</Chip></div>
      <div style={up(p('l2.b'))}><Chip on size={24}>推理：丢掉，或当投机解码的草稿（接受率 85%–90%）</Chip></div>
    </Abs>
    <Svg opacity={ease(f, at('l2'), 12)}>
      <line x1={LX0} x2={LX1} y1={LY} y2={LY} stroke={C.ink3} />
      <line x1={LX0} x2={LX0} y1={LY} y2={LY - 300} stroke={C.ink3} />
      <path d={`M${LX0} ${LY - 0.3 * 800} L${tx(10)} ${LY - 0.3 * 800} L${tx(10)} ${LY - 0.1 * 800} L${tx(TT)} ${LY - 0.1 * 800}`} fill="none"
        stroke={ACC} strokeWidth={3} strokeDasharray={`${2000 * draw} 2000`} filter="url(#vglow)" />
      <text x={LX0 + 10} y={LY - 0.3 * 800 - 14} fontFamily={F.mono} fontSize={22} fill={ACC}>λ = 0.3</text>
      <text x={tx(10) + 10} y={LY - 0.1 * 800 - 14} fontFamily={F.mono} fontSize={22} fill={ACC} opacity={draw > 0.7 ? 1 : 0}>0.1</text>
      <line x1={tx(10)} x2={tx(10)} y1={LY} y2={LY + 10} stroke={C.ink3} />
      <text x={tx(10)} y={LY + 34} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink2}>10T</text>
      <text x={LX1} y={LY + 34} textAnchor="end" fontFamily={F.mono} fontSize={18} fill={C.ink2}>14.8T token</text>
    </Svg>
    <Abs x={1000} y={760} w={800} style={{display: 'flex', alignItems: 'center', gap: 18, ...up(v41)}}>
      <Chip size={22}>V4.1：预训练不再带 MTP</Chip><Mono size={22} color={C.ink2}>→</Mono>
      <div style={up(p('l3.b'))}><Chip on size={22}>事后训练的 DSpark 草稿模块</Chip></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 06 改进链
type Stop = {id?: string; x?: number; y?: number; title: string; gist: string; key: string; above: boolean; dx?: number};
const CHAIN: Stop[] = [
  {id: 'rlhf', title: '2022-01 · RLHF 三步法', gist: 'SFT → 奖励模型 → 带 KL 的 PPO', key: 'l1.a', above: true},
  {id: 'dpo', title: '2023-05 · DPO', gist: '奖励模型藏进策略：一个二分类损失', key: 'l1.b', above: false},
  {id: 'grpo', title: '2024-02 · GRPO', gist: '去掉价值网络；7B 在 MATH 上 51.7%', key: 'l2.a', above: true, dx: -90},
  {id: 'grpo2', title: '2024-05 · V2', gist: '先推理 RL，再偏好 RL', key: 'l2.b', above: false},
  {id: 'mtp', title: '2024-12 · V3', gist: '训练时多预测一个 token', key: 'l2.c', above: true, dx: 60},
  {id: 'r1rl', title: '2025-01 · R1-Zero / R1', gist: '只用规则奖励的 GRPO；四个阶段；蒸馏 6 个小模型', key: 'l3.a', above: false},
  {id: 'hybrid', title: '2025-08 · V3.1', gist: '一个特殊 token 决定先不先思考', key: 'l4.a', above: true},
  {id: 'scalerl', title: '2025-12 · V3.2', gist: '八个专家各自 RL 再蒸馏；85,267 个智能体任务', key: 'l4.b', above: false},
  {id: 'opd', title: '2026-04 · V4', gist: '十个以上的教师在线蒸馏成一个模型', key: 'l5.a', above: true, dx: -90},
  {id: 'dspark', title: '2026-06 · DSpark', gist: '单用户生成比 MTP-1 基线快 60%–85%', key: 'l5.b', above: false},
  {x: 2420, y: 560, title: '2026-09 · V4.1-Flash', gist: '后训练没有算法创新：改动在数据和环境', key: 'l5.c', above: true, dx: 80},
];
export const B06: React.FC = () => {
  const {f, at} = useBoard();
  const rm = useRm();
  const D = laneRect(rm, 'D');
  const LANE = fit({x: 380, y: D.y - 150, w: 2170, h: D.h + 300}, 0.03);
  const W = 1450, Y = D.y + D.h / 2;
  const v: View = camAt(f, LANE, [
    {at: at('l1') + 4, dur: 34, to: {x: 560, y: Y, w: W}}, {at: at('l2') - 6, dur: 40, to: {x: 1060, y: Y, w: W}},
    {at: at('l3') - 6, dur: 40, to: {x: 1420, y: Y, w: W}}, {at: at('l4') - 6, dur: 40, to: {x: 1800, y: Y, w: W}},
    {at: at('l5') - 6, dur: 44, to: {x: 2240, y: Y, w: W}}, {at: at('l5.c') + 60, dur: 50, to: LANE},
  ]);
  const cam = project(v);
  const overview = ease(f, at('l5.c') + 70, 20);
  // a stop's card leaves when the narration moves on to the next line (the camera moves with it)
  const gone = (key: string) => {
    const next = `l${Number(key.split('.')[0].slice(1)) + 1}`;
    let t = Infinity;
    try { t = at(next); } catch { /* the last line: the overview takes the cards away */ }
    return Number.isFinite(t) ? ease(f, t - 8, 10) : overview;
  };
  const anchor = (s: Stop) => {
    if (s.id) {
      const n = nodeOf(rm, s.id);
      return {x: cam.X(n.x + n.w / 2), top: cam.Y(n.y), bottom: cam.Y(n.y + n.h)};
    }
    return {x: cam.X(s.x!), top: cam.Y(s.y!), bottom: cam.Y(s.y! + 34)};
  };
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{nodeIn: (n) => (n.lane === 'D' ? 1 : 0.12), laneIn: (k) => (k === 'D' ? 1 : 0.25), axisIn: 0.5,
      laneLit: (k) => (k === 'D' ? 0.6 : 0), nodeLit: (n) => {
        const c = CHAIN.find((s) => s.id === n.id);
        return c ? ease(f, at(c.key), 10) : 0;
      }}}>
      {CHAIN.map((s) => {
        const a = anchor(s), dx = s.dx ?? 0;
        const y0 = s.above ? a.top - 4 : a.bottom + 4, y1 = s.above ? y0 - 60 : y0 + 60;
        const k = lin(f, at(s.key), 10) * (1 - gone(s.key));
        return k > 0 ? <g key={s.title}>
          {!s.id && <rect x={a.x - 70} y={a.top} width={140} height={a.bottom - a.top} rx={6} fill="none" stroke={ACC} strokeDasharray="4 4" />}
          <path d={`M${a.x} ${y0} L${a.x} ${(y0 + y1) / 2} L${a.x + dx} ${(y0 + y1) / 2} L${a.x + dx} ${y1}`} fill="none" stroke={ACC}
            strokeWidth={1.4} pathLength={1} strokeDasharray={`${k} 1`} />
        </g> : null;
      })}
    </TreeScene>
    {CHAIN.map((s) => {
      const a = anchor(s), dx = s.dx ?? 0;
      const y = s.above ? a.top - 64 : a.bottom + 64;
      const k = ease(f, at(s.key) + 6, 12) * (1 - gone(s.key));
      if (k <= 0 || a.x < -300 || a.x > 2220) return null;
      return <Abs key={s.title} x={a.x + dx - 210} y={y} w={420} style={{opacity: k, transform: s.above ? 'translateY(-100%)' : undefined}}>
        <div style={{padding: '12px 18px', borderRadius: 10, background: rgba(C.bg, 0.94), border: `1px ${s.id ? 'solid' : 'dashed'} ${ACC}`,
          boxShadow: s.id ? BOX_GLOW : undefined}}>
          <Mono size={20} color={ACC}>{s.title}</Mono>
          <Txt size={21} style={{marginTop: 4, lineHeight: 1.3}}>{s.gist}</Txt>
        </div>
      </Abs>;
    })}
  </AbsoluteFill>;
};

// ================================================================ 07 后训练的账单
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const X0 = 360, K = 1300 / 2664;
  const zoom = ease(f, at('l1.a') + 30, 16);
  return <AbsoluteFill>
    <Abs x={160} y={160} style={up(p('l1'))}><Txt size={28} color={C.ink2}>公开的 GPU 小时：预训练与后训练</Txt></Abs>
    <Svg>
      {/* V3 */}
      <text x={X0 - 20} y={300} textAnchor="end" fontFamily={F.mono} fontSize={24} fill={C.ink}>V3</text>
      <rect x={X0} y={270} width={2664 * K * lin(f, at('l1'), 24)} height={44} fill={rgba(C.ink2, 0.4)} />
      <text x={X0 + 12} y={300} fontFamily={F.zh} fontSize={20} fill={C.ink}>预训练 2,664K</text>
      <rect x={X0 + 2664 * K} y={270} width={Math.max(2, 5 * K)} height={44} fill={ACC} opacity={ease(f, at('l1.a'), 10)} />
      <g opacity={zoom}>
        <circle cx={X0 + 2664 * K + 1} cy={292} r={34} fill="none" stroke={ACC} strokeWidth={2} />
        <line x1={X0 + 2664 * K + 25} y1={316} x2={X0 + 2664 * K + 60} y2={360} stroke={ACC} strokeWidth={2} />
        <text x={X0 + 2664 * K + 66} y={386} fontFamily={F.zh} fontSize={22} fill={ACC}>后训练 5K</text>
      </g>
      {/* R1 */}
      <g opacity={ease(f, at('l2.a'), 12)}>
        <text x={X0 - 20} y={470} textAnchor="end" fontFamily={F.mono} fontSize={24} fill={C.ink}>R1</text>
        <rect x={X0} y={440} width={147 * K * lin(f, at('l2.a') + 4, 16)} height={44} fill={ACC} filter="url(#vglow)" />
        <text x={X0 + 147 * K + 16} y={470} fontFamily={F.zh} fontSize={20} fill={C.ink}>全部 RL 与数据构造 147K</text>
      </g>
      {/* V3.2: only a ratio is given */}
      <g opacity={ease(f, at('l2.b'), 12)}>
        <text x={X0 - 20} y={610} textAnchor="end" fontFamily={F.mono} fontSize={24} fill={C.ink}>V3.2</text>
        <rect x={X0} y={580} width={266 * K * lin(f, at('l2.b') + 4, 16)} height={44} fill="none" stroke={ACC} strokeWidth={2}
          strokeDasharray="8 6" />
        <path d={`M${X0 + 266 * K} 602 L${X0 + 266 * K + 80} 602`} stroke={ACC} strokeWidth={2} strokeDasharray="4 6" opacity={ease(f, at('l3.b'), 12)} />
        <text x={X0 + 266 * K + 96} y={610} fontFamily={F.zh} fontSize={20} fill={C.ink}>后训练 &gt; 预训练的 10%</text>
      </g>
    </Svg>
    <Abs x={X0} y={330} style={up(ease(f, at('l1.a') + 30, 12))}><Tex id="v3post" size={32} parts={{r: lit(1)}} /></Abs>
    <Abs x={X0} y={500} style={up(ease(f, at('l2.a') + 30, 12))}><Tex id="r1pct" size={32} parts={{r: lit(1)}} /></Abs>
    <Abs x={X0} y={640} style={up(p('l3.b'))}><Tex id="v32min" size={32} parts={{r: lit(1)}} /></Abs>
    <Abs x={X0 + 460} y={650} style={up(p('l3.a'))}><Txt size={20} color={C.ink2}>论文只给比例、没给基数：虚框只是示意</Txt></Abs>
    <Note x={X0} y={820} p={ease(f, at('l1'), 12)}>金额都按报告自己假设的每 GPU 小时 2 美元</Note>
  </AbsoluteFill>;
};

// ================================================================ 08 代价
const COSTS: [string, string[], string][] = [
  ['奖励被钻空子', ['神经奖励模型会被讨好', 'R1：推理题只用规则奖励', '神经奖励只用于一般题、最后 400 步'], 'l1'],
  ['两条没走通的路', ['过程奖励模型、蒙特卡洛树搜索', 'R1 都没用上', '不代表它们做不出好的推理模型'], 'l2'],
  ['RL 改变了什么', ['提升 Maj@K（多数投票）', '不提升 Pass@K', '把已有的正确答案磨尖'], 'l3'],
  ['规模化的不稳定', ['后训练放大 50 多倍', '采样的模型 ≠ 训练的模型', 'V3.2：四项修补'], 'l4.a'],
  ['更长的思考', ['R1 平均每题 8,793 个思考 token', '约为 GPT-4o 的 12 倍', '注意力的账单回到主线 A'], 'l4.b'],
];
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  const cur = [...COSTS].reverse().find((c) => f >= at(c[2]));
  return <AbsoluteFill>
    <Abs x={120} y={160} w={1680} style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 22}}>
      {COSTS.map(([title, lines, key]) => {
        const on = !!cur && cur[0] === title;
        return <div key={title} style={up(p(key))}>
          <Card on={on} style={{height: 300, boxSizing: 'border-box'}}>
            <Txt size={32} color={on ? ACC : C.ink} weight={400}>{title}</Txt>
            {lines.map((l, i) => <Txt key={l} size={22} color={C.ink2} style={{marginTop: i ? 6 : 14, opacity: ease(f, at(key) + 10 + i * 12, 10)}}>{l}</Txt>)}
            {title === 'RL 改变了什么' && <div style={{display: 'flex', alignItems: 'flex-end', gap: 30, marginTop: 14, height: 64}}>
              {[['Maj@K', 0.75], ['Pass@K', 0.45]].map(([n, after], i) => {
                const h2 = i === 0 ? mix(0.45, after as number, ease(f, at('l3.a') + 6, 16)) : 0.45;
                return <div key={n as string} style={{display: 'flex', alignItems: 'flex-end', gap: 6}}>
                  <div style={{width: 24, height: 60 * 0.45, background: rgba(C.ink2, 0.4)}} />
                  <div style={{width: 24, height: 60 * h2, background: i === 0 ? ACC : rgba(C.ink2, 0.4)}} />
                  <Mono size={16} color={C.ink3}>{n as string}</Mono>
                </div>;
              })}
              <Mono size={14} color={C.ink3}>RL 前 / 后 · 示意</Mono>
            </div>}
          </Card>
        </div>;
      })}
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 09 三个问题的答案
export const B09: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const out = ease(f, at('l3.a') - 4, 16);
  const v = camAt(f, {...FULLV, w: FULLV.w * 0.85}, [{at: at('l3.a'), dur: 60, to: FULLV}]);
  const items: [number, string, string, string, number][] = [
    [1, '强化学习：按奖励打分，再用 KL 拴住', 'PPO 要同时放四个模型：7B 规模约 252 GB', 'l1', at('l1.b')],
    [2, 'GRPO：同题其他回答的均值当基线', '去掉价值网络；R1：只看最后答案对不对，也能教会一步一步地想', 'l2', at('l2.b')],
    [3, '0.19% → 10% 以上', '代价：奖励被钻空子、训练不稳、更贵的推理', 'l3', at('l3') + 30],
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
      <TreeScene cam={project(v)} look={{laneLit: (k) => (k === 'E' ? ease(f, at('l3.a') + 30, 12) : 0)}} />
      <Abs x={0} y={120} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 主线 E：深度方向的“1”，以及记在哪里</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};
