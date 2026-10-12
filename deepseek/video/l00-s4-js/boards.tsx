// 第 0 课 §4 · 主线 C：数值与系统 —— 9 段黑板。数字全部来自 lesson-00 §4（研究笔记 lesson-00-doc.md）；
// 03 段的 8 个数、02 段的尺子刻度、07 段的误差曲线是示意，画面上标了"示意"；它们的舍入按 E4M3 的真实网格算。
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

// ================================================================ 01 557.6 万美元
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const X0 = 260, W = 1460, K = W / 2788;
  const seg: [string, number, string, string][] = [['预训练', 2664, '2,664K', 'l1.a'], ['上下文扩展', 119, '119K', 'l1.b'], ['后训练', 5, '5K', 'l1.c']];
  let x = X0;
  const roads = 1 - ease(f, at('l3'), 14);
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: roads}}>
      <Abs x={X0} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>V3 全部正式训练的 GPU 小时（H800）</Txt></Abs>
      <Svg>
        {seg.map(([name, v, t, key], i) => {
          const w = Math.max(3, v * K) * lin(f, at(key), 18), x0 = x;
          x += v * K;
          const below = i > 0;
          return <g key={name} opacity={ease(f, at(key), 8)}>
            <rect x={x0} y={300} width={w} height={80} fill={i === 0 ? rgba(ACC, 0.45) : i === 1 ? ACC : C.ink} />
            <line x1={x0 + Math.max(3, v * K) / 2} x2={x0 + Math.max(3, v * K) / 2 + (i === 2 ? 40 : 0)} y1={below ? 380 : 300}
              y2={below ? 420 + i * 46 : 270} stroke={C.ink2} />
            <text x={x0 + Math.max(3, v * K) / 2 + (i === 2 ? 40 : 0)} y={below ? 446 + i * 46 : 258} textAnchor={i === 0 ? 'middle' : 'end'}
              fontFamily={F.zh} fontSize={24} fill={C.ink}>{name} {t}</text>
          </g>;
        })}
        {/* research and ablations before the formal run: not in the bill */}
        <g opacity={ease(f, at('l2.b'), 14)}>
          <rect x={X0 - 230} y={300} width={210} height={80} fill="none" stroke={C.ink3} strokeDasharray="6 6" />
          <text x={X0 - 125} y={348} textAnchor="middle" fontFamily={F.zh} fontSize={20} fill={C.ink3}>此前的研究、消融</text>
          <text x={X0 - 125} y={410} textAnchor="middle" fontFamily={F.zh} fontSize={20} fill={ACC}>不在账单里</text>
        </g>
      </Svg>
      <Abs x={X0} y={560} style={{display: 'flex', alignItems: 'baseline', gap: 26, ...up(p('l2'))}}>
        <Txt size={30}>合计</Txt><Mono size={52} color={C.ink}>2,788K</Mono><Txt size={26} color={C.ink2}>GPU 小时</Txt>
      </Abs>
      <Abs x={X0} y={660} style={up(p('l2.a'))}><Tex id="bill" size={52} parts={{r: lit(1)}} /></Abs>
      <Abs x={X0 + 760} y={672} style={up(p('l2.a'))}><Txt size={34} color={ACC} glow>= 557.6 万美元</Txt></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - roads}}>
      <Abs x={160} y={170} style={up(p('l3'))}><Txt size={28} color={C.ink2}>让账单变小，只有两条路</Txt></Abs>
      <Abs x={160} y={250} w={760} style={up(p('l3.a'))}>
        <Card on={f >= at('l3.a') && f < at('l3.b')}>
          <Txt size={30}>① 更短的数</Txt>
          <Txt size={22} color={C.ink2} style={{marginTop: 4}}>每次矩阵乘法更便宜</Txt>
          <div style={{display: 'flex', alignItems: 'center', gap: 18, marginTop: 22}}>
            <div style={{width: 320, height: 40, borderRadius: 4, background: rgba(C.ink2, 0.35), display: 'flex', alignItems: 'center',
              paddingLeft: 12}}><Mono size={18}>16 位</Mono></div>
            <Mono size={22} color={C.ink2}>→</Mono>
            <div style={{width: 320 * mix(1, 0.5, swing(f, at('l3.a') + 20, 20)), height: 40, borderRadius: 4, background: rgba(ACC, 0.5),
              display: 'flex', alignItems: 'center', paddingLeft: 12, boxShadow: BOX_GLOW}}><Mono size={18}>8 位</Mono></div>
          </div>
        </Card>
      </Abs>
      <Abs x={980} y={250} w={780} style={up(p('l3.b'))}>
        <Card on={f >= at('l3.b') && f < at('l4')}>
          <Txt size={30}>② 不算的时间更少</Txt>
          <Txt size={22} color={C.ink2} style={{marginTop: 4}}>把通信藏到计算后面</Txt>
          <Svg w={720} h={150} y={110}>
            {(() => {
              const k = swing(f, at('l3.b') + 24, 30);
              return <g>
                <text x={0} y={34} fontFamily={F.zh} fontSize={18} fill={C.ink3}>计算</text>
                <text x={0} y={84} fontFamily={F.zh} fontSize={18} fill={C.ink3}>通信</text>
                {[0, 1, 2].map((i) => <rect key={`c${i}`} x={60 + i * mix(200, 140, k)} y={16} width={120} height={28} rx={3} fill={rgba(ACC, 0.45)} />)}
                {[0, 1, 2].map((i) => <rect key={`m${i}`} x={60 + i * mix(200, 140, k) + mix(120, 10, k)} y={66} width={80} height={28} rx={3}
                  fill={rgba(C.ink2, 0.45)} />)}
                <line x1={60} x2={60 + mix(2 * 200 + 200, 2 * 140 + 130, k)} y1={124} y2={124} stroke={ACC} strokeWidth={2} />
                <text x={60 + mix(600, 410, k) + 10} y={130} fontFamily={F.zh} fontSize={18} fill={ACC}>一步的时间</text>
              </g>;
            })()}
          </Svg>
          <div style={{height: 150}} />
        </Card>
      </Abs>
      <Note x={980} y={540} p={ease(f, at('l3.b'), 12)}>示意：重叠前后的时间线</Note>
      <Abs x={160} y={640} style={up(p('l4'))}>
        <Chip on={f >= at('l4')} size={28}>③ 推理：显存装得下多少，每一步读多少字节</Chip></Abs>
      <Abs x={160} y={730} style={up(p('l4.b'))}><Txt size={26} color={C.ink2}>→ 这又回到了主线 A</Txt></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 02 短尺子和放大镜
const RAW = [3.1, 5.2, 2.0, 6.8, 4.4];   // 示意: one group of small numbers; the outlier is 900
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const X0 = 300, X1 = 1640, TICK = 28;                       // the ruler: 0…448, a tick every 28 (示意, coarser than FP8)
  const X = (v: number) => X0 + ((X1 - X0) * v) / 448;
  const snap = (v: number) => Math.round(v / TICK) * TICK;
  const lens = swing(f, at('l1.b') + 6, 30);
  const mapA = swing(f, at('l2.b') + 10, 26), snapA = swing(f, at('l2.c') + 6, 18);
  const ruler = (y: number, o: number, key: string) => <g key={key} opacity={o}>
    <line x1={X0} x2={X1} y1={y} y2={y} stroke={C.ink2} strokeWidth={2} />
    {Array.from({length: 17}, (_, i) => <line key={i} x1={X(i * TICK)} x2={X(i * TICK)} y1={y - 12} y2={y + 12} stroke={C.ink2} strokeWidth={1.4} />)}
    <text x={X0} y={y + 40} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>0</text>
    <text x={X1} y={y + 40} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>448</text>
  </g>;
  const two = ease(f, at('l2'), 14), three = ease(f, at('l3'), 14);
  const sB = 6.8 / 448, sA = 900 / 448;
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: 1 - two}}>
      <Abs x={X0} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>FP8 = 一把刻度很少的尺子</Txt></Abs>
      <Abs x={X0} y={226} style={up(p('l1.a'))}><Txt size={24} color={ACC}>一共只有 256 个码</Txt></Abs>
      <Svg>
        {ruler(520, ease(f, at('l1'), 14), 'r1')}
        {RAW.map((v, i) => {
          const x = mix(X0 + v * 3, X(v / sB), lens);
          return <circle key={i} cx={x} cy={520} r={9} fill={ACC} filter="url(#vglow)" opacity={ease(f, at('l1.b'), 10)} />;
        })}
        <g opacity={ease(f, at('l1.b'), 12)}>
          <circle cx={mix(X0 + 40, X(300), lens)} cy={400} r={48} fill="none" stroke={C.ink} strokeWidth={3} />
          <line x1={mix(X0 + 40, X(300), lens) + 34} y1={434} x2={mix(X0 + 40, X(300), lens) + 70} y2={470} stroke={C.ink} strokeWidth={6}
            strokeLinecap="round" />
          <text x={mix(X0 + 40, X(300), lens) + 70} y={390} fontFamily={F.zh} fontSize={24} fill={C.ink}>缩放因子 = 放大镜</text>
        </g>
      </Svg>
      <Abs x={X0} y={620} w={1340} style={up(ease(f, at('l1.b') + 30, 14))}>
        <Txt size={24} color={C.ink2}>放大以后，这一段的数正好落在尺子最精细的区间里</Txt></Abs>
      <Note x={X0} y={700} p={ease(f, at('l1'), 12)}>示意：画了 16 格；真的 FP8 有 256 个码，刻度也不均匀</Note>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: two * (1 - three)}}>
      <Abs x={X0} y={170} style={up(p('l2.a'))}><Txt size={28} color={C.ink2}>放大镜配多密？</Txt></Abs>
      <Abs x={120} y={300} w={170}><Txt size={22} color={C.ink2}>整个张量一个</Txt></Abs>
      <Abs x={120} y={560} w={170}><Txt size={22} color={ACC}>每 128 个数一个</Txt></Abs>
      <Svg>
        {ruler(330, 1, 'ra')}
        {ruler(590, ease(f, at('l2.a'), 14), 'rb')}
        {/* per-tensor: the outlier sets the scale; everyone else is crushed into the first tick */}
        {[...RAW, 900].map((v, i) => {
          const tgt = v / sA;
          const xv = mix(X0 + (v / 900) * 60, X(mix(tgt, snap(tgt), snapA)), mapA);
          return <circle key={i} cx={xv} cy={330} r={9} fill={v === 900 ? C.ink : ACC} opacity={ease(f, at('l2.b'), 10)} />;
        })}
        {snapA > 0.5 && <text x={X(0) + 14} y={300} fontFamily={F.zh} fontSize={22} fill={ACC} opacity={snapA}>五个数全挤在 0 上</text>}
        <text x={X(448)} y={300} textAnchor="middle" fontFamily={F.zh} fontSize={20} fill={C.ink} opacity={ease(f, at('l2.b'), 12)}>离群值</text>
        {/* fine-grained: this group gets its own magnifier */}
        {RAW.map((v, i) => {
          const tgt = v / sB;
          return <circle key={i} cx={mix(X0 + (v / 900) * 60, X(snap(tgt)), mapA)} cy={590} r={9} fill={ACC} filter="url(#vglow)"
            opacity={ease(f, at('l2.b'), 10)} />;
        })}
      </Svg>
      <Abs x={X0} y={700} w={1340} style={up(p('l2.c'))}><Txt size={24}>放大镜只能照顾最大的那个数，其余的数都被挤到粗刻度上</Txt></Abs>
      <Note x={X0} y={770} p={ease(f, at('l2.b'), 12)}>示意：五个小数和一个离群值 900</Note>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: three}}>
      <Abs x={160} y={200} style={up(p('l3'))}><Txt size={30} color={C.ink2}>比方的边界</Txt></Abs>
      <Abs x={160} y={290} w={780} style={up(p('l3.a'))}>
        <Card on={f < at('l3.b')}><Txt size={28}>放大镜只能整体缩放</Txt>
          <Txt size={22} color={C.ink2} style={{marginTop: 6}}>同一小段里大小悬殊的数，仍然有一个被量得很粗</Txt></Card></Abs>
      <Abs x={1000} y={290} w={780} style={up(p('l3.b'))}>
        <Card on={f >= at('l3.b')}><Txt size={28}>缩放因子本身也要存、要乘</Txt>
          <Txt size={22} color={C.ink2} style={{marginTop: 6}}>配得越密，额外的字节和运算越多</Txt></Card></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 03 一组数共用一个缩放因子
// 示意: 8 numbers in one group (V3 uses 128). s = 3.6 / 448; x / s rounded to the E4M3 grid (step 0.5 in [4, 8), 4 in [32, 64),
// 8 in [64, 128), 16 in [128, 256), 32 in [256, 448]).
const XS = [0.8, -2.1, 0.05, 3.6, -0.4, 1.2, 0.3, -1.7];
const QV = [96, -256, 6, 448, -48, 144, 36, -208];
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const s = 3.6 / 448;
  const tile = ease(f, at('l2.b'), 14);
  const rows: [string, (i: number) => string, string][] = [
    ['x', (i) => XS[i].toString(), 'l1'], ['x / s', (i) => (XS[i] / s).toFixed(1), 'l1.b'], ['舍入到 E4M3', (i) => QV[i].toString(), 'l1.b'],
    ['x̂ = s · 舍入', (i) => (QV[i] * s).toFixed(3), 'l1.b'], ['相对误差', (i) => `${(Math.abs(QV[i] * s - XS[i]) / Math.abs(XS[i]) * 100).toFixed(1)}%`, 'l1.b'],
  ];
  return <AbsoluteFill>
    <Abs x={160} y={160} style={{display: 'flex', gap: 80, alignItems: 'center'}}>
      <div style={up(p('l1.a'))}><Tex id="qs" size={46} parts={{q: f >= at('l2.a') ? lit(1) : {}}} /></div>
      <div style={up(p('l1.b'))}><Tex id="qx" size={46} /></div>
    </Abs>
    <AbsoluteFill style={{opacity: 1 - tile}}>
      <Abs x={120} y={360} w={1700}>
        {rows.map(([label, val, key], r) => <div key={label} style={{display: 'flex', alignItems: 'center', height: 64,
          borderBottom: `1px solid ${C.rule}`, opacity: ease(f, at(key) + r * 8, 12)}}>
          <Txt size={22} color={C.ink2} style={{width: 220}}>{label}</Txt>
          {XS.map((_, i) => <Mono key={i} size={24} color={i === 3 && r === 0 ? ACC : r === 4 ? C.ink2 : C.ink} glow={i === 3 && r === 0}
            style={{width: 180, textAlign: 'right'}}>{val(i)}</Mono>)}
        </div>)}
      </Abs>
      <Abs x={120} y={700} style={up(p('l1.a'))}><Txt size={22} color={C.ink2}>
        组里的最大绝对值是 <span style={{color: ACC}}>3.6</span>，所以 s = 3.6 / 448；它自己正好落在 448 上</Txt></Abs>
      <Note x={120} y={760} p={ease(f, at('l1'), 12)}>示意：8 个数一组（V3 实际是 128 个）；舍入按 E4M3 的真实网格</Note>
      <Abs x={120} y={800} style={{display: 'flex', gap: 18, ...up(p('l2.a'))}}>
        <Chip on size={24}>FP8 E4M3：q_max = 448</Chip><Chip size={24}>FP4 E2M1：q_max = 6</Chip></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: tile}}>
      <Abs x={160} y={330}><Txt size={26} color={C.ink2}>V3 训练时怎么分组</Txt></Abs>
      <Svg>
        {/* activations: rows of 512, cut into 1 × 128 tiles, one scale each */}
        {Array.from({length: 4}, (_, r) => Array.from({length: 4}, (_, c) => <g key={`${r}-${c}`} opacity={ease(f, at('l2.b') + (r * 4 + c) * 2, 10)}>
          <rect x={160 + c * 180} y={420 + r * 44} width={172} height={34} rx={3} fill={rgba(ACC, 0.14)} stroke={ACC} strokeWidth={1} />
          <circle cx={160 + c * 180 + 160} cy={420 + r * 44 + 17} r={6} fill={ACC} />
        </g>))}
        <text x={160} y={630} fontFamily={F.zh} fontSize={22} fill={C.ink}>激活：1 × 128 一组</text>
        {/* weights: 256 × 256, cut into 128 × 128 blocks */}
        {[0, 1].map((r) => [0, 1].map((c) => <g key={`w${r}${c}`} opacity={ease(f, at('l2.b') + 20 + (r * 2 + c) * 4, 10)}>
          <rect x={1060 + c * 180} y={400 + r * 180} width={172} height={172} rx={4} fill={rgba(C.ink2, 0.16)} stroke={C.ink2} />
          <circle cx={1060 + c * 180 + 150} cy={400 + r * 180 + 22} r={8} fill={ACC} />
        </g>))}
        <text x={1060} y={790} fontFamily={F.zh} fontSize={22} fill={C.ink}>权重：128 × 128 一组</text>
        <circle cx={1520} cy={460} r={8} fill={ACC} />
        <text x={1540} y={468} fontFamily={F.zh} fontSize={22} fill={C.ink}>= 一个缩放因子（FP32）</text>
      </Svg>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 04 只取 2 的幂
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  // log2 axis for s, from 2^-9 to 2^-3
  const LX0 = 220, LX1 = 1420, L0 = -9, L1 = -3;
  const LX = (e: number) => LX0 + ((e - L0) / (L1 - L0)) * (LX1 - LX0);
  const raw = Math.log2(3.6 / 448);           // −6.96
  const ceilK = swing(f, at('l1.b') + 6, 22);
  const use = ease(f, at('l2.b'), 14);
  return <AbsoluteFill>
    <Abs x={160} y={160} style={up(p('l1'))}><Tex id="ue" size={48} parts={{c: f >= at('l1.b') && f < at('l2') ? lit(1) : {}}} /></Abs>
    <Abs x={1220} y={170} style={up(p('l1.a'))}>
      <div style={{display: 'flex', gap: 4}}>
        {Array.from({length: 8}, (_, i) => <div key={i} style={{width: 44, height: 44, borderRadius: 4, border: `1px solid ${ACC}`,
          background: rgba(ACC, 0.15), display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: 20,
          color: ACC}}>E</div>)}
      </div>
      <Txt size={20} color={C.ink2} style={{marginTop: 8}}>UE8M0：8 位全是指数，没有尾数</Txt>
    </Abs>
    <Svg opacity={1 - use}>
      <line x1={LX0} x2={LX1} y1={420} y2={420} stroke={C.ink2} strokeWidth={2} opacity={ease(f, at('l1'), 12)} />
      {Array.from({length: L1 - L0 + 1}, (_, i) => <g key={i} opacity={ease(f, at('l1'), 12)}>
        <line x1={LX(L0 + i)} x2={LX(L0 + i)} y1={406} y2={434} stroke={C.ink2} />
        <text x={LX(L0 + i)} y={466} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink2}>2^{L0 + i}</text>
      </g>)}
      <g opacity={ease(f, at('l1.a'), 12)}>
        <circle cx={LX(raw)} cy={420} r={9} fill={C.ink} />
        <text x={LX(raw)} y={380} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink}>a/448 ≈ 0.00804</text>
      </g>
      {ceilK > 0 && <g>
        <path d={`M${LX(raw)} 430 Q${(LX(raw) + LX(-6)) / 2} 500 ${LX(raw) + (LX(-6) - LX(raw)) * ceilK} 430`} fill="none" stroke={ACC}
          strokeWidth={2} />
        <circle cx={LX(raw) + (LX(-6) - LX(raw)) * ceilK} cy={420} r={11} fill={ACC} filter="url(#vglow)" />
        <text x={LX(-6) + 16} y={530} fontFamily={F.zh} fontSize={22} fill={ACC} opacity={ceilK}>向上取整到 2^-6：不会溢出</text>
      </g>}
    </Svg>
    <Abs x={160} y={258} style={{display: 'flex', gap: 18, ...up(p('l2.a'))}}>
      <Chip on size={24}>缩放因子只占 1 字节</Chip><Chip on size={24}>乘 2ⁿ：又准又便宜</Chip></Abs>
    <AbsoluteFill style={{opacity: use}}>
      <Abs x={160} y={330}><Txt size={24} color={C.ink2}>代价：组里最大的数，用不满 E4M3 的范围</Txt></Abs>
      <Svg>
        <rect x={220} y={420} width={1200} height={50} rx={4} fill="none" stroke={C.ink2} />
        <rect x={220 + 600} y={420} width={600} height={50} fill={rgba(ACC, 0.12)} />
        <text x={220 + 900} y={500} textAnchor="middle" fontFamily={F.zh} fontSize={20} fill={ACC}>最大的数会落在这一半里</text>
        <line x1={220 + 1200 * 0.514 * lin(f, at('l2.b') + 10, 20)} x2={220 + 1200 * 0.514 * lin(f, at('l2.b') + 10, 20)} y1={404} y2={486}
          stroke={ACC} strokeWidth={3} filter="url(#vglow)" />
        <text x={220} y={404} fontFamily={F.mono} fontSize={18} fill={C.ink3}>0</text>
        <text x={220 + 600} y={404} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>224</text>
        <text x={1420} y={404} textAnchor="end" fontFamily={F.mono} fontSize={18} fill={C.ink3}>448</text>
      </Svg>
      <Abs x={220} y={540} style={up(ease(f, at('l2.b') + 20, 14))}><Tex id="ex2" size={36} /></Abs>
      <Abs x={220} y={640} style={up(p('l2.c'))}><Tex id="avg" size={44} parts={{r: lit(1)}} /></Abs>
      <Abs x={980} y={660} style={up(p('l2.c'))}><Txt size={22} color={C.ink2}>假设 log₂ r 在 (−1, 0] 上均匀</Txt></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 05 改进链
type Stop = {id?: string; x?: number; title: string; gist: string; key: string; above: boolean; dx?: number};
const CHAIN: Stop[] = [
  {x: 470, title: '2022-03 · BF16 混合精度', gist: 'Chinchilla 论文的写法；不是第一次使用', key: 'l1.a', above: true},
  {id: 'fp8fmt', title: '2022-09 · FP8 格式', gist: 'E4M3 最大 448，E5M2 最大 57,344', key: 'l1.b', above: false},
  {id: 'fireflyer', title: '2024-08 · Fire-Flyer', gist: '1 万张 PCIe A100 的软硬件协同', key: 'l2.a', above: true},
  {id: 'fp8train', title: '2024-12 · V3', gist: '细粒度 FP8；DualPipe；all-to-all 只占 20 个 SM', key: 'l2.b', above: false},
  {id: 'openweek', title: '2025-02 · 开源周', gist: 'FlashMLA、DeepEP、DeepGEMM……', key: 'l3.a', above: false, dx: -90},
  {id: 'insights', title: '2025-05 · ISCA 论文', gist: '2,048 张 H800 上的经验', key: 'l3.b', above: true},
  {id: 'ue8m0', title: '2025-08 · V3.1', gist: 'UE8M0 缩放：只取 2 的幂', key: 'l3.c', above: false, dx: 90},
  {id: 'fp4', title: '2026-04 · V4', gist: '专家权重、indexer 的 QK 路径：FP4 量化感知训练', key: 'l4.a', above: true},
  {id: 'fp4kv', title: '2026-09 · V4.1-Flash', gist: '主 KV 存成 FP4，每 16 维一个缩放', key: 'l4.b', above: false},
  {x: 2520, title: '2026-09 · DSec', gist: '智能体 RL 的沙箱平台，每天约 300 万个', key: 'l4.c', above: true},
];
export const B05: React.FC = () => {
  const {f, at} = useBoard();
  const rm = useRm();
  const Cl = laneRect(rm, 'C');
  const LANE = fit({x: 330, y: Cl.y - 150, w: 2300, h: Cl.h + 300}, 0.03);
  const W = 1450, Y = Cl.y + Cl.h / 2;
  const v: View = camAt(f, LANE, [
    {at: at('l1') + 4, dur: 34, to: {x: 560, y: Y, w: W}}, {at: at('l2') - 6, dur: 40, to: {x: 1300, y: Y, w: W}},
    {at: at('l3') - 6, dur: 40, to: {x: 1620, y: Y, w: W}}, {at: at('l4') - 6, dur: 48, to: {x: 2330, y: Y, w: W}},
    {at: at('l4.c') + 50, dur: 50, to: LANE},
  ]);
  const cam = project(v);
  const overview = ease(f, at('l4.c') + 60, 20);
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
    return {x: cam.X(s.x!), top: cam.Y(Cl.y + 18), bottom: cam.Y(Cl.y + 52)};
  };
  return <AbsoluteFill>
    <TreeScene cam={cam} look={{nodeIn: (n) => (n.lane === 'C' ? 1 : 0.12), laneIn: (k) => (k === 'C' ? 1 : 0.25), axisIn: 0.5,
      laneLit: (k) => (k === 'C' ? 0.6 : 0),
      nodeLit: (n) => {
        const c = CHAIN.find((s) => s.id === n.id);
        return c ? ease(f, at(c.key), 10) : 0;
      }}}>
      {CHAIN.map((s) => {
        const a = anchor(s), dx = s.dx ?? 0;
        const y0 = s.above ? a.top - 4 : a.bottom + 4, y1 = s.above ? y0 - 60 : y0 + 60;
        const k = lin(f, at(s.key), 10) * (1 - gone(s.key));
        return k > 0 ? <g key={s.title}>
          {!s.id && <rect x={a.x - 50} y={a.top} width={100} height={a.bottom - a.top} rx={6} fill="none" stroke={ACC} strokeDasharray="4 4" />}
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
    <Note x={120} y={880} p={ease(f, at('l1.a'), 10) * (1 - ease(f, at('l1.b'), 10))}>虚线框：不在这条线上的节点（BF16 在主线 F 的 Chinchilla 旁）</Note>
  </AbsoluteFill>;
};

// ================================================================ 06 每维几位
const ENTRIES: [string, number, number, number, string, string][] = [
  // name, bytes, dims, bits per dim, formula, mark
  ['V2 / V3', 1152, 576, 16, 'b16', 'l2.a'], ['V3.2', 656, 576, 9.1, 'b91', 'l2.b'], ['V4', 584, 512, 9.1, 'b91v4', 'l2.c'],
  ['V4.1', 288, 512, 4.5, 'b45', 'l2.d'],
];
export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const BASE = 740, PX = 0.56, PY = 17;
  const cur = [...ENTRIES].reverse().find((e) => f >= at(e[5]));
  return <AbsoluteFill>
    <Abs x={0} y={150} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(p('l1'))}}>
      <Tex id="bkv" size={52} parts={{c: f >= at('l1.a') && f < at('l1.b') ? lit(1) : {}, m: f >= at('l1.a') && f < at('l1.b') ? lit(1) : {},
        b: f >= at('l1.b') ? lit(1) : {}}} />
    </Abs>
    <Abs x={560} y={262} style={{display: 'flex', gap: 300}}>
      <div style={up(p('l1.a'))}><Txt size={22} color={C.ink2}>主线 A 管：存几份、每几个 token 一条</Txt></div>
      <div style={up(p('l1.b'))}><Txt size={22} color={ACC}>主线 C 管：每个数几位</Txt></div>
    </Abs>
    <Svg>
      {ENTRIES.map(([name, bytes, dims, bits, , key], i) => {
        const x = 170 + i * 420, w = dims * PX, h = bits * PY * swing(f, at(key), 18);
        const on = !!cur && cur[0] === name;
        return <g key={name} opacity={ease(f, at(key), 10)}>
          <rect x={x} y={BASE - h} width={w} height={h} rx={3} fill={on ? rgba(ACC, 0.5) : rgba(ACC, 0.22)} stroke={ACC}
            strokeWidth={on ? 2 : 1} filter={on ? 'url(#vglow)' : undefined} />
          <text x={x} y={BASE - 16 * PY - 46} fontFamily={F.mono} fontSize={26} fill={on ? ACC : C.ink}>{name}</text>
          <text x={x} y={BASE - 16 * PY - 16} fontFamily={F.zh} fontSize={20} fill={C.ink2}>{bytes.toLocaleString('en-US')} 字节 · {dims} 维</text>
          <text x={x + w / 2} y={BASE - h + 34} textAnchor="middle" fontFamily={F.mono} fontSize={26} fill={C.ink}>{bits} 位/维</text>
          <line x1={x} x2={x + w} y1={BASE} y2={BASE} stroke={C.ink3} />
        </g>;
      })}
      <text x={170} y={BASE + 34} fontFamily={F.zh} fontSize={18} fill={C.ink3} opacity={ease(f, at('l2.a'), 12)}>
        宽 = 维数，高 = 每维几位，面积 = 字节</text>
    </Svg>
    <Abs x={170} y={BASE + 60} style={{display: 'flex', gap: 60}}>
      {ENTRIES.map(([name, , , , fm, key]) => <div key={name} style={{width: 360, ...up(p(key))}}>
        <Tex id={fm} size={24} color={cur && cur[0] === name ? ACC : C.ink2} /></div>)}
    </Abs>
    <Abs x={0} y={318} w={1920} style={{display: 'flex', justifyContent: 'center', gap: 18}}>
      <div style={up(p('l3.a'))}><Chip size={22}>缩放因子：一直只占很小一块</Chip></div>
      <div style={up(p('l3.b'))}><Chip on size={22}>RoPE 维：直到 V4.1 才跟着量化</Chip></div>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 07 代价
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const ph = (a: string, b?: string) => ease(f, at(a), 14) * (b ? 1 - ease(f, at(b), 14) : 1);
  // E4M3 between 256 and 448: a code every 32
  const RX0 = 260, RX1 = 1660, RX = (v: number) => RX0 + ((v - 224) / (448 - 224)) * (RX1 - RX0);
  const K = 1 - Math.pow(1 - lin(f, at('l2.a') + 10, 80), 1.2);
  return <AbsoluteFill>
    <AbsoluteFill style={{opacity: ph('l1', 'l2')}}>
      <Abs x={RX0} y={170} style={up(p('l1'))}><Txt size={28} color={C.ink2}>E4M3 只有 256 个码</Txt></Abs>
      <Svg>
        <line x1={RX0} x2={RX1} y1={380} y2={380} stroke={C.ink2} strokeWidth={2} opacity={ease(f, at('l1'), 12)} />
        {[224, 240, 256, 288, 320, 352, 384, 416, 448].map((v) => <g key={v} opacity={ease(f, at('l1'), 12)}>
          <line x1={RX(v)} x2={RX(v)} y1={364} y2={396} stroke={C.ink} strokeWidth={1.6} />
          <text x={RX(v)} y={430} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink2}>{v}</text>
        </g>)}
        <g opacity={ease(f, at('l1.a'), 12)}>
          <circle cx={RX(272)} cy={380} r={9} fill={ACC} filter="url(#vglow)" />
          <path d={`M${RX(272)} 350 L${RX(272)} 320 L${RX(256)} 320 L${RX(256)} 350`} fill="none" stroke={ACC} strokeWidth={1.6} />
          <text x={RX(264)} y={306} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={ACC}>差 16</text>
          <text x={RX(272) + 14} y={360} fontFamily={F.mono} fontSize={18} fill={ACC}>272</text>
        </g>
      </Svg>
      <Abs x={RX0} y={480} style={up(p('l1.a'))}><Tex id="e625" size={40} parts={{r: lit(1)}} /></Abs>
      <Abs x={RX0 + 520} y={492} style={up(p('l1.a'))}><Txt size={22} color={C.ink2}>落在两个码正中间的数，误差最大</Txt></Abs>
      <Abs x={RX0} y={580} style={up(p('l1.b'))}><Tex id="r3w" size={40} parts={{r: lit(1)}} /></Abs>
      <Abs x={RX0 + 520} y={600} style={up(p('l1.b'))}><Txt size={22} color={C.ink2}>比组里最大值小这么多倍，就开始失真</Txt></Abs>
      <Abs x={RX0} y={720} style={up(p('l1.c'))}><Chip on size={26}>离群值决定整组的缩放 → 所以要细粒度</Chip></Abs>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: ph('l2', 'l3')}}>
      <Abs x={260} y={170} style={up(p('l2'))}><Txt size={28} color={C.ink2}>累加也会出错</Txt></Abs>
      <Svg>
        <line x1={260} x2={1500} y1={760} y2={760} stroke={C.ink3} />
        <line x1={260} x2={260} y1={300} y2={760} stroke={C.ink3} />
        <text x={1500} y={796} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={C.ink3}>内维（要加起来的项数）→</text>
        <text x={270} y={290} fontFamily={F.zh} fontSize={20} fill={C.ink3}>累加误差 ↑</text>
        <line x1={260} y1={760} x2={260 + 1240 * K} y2={760 - 420 * K} stroke={C.ink} strokeWidth={2.4} />
        <text x={260 + 1240 * K - 10} y={760 - 420 * K - 14} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={C.ink}
          opacity={K > 0.3 ? 1 : 0}>只在 FP8 张量核心里累加（约 14 位有效位）</text>
        {f >= at('l2.c') && <path d={Array.from({length: 11}, (_, i) => {
          const x0 = 260 + i * 124;
          return `${i === 0 ? 'M' : 'L'}${x0} 760 L${x0 + 124} ${760 - 42}`;
        }).join(' ')} fill="none" stroke={ACC} strokeWidth={2.4} filter="url(#vglow)" opacity={ease(f, at('l2.c'), 14)}
          strokeDasharray={`${3000 * lin(f, at('l2.c'), 40)} 3000`} />}
        {f >= at('l2.c') && <text x={1500} y={700} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={ACC} opacity={ease(f, at('l2.c') + 30, 12)}>
          V3：每 128 个元素把部分和搬到 FP32</text>}
      </Svg>
      <Abs x={1540} y={420} w={300} style={up(p('l2.b'))}><Txt size={22} color={C.ink2}>误差随内维线性变大</Txt></Abs>
      <Note x={260} y={820} p={ease(f, at('l2'), 12)}>示意：误差曲线的形状，不是测量值</Note>
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: ph('l3')}}>
      <Abs x={260} y={170} style={up(p('l3'))}><Txt size={28} color={C.ink2}>V4.1 省掉了 FP4 的第二级全局缩放</Txt></Abs>
      <Abs x={260} y={250} style={up(p('l3.a'))}><Chip size={26}>前提：RMSNorm 的权重大约为 1</Chip></Abs>
      <Abs x={260} y={360} style={up(p('l3.b'))}><Tex id="sq" size={46} parts={{a: lit(1)}} /></Abs>
      <Svg opacity={ease(f, at('l3.b'), 14)}>
        {[[22.6, '每个分量 ≤ 约 22.6', true], [2688, '448 × 6 = 2,688', false]].map(([v, t, on], i) => {
          const w = (1300 * Math.log10(v as number)) / Math.log10(3000) * lin(f, at('l3.b') + 10 + i * 10, 20);
          return <g key={t as string}>
            <rect x={260} y={500 + i * 70} width={w} height={40} rx={3} fill={on ? ACC : rgba(C.ink2, 0.35)} filter={on ? 'url(#vglow)' : undefined} />
            <text x={270 + w} y={528 + i * 70} fontFamily={F.zh} fontSize={22} fill={C.ink}>{t as string}</text>
          </g>;
        })}
        <text x={260} y={660} fontFamily={F.zh} fontSize={18} fill={C.ink3}>对数长度</text>
      </Svg>
      <Abs x={260} y={720} style={up(p('l3.c'))}><Chip on size={28}>这个前提不成立，论证就不成立</Chip></Abs>
    </AbsoluteFill>
  </AbsoluteFill>;
};

// ================================================================ 08 看清口径
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  return <AbsoluteFill>
    <Abs x={160} y={170} w={780} style={up(p('l1'))}>
      <Card on={f < at('l2')}>
        <Txt size={32} weight={400}>“557.6 万美元”</Txt>
        <Txt size={22} color={C.ink2} style={{marginTop: 10, ...up(p('l1.a'))}}>= 正式训练的 GPU 小时 × 2 美元</Txt>
        <Txt size={22} color={ACC} style={{marginTop: 6, ...up(p('l1.b'))}}>不含此前的研究和消融实验</Txt>
      </Card>
    </Abs>
    <Abs x={1000} y={170} w={780} style={up(p('l2'))}>
      <Card on={f >= at('l2') && f < at('l3')}>
        <Txt size={32} weight={400}>“成本利润率 545%”</Txt>
        <Txt size={22} color={C.ink2} style={{marginTop: 10, ...up(p('l2.a'))}}>开源周第六天公布</Txt>
        <Txt size={22} color={ACC} style={{marginTop: 6, ...up(p('l2.b'))}}>按 R1 的价格算的理论值，不是实际收入</Txt>
      </Card>
    </Abs>
    <Abs x={160} y={500} w={1620} style={up(p('l3'))}>
      <Card on={f >= at('l3')}>
        <Txt size={26} color={C.ink2}>能复算的数，就自己算一遍</Txt>
        <div style={{display: 'flex', alignItems: 'center', gap: 40, marginTop: 16}}>
          <Txt size={22} color={C.ink2} style={{...up(p('l3.a'))}}>每万亿 token 180K GPU 小时，2,048 张卡：</Txt>
          <div style={up(p('l3.a'))}><Tex id="days" size={40} parts={{r: lit(1)}} /></div>
          <Txt size={24} style={{...up(p('l3.b'))}}>天　·　原文：约 3.7 天</Txt>
        </div>
      </Card>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 09 三个问题的答案
export const B09: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm();
  const out = ease(f, at('l3.a') - 4, 16);
  const D = laneRect(rm, 'D');
  const v = camAt(f, FULLV, [{at: at('l3.a'), dur: 60, to: fit({x: -330, y: D.y - 200, w: 2880, h: D.h + 400}, 0.03)}]);
  const items: [number, string, string, string, number][] = [
    [1, '2,788K GPU 小时，557.6 万美元', '更短的数；把通信藏到计算后面', 'l1', at('l1.b')],
    [2, '一小组数共用一个缩放因子', 'V3：1 × 128 / 128 × 128；V3.1：缩放因子只取 2 的幂', 'l2', at('l2.b')],
    [3, '舍入误差、累加误差、必须成立的前提', '6.25% · 约 14 位有效位 · RMSNorm 权重 ≈ 1', 'l3', at('l3') + 30],
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
      <TreeScene cam={project(v)} look={{laneLit: (k) => (k === 'D' ? ease(f, at('l3.a') + 30, 12) : 0)}} />
      <Abs x={0} y={800} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 主线 D：从“像语料”到“做得对”，以及它的算力</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};
