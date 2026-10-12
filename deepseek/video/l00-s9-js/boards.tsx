// 第 0 课 §9 · 怎么读这门课 —— 8 段黑板。内容全部来自 lesson-00 §9（研究笔记 lesson-00-doc.md）：
// 五个部分的阅读与练习分钟、两条跳读路线、数学对照表（03 段的矩阵由对照表逐格读出）、图色语义、八个类比、
// 研究笔记与复核、批注的写法。04 段的色块是课程配图的深色主题取值（--fig-*）。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Check, Chip, F, Mono, Roadmap, Svg, Tex, Txt, camAt, ease, lin, lit, project, rgba, up, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, TreeScene} from '../l00-shared/scene';

const Card: React.FC<{on?: boolean; dashed?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, dashed, style, children}) =>
  <div style={{padding: '16px 22px', borderRadius: 12, background: rgba(C.bg, 0.92),
    border: `1px ${dashed ? 'dashed' : 'solid'} ${on ? rgba(ACC, 0.7) : C.line}`, boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;

// ================================================================ 01 按课号读
const PARTS: [string, string, string, number, number, string][] = [
  ['一', 'GPT 的地基', '1–5', 910, 250, 'l2.a'], ['二', 'DeepSeek 的起点', '6–8', 530, 145, 'l2.b'], ['三', 'V2 与 V3', '9–12', 835, 200, 'l2.c'],
  ['四', '推理模型与稀疏注意力', '13–15', 520, 145, 'l2.d'], ['五', '新架构', '16–19', 830, 195, 'l2.e'],
];
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  return <AbsoluteFill>
    <Abs x={120} y={150} style={{display: 'flex', gap: 18, alignItems: 'center', ...up(p('l1'))}}>
      <Chip on size={26}>默认：按课号顺序读</Chip>
      <div style={up(p('l1.a'))}><Chip size={22}>每一课接着上一课留下的问题开头</Chip></div>
      <div style={up(p('l1.b'))}><Chip size={22}>封面：科技树上的位置 + 前置课</Chip></div>
    </Abs>
    <Abs x={120} y={260} w={1680} style={{display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16}}>
      {PARTS.map(([n, name, range, rd, pr, key]) => {
        const k = ease(f, at(key), 12);
        return <div key={n} style={up(k)}><Card on={f >= at(key) && f < at(key) + 60} style={{height: 330, boxSizing: 'border-box'}}>
          <Mono size={22} color={ACC}>{n} · 第 {range} 课</Mono>
          <Txt size={28} weight={400} style={{marginTop: 6, height: 76}}>{name}</Txt>
          <Txt size={18} color={C.ink3} style={{marginTop: 8}}>阅读</Txt>
          <div style={{display: 'flex', alignItems: 'center', gap: 8}}>
            <div style={{width: (180 * rd) / 910 * lin(f, at(key) + 6, 16), height: 18, borderRadius: 3, background: ACC}} /><Mono size={18}>{rd}′</Mono></div>
          <Txt size={18} color={C.ink3} style={{marginTop: 8}}>练习</Txt>
          <div style={{display: 'flex', alignItems: 'center', gap: 8}}>
            <div style={{width: (180 * pr) / 910 * lin(f, at(key) + 10, 16), height: 18, borderRadius: 3, background: rgba(C.ink2, 0.6)}} /><Mono size={18}>{pr}′</Mono></div>
        </Card></div>;
      })}
    </Abs>
    <Abs x={120} y={630} style={{display: 'flex', flexDirection: 'column', gap: 10, ...up(p('l3.a'))}}>
      <Tex id="read" size={34} parts={{r: lit(1)}} /><Tex id="prac" size={34} parts={{r: lit(0.6)}} /></Abs>
    <Abs x={1000} y={640} style={up(p('l3.a'))}><Txt size={30}>≈ <span style={{color: ACC}}>六十多个小时</span>（第 1–19 课）</Txt></Abs>
    <Abs x={1000} y={710} w={800} style={up(p('l3.b'))}><Txt size={22} color={C.ink2}>十一课数学补课：约 1,710 分钟阅读、505 分钟练习；按节补，不必整课读</Txt></Abs>
  </AbsoluteFill>;
};

// ================================================================ 02 只追一条主线
const ROUTES: [string, number[], string][] = [['A', [1, 4, 5, 9, 14, 18, 19], 'l1.a'], ['D', [1, 2, 3, 8, 13, 15], 'l1.b']];
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const X = (n: number) => 200 + (n - 1) * 86, RY = [380, 560];
  const pre = ease(f, at('l2.a'), 14);
  return <AbsoluteFill>
    <Abs x={120} y={150} style={up(p('l1'))}><Txt size={28} color={C.ink2}>只关心一条主线：沿科技树的一条色带跳读</Txt></Abs>
    <Svg>
      {Array.from({length: 19}, (_, i) => <text key={i} x={X(i + 1)} y={260} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}
        opacity={ease(f, at('l1'), 12)}>{i + 1}</text>)}
      {ROUTES.map(([lane, ls, key], r) => {
        const k = lin(f, at(key), 30);
        const y = RY[r];
        const last = ls[ls.length - 1];
        return <g key={lane} opacity={ease(f, at(key), 10)}>
          <text x={150} y={y + 8} textAnchor="end" fontFamily={F.mono} fontSize={30} fill={ACC}>{lane}</text>
          <line x1={X(ls[0])} x2={X(ls[0]) + (X(last) - X(ls[0])) * k} y1={y} y2={y} stroke={ACC} strokeWidth={4} />
          {ls.map((n) => {
            const on = X(n) <= X(ls[0]) + (X(last) - X(ls[0])) * k + 1;
            return <g key={n} opacity={on ? 1 : 0.2}>
              <circle cx={X(n)} cy={y} r={22} fill={C.bg} stroke={ACC} strokeWidth={3} />
              <text x={X(n)} y={y + 7} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink}>{n}</text>
            </g>;
          })}
        </g>;
      })}
      {/* lesson 18 also needs 11 and 16 */}
      {pre > 0 && [11, 16].map((n, i) => {
        const k = lin(f, at('l2.a') + 6 + i * 16, 20);
        return <g key={n} opacity={pre}>
          <circle cx={X(n)} cy={RY[0] - 120} r={22} fill={C.bg} stroke={C.ink2} strokeWidth={2} strokeDasharray="4 4" />
          <text x={X(n)} y={RY[0] - 113} textAnchor="middle" fontFamily={F.mono} fontSize={20} fill={C.ink2}>{n}</text>
          <path d={`M${X(n) + 18} ${RY[0] - 105} Q${(X(n) + X(18)) / 2} ${RY[0] - 150} ${X(18)} ${RY[0] - 24}`} fill="none" stroke={C.ink2}
            strokeWidth={1.6} strokeDasharray={`${400 * k} 400`} />
        </g>;
      })}
    </Svg>
    <Abs x={X(11) - 120} y={RY[0] - 205} w={700} style={up(pre)}><Txt size={22} color={C.ink2}>跳过的前置：第 18 课还要求第 11 课（MoE 与 MTP）和第 16 课（mHC）</Txt></Abs>
    <Abs x={120} y={720} w={1600} style={up(p('l3.a'))}>
      <Card dashed style={{borderColor: '#DFB44E'}}><Txt size={24}><span style={{color: '#DFB44E'}}>黄色的“前置知识”框</span>会就地补一点<span
        style={{opacity: ease(f, at('l3.b'), 12)}}>，但不如完整读来得扎实</span></Txt></Card></Abs>
  </AbsoluteFill>;
};

// ================================================================ 03 数学补哪一节
// sections each lesson's text links to, per maths lesson (M01…M11), read off the table in lesson 0 §9; * = on the cover
const MODS = ['M01 矩阵与形状', 'M02 低秩与 SVD', 'M03 旋转与复数', 'M04 概率与方差', 'M05 熵与 KL', 'M06 反向传播', 'M07 优化器',
  'M08 浮点与量化', 'M09 策略梯度', 'M10 算力算术', 'M11 Sinkhorn'];
const LINKS: Record<number, Record<number, number>> = {
  1: {1: 6, 4: 3, 6: 3}, 2: {1: 1, 5: 3, 6: 1, 7: 2, 8: 1}, 3: {4: 3, 5: 2, 6: 1, 9: 3, 10: 2}, 4: {1: 4, 4: 4, 5: 1, 10: 3},
  5: {1: 3, 2: 2, 3: 6}, 6: {1: 1, 4: 1, 5: 1, 7: 3, 10: 1}, 7: {1: 2, 4: 1, 5: 1, 6: 1, 10: 2}, 8: {4: 4, 5: 3, 9: 4, 10: 1},
  9: {1: 5, 2: 2, 3: 1, 6: 1, 8: 1, 10: 1}, 10: {1: 2, 4: 2, 6: 1}, 11: {4: 3, 5: 3, 6: 2, 7: 1, 8: 2, 10: 2}, 12: {1: 1, 8: 4, 10: 4},
  13: {4: 1, 5: 3, 6: 1, 9: 4}, 14: {1: 1, 5: 2, 6: 1, 8: 1, 10: 1}, 15: {5: 2, 8: 2, 9: 2}, 16: {2: 2, 5: 1, 6: 3, 11: 4},
  17: {1: 1, 4: 1, 7: 3, 8: 1, 11: 1}, 18: {1: 1, 2: 3, 3: 2, 5: 2, 7: 2, 8: 2, 10: 1, 11: 1}, 19: {1: 1, 3: 1, 8: 2, 10: 1},
};
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const X0 = 400, Y0 = 220, CW = 62, RH = 54;
  const hotCols: [number[], string][] = [[[1], 'l2.a'], [[4, 5], 'l2.b'], [[8, 10], 'l2.c']];
  const hot = [...hotCols].reverse().find(([, k]) => f >= at(k));
  const row1 = f >= at('l1.b') && f < at('l2');
  return <AbsoluteFill>
    <Abs x={120} y={140} style={up(p('l1'))}><Txt size={24} color={C.ink2}>每一课正文链接到的数学小节（数字 = 链接了几节）</Txt></Abs>
    <Svg>
      {MODS.map((m, j) => <g key={m} opacity={ease(f, at('l1'), 12)}>
        <text x={X0 - 20} y={Y0 + j * RH + RH * 0.62} textAnchor="end" fontFamily={F.zh} fontSize={20}
          fill={hot && hot[0].includes(j + 1) ? ACC : C.ink2}>{m}</text>
      </g>)}
      {Array.from({length: 19}, (_, i) => <text key={i} x={X0 + i * CW + CW / 2} y={Y0 - 12} textAnchor="middle" fontFamily={F.mono} fontSize={18}
        fill={row1 && i === 0 ? ACC : C.ink3} opacity={ease(f, at('l1'), 12)}>{i + 1}</text>)}
      {Array.from({length: 19}, (_, i) => MODS.map((_, j) => {
        const v = LINKS[i + 1][j + 1] ?? 0;
        const k = ease(f, at('l1.a') + i * 2, 10);
        const on = (hot && hot[0].includes(j + 1)) || (row1 && i === 0);
        return <g key={`${i}-${j}`} opacity={k}>
          <rect x={X0 + i * CW + 3} y={Y0 + j * RH + 4} width={CW - 6} height={RH - 8} rx={4}
            fill={v ? rgba(on ? ACC : C.ink2, Math.min(0.85, 0.15 + v * 0.12)) : rgba(C.ink, 0.03)} stroke={on && v ? ACC : 'none'} />
          {v > 0 && <text x={X0 + i * CW + CW / 2} y={Y0 + j * RH + RH * 0.64} textAnchor="middle" fontFamily={F.mono} fontSize={18}
            fill={on ? C.bg : C.ink}>{v}</text>}
        </g>;
      }))}
    </Svg>
    <Abs x={120} y={840} style={{display: 'flex', gap: 14, ...up(p('l2'))}}>
      <Chip on={!!hot && hot[1] === 'l2.a'} size={20}>M01：几乎每一课</Chip>
      <Chip on={!!hot && hot[1] === 'l2.b'} size={20}>M04、M05：第 3、8、13、15 课的强化学习</Chip>
      <Chip on={!!hot && hot[1] === 'l2.c'} size={20}>M08、M10：第 12 课以后的系统部分</Chip>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 04 图的颜色
const COLORS: [string, string, string, string][] = [
  ['#7FAAF0', '蓝', '查询 Q、当前 token、正在计算的东西、被激活的路由专家', 'l1.a'],
  ['#BB9BF2', '紫', '键 K、路由打分（含 indexer）、共享专家', 'l1.b'],
  ['#5EC191', '绿', '值 V、输出、正奖励、“好的 / 保留的”', 'l1.c'],
  ['#DFB44E', '黄褐', '缓存、存储、记忆（KV cache、状态、内存层级）', 'l2.a'],
  ['#F0A26E', '橙', '压缩后的潜在表示、量化后的低精度数据', 'l2.b'],
  ['#F0928B', '红', '被掩掉、被丢弃、错误、负奖励', 'l2.c'],
];
const LANE_COLORS: [string, string][] = [['A', '#7FAAF0'], ['B', '#BB9BF2'], ['C', '#F0A26E'], ['D', '#5EC191'], ['E', '#DFB44E'], ['F', '#ECEAE5']];
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const lanes = ease(f, at('l3.a'), 14);
  return <AbsoluteFill>
    <Abs x={120} y={150} style={up(p('l1'))}><Txt size={26} color={C.ink2}>全课的图用同一套颜色：颜色本身带含义</Txt></Abs>
    <Abs x={120} y={220} w={1000} style={{display: 'flex', flexDirection: 'column', gap: 14}}>
      {COLORS.map(([hex, name, what, key]) => <div key={name} style={{display: 'flex', alignItems: 'center', gap: 20, ...up(p(key))}}>
        <div style={{width: 64, height: 44, borderRadius: 6, background: hex, boxShadow: f >= at(key) && f < at(key) + 50 ? `0 0 18px ${hex}` : undefined}} />
        <Txt size={26} style={{width: 70, color: hex}}>{name}</Txt>
        <Txt size={23} color={C.ink}>{what}</Txt>
      </div>)}
    </Abs>
    <Abs x={1240} y={220} w={560} style={up(lanes)}>
      <Txt size={22} color={C.ink2}>科技树的六条色带</Txt>
      {LANE_COLORS.map(([k, hex], i) => <div key={k} style={{display: 'flex', alignItems: 'center', gap: 14, marginTop: 12,
        opacity: ease(f, at('l3.a') + 6 + i * 5, 10)}}>
        <Mono size={24} color={hex}>{k}</Mono>
        <div style={{flexGrow: 1, height: 26, borderRadius: 6, background: rgba(hex, 0.25), border: `1px solid ${hex}`}} />
      </div>)}
    </Abs>
    <Abs x={120} y={760} w={1680} style={up(p('l3.b'))}>
      <Card dashed><Txt size={22}>颜色写成主题变量（--fig-*）：深色模式下自动换成对应的浅色。本页色块是深色主题的取值。</Txt></Card></Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 八个类比
const ANA: [string, string, string, string][] = [
  ['注意力', '图书馆检索', '查询 = 检索问题，键 = 书脊标签，值 = 书的内容', 'l1.a'],
  ['KV cache', '会议纪要本', '秘书把“标签 + 要点”记下，后来的人只翻本子', 'l1.b'],
  ['MLA', '速记', '纪要改记成速记符号，要用时再展开', 'l1.c'],
  ['MoE', '医院', '分诊台、专科医生、全科医生', 'l1.d'],
  ['GRPO', '同题多答', '按比组平均高几个标准差给分', 'l2.a'],
  ['残差流', '传送带', '工位取件、加工、放回增量', 'l2.b'],
  ['稀疏注意力', '先查目录再精读', '便宜的目录，只翻开得分最高的 k 页', 'l2.c'],
  ['低精度数值', '短尺子 + 放大镜', '刻度很少的尺子，每一小段配一个放大镜', 'l2.d'],
];
export const B05: React.FC = () => {
  const {f, at, p} = useBoard();
  const edge = ease(f, at('l3.b'), 14);
  return <AbsoluteFill>
    <Abs x={120} y={140} style={up(p('l1'))}><Txt size={26} color={C.ink2}>全课只用这八个类比</Txt></Abs>
    <Abs x={120} y={200} w={1680} style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16}}>
      {ANA.map(([what, like, map, key], i) => {
        const flip = i === 0 ? edge : 0;
        return <div key={what} style={up(p(key))}><Card on={i === 0 ? edge > 0.5 : f >= at(key) && f < at(key) + 45} style={{height: 250, boxSizing: 'border-box'}}>
          <Mono size={20} color={C.ink3}>{what} =</Mono>
          <Txt size={30} weight={400} color={ACC}>{like}</Txt>
          <Txt size={20} color={C.ink2} style={{marginTop: 10, opacity: 1 - flip}}>{map}</Txt>
          {i === 0 && <Txt size={20} color={C.ink} style={{marginTop: -54, opacity: flip}}>失效处：真检索是挑一本；注意力是把所有书加权混合</Txt>}
        </Card></div>;
      })}
    </Abs>
    <Abs x={120} y={770} style={up(p('l3.a'))}><Chip on size={24}>每个类比都有它失效的地方：表里把“边界”一栏也列了出来</Chip></Abs>
  </AbsoluteFill>;
};

// ================================================================ 06 数字从哪来
export const B06: React.FC = () => {
  const {f, at, p} = useBoard();
  const NOTE: [string, string][] = [['一手来源 + 读的是哪一版', 'l1.a'], ['承载事实的那一句原文', 'l1.b'], ['派生数字的计算式（Python 复算）', 'l1.c'],
    ['勘误 + “未能核实”清单', 'l1.d']];
  return <AbsoluteFill>
    <Abs x={120} y={150} style={up(p('l1'))}><Txt size={26} color={C.ink2}>先有研究笔记，再有课文</Txt></Abs>
    <Abs x={120} y={220} w={560} style={up(p('l1'))}>
      <Card on={f < at('l2')}><Mono size={20} color={ACC}>reference-docs/lesson-NN-doc.md</Mono>
        {NOTE.map(([t, key]) => <div key={t} style={{display: 'flex', gap: 10, alignItems: 'center', marginTop: 12, ...up(p(key))}}>
          <Mono size={20} color={ACC}>✓</Mono><Txt size={22}>{t}</Txt></div>)}
      </Card></Abs>
    <Svg>
      {[[700, 'l2.a'], [1240, 'l2.b']].map(([x, key]) => <path key={key as string} d={`M${(x as number) - 10} 330 L${(x as number) + 30} 330`} stroke={C.ink2}
        strokeWidth={2} opacity={ease(f, at(key as string), 12)} />)}
    </Svg>
    <Abs x={740} y={220} w={480} style={up(p('l2.a'))}>
      <Card on={f >= at('l2.a') && f < at('l2.b')}><Txt size={26}>课文</Txt><Txt size={20} color={C.ink2} style={{marginTop: 8}}>数字只能来自研究笔记；标“未能核实”的不进课文，进了也要写明</Txt></Card></Abs>
    <Abs x={1280} y={220} w={520} style={up(p('l2.b'))}>
      <Card on={f >= at('l2.b') && f < at('l3')}><Txt size={26}>对抗式复核</Txt><Txt size={20} color={C.ink2} style={{marginTop: 8}}>复核人回到一手来源，逐条重查</Txt>
        <div style={{display: 'flex', gap: 24, marginTop: 14, ...up(p('l2.c'))}}>
          <div><Mono size={44} color={ACC}>25</Mono><Txt size={18} color={C.ink2}>项核对无误</Txt></div>
          <div><Mono size={44} color={C.ink}>8</Mono><Txt size={18} color={C.ink2}>处修改</Txt></div></div></Card></Abs>
    <Abs x={120} y={600} w={1680} style={up(p('l3'))}>
      <Card on={f >= at('l3')}><div style={{display: 'flex', alignItems: 'center', gap: 30}}>
        <Txt size={24}>图：数据表</Txt><Mono size={24} color={C.ink2}>→</Mono><Txt size={24}>figures/gen/ 下的脚本</Txt><Mono size={24} color={C.ink2}>→</Mono>
        <Txt size={24}>由数据决定形状的图，不手画</Txt>
        <span style={{opacity: ease(f, at('l3.b'), 12), marginLeft: 'auto'}}><Txt size={24} color={ACC}>这一课：9 张图 ↔ 9 个 l00_*.py</Txt></span>
      </div></Card></Abs>
  </AbsoluteFill>;
};

// ================================================================ 07 怎样留批注
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const code = (t: string, c: string = C.ink) => <div style={{fontFamily: F.mono, fontSize: 24, color: c, whiteSpace: 'pre'}}>{t}</div>;
  return <AbsoluteFill>
    <Abs x={120} y={150} style={up(p('l1'))}><Txt size={26} color={C.ink2}>源文件是 Markdown：deepseek/src/、math/src/</Txt></Abs>
    <Abs x={120} y={230} w={1680} style={up(p('l1.a'))}>
      <Card style={{background: rgba(C.ink, 0.04)}}>
        {code('<!-- 批注(Harry): （示例）这里的 3,514 是只算全局 KV 吗？SWA 状态算不算？ -->', '#DFB44E')}
        <div style={{opacity: ease(f, at('l1.b'), 12), marginTop: 10}}>{code('<!-- 回复(Claude): 只算全局 KV；SWA 状态是每个请求的常数，见式 (0.1) 的说明。 -->', ACC)}</div>
      </Card>
      <Txt size={20} color={C.ink3} style={{marginTop: 10}}>HTML 注释：构建时不会出现在网页上</Txt>
    </Abs>
    <Abs x={120} y={500} w={1680} style={up(p('l2.a'))}>
      <Card style={{background: rgba(C.ink, 0.04)}}>{code('$ grep -rn "批注(" deepseek/src math/src', C.ink)}
        <Txt size={20} color={C.ink3} style={{marginTop: 8}}>列出所有还没处理的批注</Txt></Card></Abs>
    <Abs x={120} y={700} style={up(p('l2.b'))}><Chip on size={24}>涉及事实的修改：先改研究笔记（来源 + 原文），再改课文</Chip></Abs>
  </AbsoluteFill>;
};

// ================================================================ 08 三个问题的答案
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  const out = ease(f, at('l3.a') - 4, 16);
  const v = camAt(f, {...FULLV, w: FULLV.w * 0.85}, [{at: at('l3.a'), dur: 60, to: FULLV}]);
  const items: [number, string, string, string, number][] = [
    [1, '默认按课号读', '约 60 小时阅读、16 小时练习；只追一条主线时，先看前置', 'l1', at('l1.b')],
    [2, '数学按节补', 'M01、M04、M05、M08、M10 最常用', 'l2', at('l2.a') + 20],
    [3, '颜色有含义，类比有边界，数字有出处', '每个数字都能追到研究笔记', 'l3', at('l3') + 30],
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
      <TreeScene cam={project(v)} look={{alpha: 0.7}} />
      <Abs x={0} y={120} w={1920} style={{display: 'flex', justifyContent: 'center', ...up(ease(f, at('l3.a') + 40, 14))}}>
        <div style={{padding: '12px 28px', borderRadius: 12, background: rgba(C.bg, 0.92), border: `1px solid ${ACC}`, boxShadow: BOX_GLOW}}>
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 最后一页：五个贯穿全课的问题</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};

export type {Roadmap};
