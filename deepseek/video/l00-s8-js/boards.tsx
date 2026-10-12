// 第 0 课 §8 · 规格总表 —— 8 段黑板。两张规格表、完整时间线与常见误读表的内容全部来自 lesson-00 §8
// （研究笔记 lesson-00-doc.md）；03 段的配置数组按课文的描述画出（前 2 层纯 SWA，之后 CSA 与 HCA 交替）。
import React from 'react';
import {AbsoluteFill} from 'remotion';
import {
  ACC, Abs, BOX_GLOW, C, Check, Chip, F, Mono, Roadmap, Svg, Tex, Txt, camAt, ease, lin, lit, project, rgba, up, useBoard, useEp,
} from '../../../.cursor/skills/lesson-video/js/kit';
import {FULLV, TreeScene} from '../l00-shared/scene';

const useRm = () => useEp().roadmap as Roadmap;
const Card: React.FC<{on?: boolean; dashed?: boolean; style?: React.CSSProperties; children?: React.ReactNode}> = ({on, dashed, style, children}) =>
  <div style={{padding: '16px 22px', borderRadius: 12, background: rgba(C.bg, 0.92),
    border: `1px ${dashed ? 'dashed' : 'solid'} ${on ? rgba(ACC, 0.7) : C.line}`, boxShadow: on ? BOX_GLOW : undefined, ...style}}>{children}</div>;
const Note: React.FC<{x: number; y: number; p?: number; children?: React.ReactNode}> = ({x, y, p = 1, children}) =>
  <Abs x={x} y={y} style={{opacity: p}}><Mono size={16} color={C.ink3} style={{letterSpacing: '0.08em'}}>{children}</Mono></Abs>;

type Src = '论文' | '配置' | '模型卡' | '笔者';
const Badge: React.FC<{s: Src; p?: number}> = ({s, p = 1}) =>
  <span style={{display: 'inline-block', padding: '2px 10px', marginRight: 6, borderRadius: 6, fontSize: 16, opacity: p,
    border: `1px ${s === '笔者' ? 'dashed' : 'solid'} ${s === '笔者' ? ACC : C.ink3}`, color: s === '笔者' ? ACC : C.ink2,
    background: s === '配置' ? rgba(C.ink2, 0.18) : 'transparent'}}>{s === '笔者' ? '笔者合计 / 推算' : s}</span>;

/** A table whose rows come in one by one; `hot(row, col)` lights a cell. */
const Table: React.FC<{x: number; y: number; cols: [string, number][]; rows: React.ReactNode[][]; rowIn: (i: number) => number;
  hot?: (r: number, c: number) => number; rowH?: number; fs?: number}> = ({x, y, cols, rows, rowIn, hot = () => 0, rowH = 50, fs = 21}) =>
  <Abs x={x} y={y}>
    <div style={{display: 'flex', borderBottom: `1px solid ${C.line}`, paddingBottom: 6}}>
      {cols.map(([h, w]) => <div key={h} style={{width: w, fontSize: 18, color: C.ink3}}>{h}</div>)}
    </div>
    {rows.map((row, r) => <div key={r} style={{display: 'flex', alignItems: 'center', height: rowH, borderBottom: `1px solid ${C.rule}`,
      opacity: rowIn(r), transform: `translateY(${(1 - rowIn(r)) * 10}px)`}}>
      {row.map((cell, c) => {
        const h = hot(r, c);
        return <div key={c} style={{width: cols[c][1], fontSize: fs, color: h > 0.5 ? ACC : c === 0 ? C.ink : C.ink2, fontFamily: c === 0 ? F.zh : undefined,
          textShadow: h > 0.5 ? `0 0 12px ${rgba(ACC, 0.5)}` : undefined}}>{cell}</div>;
      })}
    </div>)}
  </Abs>;

// ================================================================ 01 规格表一：规模与训练
const T1: [string, string, string, string, string, string, Src[]][] = [
  ['DeepSeek LLM 7B', '30', '4096', '稠密 7B', '2.0T', '4096', ['论文']],
  ['DeepSeek LLM 67B', '95', '8192', '稠密 67B', '2.0T', '4096', ['论文', '配置']],
  ['DeepSeekMoE 16B', '28', '2048', '16.4B / 2.8B', '2T', '4K', ['论文']],
  ['DeepSeek-V2', '60', '5120', '236B / 21B', '8.1T', '128K', ['论文']],
  ['DeepSeek-V2-Lite', '27', '2048', '15.7B / 2.4B', '5.7T', '32K', ['论文']],
  ['DeepSeek-V3', '61', '7168', '671B / 37B', '14.8T', '128K', ['论文']],
  ['DeepSeek-R1', '同 V3', '同 V3', '671B / 37B', '在 V3-Base 上做 RL', '128K', ['论文']],
  ['DeepSeek-V3.1', '同 V3', '同 V3', '671B / 37B', 'V3 基座 + 839B 续训', '128K', ['模型卡']],
  ['DeepSeek-V3.2', '61', '7168', '671B / 37B', '约 16.6T', '128K', ['论文', '配置', '笔者']],
  ['DeepSeek-V4-Flash', '43', '4096', '284B / 13B', '32T', '1M', ['论文']],
  ['DeepSeek-V4-Pro', '61', '7168', '1.6T / 49B', '33T', '1M', ['论文']],
  ['DeepSeek-V4.1-Flash', '40', '5120', '552B + 196B Engram / 8B · 16B', '45T 多模态', '1M', ['论文']],
];
export const B01: React.FC = () => {
  const {f, at, p} = useBoard();
  const src = ease(f, at('l1.b'), 12);
  const v32 = f >= at('l2') && f < at('l3');
  return <AbsoluteFill>
    <Abs x={120} y={140} style={up(p('l1'))}><Txt size={24} color={C.ink2}>规格表一：规模与训练</Txt></Abs>
    <Table x={120} y={190} rowH={46} fs={19}
      cols={[['模型', 290], ['层数', 90], ['d_model', 100], ['总 / 激活参数', 420], ['训练 tokens', 280], ['上下文', 110], ['来源', 390]]}
      rows={T1.map(([m, l, d, n, t, c, s]) => [m, l, d, n, t, c, <span key="s">{s.map((x) => <Badge key={x} s={x} p={src} />)}</span>])}
      rowIn={(i) => ease(f, at('l1') + 4 + i * 4, 10)}
      hot={(r, c) => (v32 && r === 8 && (c === 4 || c === 0) ? 1 : f >= at('l3') && c === 5 ? (f < at('l3.b') ? 1 : 0) : f >= at('l1.a') && f < at('l1.b') && c >= 1 && c <= 5 ? 0.6 : 0)} />
    <Abs x={1040} y={786} w={780} style={{...up(p('l2.b')), opacity: p('l2.b') * (1 - ease(f, at('l3'), 10))}}>
      <Card on style={{padding: '10px 18px'}}><Tex id="sum166" size={28} parts={{r: lit(1)}} /></Card></Abs>
    <Abs x={1040} y={786} w={780} style={{opacity: ease(f, at('l3.b'), 12)}}>
      <Card dashed><Txt size={22}>配置文件里的位置上限另有含义 → 本集第 06 段</Txt></Card></Abs>
  </AbsoluteFill>;
};

// ================================================================ 02 规格表二：注意力、KV 与专家
const T2: [string, string, string, string, string][] = [
  // model, attention, KV bytes, how it was checked, experts
  ['DeepSeek LLM 7B', 'MHA，32 头 × 128', '491,520', '', '稠密'],
  ['DeepSeek LLM 67B', 'GQA：64 查询头、8 个 KV 头', '389,120', 'fig', '稠密'],
  ['DeepSeekMoE 16B', 'MHA，16 头 × 128', '229,376', '', '2 + 64，激活 6'],
  ['DeepSeek-V2', 'MLA', '69,120（BF16）', '', '2 + 160，激活 6'],
  ['DeepSeek-V2-Lite', 'MLA', '31,104', '', '2 + 64，激活 6'],
  ['DeepSeek-V3', 'MLA，维度同 V2', '70,272（BF16）', 'isca', '1 + 256，激活 8'],
  ['DeepSeek-V3.2', 'MLA + DSA', '48,068', 'fig', '同 V3'],
  ['DeepSeek-V4-Flash', 'CSA / HCA 交替', '3,514', 'fig', '1 + 256，激活 6'],
  ['DeepSeek-V4-Pro', 'CSA / HCA 交替', '5,031', 'ours', '1 + 384，激活 6'],
  ['DeepSeek-V4.1-Flash', 'CSA2；主 KV 用 FP4', '890', 'fig', '1 + 384，激活 6'],
];
export const B02: React.FC = () => {
  const {f, at, p} = useBoard();
  const mark = (k: string) => (k === 'fig' ? ease(f, at('l2.a'), 12) : k === 'isca' ? ease(f, at('l2.b'), 12) : k === 'ours' ? ease(f, at('l2.c'), 12) : 0);
  return <AbsoluteFill>
    <Abs x={120} y={140} style={up(p('l1'))}><Txt size={24} color={C.ink2}>规格表二：注意力、KV 缓存与专家</Txt></Abs>
    <Table x={120} y={190} rowH={60} fs={21}
      cols={[['模型', 300], ['注意力', 420], ['每 token 全局 KV（字节）', 300], ['核对', 300], ['专家（共享 + 路由）', 360]]}
      rows={T2.map(([m, a, kv, k, e]) => [m, a, <Mono key="kv" size={22} color={k ? C.ink : C.ink2}>{kv}</Mono>,
        <span key="k" style={{opacity: mark(k), fontSize: 18, color: k === 'ours' ? ACC : C.ink}}>
          {k === 'fig' ? '✓ = V4.1 论文图 1(b)' : k === 'isca' ? '✓ = ISCA 表 1（70.272 KB）' : k === 'ours' ? '笔者推算' : ''}</span>, e])}
      rowIn={(i) => ease(f, at('l1') + 4 + i * 4, 10)}
      hot={(r, c) => (c === 2 && f >= at('l1.a') && f < at('l2') ? 0.6 : c === 2 && T2[r][3] && mark(T2[r][3]) > 0.5 && f < at('l2') + 9999 ? 1 : 0)} />
    <Note x={120} y={850} p={ease(f, at('l1.a'), 12)}>KV 一栏由式 (0.2)–(0.5) 复算</Note>
  </AbsoluteFill>;
};

// ================================================================ 03 从配置文件复现
// V4-Flash compress_ratios: 2 × 0 (pure SWA), then 21 × 4 (CSA) alternating with 20 × 128 (HCA), then the MTP layer's 0.
const CR: number[] = [0, 0, ...Array.from({length: 41}, (_, i) => (i % 2 === 0 ? 4 : 128)), 0];
const KVSRC = [2, 8, 14, 20];
export const B03: React.FC = () => {
  const {f, at, p} = useBoard();
  const S = 36, G = 2.4;
  return <AbsoluteFill>
    <Abs x={120} y={150} style={up(p('l1.a'))}><Txt size={24} color={C.ink2}>V4-Flash 的 compress_ratios：44 项</Txt></Abs>
    <Abs x={120} y={200} style={{display: 'flex', gap: G}}>
      {CR.map((v, i) => {
        const mtp = i === 43;
        const k = ease(f, at('l1.a') + 6 + i * 0.8, 8);
        const on = f >= at('l1.b') && !mtp && v === 4;
        return <div key={i} style={{width: S, height: S + 10, borderRadius: 4, opacity: k * (mtp && f >= at('l1.c') ? 0.35 : 1),
          border: `1px ${mtp ? 'dashed' : 'solid'} ${on ? ACC : C.line}`, background: v === 4 ? rgba(ACC, on ? 0.35 : 0.12) : v === 128 ? rgba(C.ink2, 0.3) : 'transparent',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: v === 128 ? 12 : 15,
          color: v === 4 ? ACC : C.ink2}}>{v}</div>;
      })}
    </Abs>
    <Abs x={120} y={262} style={{display: 'flex', gap: 30, ...up(p('l1.b'))}}>
      <Mono size={18} color={C.ink2}>前 2 项 0：纯 SWA</Mono><Mono size={18} color={ACC}>21 个 4：CSA</Mono>
      <Mono size={18} color={C.ink2}>20 个 128：HCA</Mono>
      <span style={{opacity: ease(f, at('l1.c'), 12)}}><Mono size={18} color={C.ink3}>第 44 项（虚线）：MTP 层，不计</Mono></span>
    </Abs>
    <Abs x={120} y={320} style={up(p('l2.a'))}><Tex id="f4" size={40} parts={{r: lit(1)}} /></Abs>
    <Abs x={120} y={470} style={up(p('l3.a'))}><Txt size={24} color={C.ink2}>V4.1-Flash 的 kv_source_layer_ids = [2, 8, 14, 20]：40 层里只有 4 层自己压缩 KV</Txt></Abs>
    <Abs x={120} y={520} style={{display: 'flex', gap: G}}>
      {Array.from({length: 40}, (_, i) => {
        const src = KVSRC.includes(i);
        const k = ease(f, at('l3.a') + 6 + i * 0.8, 8);
        return <div key={i} style={{width: 38.5, height: 46, borderRadius: 4, opacity: k, marginLeft: i === 20 ? 14 : 0,
          border: `1px solid ${src ? ACC : C.line}`, background: src ? rgba(ACC, 0.35) : i < 2 ? 'transparent' : rgba(C.ink2, 0.12),
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: 13, color: src ? ACC : C.ink3}}>
          {i}{src && <span style={{fontSize: 12, opacity: ease(f, at('l3.b'), 10)}}>m={i < 20 ? 2 : 1}</span>}</div>;
      })}
    </Abs>
    <Abs x={120} y={576} style={{display: 'flex', gap: 640, ...up(p('l3.b'))}}>
      <Mono size={18} color={C.ink2}>编码器 20 层（0、1 层纯 SWA）</Mono><Mono size={18} color={C.ink2}>解码器 20 层</Mono></Abs>
    <Abs x={120} y={640} style={up(p('l3.c'))}><Tex id="f41" size={40} parts={{r: lit(1)}} /></Abs>
    <Note x={120} y={740} p={ease(f, at('l1'), 12)}>数组按课文的描述画：前 2 层纯 SWA，之后 CSA 与 HCA 交替</Note>
  </AbsoluteFill>;
};

// ================================================================ 04 50 行里的 7 个主干
export const B04: React.FC = () => {
  const {f, at, p} = useBoard();
  const rm = useRm() as any;
  const rows: {date: string; name: string; trunk: boolean; paper: boolean; lanes: string[]}[] = rm.timeline;
  const t0 = Date.UTC(2023, 9, 1), t1 = Date.UTC(2026, 9, 1);
  const X0 = 160, X1 = 1760, AY = 560;
  const xd = (s: string) => {
    const [y, m, d] = s.split('-').map(Number);
    return X0 + ((X1 - X0) * (Date.UTC(y, m - 1, d) - t0)) / (t1 - t0);
  };
  const lastX: number[] = [];
  const stack = rows.map((e) => {
    const x = xd(e.date);
    let k = 0;
    while (lastX[k] !== undefined && x - lastX[k] < 16) k++;
    lastX[k] = x;
    return k;
  });
  const trunkOn = ease(f, at('l2.a'), 14), rest = ease(f, at('l2.b'), 14);
  const SHORT: Record<string, string> = {'DeepSeek LLM 7B / 67B': 'DeepSeek LLM', 'DeepSeek-V2': 'V2', 'DeepSeek-V3': 'V3',
    'DeepSeek-R1、R1-Zero 与 6 个蒸馏模型': 'R1', 'DeepSeek-V3.2、V3.2-Speciale': 'V3.2', 'DeepSeek-V4 预览版（Pro、Flash）': 'V4',
    'DeepSeek-V4.1-Flash': 'V4.1-Flash'};
  let ti = 0;
  return <AbsoluteFill>
    <Abs x={X0} y={150} style={up(p('l1'))}><Txt size={26} color={C.ink2}>完整时间线：50 行</Txt></Abs>
    <Svg>
      <line x1={X0} x2={X1} y1={AY} y2={AY} stroke={C.ink3} strokeWidth={2} opacity={ease(f, at('l1'), 12)} />
      {[2024, 2025, 2026].map((yr) => <g key={yr} opacity={ease(f, at('l1'), 12)}>
        <line x1={xd(`${yr}-01-01`)} x2={xd(`${yr}-01-01`)} y1={AY - 8} y2={AY + 8} stroke={C.ink3} />
        <text x={xd(`${yr}-01-01`)} y={AY + 34} textAnchor="middle" fontFamily={F.mono} fontSize={18} fill={C.ink3}>{yr}</text>
      </g>)}
      {rows.map((e, i) => {
        const x = xd(e.date), y = AY - 24 - stack[i] * 18;
        const k = ease(f, at('l1') + 10 + i * 1.6, 8);
        const tr = e.trunk;
        const r = tr ? 9 + 2 * trunkOn : 6;
        return <g key={i} opacity={k * (tr ? 1 : 1 - 0.55 * trunkOn * (1 - rest))}>
          <circle cx={x} cy={y} r={r} fill={tr && trunkOn > 0 ? ACC : e.paper ? 'none' : C.ink2} stroke={e.paper ? C.ink2 : 'none'} strokeWidth={1.5}
            filter={tr && trunkOn > 0.5 ? 'url(#vglow)' : undefined} />
          {tr && (() => {
            const j = ti++;
            const up2 = j % 2 === 0;
            return <g opacity={trunkOn}>
              <line x1={x} x2={x} y1={y - 12} y2={up2 ? y - 110 : y - 60} stroke={ACC} strokeWidth={1.2} />
              <text x={x} y={up2 ? y - 118 : y - 68} textAnchor="middle" fontFamily={F.zh} fontSize={22} fill={ACC}>{SHORT[e.name] ?? e.name}</text>
            </g>;
          })()}
        </g>;
      })}
      <text x={X0} y={AY + 80} fontFamily={F.zh} fontSize={20} fill={C.ink2} opacity={ease(f, at('l1.a'), 12)}>2023-11-02 · DeepSeek-Coder</text>
      <text x={X1} y={AY + 80} textAnchor="end" fontFamily={F.zh} fontSize={20} fill={C.ink2} opacity={ease(f, at('l1.b'), 12)}>2026-09-19 · DSec</text>
    </Svg>
    <Abs x={X0} y={700} style={{display: 'flex', gap: 16, ...up(rest)}}>
      <Chip on size={22}>主干：7 个</Chip><Chip size={22}>同结构的更新</Chip><Chip size={22}>旁支模型</Chip><Chip size={22}>论文（空心点）</Chip><Chip size={22}>开源制品</Chip></Abs>
    <Abs x={X0} y={790} style={{display: 'flex', gap: 16, alignItems: 'center', ...up(p('l3.a'))}}>
      {['A', 'B', 'C', 'D', 'E', 'F'].map((k) => <Mono key={k} size={22} color={ACC}>{k}</Mono>)}
      <Txt size={22} color={C.ink2}>六条主线</Txt><Mono size={22} color={C.ink}>M</Mono><Txt size={22} color={C.ink2}>多模态旁支</Txt>
      <Mono size={22} color={C.ink}>P</Mono><Txt size={22} color={C.ink2}>形式证明旁支</Txt>
      <span style={{opacity: ease(f, at('l3.b'), 12), marginLeft: 20}}><Txt size={22} color={C.ink2}>“（论文）”= arXiv v1 的日期</Txt></span>
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 05 / 06 常见误读
type Myth = {say: string; fix: React.ReactNode; key: string};
const Myths: React.FC<{items: Myth[]}> = ({items}) => {
  const {f, at} = useBoard();
  return <Abs x={120} y={160} w={1680} style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 22}}>
    {items.map((m) => {
      const b = `${m.key}.b`;
      const fixed = ease(f, at(b), 12);
      const on = f >= at(m.key) && (items.indexOf(m) === items.length - 1 || f < at(items[items.indexOf(m) + 1].key));
      return <div key={m.key} style={up(ease(f, at(m.key), 12))}>
        <Card on={on} style={{height: 330, boxSizing: 'border-box'}}>
          <div style={{position: 'relative', display: 'inline-block'}}>
            <Txt size={27} color={fixed > 0.5 ? C.ink3 : C.ink}>“{m.say}”</Txt>
            <div style={{position: 'absolute', left: -4, right: -4, top: '52%', height: 2, background: ACC, transform: `scaleX(${fixed})`,
              transformOrigin: 'left'}} />
          </div>
          <div style={{marginTop: 14, opacity: fixed, transform: `translateY(${(1 - fixed) * 10}px)`}}>{m.fix}</div>
        </Card>
      </div>;
    })}
  </Abs>;
};
export const B05: React.FC = () => <Myths items={[
  {say: 'V4 论文是 2026 年 6 月发的', key: 'l1', fix: <><Txt size={22}>编号 2606.19348，但 arXiv 提交于 2026-04-26</Txt>
    <Txt size={22} color={ACC}>权重与技术报告 04-24 已在 Hugging Face 公开</Txt></>},
  {say: 'V4.1-Flash 是 552B 的模型', key: 'l2', fix: <><Txt size={22}>552B 是主干，另有 196B Engram 参数</Txt>
    <Txt size={22} color={ACC}>激活参数：预填充 8B，解码 16B</Txt></>},
  {say: 'V4.1 的激活参数只有 V4-Pro 的 1/4', key: 'l3', fix: <><Txt size={22}>论文原话如此，但</Txt><Tex id="r33" size={30} />
    <Txt size={22} color={ACC}>直接写 8B、16B、49B 更稳妥</Txt></>},
  {say: 'DeepSeek-V1 是一次独立发布', key: 'l4', fix: <><Txt size={22}>官方后来给 DeepSeek LLM 67B 起的别名</Txt>
    <Txt size={22} color={ACC}>最早公开的 DeepSeek 语言模型是 DeepSeek-Coder</Txt></>},
]} />;
export const B06: React.FC = () => <Myths items={[
  {say: 'config 写 163,840，说明支持 160K 上下文', key: 'l1', fix: <><Tex id="yarn" size={30} /><Txt size={22}>是 YaRN 的位置上限</Txt>
    <Txt size={22} color={ACC}>V2 / V3 系列的论文与模型卡都写 128K</Txt></>},
  {say: '小模型的 KV cache 一定更小', key: 'l2', fix: <><Txt size={22}>DeepSeek LLM 7B 用 MHA：每 token 491,520 字节</Txt>
    <Txt size={22} color={ACC}>比用 GQA-8 的 67B（389,120）还大</Txt></>},
  {say: 'V3 只花了 557.6 万美元', key: 'l3', fix: <><Txt size={22}>只是正式训练的 GPU 小时按租价折算</Txt>
    <Txt size={22} color={ACC}>R1 的 29.4 万美元也不含 V3-Base 本身</Txt></>},
  {say: 'V4 用 32T tokens 训练 / R2 已发布', key: 'l4', fix: <><Txt size={22}>32T 只是 Flash；Pro 33T；V4.1-Flash 45T 多模态</Txt>
    <Txt size={22} color={ACC}>截至 2026-09-24 的核查：官方渠道里没有 R2</Txt></>},
]} />;

// ================================================================ 07 先问三件事
export const B07: React.FC = () => {
  const {f, at, p} = useBoard();
  const ASK: [string, string, string[], string][] = [
    ['1', '哪一版', ['预览，还是正式', 'arXiv 的哪一版'], 'l1.a'],
    ['2', '什么口径', ['主干，还是含 Engram', '预填充，还是解码'], 'l2.a'],
    ['3', '原文还是推算', ['原文', '笔者推算 / 合计'], 'l2.b'],
  ];
  return <AbsoluteFill>
    <Abs x={120} y={170} style={up(p('l1'))}><Txt size={30}>引用一个数之前，先问三件事</Txt></Abs>
    <Abs x={120} y={280} w={1680} style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 26}}>
      {ASK.map(([n, head, lines, key], i) => {
        const on = f >= at(key) && (i === ASK.length - 1 || f < at(ASK[i + 1][3]));
        return <div key={n} style={up(p(key))}><Card on={on} style={{height: 330, boxSizing: 'border-box'}}>
          <Mono size={60} color={ACC} glow>{n}</Mono>
          <Txt size={36} weight={400} style={{marginTop: 6}}>{head}？</Txt>
          {lines.map((l) => <Txt key={l} size={24} color={C.ink2} style={{marginTop: 10}}>· {l}</Txt>)}
        </Card></div>;
      })}
    </Abs>
  </AbsoluteFill>;
};

// ================================================================ 08 三个问题的答案
export const B08: React.FC = () => {
  const {f, at, p} = useBoard();
  const out = ease(f, at('l3.a') - 4, 16);
  const v = camAt(f, {...FULLV, w: FULLV.w * 0.85}, [{at: at('l3.a'), dur: 60, to: FULLV}]);
  const items: [number, string, string, string, number][] = [
    [1, '每一格都能追到论文、配置文件或复算', '推算与合计的格子都标明了（如 V3.2 的约 16.6T、V4-Pro 的 5,031）', 'l1', at('l1.b')],
    [2, '50 行里，主干只有 7 个', 'DeepSeek LLM、V2、V3、R1、V3.2、V4、V4.1-Flash', 'l2', at('l2.a') + 30],
    [3, '先问版本、口径和出处', '预览还是正式；主干还是含 Engram；原文还是推算', 'l3', at('l3') + 30],
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
          <Txt size={34} color={ACC} glow weight={400}>下一集 · 怎么读这门课：推荐顺序、数学对照、图的颜色、八个类比</Txt></div></Abs>
    </AbsoluteFill>}
  </AbsoluteFill>;
};
