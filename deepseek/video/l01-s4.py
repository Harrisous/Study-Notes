"""第 1 课 §4 视频试样：注意力（二）——缩放、因果掩码与多头。

口播和板书都取自 deepseek/src/lesson-01-transformer-anatomy.md 的 §4。数字全部来自那一页（研究笔记
reference-docs/lesson-01-doc.md 已核实），这里不引入新的事实；512 个位置的尖峰图是一次抽样示意。
构建：python .cursor/skills/lesson-video/scripts/make_video.py all deepseek/video/l01-s4.py [--preview]
"""
from math import comb

from blackboard import *

EPISODE = {"title": "第 1 课 §4 · 注意力（二）：缩放、因果掩码与多头"}


def coin_hist(unit=0.19, max_h=2.3, base=-2.2):
    """Exact distribution of the sum of 64 fair ±1 coins, as chalk bars."""
    p = {s: comb(64, (s + 64) // 2) / 2 ** 64 for s in range(-32, 33, 2)}
    bars_ = VGroup()
    for s, v in p.items():
        h = max(0.01, max_h * v / p[0])
        c = AMBER if abs(s) <= 8 else MUTED
        bars_.add(Rectangle(width=unit * 1.6, height=h, stroke_width=1.5, stroke_color=c, fill_color=c,
                            fill_opacity=0.5).move_to([s * unit, base + h / 2, 0]))
    axis = Line([-32 * unit - 0.3, base, 0], [32 * unit + 0.3, base, 0], color=INK, stroke_width=2)
    ticks = VGroup(*[M(str(s), size=30, color=MUTED).move_to([s * unit, base - 0.3, 0]) for s in (-32, -8, 0, 8, 32)])
    return bars_, axis, ticks


def spikes(scale, x0, width=5.4, max_h=2.3, base=-2.3, seed=3):
    """softmax of 512 independent N(0, scale^2) scores, drawn as one comb (a single sample, illustrative)."""
    z = np.random.default_rng(seed).normal(0, scale, 512)
    p = np.exp(z - z.max())
    p /= p.sum()
    pts = []
    for i, v in enumerate(p):
        x = x0 + i * width / 511
        pts += [[x, base, 0], [x, base + max_h * v, 0], [x, base, 0]]
    comb_ = VMobject(stroke_color=AMBER, stroke_width=1.6).set_points_as_corners(pts)
    axis = Line([x0 - 0.1, base, 0], [x0 + width + 0.1, base, 0], color=MUTED, stroke_width=2)
    return VGroup(axis, comb_), float(p.max())


def token_grid(words=("The", "cat", "sat", "on"), size=0.85):
    cells = VGroup(*[Square(size, stroke_color=MUTED, stroke_width=2).move_to([j * size, -t * size, 0])
                     for t in range(len(words)) for j in range(len(words))])
    rows = VGroup(*[T(w, 30, Q_BLUE).next_to(cells[t * len(words)], LEFT, buff=0.25) for t, w in enumerate(words)])
    cols = VGroup(*[T(w, 30, K_VIOLET).next_to(cells[j], UP, buff=0.18) for j, w in enumerate(words)])
    return VGroup(cells, rows, cols)


class S01Coins(Segment):
    level, title = ZERO, "开场：抛 64 次硬币"

    def header(self):
        kicker = T("DeepSeek 科技树 · 第 1 课 §4 · 黑板课", 28, MUTED)
        name = T("注意力（二）：缩放、因果掩码与多头", 54)
        legend = VGroup(
            VGroup(badge(ZERO, 26), T("谁都能听懂：比方、手算、结论", 30)).arrange(RIGHT, buff=0.35),
            VGroup(badge(ENG, 26), T("要一点线性代数、概率和深度学习的底子，第一次看可以跳过", 30)).arrange(RIGHT, buff=0.35),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        card = VGroup(kicker, name, legend).arrange(DOWN, buff=0.6)
        with self.say("这是 DeepSeek 科技树第一课第四页的黑板课。{a}视频分段标了难度：绿色“零基础”，谁都能听懂；"
                      "{b}橙色“工程师向”，要一点线性代数和深度学习的底子，第一次看可以跳过。") as v:
            self.play(FadeIn(kicker), Write(name), run_time=1.5)
            v.until("a")
            self.play(FadeIn(legend[0], shift=RIGHT * 0.2))
            v.until("b")
            self.play(FadeIn(legend[1], shift=RIGHT * 0.2))
        self.play(FadeOut(card), run_time=0.6)
        super().header()

    def body(self):
        with self.say("先请大家做一个思想实验。{a}抛一枚硬币，正面记加一，反面记减一。") as v:
            coin = VGroup(L(r"正面 $\to +1$", 44, V_GREEN), L(r"反面 $\to -1$", 44, RED)).arrange(RIGHT, buff=1.5)
            self.put(coin, center=True, first_y=2.7)
            v.until("a")
            self.play(Write(coin[0]), run_time=0.8)
            self.play(Write(coin[1]), run_time=0.8)
        with self.say("抛 64 次，把结果全部加起来。{a}大家猜一猜，这个和通常落在什么范围？{b}[±64|正负六十四]？{c}还是[±32|正负三十二]？") as v:
            q = self.put(L(r"抛 64 次，和 $S$ 通常落在哪里？", 44), center=True, gap=0.6)
            v.until("a")
            self.play(Write(q), run_time=1.2)
            v.until("b")
            g = VGroup(M(r"\pm 64\ ?", size=60, color=MUTED), M(r"\pm 32\ ?", size=60, color=MUTED)).arrange(RIGHT, buff=2.2)
            self.put(g, center=True, gap=0.8)
            self.play(FadeIn(g[0], shift=UP * 0.2))
            v.until("c")
            self.play(FadeIn(g[1], shift=UP * 0.2))
        bars_, axis, ticks = coin_hist()
        with self.say("都不是。{a}它通常只在[±8|正负八]上下：大约四分之三的机会，落在负八到八之间；{b}绝对值达到三十二的机会，连万分之一都不到。") as v:
            self.play(FadeOut(coin), FadeOut(g), q.animate.set_y(2.75), run_time=0.6)
            self.play(Create(axis), FadeIn(ticks), run_time=0.6)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars_], lag_ratio=0.04), run_time=1.6)
            v.until("a")
            band = Rectangle(width=16 * 0.19 + 0.36, height=2.75, stroke_width=0, fill_color=AMBER,
                             fill_opacity=0.12).move_to([0, -2.2 + 1.375, 0])
            frac = L(r"约 $3/4$", 36, AMBER).next_to(band, UP, buff=0.12)
            self.play(FadeIn(band), Write(frac))
            v.until("b")
            far = VGroup(*[Line([s * 0.19, -2.2, 0], [s * 0.19, -1.3, 0], color=RED, stroke_width=4) for s in (-32, 32)])
            far_t = L(r"$|S|\ge 32$：不到万分之一", 30, RED).move_to([4.3, -0.9, 0])
            self.play(Create(far), Write(far_t))
        with self.say("为什么偏偏是八？{a}因为八等于根号六十四。") as v:
            eq = M(r"8=\sqrt{64}", size=66, color=AMBER).move_to([-4.3, 0.9, 0])
            v.until("a")
            self.play(Write(eq))
            self.play(Create(chalk_box(eq)))
        self.wipe()
        with self.say("这跟注意力有什么关系？{a}注意力的打分，q 和 k 的点积，{b}恰好就是这样一个“六十四项相加”。") as v:
            dot = M(r"\mathbf q^\top", r"\mathbf k", "=", r"\sum_{i=1}^{64}", r"q_i", r"k_i", size=62)
            dot[0].set_color(Q_BLUE), dot[1].set_color(K_VIOLET), dot[4].set_color(Q_BLUE), dot[5].set_color(K_VIOLET)
            self.put(dot, center=True, first_y=2.85)
            v.until("a")
            self.play(Write(dot), run_time=1.5)
            v.until("b")
            br = Brace(dot[3:], DOWN, color=AMBER)
            lab = T("64 项相加，就像抛 64 次硬币", 34, AMBER).next_to(br, DOWN, buff=0.15)
            self.play(GrowFromCenter(br), Write(lab))
        with self.say("所以这一页回答三个问题：{a}为什么打分要除以[√d_k|根号 d k]；{b}为什么要在 softmax 之前加上负无穷；"
                      "{c}为什么 GPT 2 用十二个 64 维的头，而不是一个 768 维的大头。") as v:
            qs = VGroup(
                L(r"① 为什么要除以 $\sqrt{d_k}$？", 34),
                L(r"② 为什么 softmax 之前要加 $-\infty$？", 34),
                L(r"③ 为什么用 12 个 64 维的头，不用 1 个 768 维的头？", 34),
            ).arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to([0, -1.6, 0])
            for key, item in zip("abc", qs):
                v.until(key)
                self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.6)


class S02FourCoins(Segment):
    level, title = ZERO, "黑板上：4 维点积 = 抛 4 次硬币"

    def body(self):
        self.narrate("先把 64 缩成 4，在黑板上一种一种地数。")
        with self.say("取[d_k|d k]等于四。q 和 k 的每个分量，都随机取加一或者减一，各占一半，互相独立。"):
            l1 = self.put(L(r"取 $d_k=4$：$q_i,\ k_i$ 随机取 $+1$ 或 $-1$，各占一半，互相独立"))
            self.play(Write(l1), run_time=1.8)
        with self.say("那么每个乘积 q i 乘 k i，也是加一或者减一，各占一半。{a}点积就是四个正负一相加，和抛四次硬币一模一样。") as v:
            l2 = self.put(L(r"$q_ik_i=\pm1$，点积 $=$ 4 个 $\pm1$ 相加 $=$ 抛 4 次硬币"))
            v.until("a")
            self.play(Write(l2), run_time=1.8)
        tb = table([["点积的值", "有几种", "值的平方"], ["$-4$", "1", "16"], ["$-2$", "4", "4"], ["$0$", "6", "0"],
                    ["$2$", "4", "4"], ["$4$", "1", "16"]], size=30, h_buff=0.45, v_buff=0.16)
        tb.move_to([-3.9, -0.8, 0])
        with self.say("四次硬币，一共十六种等可能的结果。{a}按点积的值分一分：{b}负四有一种，负二有四种，零有六种，二有四种，四有一种。") as v:
            v.until("a")
            self.play(FadeIn(tb[0]), Create(tb[-1]))
            v.until("b")
            self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.1) for r in tb[1:-1]], lag_ratio=0.6), run_time=4.0)
        x0 = -0.95
        with self.say("均值是零，因为左右对称。{a}方差是“平方的平均”：{b}十六乘一，加四乘四，加零，加四乘四，加十六乘一，一共六十四，{c}再除以十六，等于四。") as v:
            m0 = L(r"均值 $=0$（左右对称）", 34)
            m0.move_to([x0 + m0.width / 2, 1.05, 0])
            self.play(Write(m0))
            v.until("a")
            var = L(r"方差 $=\dfrac{16\cdot1+4\cdot4+0\cdot6+4\cdot4+16\cdot1}{16}$", 30)
            var.move_to([x0 + var.width / 2, 0.15, 0])
            self.play(Write(var[0]))
            v.until("b")
            self.play(Write(var[1]), run_time=2.5)
            v.until("c")
            res = L(r"$=\dfrac{64}{16}=4$", 34, AMBER)
            res.next_to(var, DOWN, buff=0.2).align_to(var[1], LEFT)
            self.play(Write(res))
        with self.say("方差恰好等于[d_k|d k]，也就是四；标准差是根号四，等于二。{a}如果每个分数都除以二，取值就变成负二、负一、零、一、二，{b}方差回到一。") as v:
            sd = L(r"标准差 $=\sqrt4=2$", 34)
            sd.move_to([x0 + sd.width / 2, -1.65, 0])
            self.play(Write(sd))
            v.until("a")
            fix = L(r"都除以 2：$-2,-1,0,1,2$，方差 $=1$", 34, AMBER)
            fix.move_to([x0 + fix.width / 2, -2.3, 0])
            self.play(Write(fix))
            v.until("b")
            self.play(Create(chalk_box(fix)))


class S03Variance(Segment):
    level, title = ENG, "推广：点积的方差为什么等于维度"

    def body(self):
        with self.say("好，现在把四换成一般的[d_k|d k]，把硬币换成一般的随机数。{a}需要两条关于方差的事实。") as v:
            pre = self.put(L(r"前提：$q_i,\ k_i$ 相互独立，均值 0，方差 1", 36, AMBER))
            self.play(Write(pre), run_time=1.5)
        with self.say("第一条：两个独立的随机变量，乘积的期望等于期望的乘积。{a}第二条：独立的随机变量相加，方差也相加。") as v:
            f1 = self.put(L(r"① $X,Y$ 独立 $\Rightarrow\ \mathrm E[XY]=\mathrm E[X]\,\mathrm E[Y]$", 36), gap=0.45)
            self.play(Write(f1), run_time=1.5)
            v.until("a")
            f2 = self.put(L(r"② 独立 $\Rightarrow\ \mathrm{Var}(X_1+\cdots+X_n)=\mathrm{Var}(X_1)+\cdots+\mathrm{Var}(X_n)$", 36))
            self.play(Write(f2), run_time=1.8)
        self.narrate("原论文的脚注，就是在这个前提下做的估算。黑板上的正负一硬币，正好满足这个前提。")
        self.wipe()
        x = SAFE_LEFT + 0.25
        rows = []

        def step(tex, reason, gap=0.3):
            m = self.put(M(tex, size=36), gap=gap, x=x)
            r = why(m, reason, x=2.4)
            rows.append((m, r))
            return m, r

        with self.say("先算均值：{a}期望可以拆开，q i 和 k i 又独立，每一项都是零乘零，{b}所以点积的均值是零。") as v:
            m, r = step(r"\mathrm E[\mathbf q^\top\mathbf k]=\sum_{i=1}^{d_k}\mathrm E[q_i]\,\mathrm E[k_i]=0", "期望可拆；独立；均值 0", 0.55)
            v.until("a")
            self.play(Write(m), run_time=1.8)
            v.until("b")
            self.play(FadeIn(r))
        with self.say("再算方差：{a}[d_k|d k]个乘积相互独立，方差相加。{b}每一项的方差，是 q i 平方的期望乘 k i 平方的期望，一乘一，等于一。{c}[d_k|d k]个一相加，就是[d_k|d k]。") as v:
            m2, r2 = step(r"\mathrm{Var}(\mathbf q^\top\mathbf k)=\sum_{i=1}^{d_k}\mathrm{Var}(q_ik_i)", "乘积相互独立，方差相加")
            m3, r3 = step(r"\mathrm{Var}(q_ik_i)=\mathrm E[q_i^2]\,\mathrm E[k_i^2]-0=1\times1", "方差的定义")
            m4, r4 = step(r"\mathrm{Var}(\mathbf q^\top\mathbf k)=d_k", "$d_k$ 个 1 相加")
            v.until("a")
            self.play(Write(m2), FadeIn(r2), run_time=1.6)
            v.until("b")
            self.play(Write(m3), FadeIn(r3), run_time=1.6)
            v.until("c")
            self.play(Write(m4), FadeIn(r4), run_time=1.0)
        with self.say("最后，除以[√d_k|根号 d k]，方差要除以[d_k|d k]，{a}正好回到一。") as v:
            m5, r5 = step(r"\mathrm{Var}\Big(\frac{\mathbf q^\top\mathbf k}{\sqrt{d_k}}\Big)=\frac{1}{d_k}\cdot d_k=1", "常数提出来要平方", 0.42)
            self.play(Write(m5), FadeIn(r5), run_time=1.6)
            v.until("a")
            self.play(Create(chalk_box(m5)))
        self.wipe()
        with self.say("代回来看：[d_k|d k]等于四，标准差二，和黑板上数的一样。{a}等于 64，标准差八，正是开头那 64 枚硬币；{b}等于 128，标准差约 11.31。") as v:
            t = table([["$d_k$", "点积的标准差", "除以 $\\sqrt{d_k}$ 后"], ["$4$", "$2$", "$1$"], ["$64$", "$8$", "$1$"],
                       ["$128$", "$\\sqrt{128}\\approx11.31$", "$1$"]], size=34, h_buff=1.0, v_buff=0.3)
            self.put(t, center=True, first_y=2.8)
            self.play(FadeIn(t[0]), Create(t[-1]), FadeIn(t[1]))
            v.until("a")
            self.play(FadeIn(t[2]))
            v.until("b")
            self.play(FadeIn(t[3]))
        with self.say("我们也用 Python 验证了一下：{a}二十万对 64 维的随机向量，点积的样本方差是 63.77，理论值 64；{b}除以八之后，方差是 0.996。") as v:
            s1 = self.put(L(r"200,000 对 64 维随机向量（分量取自标准正态分布）", 32, MUTED), center=True, gap=0.7)
            self.play(FadeIn(s1))
            v.until("a")
            s2 = self.put(L(r"点积的样本方差 $63.77$（理论 $64$），标准差 $7.985$（理论 $8$）", 34), center=True)
            self.play(Write(s2), run_time=1.5)
            v.until("b")
            s3 = self.put(L(r"除以 8 之后：方差 $0.996$", 34, AMBER), center=True)
            self.play(Write(s3))


class S04Temperature(Segment):
    level, title = ZERO, "温度：分数一放大，第一名通吃"

    def body(self):
        self.narrate("方差大一点，有什么要紧？要紧在 softmax。")
        with self.say("我们拿一组固定的分数：二、一、零点一。{a}把它们放大不同的倍数，再送进 softmax，看看结果。") as v:
            z = self.put(L(r"分数 $\mathbf z=(2,\ 1,\ 0.1)$，放大几倍，再做 softmax", 36), center=True)
            self.play(Write(z), run_time=1.5)
        data = [(1, (0.6590, 0.2424, 0.0986)), (2, (0.8638, 0.1169, 0.0193)), (4, (0.9815, 0.0180, 0.0005)),
                (8, (0.9997, 0.0003, 0.0000))]
        groups = VGroup()
        for m, vals in data:
            g = bars(vals, labels=["2", "1", "0.1"], colors=[AMBER, INK, INK], max_h=2.1, vmax=1, fmt="{:.4f}",
                     size=19, bar_w=0.5, gap=0.45)
            cap = M(rf"\times{m}", size=40).next_to(g, DOWN, buff=0.5)
            groups.add(VGroup(g, cap))
        groups.arrange(RIGHT, buff=0.45, aligned_edge=DOWN).move_to([0, -0.85, 0])
        with self.say("不放大的时候，三个概率是 0.66、0.24、0.10。{a}放大两倍，第一名涨到 0.86。{b}放大四倍，0.98。{c}放大八倍，0.9997，剩下两个几乎是零。") as v:
            self.play(FadeIn(groups[0], shift=UP * 0.2))
            for key, g in zip("abc", groups[1:]):
                v.until(key)
                self.play(FadeIn(g, shift=UP * 0.2))
        with self.say("大家从左往右看：分数一放大，第一名就把别人全挤没了。{a}[d_k|d k]等于 64 时不缩放，就相当于把分数放大八倍。") as v:
            self.play(Indicate(VGroup(*[g[0][0][0] for g in groups]), color=AMBER, scale_factor=1.05), run_time=1.5)
            v.until("a")
            box = chalk_box(groups[3], RED)
            note = L(r"$d_k=64$ 不缩放", 30, RED).next_to(box, UP, buff=0.1)
            self.play(Create(box), Write(note))
        self.wipe()
        with self.say("所以，除以[√d_k|根号 d k]，就是把 softmax 的温度，固定在“分数的标准差约为一”的地方；{a}不除，头越宽，分布越尖。") as v:
            k = self.put(L(r"除以 $\sqrt{d_k}$ $=$ 把 softmax 的“温度”固定在分数标准差 $\approx1$ 处", 36, AMBER), center=True)
            self.play(Write(k), run_time=1.8)
            v.until("a")
            k2 = self.put(L(r"不除：头越宽 $\Rightarrow$ 分数越大 $\Rightarrow$ 分布越尖", 36), center=True, gap=0.35)
            self.play(Write(k2), run_time=1.5)
        left, pmax1 = spikes(1, -6.4)
        right, pmax8 = spikes(8, 1.0)
        with self.say("拿 512 个位置试一试：{a}分数的标准差是一的时候，最大的那个权重，平均只有 0.027；{b}标准差变成八，平均涨到大约 0.73。") as v:
            cap1 = L(r"标准差 1：最大权重平均 $0.027$", 30).next_to(left, UP, buff=0.25)
            cap2 = L(r"标准差 8：最大权重平均 $\approx0.73$", 30, AMBER).next_to(right, UP, buff=0.25).set_y(cap1.get_y())
            note = T("（各为一次抽样示意，512 个位置）", 24, MUTED).move_to([0, -2.6, 0])
            v.until("a")
            self.play(Create(left), Write(cap1), FadeIn(note), run_time=1.4)
            v.until("b")
            self.play(Create(right), Write(cap2), run_time=1.4)


class S05Saturation(Segment):
    level, title = ENG, "饱和了为什么学不动"

    def body(self):
        with self.say("分布尖一点，有什么坏处？坏在训练。{a}我们看 softmax 的导数。") as v:
            v.until("a")
        with self.say("第 i 个输出对第 j 个输入的导数，{a}等于 p i 乘以，delta i j 减去 p j。") as v:
            f = M(r"\frac{\partial p_i}{\partial z_j}=p_i\,(\delta_{ij}-p_j)", size=58)
            self.put(f, center=True, first_y=2.9)
            v.until("a")
            self.play(Write(f), run_time=1.6)
        d1 = bars((0.2247, 0.1837, 0.0889), labels=["1", "2", "3"], colors=V_GREEN, max_h=1.6, vmax=0.25, fmt="{:.4f}",
                  size=21, bar_w=0.6, gap=0.75)
        d8 = bars((0.00034, 0.00034, 2.5e-7), labels=["1", "2", "3"], colors=RED, max_h=1.6, vmax=0.25,
                  texts=["0.00034", "0.00034", "2.5e-7"], size=21, bar_w=0.6, gap=0.75)
        VGroup(d1, d8).arrange(RIGHT, buff=3.2, aligned_edge=DOWN).move_to([0, -0.4, 0])
        c1 = VGroup(T("不放大", 28, V_GREEN), M(r"\mathbf p=(0.6590,\ 0.2424,\ 0.0986)", size=28)).arrange(DOWN, buff=0.12)
        c8 = VGroup(T("放大 8 倍", 28, RED), M(r"\mathbf p\approx(0.99966,\ 0.00034,\ 2.5\times10^{-7})", size=28)).arrange(DOWN, buff=0.12)
        c1.next_to(d1, DOWN, buff=0.25)
        c8.next_to(d8, DOWN, buff=0.25)
        with self.say("看对角线，也就是 p i 乘以一减 p i。{a}不放大的时候，三个值是 0.22、0.18、0.09：输入动一点，输出就跟着动。") as v:
            diag = L(r"对角线 $p_i(1-p_i)$：", 34).next_to(f, DOWN, buff=0.35)
            self.play(Write(diag))
            v.until("a")
            self.play(FadeIn(d1, shift=UP * 0.2), Write(c1))
        with self.say("放大八倍以后，只剩万分之三点四、万分之三点四，和千万分之二点五。{a}输入再怎么动，输出几乎纹丝不动。") as v:
            self.play(FadeIn(d8, shift=UP * 0.2), Write(c8))
        self.wipe()
        with self.say("到了极限，softmax 的输出变成独热分布，它对所有分数的导数都是零，{a}梯度传不回 q 和 k，{b}这个头的打分方式就学不动了。") as v:
            chain = self.put(L(r"输出变成独热 $\Rightarrow$ 对所有分数的导数 $=0$", 38), center=True, first_y=2.6)
            self.play(Write(chain), run_time=1.6)
            v.until("a")
            chain2 = self.put(L(r"$\Rightarrow$ 梯度传不回 $\mathbf q,\ \mathbf k$ $\Rightarrow$ 这个头学不动", 38, RED), center=True)
            self.play(Write(chain2), run_time=1.4)
        with self.say("这里要诚实地补两句。{a}第一，这是估算，不是证明，论文自己用的词是“我们猜测”。{b}第二，严格为零只在完全独热的极限；"
                      "更准确的说法是：不缩放时，softmax 的温度被头的维度绑架了。") as v:
            h0 = self.put(T("两处要诚实的地方", 34, AMBER), gap=0.7)
            self.play(FadeIn(h0))
            v.until("a")
            h1 = self.put(L(r"① 这是估算，不是“除以 $\sqrt{d_k}$ 最好”的证明", 32))
            h1b = self.put(T("原文用词：We suspect（我们猜测）", 28, MUTED), gap=0.15, x=SAFE_LEFT + 0.85)
            self.play(Write(h1), FadeIn(h1b), run_time=1.5)
            v.until("b")
            h2 = self.put(T("② 严格为 0 只在完全独热的极限", 32), gap=0.4)
            h2b = self.put(T("更准确：不缩放时，softmax 的温度被头的维度绑架", 28, MUTED), gap=0.15, x=SAFE_LEFT + 0.85)
            self.play(Write(h2), FadeIn(h2b), run_time=1.5)
        self.wipe()
        with self.say("还有个小故事：这个脚注一开始并没有。{a}2017 年 6 月，[arXiv|archive]上的第一版只有一句“点积可能过大”的猜测；{b}方差估算是一周后的第二版才补上的。") as v:
            s0 = self.put(T("小故事 · 方差脚注是一周后补上的", 36, AMBER), center=True)
            self.play(Write(s0))
            v.until("a")
            s1 = self.put(T("2017 年 6 月 arXiv 第一版：只有一句猜测，“点积可能过大”", 30), center=True, gap=0.45)
            self.play(FadeIn(s1))
            v.until("b")
            s2 = self.put(T("一周后的第二版：补上方差估算的脚注", 30), center=True)
            self.play(FadeIn(s2))
        with self.say("在 GPT 2 的代码里，这件事只有一行：{a}把分数乘以一除以根号 64，也就是乘八分之一。") as v:
            c = code("w = w * tf.rsqrt(v.shape[-1])", 30)
            eqc = M(r"=\ w\times\frac{1}{\sqrt{64}}=\frac{w}{8}", size=42, color=AMBER)
            row = VGroup(c, eqc).arrange(RIGHT, buff=0.4)
            self.put(row, center=True, gap=0.7)
            self.play(FadeIn(c))
            v.until("a")
            self.play(Write(eqc))


class S06Mask(Segment):
    level, title = ZERO, "因果掩码：盖住答案的那张纸"

    def body(self):
        words = [("apple", "苹果"), ("book", "书"), ("cat", "猫"), ("dog", "狗")]
        rows = VGroup(*[VGroup(T(e, 40, Q_BLUE), T(c, 40)) for e, c in words])
        for i, r in enumerate(rows):
            r[0].move_to([-5.2, 1.6 - i * 0.8, 0], aligned_edge=LEFT)
            r[1].move_to([-2.6, 1.6 - i * 0.8, 0], aligned_edge=LEFT)
        def cover(k):  # a sheet over rows k..3
            top, bottom = 1.6 - (k - 0.5) * 0.8, 1.6 - 3.5 * 0.8
            return Rectangle(width=4.6, height=top - bottom, fill_color="#D9D3C2", fill_opacity=0.96,
                             stroke_width=0).move_to([-3.35, (top + bottom) / 2, 0])

        paper = cover(1)
        with self.say("换个话题。小时候背单词，大家多半用过这一招：{a}拿一张纸，盖住下面还没背到的行；{b}背一行，往下挪一行，对一次答案。") as v:
            self.play(FadeIn(rows))
            v.until("a")
            self.play(FadeIn(paper, shift=DOWN * 0.3))
            v.until("b")
            for k in (2, 3):
                self.play(Transform(paper, cover(k)), run_time=0.8)
                self.wait(0.3)
        with self.say("那张纸，就是这一节的主角。") as v:
            lab = T("那张纸 = 因果掩码", 40, AMBER).move_to([3.4, 0.6, 0])
            self.play(Write(lab))
        self.wipe()
        toks = VGroup(*[VGroup(RoundedRectangle(width=1.6, height=0.8, corner_radius=0.1, stroke_color=Q_BLUE), T(w, 36))
                        for w in ("The", "cat", "sat", "on")]).arrange(RIGHT, buff=0.7).move_to([0, 1.7, 0])
        for b in toks:
            b[1].move_to(b[0])
        preds = VGroup(*[T(w, 32, V_GREEN) for w in ("cat", "sat", "on", "…")])
        arrows = VGroup()
        for b, p in zip(toks, preds):
            p.next_to(b, DOWN, buff=1.0)
            arrows.add(Arrow(b.get_bottom(), p.get_top(), buff=0.1, color=MUTED, stroke_width=3))
        with self.say("训练的时候，一句话里的所有 token 是同时喂进去的，{a}每个位置都要预测它的下一个 token。") as v:
            self.play(FadeIn(toks, shift=DOWN * 0.2))
            v.until("a")
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2), FadeIn(preds), run_time=1.4)
        with self.say("问题来了：如果位置一的注意力能看到位置二，{a}它只要把位置二的输入抄过来，就能完美地“预测”出位置二。{b}损失降得飞快，模型却什么都没学会。") as v:
            v.until("a")
            cheat = CurvedArrow(toks[1].get_bottom() + DOWN * 0.1, preds[0].get_right() + RIGHT * 0.1, color=RED,
                                angle=-TAU / 5)
            ct = T("抄答案！", 34, RED).next_to(cheat, DOWN, buff=0.15)
            self.play(Create(cheat), Write(ct))
            v.until("b")
            loss = L(r"训练损失 $\downarrow$，真本事 $=0$", 34, RED).move_to([0, -2.2, 0])
            self.play(Write(loss))
        self.wipe()
        g = token_grid().move_to([-2.2, -0.3, 0])
        cells, rlab, clab = g
        n = 4
        masked = [cells[t * n + j] for t in range(n) for j in range(n) if j > t]
        infs = VGroup(*[M(r"-\infty", size=28, color=RED).move_to(c) for c in masked])
        side = VGroup(T("行：谁在看（查询）", 28, Q_BLUE), T("列：被看的位置（键）", 28, K_VIOLET),
                      L(r"右上角 $j>t$：未来，不许看", 28, RED)).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([3.6, 0.3, 0])
        with self.say("办法就是那张纸：在打分表上，把不许看的格子，也就是右上角那一半，{a}全部加上负无穷。") as v:
            self.play(Create(cells), FadeIn(rlab), FadeIn(clab), FadeIn(side), run_time=1.4)
            v.until("a")
            self.play(*[c.animate.set_fill(RED, 0.25) for c in masked], Write(infs), run_time=1.2)
        with self.say("但是背单词这个比方，有个地方不对：{a}背单词时，纸是一行一行往下挪的；{b}注意力里，所有行是同时算的，相当于给每一行各发一张纸。") as v:
            v.until("b")
            papers = VGroup()
            for t in range(n - 1):
                first = cells[t * n + t + 1]
                w = (n - 1 - t) * 0.85
                papers.add(Rectangle(width=w - 0.08, height=0.85 - 0.12, fill_color="#D9D3C2", fill_opacity=0.5,
                                     stroke_width=0).move_to(first.get_left() + RIGHT * w / 2))
            self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.2) for p in papers], lag_ratio=0.0), run_time=0.9)
            note = T("每一行各一张纸，同时盖好", 30, AMBER).next_to(side, DOWN, buff=0.5).align_to(side, LEFT)
            self.play(Write(note))


class S07MaskMath(Segment):
    level, title = ENG, "掩码的数学：先加负无穷，再做 softmax"

    def body(self):
        x = SAFE_LEFT + 0.25

        def line(label, tex, color=INK, gap=0.38):
            row = VGroup(T(label, 32, MUTED), M(*tex, size=42, color=color)).arrange(RIGHT, buff=0.3)
            return self.put(row, gap=gap, x=x)

        with self.say("看一行具体的数。上一页 cat 那一行，缩放后的分数是[1.4142|一点四一四二]、[0|零]、[1.4142|一点四一四二]，{a}第三格是“未来”的 sat。") as v:
            r0 = line("cat 这一行，缩放后的分数：", [r"(1.4142,\ 0,\ ", r"1.4142", r")"])
            self.play(Write(r0), run_time=1.6)
            v.until("a")
            fut = T("← sat，未来", 30, RED).next_to(r0, RIGHT, buff=0.3)
            self.play(r0[1][1].animate.set_color(RED), FadeIn(fut))
        with self.say("加掩码，第三格变成负无穷。{a}取指数：[4.1133|四点一一三三]、[1|一]、[0|零]，{b}加起来[5.1133|五点一一三三]。"
                      "{c}再除以和：[0.8044|零点八零四四]、[0.1956|零点一九五六]、[0|零]。") as v:
            r1 = line("加掩码：", [r"(1.4142,\ 0,\ ", r"-\infty", r")"])
            r1[1][1].set_color(RED)
            self.play(Write(r1), run_time=1.2)
            v.until("a")
            r2 = line("取指数：", [r"(4.1133,\ 1,\ 0)"])
            self.play(Write(r2), run_time=1.2)
            v.until("b")
            s = M(r"\text{sum}=5.1133", size=38, color=MUTED).next_to(r2, RIGHT, buff=0.5)
            self.play(Write(s))
            v.until("c")
            r3 = line("除以和：", [r"(0.8044,\ 0.1956,\ 0)"], color=AMBER)
            self.play(Write(r3), run_time=1.2)
            self.play(Create(chalk_box(r3[1])))
        with self.say("被盖住的那一格，权重正好是零，剩下两格加起来仍然是一，{a}不用再做任何修补。") as v:
            ok = self.put(L(r"盖住的一格 $=0$；其余 $0.8044+0.1956=1$，不用修补", 32, V_GREEN), gap=0.5, x=x)
            self.play(Write(ok), run_time=1.4)
        self.wipe()
        with self.say("如果反过来，先做 softmax 再清零，每一行就加不到一了，还得再归一化一次。{a}先加负无穷，一步到位。") as v:
            bad = self.put(L(r"反过来：先 softmax 再清零 $\Rightarrow$ 行和 $\ne1$，还要再归一化", 34, RED), center=True,
                           first_y=2.2)
            self.play(Write(bad), run_time=1.4)
        with self.say("GPT 2 的代码没有用真正的负无穷，而是减去十的十次方；{a}e 的负十的十次方，在浮点数里就是零。") as v:
            c = code("w = w*b - 1e10*(1-b)", 28)
            e = L(r"$e^{-10^{10}}$ 在浮点数里就是 0", 30, MUTED)
            row = VGroup(c, e).arrange(RIGHT, buff=0.5)
            self.put(row, gap=0.9, center=True)
            self.play(FadeIn(c))
            v.until("a")
            self.play(Write(e))
        self.wipe()
        with self.say("更要紧的是一个结论：{a}整张网络在位置 t 的每个中间结果，都只依赖前 t 个 token，{b}因为只有注意力会混合位置，而掩码恰好在那里挡住了后面。") as v:
            v.until("a")
            big = M(r"\mathbf h^{(l)}_t=f(x_1,\dots,x_t)", size=62, color=AMBER)
            self.put(big, center=True, first_y=2.8)
            self.play(Write(big), run_time=1.5)
            v.until("b")
            n1 = self.put(L(r"只有注意力会混合位置；掩码让 $j>t$ 的权重为 0（对层数归纳）", 32, MUTED), center=True)
            self.play(FadeIn(n1))
        with self.say("这带来两个后果。第一，训练是诚实的，而且是并行的：{a}一段 1024 个 token 的文本，一次前向，就给出 1024 个合法的训练信号。") as v:
            c1 = self.put(L(r"① 训练诚实又并行：1024 个 token $\Rightarrow$ 一次前向 1024 个训练信号", 34), gap=0.6, center=True)
            v.until("a")
            self.play(Write(c1), run_time=1.8)
        with self.say("第二，旧位置的结果永远不变。{a}生成时，可以把旧位置的 K 和 V 存起来反复用，这就是第四课的 KV cache，{b}而且它是精确的优化，不是近似。") as v:
            c2 = self.put(L(r"② 旧位置的 $K,\ V$ 永远不变 $\Rightarrow$ KV cache（第 4 课）", 34, AMBER), center=True)
            v.until("a")
            self.play(Write(c2), run_time=1.6)
            v.until("b")
            c3 = self.put(T("精确的优化，不是近似", 32, AMBER), center=True, gap=0.25)
            self.play(Write(c3))


class S08Heads(Segment):
    level, title = ZERO, "多头：几位评委一起打分"

    def body(self):
        one = VGroup(T("1 位评委", 38), T("→ 一个总分", 34, MUTED)).arrange(DOWN, buff=0.3)
        three = VGroup(T("3 位评委", 38), VGroup(T("内容", 32, Q_BLUE), T("结构", 32, AMBER), T("文笔", 32, V_GREEN))
                       .arrange(RIGHT, buff=0.5)).arrange(DOWN, buff=0.3)
        VGroup(one, three).arrange(RIGHT, buff=3.0).move_to([0, 1.2, 0])
        with self.say("最后一个问题：多头。大家看过作文比赛打分吧。{a}只请一位评委，他只能打一个总分，把内容、结构、文笔全揉在一起；"
                      "{b}请三位评委，一位盯内容，一位盯结构，一位盯文笔，信息就多得多。") as v:
            v.until("a")
            self.play(FadeIn(one, shift=UP * 0.2))
            v.until("b")
            self.play(FadeIn(three, shift=UP * 0.2))
        with self.say("注意力里，一个头只有一套 softmax 权重。{a}也就是说，一个位置在一个头里，只能按一种标准分配注意力。") as v:
            a1 = L(r"一个头 $=$ 一套 softmax 权重 $=$ 一种标准", 36).move_to([0, -0.6, 0])
            self.play(Write(a1), run_time=1.5)
        with self.say("多头注意力，就是把一个宽的头，拆成几个窄的头，{a}每个头各自打分、各自 softmax、各自加权平均。") as v:
            a2 = T("多头：拆成几个窄头，各自打分 → softmax → 加权平均", 36, AMBER).move_to([0, -1.6, 0])
            v.until("a")
            self.play(Write(a2), run_time=1.6)
        self.wipe()
        w1, w2, ws = (0.0159, 0.8668, 0.1173), (0.4223, 0.1554, 0.4223), (0.0743, 0.6200, 0.3057)
        names = ["The", "cat", "sat"]
        b1 = bars(w1, names, Q_BLUE, max_h=2.2, vmax=1, fmt="{:.4f}", size=22, bar_w=0.6, gap=0.45)
        b2 = bars(w2, names, AMBER, max_h=2.2, vmax=1, fmt="{:.4f}", size=22, bar_w=0.6, gap=0.45)
        bs = bars(ws, names, MUTED, max_h=2.2, vmax=1, fmt="{:.4f}", size=22, bar_w=0.6, gap=0.45)
        VGroup(b1, b2, bs).arrange(RIGHT, buff=1.4, aligned_edge=DOWN).move_to([0, -0.45, 0])
        caps = VGroup(T("头 1", 32, Q_BLUE).next_to(b1, DOWN, buff=0.3), T("头 2", 32, AMBER).next_to(b2, DOWN, buff=0.3),
                      T("只有一个头（§3）", 30, MUTED).next_to(bs, DOWN, buff=0.3))
        with self.say("黑板上试一下：还是 The cat sat 这三个词，看 sat 这一行，{a}把一个二维的头，拆成两个一维的头。") as v:
            t0 = self.put(L(r"“The cat sat”，看 sat 这一行：$d=2$ 拆成 2 个头，每头 $d_h=1$", 34), center=True)
            self.play(Write(t0), run_time=1.6)
        with self.say("头一，把[87%|百分之八十七]的权重给了 cat；{a}头二，只给 cat [16%|百分之十六]，把其余的平分给 The 和 sat 自己。{b}两位评委，两套看法。") as v:
            self.play(FadeIn(b1, shift=UP * 0.2), FadeIn(caps[0]))
            v.until("a")
            self.play(FadeIn(b2, shift=UP * 0.2), FadeIn(caps[1]))
        with self.say("而只有一个头的时候，只能给出一套折中的权重：0.07、0.62、0.31。") as v:
            self.play(FadeIn(bs, shift=UP * 0.2), FadeIn(caps[2]))
        with self.say("不过这个比方也有不对的地方：{a}评委的分工是事先定好的，头的分工是训练出来的；{b}而且每个头交出的不是一个分数，是一整套权重。") as v:
            v.until("a")
            self.wipe()
            n1 = self.put(T("比方不对的地方", 34, AMBER), center=True, first_y=1.5)
            n2 = self.put(T("① 评委的分工事先定好；头的分工是训练出来的", 34), center=True, gap=0.45)
            self.play(FadeIn(n1), Write(n2), run_time=1.4)
            v.until("b")
            n3 = self.put(T("② 每个头交出的不是一个分数，而是一整套权重", 34), center=True)
            self.play(Write(n3), run_time=1.4)


class S09Merge(Segment):
    level, title = ENG, "拆头与合头：不多花一次运算"

    def body(self):
        with self.say("把头拆开，要不要多花计算？{a}不用。") as v:
            v.until("a")
        with self.say("把十二个头的 W Q i [按行|一行一行地]叠起来，就是一个[768|七百六十八]乘[768|七百六十八]的大矩阵。"
                      "{a}大矩阵乘一次，得到[768|七百六十八]个数，{b}每[64|六十四]个切一段，第 i 段就是第 i 个头的查询。") as v:
            m = self.put(L(r"12 个头的 $W^Q_i$ 按行叠起来 $=$ 一个 $768\times768$ 的 $W^Q$", 36), center=True)
            self.play(Write(m), run_time=1.6)
            strip = VGroup(*[Rectangle(width=0.95, height=0.55, stroke_color=INK, stroke_width=1.5,
                                       fill_color=[Q_BLUE, AMBER][i % 2], fill_opacity=0.35) for i in range(12)]).arrange(RIGHT, buff=0)
            self.put(strip, center=True, gap=0.9)
            lab = T("一次乘法得到 768 个数", 30, MUTED).next_to(strip, UP, buff=0.15)
            v.until("a")
            self.play(FadeIn(strip), FadeIn(lab))
            v.until("b")
            nums = VGroup(*[T(str(i + 1), 22).move_to(r) for i, r in enumerate(strip)])
            sub = T("每 64 个一段：第 i 段 = 第 i 个头的查询", 30, AMBER).next_to(strip, DOWN, buff=0.2)
            self.play(FadeIn(nums), Write(sub))
            self.reset_cursor(sub.get_bottom()[1])
        with self.say("合头靠 W O。把 W O 按列切成十二块，{a}拼接再乘 W O，{b}就等于每个头用自己那一块，写回 768 维，再相加。") as v:
            f = M(r"W^O\begin{pmatrix}\mathbf o_1\\ \vdots\\ \mathbf o_{12}\end{pmatrix}", "=", r"\sum_{i=1}^{12}W^O_i\,\mathbf o_i",
                  size=44)
            self.put(f, center=True, gap=0.55)
            v.until("a")
            self.play(Write(f[0]), run_time=1.2)
            v.until("b")
            self.play(Write(f[1:]), run_time=1.2)
            self.play(Create(chalk_box(f[2])))
        self.wipe()
        plane = NumberPlane(x_range=[-0.25, 0.75, 0.25], y_range=[-0.25, 0.75, 0.25], x_length=3.8, y_length=3.8,
                            background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.35, "stroke_width": 1},
                            axis_config={"stroke_color": MUTED}).move_to([3.7, 0.6, 0])
        o = (0.1332, 0.5777)
        h1 = (o[0] * 1, o[0] * -1)
        h2 = (o[1] * 0.5, o[1] * 1)
        a1 = Arrow(plane.c2p(0, 0), plane.c2p(*h1), buff=0, color=Q_BLUE, stroke_width=5)
        a2 = Arrow(plane.c2p(*h1), plane.c2p(h1[0] + h2[0], h1[1] + h2[1]), buff=0, color=AMBER, stroke_width=5)
        a3 = Arrow(plane.c2p(0, 0), plane.c2p(0.4220, 0.4445), buff=0, color=V_GREEN, stroke_width=5)
        x = SAFE_LEFT + 0.25
        with self.say("回到黑板上的两个头。拼接得到[0.1332|零点一三三二]和[0.5777|零点五七七七]。{a}取 W O 等于这个二乘二的矩阵。"
                      "{b}头一沿第一列写入，头二沿第二列写入，{c}相加得到[0.4220|零点四二二零]、[0.4445|零点四四四五]。") as v:
            e1 = self.put(M(r"[o_{3,1};\ o_{3,2}]=(0.1332,\ 0.5777)", size=40), x=x, first_y=2.6)
            self.play(Write(e1), run_time=1.4)
            v.until("a")
            e2 = self.put(M(r"W^O=\begin{pmatrix}1&0.5\\-1&1\end{pmatrix}", size=40), x=x, gap=0.4)
            self.play(Write(e2), Create(plane), run_time=1.4)
            v.until("b")
            e3 = self.put(M(r"\mathbf u_3=", r"0.1332\begin{pmatrix}1\\-1\end{pmatrix}", "+", r"0.5777\begin{pmatrix}0.5\\1\end{pmatrix}",
                            size=38), x=x, gap=0.4)
            e3[1].set_color(Q_BLUE), e3[3].set_color(AMBER)
            la1 = T("头 1", 24, Q_BLUE).next_to(a1.get_end(), RIGHT, buff=0.1)
            la2 = T("头 2", 24, AMBER).next_to(a2, RIGHT, buff=0.08)
            self.play(Write(e3[:2]), GrowArrow(a1), FadeIn(la1))
            self.play(Write(e3[2:]), GrowArrow(a2), FadeIn(la2))
            v.until("c")
            e4 = self.put(M(r"=\begin{pmatrix}0.4220\\0.4445\end{pmatrix}", size=38, color=V_GREEN), x=x + 1.0, gap=0.3)
            la3 = M(r"\mathbf u_3", size=30, color=V_GREEN).next_to(a3.get_center(), LEFT, buff=0.15)
            self.play(Write(e4), GrowArrow(a3), FadeIn(la3))
        with self.say("所以 W O 做两件事：{a}把每个头的结果，放回残差流的任意方向；{b}让不同头的结果，在同一个方向上叠加。") as v:
            v.until("a")
            w1 = T("① 放回残差流的任意方向", 28, AMBER).next_to(plane, DOWN, buff=0.3)
            self.play(Write(w1))
            v.until("b")
            w2 = T("② 不同头的结果在同一方向上叠加", 28, AMBER).next_to(w1, DOWN, buff=0.15)
            self.play(Write(w2))
        self.wipe()
        with self.say("算一下账：一个 768 维的大头，打分要二 T 平方乘 768 次运算；{a}十二个 64 维的小头，是十二乘二 T 平方乘 64，{b}一样多。") as v:
            c1 = self.put(L(r"1 个 768 维的头：$2T^2\cdot768$", 36), center=True, first_y=2.8)
            self.play(Write(c1))
            v.until("a")
            c2 = self.put(L(r"12 个 64 维的头：$12\times2T^2\cdot64=2T^2\cdot768$", 36), center=True)
            self.play(Write(c2), run_time=1.5)
            v.until("b")
            self.play(Create(chalk_box(c2, V_GREEN)))
        with self.say("那头是不是越多越好？原论文在机器翻译上试过，总计算量不变：{a}8 个头，BLEU 25.8；1 个头，降到 24.9；"
                      "{b}16 个头持平；{c}32 个头、每头 16 维，又降到 25.4。") as v:
            t = table([["头数", "每头维度", "BLEU"], ["1", "512", "24.9"], ["8", "64", "25.8（基线）"], ["16", "32", "持平"],
                       ["32", "16", "25.4"]], size=32, h_buff=1.2)
            self.put(t, center=True, gap=0.55)
            v.until("a")
            self.play(FadeIn(t[0]), Create(t[-1]), FadeIn(t[2]), FadeIn(t[1]))
            v.until("b")
            self.play(FadeIn(t[3]))
            v.until("c")
            self.play(FadeIn(t[4]))
        with self.say("头太少，只能按一种标准取信息；头太多，每个头的窗口又太窄。{a}最后记一笔账：GPT 2 small 一层注意力，一共[2,362,368|二百三十六万两千三百六十八]个参数，约等于四 d 方。") as v:
            v.until("a")
            self.wipe()
            p1 = self.put(T("GPT-2 small 一层注意力的参数", 34, MUTED), center=True, first_y=1.6)
            p2 = self.put(L(r"$\underbrace{768\times2304+2304}_{\texttt{c\_attn}}+\underbrace{768\times768+768}_{W^O}=2{,}362{,}368$", 38),
                          center=True, gap=0.45)
            p3 = self.put(L(r"$\approx4d^2=4\times768^2=2{,}359{,}296$", 38, AMBER), center=True, gap=0.4)
            self.play(FadeIn(p1), Write(p2), run_time=1.8)
            self.play(Write(p3))


class S10Recap(Segment):
    level, title = ZERO, "这一页讲了三件事"

    def item(self, v, head, sub, first=False):
        h = self.put(L(head, 36), gap=0.3 if first else 0.45, first_y=2.8)
        self.play(Write(h), run_time=1.4)
        v.until("a")
        s = self.put(L(sub, 30, AMBER), gap=0.16, x=SAFE_LEFT + 0.85)
        self.play(FadeIn(s, shift=RIGHT * 0.2))

    def body(self):
        self.narrate("好，这一页就讲了三件事。")
        with self.say("第一，除以[√d_k|根号 d k]，是为了让分数不随头宽变尖：{a}就像抛硬币，抛 64 次，标准差是八。") as v:
            self.item(v, r"① 除以 $\sqrt{d_k}$：分数方差回到 1，softmax 不随头宽变尖", r"64 次硬币，标准差 $\sqrt{64}=8$", True)
        with self.say("第二，因果掩码，是在 softmax 之前加的负无穷：{a}训练又并行又诚实，旧位置的 K 和 V 可以精确地缓存下来。") as v:
            self.item(v, r"② 因果掩码：softmax 之前加 $-\infty$", r"训练并行且诚实；旧位置的 $K,\ V$ 可精确缓存")
        with self.say("第三，多头就是几套独立的 softmax，总运算量不变；{a}W O 让每个头把结果写回残差流。") as v:
            self.item(v, r"③ 多头：几套独立的 softmax，总运算量不变", r"$W^O$ 让每个头写回残差流：$\mathbf u_t=\sum_iW^O_i\mathbf o_{t,i}$")
        with self.say("下一页，我们看注意力之后的另一半：前馈网络。") as v:
            nxt = self.put(T("下一页：前馈网络 →", 34, MUTED), center=True, gap=0.5)
            self.play(FadeIn(nxt))
