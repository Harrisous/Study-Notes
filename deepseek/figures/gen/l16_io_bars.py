"""Figure 16.15: memory traffic of the residual stream per token per sublayer (bill figure, computed).

Element counts in units of d (n = 4):
  plain residual                 read 2d, write d                         -> 3d          (mHC Table 2)
  HC, unfused (mHC Table 2)      read (5n+1)d + n^2+2n, write (3n+1)d + n^2+2n -> 34d + 48
  V4 mHC, multi-pass (V4.1 2.4.1) (4n+4)d = 20d  (three kernels + the sublayer's pre-norm)
  Mega-mHC, one kernel (V4.1)    (3n+2)d = 14d
  Single-Pass mHC (V4.1)         (2n+2)d = 10d  = the lower bound
Bytes: V4.1-Flash d = 5120, assuming bf16 (2 bytes/element; the author's assumption -- V4.1 mentions an FP8
conversion inside Mega-mHC, so real bytes may be smaller). Research notes N10, N11, N16.
Note the scopes differ: Table 2 excludes the sublayer's own pre-norm; the V4.1 counts include it.
Writes deepseek/figures/l16-io-bars.svg
"""

from l16_svg import SVG

n, d = 4, 5120
ROWS = [
    ("普通残差", "读 2d + 写 d", 3, 0, "green"),
    ("HC，未融合（mHC 表 2）", "读 21d + 写 13d，另 48 个系数", 34, 48, "red"),
    ("V4 的 mHC：多趟（三个核）", "(4n+4)d，含子层的 pre-norm", 4 * n + 4, 0, "amber"),
    ("Mega-mHC：单核", "(3n+2)d", 3 * n + 2, 0, "amber"),
    ("V4.1 Single-Pass mHC", "(2n+2)d = 下界", 2 * n + 2, 0, "green"),
]
assert (5 * n + 1) + (3 * n + 1) == 34 and 2 * (n * n + 2 * n) == 48

W, H = 760, 380
svg = SVG(W, H, "残差流每个 token、每个子层的访存量（n = 4）")
X0, X1 = 250, 560
UNIT = (X1 - X0) / 36.0
Y0 = 74
RH, GAP = 30, 22

svg.text(20, 24, "每个 token、每个子层读写的元素数（以 d 为单位，n = 4）", size=14, weight="700")
for v in range(0, 37, 6):
    x = X0 + v * UNIT
    svg.line(x, Y0 - 10, x, Y0 + 5 * (RH + GAP) - GAP + 6, color="grid", width=1)
    svg.text(x, Y0 + 5 * (RH + GAP) - GAP + 22, "%dd" % v if v else "0", size=11, anchor="middle", color="muted")
svg.text(X1 + 70, Y0 - 30, "V4.1-Flash 的字节数", size=12, color="muted")
svg.text(X1 + 70, Y0 - 14, "（d = 5120，bf16）", size=12, color="muted")

for k, (name, sub, units, extra, col) in enumerate(ROWS):
    y = Y0 + k * (RH + GAP)
    svg.text(X0 - 10, y + 13, name, size=13, anchor="end", weight="600")
    svg.text(X0 - 10, y + 29, sub, size=11, anchor="end", color="muted")
    svg.rect(X0, y, units * UNIT, RH, fill=col, fill_opacity=0.55, rx=3)
    svg.text(X0 + units * UNIT + 6, y + 20, "%dd" % units + (" + %d" % extra if extra else ""), size=12,
             weight="600")
    nbytes = (units * d + extra) * 2
    svg.text(X1 + 70, y + 20, "%s 字节" % format(nbytes, ","), size=12)

yb = Y0 + 5 * (RH + GAP) + 34
svg.text(20, yb, "表 2 不含子层自己的 pre-norm；V4.1 的三个数含 pre-norm 的一读一写（2d）。字节数按 bf16 换算，是笔者的假设。", size=12,
         color="muted")
svg.save("l16-io-bars.svg")
