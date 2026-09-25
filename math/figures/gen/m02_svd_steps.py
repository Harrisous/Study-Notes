"""Figure M2.3: A = U Sigma V^T as rotate - stretch - rotate, for A = [[3, 0], [4, 5]].

Hand SVD (M02 §3): sigma = (3 sqrt5, sqrt5) = (6.708, 2.236), v1 = (1, 1)/sqrt2, v2 = (-1, 1)/sqrt2,
u1 = (1, 3)/sqrt10, u2 = (-3, 1)/sqrt10. V^T is a rotation by -45 deg, U a rotation by +71.57 deg.
Each panel shows the unit circle (or its image), the two right singular vectors carried along,
and the point e1 = (1, 0) so the reader can follow one point: e1 -> V^T e1 -> Sigma V^T e1 -> A e1 = (3, 4).
Panels 1-2 are drawn at 3x the scale of panels 3-4 (the stretch makes the ellipse 6.7 units long).
Writes math/figures/m02-svd-steps.svg
"""

import math

import numpy as np

from m02_svg import SVG, Plane, polyline_d

A = np.array([[3.0, 0.0], [4.0, 5.0]])
s1, s2 = 3 * math.sqrt(5), math.sqrt(5)
v1, v2 = np.array([1, 1]) / math.sqrt(2), np.array([-1, 1]) / math.sqrt(2)
u1, u2 = np.array([1, 3]) / math.sqrt(10), np.array([-3, 1]) / math.sqrt(10)
U = np.column_stack([u1, u2])
V = np.column_stack([v1, v2])
S = np.diag([s1, s2])
assert np.allclose(U @ S @ V.T, A)
assert abs(np.linalg.det(U) - 1) < 1e-12 and abs(np.linalg.det(V) - 1) < 1e-12
angU = math.degrees(math.atan2(u1[1], u1[0]))       # 71.565
angV = math.degrees(math.atan2(v1[1], v1[0]))       # 45

ts = np.linspace(0, 2 * np.pi, 241)
C = np.stack([np.cos(ts), np.sin(ts)])
e1 = np.array([1.0, 0.0])

W, H = 780, 330
svg = SVG(W, H, "奇异值分解的几何：A = UΣVᵀ 等于先旋转、再沿坐标轴拉伸、再旋转")

stages = [
    ("① 输入：单位圆", np.eye(2), 34, "*v*_{1}、*v*_{2} 两两垂直"),
    ("② 乘 *V*^{⊤}：旋转 −45°", V.T, 34, "*v*_{1} → *e*_{1}，*v*_{2} → *e*_{2}"),
    ("③ 乘 Σ：沿坐标轴拉伸", S @ V.T, 11.3, "横轴 ×6.71，纵轴 ×2.24"),
    ("④ 乘 *U*：旋转 +71.6°", A, 11.3, "长轴沿 *u*_{1}，短轴沿 *u*_{2}"),
]
cx = [95, 290, 490, 690]
cy = 172
BOX = 82
for k, (title, Mk, sc, sub) in enumerate(stages):
    P = Plane(cx[k], cy, sc)
    svg.rect(cx[k] - BOX, cy - BOX - 18, 2 * BOX, 2 * BOX + 36, stroke="rule", width=1, rx=6)
    svg.text(cx[k] - BOX, 30, title, size=13, weight="600")
    svg.text(cx[k] - BOX, 48, sub, size=12, color="muted")
    svg.line(cx[k] - BOX + 4, cy, cx[k] + BOX - 4, cy, color="grid", width=1.1)
    svg.line(cx[k], cy - BOX - 14, cx[k], cy + BOX + 14, color="grid", width=1.1)
    img = Mk @ C
    fill = "green" if k >= 2 else None
    svg.path(polyline_d([P(*p) for p in img.T]) + " Z", stroke="green" if k >= 2 else "muted", width=1.8,
             fill=fill, fill_opacity=0.08)
    # images of v1, v2 (blue, violet)
    a1, a2 = Mk @ v1, Mk @ v2
    svg.arrow(*P(0, 0), *P(*a1), color="blue", width=2.2)
    svg.arrow(*P(0, 0), *P(*a2), color="violet", width=2.2)
    # follow e1
    pe = Mk @ e1
    svg.circle(*P(*pe), 4, fill="ink")
    # scale bar: 1 unit
    bx, by = cx[k] + BOX - 12 - sc, cy + BOX + 10
    svg.line(bx, by, bx + sc, by, color="ink", width=1.6)
    svg.text(bx + sc / 2, by - 5, "1", size=11, color="muted", anchor="middle")
    if k == 0:
        svg.text(*(np.array(P(*a1)) + (4, -4)), "*v*_{1}", size=13, color="blue")
        svg.text(*(np.array(P(*a2)) + (-4, -4)), "*v*_{2}", size=13, color="violet", anchor="end")
        svg.text(*(np.array(P(*pe)) + (6, 14)), "*e*_{1}", size=12)
    elif k == 1:
        svg.text(*(np.array(P(*a1)) + (4, 14)), "*e*_{1}", size=12, color="blue")
        svg.text(*(np.array(P(*a2)) + (8, 16)), "*e*_{2}", size=12, color="violet")
        svg.text(*(np.array(P(*pe)) + (8, 12)), "*V*^{⊤}*e*_{1}", size=12)
    elif k == 2:
        # kept 14 units inside the ellipse's right end so the label does not touch the outline
        svg.text(*(np.array(P(*a1)) + (-14, -13)), "σ_{1} = 6.71", size=12, color="blue", anchor="end")
        svg.text(*(np.array(P(*a2)) + (0, -10)), "σ_{2} = 2.24", size=12, color="violet", anchor="middle")
    else:
        # left of the long axis' tip lies outside the ellipse, so the label does not cross the outline
        svg.text(*(np.array(P(*a1)) + (-13, -6)), "σ_{1}*u*_{1}", size=12, color="blue", anchor="end")
        svg.text(*(np.array(P(*a2)) + (-4, -8)), "σ_{2}*u*_{2}", size=12, color="violet", anchor="end")
        svg.text(*(np.array(P(*pe)) + (7, 16)), "*Ae*_{1}", size=12)
    if k < 3:
        xa = cx[k] + BOX + 3
        xb = cx[k + 1] - BOX - 3
        svg.arrow(xa, cy, xb, cy, color="ink", width=1.6)

svg.text(12, H - 36, "黑点跟踪同一个输入 *e*_{1} = (1, 0)：(1, 0) → (0.71, −0.71) → (4.74, −1.58) → (3, 4)，最后一步正是 *A* 的第一列。",
         size=12)
svg.text(12, H - 16, "③④ 的比例尺是 ①② 的 1/3（见各面板右下角的“1”）。蓝、紫箭头是 *v*_{1}、*v*_{2} 在每一步的像。", size=12,
         color="muted")

svg.save("m02-svd-steps.svg")
print("angles", angU, angV, "V^T e1", V.T @ e1, "S V^T e1", S @ V.T @ e1)
