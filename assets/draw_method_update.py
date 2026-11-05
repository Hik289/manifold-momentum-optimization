"""Draw the detailed BuresMomentum method figure used in the README.

The diagram uses the same visual grammar as the ManifoldFlow GitHub method
figure: two stacked rows, colored framed steps, matrix tiles, and a feedback
loop.  All equations are typeset by Matplotlib and the exported asset is PNG.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("fig_buresmomentum_update_detailed.png")
C = {
    "ink": "#171717", "muted": "#4B5563", "line": "#D8E1EC",
    "blue": "#1479D4", "blue_bg": "#F2F8FF",
    "orange": "#E87925", "orange_bg": "#FFF8F1",
    "amber": "#E99B24", "amber_bg": "#FFFBF2",
    "green": "#09A979", "green_bg": "#F2FCF8",
    "purple": "#A45AD5", "purple_bg": "#FBF6FF",
    "gray": "#8F9CAE",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(18, 8.0), dpi=205)
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 18), ylim=(-.43, 7.6))
ax.axis("off")


def txt(x, y, value, size=11, color="ink", weight="normal", ha="center"):
    ax.text(x, y, value, fontsize=size, color=C.get(color, color), fontweight=weight,
            va="center", ha=ha)


def box(x, y, w, h, accent, title, subtitle):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.018,rounding_size=0.065",
                                facecolor="white", edgecolor=C[accent], linewidth=1.75))
    ax.add_patch(FancyBboxPatch((x + .10, y + .13), w - .20, h - .78,
                                boxstyle="round,pad=0.012,rounding_size=0.045",
                                facecolor=C[accent + "_bg"], edgecolor=C[accent],
                                linewidth=.66, alpha=.86))
    txt(x + .14, y + h - .27, title, 12.2, "ink", "bold", ha="left")
    txt(x + .14, y + h - .53, subtitle, 8.7, "muted", ha="left")


def matrix(x, y, w=.64, h=.85, color="blue", pattern=0):
    cols, rows = (4, 4) if abs(w-h) < .15 else (4, 5)
    cw, ch = w / cols, h / rows
    for i in range(rows):
        for j in range(cols):
            alpha = .34 + .15 * ((2*i + j + pattern) % 4)
            ax.add_patch(Rectangle((x + j*cw + .012, y + i*ch + .012),
                                   cw - .025, ch - .025, facecolor=C[color],
                                   edgecolor="none", alpha=alpha))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=C[color], linewidth=1.35))


def arrow(start, end, color="gray", width=1.65, style="-|>", scale=13,
          connection="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale,
                                 linewidth=width, color=C[color],
                                 connectionstyle=connection))


# Top row: the full optimizer update.
top_y, top_h, top_w = 4.04, 3.31, 3.18
top_x = [.28, 3.89, 7.50, 11.11, 14.72]
for args in [
    (top_x[0], "blue", "Matrix gradient", "Current parameter tile"),
    (top_x[1], "orange", "First moment", "Bias-corrected gradient memory"),
    (top_x[2], "amber", "SPD statistics", "Left and right covariance"),
    (top_x[3], "blue", "Precondition + match", "Two-sided inverse-root step"),
    (top_x[4], "purple", "Parameter update", "Shared decay and weight step"),
]:
    box(args[0], top_y, top_w, top_h, args[1], args[2], args[3])
for x in [3.49, 7.10, 10.71, 14.32]:
    arrow((x, 5.73), (x+.33, 5.73), "blue")

# 1: gradient and parameter tile.
matrix(.82, 5.31, .75, 1.04, "orange", 0)
matrix(2.15, 5.31, .75, 1.04, "blue", 1)
txt(1.19, 5.10, r"$G_t$", 12, "orange")
txt(2.52, 5.10, r"$W_t$", 12, "blue")
txt(1.87, 4.54, r"$G_t=\nabla_{W}\mathcal{L}_t$", 13)

# 2: shared momentum, visualized as old state + gradient -> new state.
matrix(4.20, 5.38, .52, .83, "blue", 1)
matrix(5.15, 5.38, .52, .83, "orange", 2)
matrix(6.11, 5.38, .52, .83, "blue", 3)
txt(4.46, 5.20, r"$M_{t-1}$", 10, "blue")
txt(5.41, 5.20, r"$G_t$", 10, "orange")
txt(6.37, 5.20, r"$M_t$", 10, "blue")
txt(4.94, 5.81, "+", 17, "muted")
txt(5.90, 5.81, "=", 15, "muted")
txt(5.48, 4.67, r"$M_t=\beta_1M_{t-1}+(1-\beta_1)G_t$", 10.6)
txt(5.48, 4.32, r"$\widehat M_t=M_t/(1-\beta_1^t)$", 10.6)

# 3: the two per-axis Gram statistics.
matrix(8.11, 5.43, .70, .76, "green", 1)
matrix(9.74, 5.43, .70, .76, "green", 3)
txt(8.46, 5.19, r"$S_{L,t}$", 11, "green")
txt(10.09, 5.19, r"$S_{R,t}$", 11, "green")
txt(9.09, 4.66, r"$S_{L,t}=G_tG_t^\top/n+\varepsilon I_m$", 10.4)
txt(9.09, 4.32, r"$S_{R,t}=G_t^\top G_t/m+\varepsilon I_n$", 10.4)

# 4: preconditioning and norm matching.
for x, color, name, pat in [(11.39, "green", r"$\widetilde L_t^{-1/2}$", 0),
                            (12.26, "blue", r"$\widehat M_t$", 1),
                            (13.13, "green", r"$\widetilde R_t^{-1/2}$", 2)]:
    matrix(x, 5.48, .57, .72, color, pat)
    txt(x+.285, 5.23, name, 9.8, color)
txt(12.69, 4.69, r"$D_t^0=\widetilde L_t^{-1/2}\widehat M_t\widetilde R_t^{-1/2}$", 10.3)
txt(12.69, 4.33, r"$s_t=\mathrm{clip}_{[s_-,s_+]}\!\left(\|\widehat M_t\|_F/(\|D_t^0\|_F+\varepsilon)\right)$", 9.1)

# 5: decoupled decay and parameter update.
matrix(15.06, 5.47, .57, .72, "blue", 1)
matrix(16.00, 5.47, .57, .72, "purple", 2)
matrix(16.94, 5.47, .57, .72, "blue", 3)
txt(15.345, 5.22, r"$W_t$", 10.2, "blue")
txt(16.285, 5.22, r"$s_tD_t^0$", 9.8, "purple")
txt(17.225, 5.22, r"$W_{t+1}$", 10.2, "blue")
txt(15.80, 5.84, "−", 15, "muted")
txt(16.74, 5.84, "=", 15, "muted")
txt(16.30, 4.54, r"$W_{t+1}=(1-\eta_t\lambda_{\rm wd})W_t-\eta_ts_tD_t^0$", 10.0)

# Lower row: expand the geometric covariance branch.
low_y, low_h = .55, 3.10
box(.28, low_y, 5.40, low_h, "green", "Optimal transport map", "A: previous SPD state; B: current statistic")
box(5.90, low_y, 5.81, low_h, "green", "Bures covariance average", "The same rule updates left and right states")
box(11.93, low_y, 5.97, low_h, "green", "Spectral clipping + inverse roots", "Bounded SPD state; periodic root refresh")

# Lower 1: transport map.
for x, label_, pat in [(1.05, r"$A$", 0), (2.50, r"$T(A,B)$", 2), (3.95, r"$B$", 3)]:
    matrix(x, 1.93, .77, .83, "green", pat)
    txt(x+.385, 1.77, label_, 10.5, "green")
arrow((1.93, 2.34), (2.38, 2.34), "green", 1.6)
arrow((3.38, 2.34), (3.83, 2.34), "green", 1.6)
txt(2.98, 1.34, r"$T(A,B)=A^{-1/2}(A^{1/2}BA^{1/2})^{1/2}A^{-1/2}$", 11)
txt(2.98, .91, r"$T(A,B)\,A\,T(A,B)=B$", 10.6, "muted")

# Lower 2: geodesic rather than Euclidean averaging.
ax.add_patch(Ellipse((6.81, 2.31), .82, .52, angle=24,
                     facecolor=C["green_bg"], edgecolor=C["green"], linewidth=1.4))
ax.add_patch(Ellipse((10.76, 2.31), .65, .82, angle=-21,
                     facecolor=C["green_bg"], edgecolor=C["green"], linewidth=1.4))
txt(6.81, 2.31, r"$C_{t-1}$", 10.5, "green")
txt(10.76, 2.31, r"$S_t$", 10.5, "green")
arrow((7.28, 2.55), (10.40, 2.53), "green", 2.0, connection="arc3,rad=-.12")
ax.plot([7.27, 10.40], [2.16, 2.16], color=C["gray"], lw=1.1, ls=(0, (4, 3)))
txt(8.81, 1.97, "Euclidean EMA: the paired baseline", 8.6, "muted")
txt(8.80, 1.42, r"$H_t=\beta_2I+(1-\beta_2)T(C_{t-1},S_t)$", 10.7)
txt(8.80, 1.00, r"$C_t^{\rm raw}=H_tC_{t-1}H_t$", 10.8)

# Lower 3: clipping and inverse-root cache.
matrix(12.57, 2.00, .77, .78, "green", 1)
matrix(14.65, 2.00, .77, .78, "green", 3)
matrix(16.55, 2.00, .77, .78, "green", 2)
txt(12.955, 1.81, r"$C_t^{\rm raw}$", 10.3, "green")
txt(15.035, 1.81, r"$C_t$", 10.3, "green")
txt(16.935, 1.81, r"$C_t^{-1/2}$", 10.3, "green")
arrow((13.43, 2.39), (14.52, 2.39), "green")
arrow((15.51, 2.39), (16.43, 2.39), "green")
txt(14.93, 1.35, r"$C_t=\mathrm{ClipSPD}(C_t^{\rm raw};\lambda_{\min},\lambda_{\max})$", 10.7)
txt(14.93, .95, r"$\widetilde L_t^{-1/2}=L_{r(t)}^{-1/2},\quad\widetilde R_t^{-1/2}=R_{r(t)}^{-1/2}$", 10.7)

# Geometry subflow and the next-iteration return.
arrow((9.10, 4.01), (9.10, 3.72), "green", 2.0)
arrow((5.70, 2.09), (5.84, 2.09), "green", 1.8)
arrow((11.73, 2.09), (11.87, 2.09), "green", 1.8)
arrow((13.32, 3.70), (13.32, 3.99), "green", 2.0)
ax.plot([17.96, 17.96, .10, .10], [5.67, .12, .12, 5.67], color=C["purple"], lw=1.2)
arrow((.10, 5.67), (.24, 5.67), "purple", 1.2, scale=10)
txt(9.05, .32, "next iteration: updated weights, momentum, and SPD states", 9.0, "purple")

# Compact legend in the same place and order as the companion GitHub figure.
for lx, color, label in [
    (3.25, "blue", "Momentum / weights"),
    (6.80, "green", "SPD covariance"),
    (9.90, "orange", "Gradient"),
    (12.17, "purple", "Parameter step"),
]:
    ax.add_patch(Rectangle((lx, -.30), .22, .22, facecolor=C[color],
                           edgecolor=C[color], linewidth=1.0, alpha=.78))
    txt(lx + .30, -.19, label, 8.1, "ink", ha="left")

fig.savefig(OUT, facecolor="white", dpi=190, bbox_inches="tight", pad_inches=.055)
plt.close(fig)
print(OUT)
