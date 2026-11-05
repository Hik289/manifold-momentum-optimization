"""Render the README method diagram as a publication-style PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("buresmomentum_intuition.png")
NAVY = "#14233B"
MUTED = "#58667D"
LINE = "#D8E0E9"
TEAL = "#0D8479"
BLUE = "#3B6FCC"
ORANGE = "#D98142"
PALE = "#F7F9FC"


def label(ax, x, y, value, size=12, color=NAVY, weight="normal", **kwargs):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            va="center", ha="center", family="DejaVu Sans", **kwargs)


def card(ax, x, number, title, subtitle, accent):
    ax.add_patch(FancyBboxPatch((x, 0.95), 3.43, 3.98,
                                boxstyle="round,pad=0.025,rounding_size=0.12",
                                facecolor="white", edgecolor=LINE, linewidth=1.1))
    ax.add_patch(FancyBboxPatch((x + .18, 4.56), .36, .27,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=accent, edgecolor="none"))
    label(ax, x + .36, 4.695, str(number), 9, "white", "bold")
    ax.text(x + .19, 4.25, title, fontsize=14, color=NAVY, fontweight="bold",
            va="center", family="DejaVu Sans")
    ax.text(x + .19, 3.91, subtitle, fontsize=9.2, color=MUTED,
            va="center", family="DejaVu Sans")


def matrix(ax, x, y, color, width=.29, n=4, pattern=0):
    for i in range(n):
        for j in range(n):
            alpha = .22 + .12 * ((i * 2 + j + pattern) % 4)
            ax.add_patch(Rectangle((x + j * width, y + (n - 1 - i) * width),
                                   width - .025, width - .025,
                                   facecolor=color, edgecolor="none", alpha=alpha))
    ax.add_patch(Rectangle((x - .018, y - .018), n * width + .01, n * width + .01,
                           fill=False, edgecolor=color, linewidth=1.2))


fig, ax = plt.subplots(figsize=(16, 6), dpi=190)
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 16), ylim=(0, 6))
ax.axis("off")

ax.text(.53, 5.58, "BuresMomentum", fontsize=24, color=NAVY,
        fontweight="bold", va="center", family="DejaVu Sans")
ax.text(5.15, 5.58, "Geometric covariance averaging for matrix optimization",
        fontsize=12.2, color=MUTED, va="center", family="DejaVu Sans")
ax.plot([.53, 15.5], [5.30, 5.30], color=LINE, linewidth=1)

xs = [.53, 4.38, 8.23, 12.08]
card(ax, xs[0], 1, "Matrix gradient", "Shared first-moment update", BLUE)
card(ax, xs[1], 2, "Two-sided statistics", "One SPD state for each axis", BLUE)
card(ax, xs[2], 3, "Bures transport", "The only change from Euclidean EMA", TEAL)
card(ax, xs[3], 4, "Preconditioned step", "Shared norm matching and update", ORANGE)

for x in (3.98, 7.83, 11.68):
    ax.add_patch(FancyArrowPatch((x, 2.92), (x + .34, 2.92),
                                 arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.6, color="#A2AEC0"))

matrix(ax, 1.64, 2.45, BLUE, width=.275, pattern=1)
label(ax, 2.19, 2.13, r"$G_t=\nabla_W\mathcal{L}_t$", 15)
label(ax, 2.25, 1.53, r"$M_t=\beta_1M_{t-1}+(1-\beta_1)G_t$", 12)

matrix(ax, 4.82, 2.59, TEAL, width=.22, pattern=0)
matrix(ax, 6.50, 2.59, TEAL, width=.22, pattern=2)
label(ax, 5.28, 2.25, r"$S_L$", 14, TEAL, "bold")
label(ax, 6.96, 2.25, r"$S_R$", 14, TEAL, "bold")
label(ax, 6.10, 1.72, r"$S_L=G_tG_t^\top/n+\epsilon I$", 11.5)
label(ax, 6.10, 1.37, r"$S_R=G_t^\top G_t/m+\epsilon I$", 11.5)

ax.add_patch(Ellipse((8.95, 2.78), 1.00, .68, angle=26, facecolor="#DDEDEB",
                     edgecolor=TEAL, linewidth=1.5))
ax.add_patch(Ellipse((10.90, 2.78), .75, 1.02, angle=-28, facecolor="#DDEDEB",
                     edgecolor=TEAL, linewidth=1.5))
label(ax, 8.93, 2.76, r"$C_{t-1}$", 12, TEAL, "bold")
label(ax, 10.91, 2.75, r"$S_t$", 12, TEAL, "bold")
ax.add_patch(FancyArrowPatch((9.45, 3.06), (10.50, 3.07),
                             connectionstyle="arc3,rad=-.43", arrowstyle="-|>",
                             mutation_scale=14, linewidth=2.1, color=TEAL))
label(ax, 9.91, 3.47, "BW geodesic", 10, TEAL, "bold")
ax.plot([9.42, 10.49], [2.50, 2.50], color="#ADB6C1", linestyle=(0, (4, 3)), linewidth=1.1)
label(ax, 9.95, 2.25, "Euclidean EMA", 8.4, MUTED)
label(ax, 9.97, 1.52, r"$C_t=\mathrm{ClipSPD}\!\left(\mathrm{Bar}_{\mathrm{BW}}(C_{t-1},S_t)\right)$", 11.2)

matrix(ax, 12.61, 2.64, TEAL, width=.20, pattern=1)
matrix(ax, 13.66, 2.64, BLUE, width=.20, pattern=0)
matrix(ax, 14.71, 2.64, TEAL, width=.20, pattern=3)
label(ax, 13.03, 2.28, r"$L_t^{-1/2}$", 11, TEAL)
label(ax, 14.08, 2.28, r"$\hat M_t$", 11, BLUE)
label(ax, 15.13, 2.28, r"$R_t^{-1/2}$", 11, TEAL)
label(ax, 13.95, 1.72, r"$D_t=L_t^{-1/2}\hat M_t R_t^{-1/2}$", 12)
label(ax, 13.95, 1.34, r"$W_{t+1}=W_t-\eta_t s_tD_t$", 12)

ax.add_patch(FancyBboxPatch((.53, .18), 14.98, .48,
                            boxstyle="round,pad=0.03,rounding_size=0.09",
                            facecolor=PALE, edgecolor=LINE, linewidth=.8))
ax.add_patch(Rectangle((.72, .31), .14, .14, facecolor=TEAL, edgecolor="none"))
ax.text(1.00, .38,
        "Controlled comparison: BuresMomentum and Euclidean-Kronecker differ only in covariance-state averaging.",
        fontsize=11.2, color=NAVY, va="center", family="DejaVu Sans")

fig.savefig(OUT, dpi=190, facecolor="white", bbox_inches="tight", pad_inches=.08)
plt.close(fig)
print(OUT)
