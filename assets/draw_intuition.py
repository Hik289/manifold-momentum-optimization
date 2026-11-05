"""Render the README Figure 1 from a noncommuting SPD example.

The two covariance ellipses are computed, not hand-shaped. The drawing
follows the large blue geometry, black axes, serif mathematics and generous
white space of the companion ManifoldFlow paper's opening figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch

OUT = Path(__file__).with_name("buresmomentum_intuition.png")
BLUE = "#3479A9"
BLUE_FILL = "#D3E5F2"
INK = "#111827"


def spd_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * values**power) @ vectors.T


def bw_mean(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a_half = spd_power(a, 0.5)
    transport = spd_power(a, -0.5) @ spd_power(a_half @ b @ a_half, 0.5) @ spd_power(a, -0.5)
    assert np.allclose(transport @ a @ transport, b, atol=1e-10)
    h = (1 - alpha) * np.eye(2) + alpha * transport
    return h @ a @ h


def draw_covariance(ax, center, covariance, major_label, minor_label):
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    unit = 0.52
    major, minor = unit * np.sqrt(values)
    cx, cy = center

    # Match the companion figure: one translucent blue body, blue section
    # curves, black coordinate axes, and direct spectral labels.
    ax.add_patch(Ellipse(center, 2 * major, 2 * minor, angle=angle,
                         facecolor=BLUE_FILL, edgecolor=BLUE, linewidth=2.5,
                         alpha=0.85, zorder=2))
    ax.add_patch(Ellipse(center, 1.70 * major, 1.70 * minor, angle=angle,
                         facecolor="none", edgecolor=BLUE, linewidth=1.25,
                         alpha=0.50, zorder=3))
    ax.add_patch(Arc(center, 2 * major, 1.02 * minor, angle=angle,
                     theta1=0, theta2=180, color=BLUE, linewidth=1.4,
                     alpha=0.65, zorder=3))
    ax.add_patch(Arc(center, 2 * major, 1.02 * minor, angle=angle,
                     theta1=180, theta2=360, color=BLUE, linewidth=1.4,
                     alpha=0.65, zorder=3))

    major_direction = vectors[:, 0]
    if major_direction[0] < 0:
        major_direction = -major_direction
    minor_direction = vectors[:, 1]
    if minor_direction[1] < 0:
        minor_direction = -minor_direction
    for direction, length in ((major_direction, major + 0.36),
                              (minor_direction, minor + 0.50)):
        dx, dy = direction * length
        ax.add_patch(FancyArrowPatch((cx, cy), (cx + dx, cy + dy),
                                     arrowstyle="-|>", mutation_scale=17,
                                     linewidth=1.55, color=INK, zorder=1))
    ax.plot([cx], [cy], marker="o", markersize=3, color=INK, zorder=4)
    major_tip = np.array(center) + major_direction * (major + 0.42)
    minor_tip = np.array(center) + minor_direction * (minor + 0.53)
    ax.text(major_tip[0] + 0.08, major_tip[1] - 0.10, major_label,
            ha="left", va="center",
            fontsize=15.2, color=INK)
    ax.text(minor_tip[0] - 0.16, minor_tip[1] + 0.08, minor_label,
            ha="right", va="center",
            fontsize=15.2, color=INK)


plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 14,
    "savefig.facecolor": "white",
})

a = np.diag([16.0, 1.0])
theta = np.deg2rad(60.0)
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
b = rotation @ a @ rotation.T
alpha = 0.5
euclidean = (1 - alpha) * a + alpha * b
bures = bw_mean(a, b, alpha)

fig, ax = plt.subplots(figsize=(13.4, 7.2), dpi=205)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 13.4)
ax.set_ylim(0, 7.2)
ax.set_aspect("equal")
ax.axis("off")

ax.text(3.22, 6.76, r"(a) Euclidean: $C_\alpha=(1-\alpha)A+\alpha B$",
        ha="center", va="center", fontsize=20.1, color=INK)
ax.text(10.16, 6.76, r"(b) BuresMomentum: $C_\alpha=H_\alpha A H_\alpha$",
        ha="center", va="center", fontsize=20.1, color=INK)

draw_covariance(ax, (2.80, 3.95), euclidean,
                r"$\sqrt{\lambda_1(C^{\rm E})}$", r"$\sqrt{\lambda_2(C^{\rm E})}$")
draw_covariance(ax, (9.70, 3.95), bures,
                r"$\sqrt{\lambda_1(C^{\rm BW})}$", r"$\sqrt{\lambda_2(C^{\rm BW})}$")

ax.text(6.68, 4.34, r"$TAT=B$", ha="center", va="center",
        fontsize=17.5, color=INK)
ax.text(6.68, 3.92, r"$H_\alpha=(1-\alpha)I+\alpha T$", ha="center",
        va="center", fontsize=15.2, color=INK)
ax.add_patch(FancyArrowPatch((5.95, 3.52), (7.40, 3.52),
                             arrowstyle="-|>", mutation_scale=19,
                             linewidth=1.8, color=INK))

ax.text(3.22, 1.30, "Entrywise covariance interpolation,",
        ha="center", va="center", fontsize=19.2, color=INK)
ax.text(3.22, 0.88, "with a rounder intermediate spectrum",
        ha="center", va="center", fontsize=18.2, color=INK)
ax.text(10.16, 1.30, "Transport-aligned covariance interpolation,",
        ha="center", va="center", fontsize=19.2, color=INK)
ax.text(10.16, 0.88, "preserving the anisotropic geometry",
        ha="center", va="center", fontsize=18.2, color=INK)

ax.text(6.70, 0.25,
        r"Same noncommuting $A,B\in\mathrm{SPD}(2)$ and $\alpha=1/2$ in both panels.",
        ha="center", va="center", fontsize=11.2, color="#5B6678")

fig.savefig(OUT, dpi=205, bbox_inches="tight", pad_inches=0.10)
plt.close(fig)
print(OUT)
