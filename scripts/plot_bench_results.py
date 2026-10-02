"""Plot the benchmark results recorded in docs/claude/{parallel_envs_bench,server_and_resolution_bench}.md.

    isaacpy scripts/plot_bench_results.py   # writes docs/claude/images/{resolution_sweep,env_scaling}.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs/claude/images"
OUT.mkdir(parents=True, exist_ok=True)

# palette slots 1-3 (blue, orange, aqua; validated all-pairs), light surface
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(c - h, 0), min(c + h, 1)


def direct_labels(ax, series, x_last, dy=None):
    """Label each line at its right end (nudged apart if they collide)."""
    ys = sorted([(y, name, col) for name, (y, col) in series.items()])
    last_y = -1e9
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for y, name, col in ys:
        yy = max(y, last_y + 0.07 * span)
        ax.annotate(name, (x_last, y), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
        last_y = yy


# ---------------------------------------------------------------- resolution sweep
RES = ["320x180", "640x360", "1280x720\n(32 envs x 2)"]
TASKS = {  # task: (color, successes /64, mean returns)
    "CubeBowl": (BLUE, [50, 60, 60], [238.2, 296.3, 297.3]),
    "CanMug": (ORANGE, [0, 3, 7], [7.0, 30.9, 50.4]),
    "BananaBin": (AQUA, [3, 3, 9], [12.9, 7.0, 28.6]),
}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.45})
x = np.arange(3)
for i, (task, (col, ks, rets)) in enumerate(TASKS.items()):
    p = np.array(ks) / 64 * 100
    lo, hi = zip(*(wilson(k, 64) for k in ks))
    off = (i - 1) * 0.04
    a1.errorbar(x + off, p, yerr=[p - np.array(lo) * 100, np.array(hi) * 100 - p], color=col, marker="o", ms=7,
                lw=2, capsize=3, mec=SURFACE, mew=1.5)
    lab_dy = {"BananaBin": 7, "CanMug": -7}.get(task, 0)  # nudge the two close labels apart
    a1.annotate(task, (2 + off, p[-1]), xytext=(12, lab_dy), textcoords="offset points", va="center", fontsize=10)
    a2.plot(x, rets, color=col, marker="o", ms=7, lw=2, mec=SURFACE, mew=1.5)
    a2.annotate(task, (2, rets[-1]), xytext=(10, 0), textcoords="offset points", va="center", fontsize=10)
for ax, yl, t in [(a1, "Success rate (%)", "Success rate (64 episodes, 95% CI)"), (a2, "Mean episode return", "Mean episode return")]:
    ax.set_xticks(x, RES, fontsize=9)
    ax.set_xlim(-0.2, 2.6)
    ax.set_ylabel(yl)
    ax.set_title(t, loc="left", fontsize=12, color=INK)
    ax.set_xlabel("Camera resolution (policy always sees 224x224)", fontsize=9)
a1.set_ylim(-3, 105)
a2.set_ylim(0, 320)
fig.savefig(OUT / "resolution_sweep.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- env scaling
SERIES = {  # label: (color, marker, N, env-steps/s, Isaac MiB)
    "Tiled, 2x720p": (ORANGE, "o", [4, 16, 32], [25.5, 67.6, 88.8], [6711, 14042, 26275]),
    "Tiled, 2x360p": (BLUE, "o", [4, 32, 64, 128], [28.0, 160, 246, 334], [4587, 9618, 14474, 24996]),
    "Tiled, 2x180p": (AQUA, "o", [256, 512], [837, 1083], [14946, 26689]),
}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.28})
for label, (col, mk, n, sps, mem) in SERIES.items():
    a1.plot(n, sps, color=col, marker=mk, ms=7, lw=2, mec=SURFACE, mew=1.5, label=label)
    a2.plot(n, np.array(mem) / 1024, color=col, marker=mk, ms=7, lw=2, mec=SURFACE, mew=1.5, label=label)
# failures
a2.plot([256], [44.3], "x", color=BLUE, ms=9, mew=2)
a2.annotate("640x360 OOM at 256", (256, 44.3), xytext=(-8, -16), textcoords="offset points", fontsize=9, color=INK2, ha="right")
a2.plot([64], [44.3], "x", color=ORANGE, ms=9, mew=2)
a2.annotate("1280x720 OOM at 64", (64, 44.3), xytext=(-8, -34), textcoords="offset points", fontsize=9, color=INK2, ha="right")
a2.axhline(44.3, color=INK2, lw=1, ls=(0, (4, 3)))
a2.annotate("GPU limit (44 GiB)", (1, 44.3), xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK2)
for ax, yl, t in [(a1, "Env steps / second (all envs)", "Throughput"), (a2, "Isaac GPU memory (GiB)", "Memory")]:
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"])
    ax.minorticks_off()
    ax.set_xlabel("Parallel envs")
    ax.set_ylabel(yl)
    ax.set_title(t, loc="left", fontsize=12, color=INK)
a1.set_ylim(bottom=0)
a2.set_ylim(0, 48)
a1.legend(frameon=False, fontsize=9, loc="upper left")
fig.savefig(OUT / "env_scaling.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "resolution_sweep.png", OUT / "env_scaling.png")
