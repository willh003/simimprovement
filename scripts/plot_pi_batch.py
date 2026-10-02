"""Plot docs/claude/images/pi_batch_bench.json -> pi_batch_throughput.png (throughput and call latency vs batch size)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = Path(__file__).resolve().parent.parent / "docs/claude/images"
data = json.load(open(img / "pi_batch_bench.json"))
STYLE = {  # key: (label, colour, ok marker)
    "local_g3092_memfrac0.5": ("Local server (same node, localhost, 50% GPU mem)", "#2a78d6"),
    "remote_g3115_fullgpu": ("Remote server (other node, full GPU)", "#eb6834"),
}
INK, MUTED, BG = "#0b0b0b", "#52514e", "#fcfcfb"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.set_xscale("log", base=2)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED)
    ax.set_xticks([1, 8, 32, 64, 128, 256, 384, 512], [1, 8, 32, 64, 128, 256, 384, 512])
    ax.set_xlabel("observations per call (batch size)", color=INK)

for key, (label, col) in STYLE.items():
    rows = data[key]
    ok = [r for r in rows if "error" not in r and not r.get("chunk")]
    chunked = [r for r in rows if r.get("chunk")]
    bad = [r for r in rows if "error" in r]
    for ax, field in zip(axes, ("obs_per_s", "call_s")):
        ax.plot([r["n"] for r in ok], [r[field] for r in ok], "-o", color=col, lw=2, ms=6, mfc=col, mec=BG, mew=1.5, label=label)
        for r in chunked:  # one call split client-side into several messages: hollow marker
            ax.plot([r["n"]], [r[field]], "o", color=col, ms=8, mfc=BG, mew=2)
    for r in bad:  # server OOM at this size
        axes[0].plot([r["n"]], [0.6], "x", color=col, ms=10, mew=2.5)
        axes[0].annotate("OOM", (r["n"], 0.6), textcoords="offset points", xytext=(0, 9), ha="center", color=col, fontsize=9)

axes[0].set_ylim(0, 24)
axes[0].set_ylabel("throughput (observations / s)", color=INK)
axes[0].set_title("Throughput", loc="left", color=INK, fontweight="bold")
axes[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.3), labelcolor=INK, fontsize=9)
axes[1].set_yscale("log")
axes[1].set_ylabel("time per call (s)", color=INK)
axes[1].set_title("Latency of one call", loc="left", color=INK, fontweight="bold")
fig.text(0.01, 0.005, "Hollow marker: N=512 sent as 2 messages of 256 (client-side chunking). x: server ran out of GPU memory at that size.",
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(img / "pi_batch_throughput.png", dpi=150, facecolor=BG)
