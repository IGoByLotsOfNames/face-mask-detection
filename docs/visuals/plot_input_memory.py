"""Render the recorded memory comparison; this does not run the benchmark.

From the repository root: python docs/visuals/plot_input_memory.py
Requires matplotlib 3.10.6. All statistics are recomputed from public raw observations.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "evidence/input-memory-results.json"


def main():
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    observations = record["observations_in_execution_order"]
    counts = sorted({item["rows"] for item in observations})
    implementations = ("baseline", "indexed")
    labels = ("Decode, then shuffle", "Shuffle indices, then decode")
    colors = ("#416B8C", "#087F83")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": "#243C50",
            "text.color": "#18354A",
            "xtick.color": "#455F72",
            "ytick.color": "#455F72",
            "svg.hashsalt": "mask-input-memory-v1",
        }
    )
    fig, ax = plt.subplots(figsize=(12.8, 7.6), dpi=160)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.74, bottom=0.27)
    fig.text(
        0.10,
        0.94,
        "FACE MASK DETECTION  /  ENGINEERING MEASUREMENT",
        size=10,
        weight="bold",
        color="#087F83",
    )
    fig.text(0.10, 0.882, "A smaller input-pipeline memory footprint", size=24, weight="bold")
    fig.text(
        0.10,
        0.839,
        "Five paired seeds per image count · 30 fresh processes · Windows CPU workload",
        size=12,
        color="#536C7D",
    )

    width = 0.29
    for offset, (kind, label, color) in enumerate(zip(implementations, labels, colors)):
        for j, rows in enumerate(counts):
            values = [
                r["peak_rss_bytes"] / 2**20
                for r in observations
                if r["rows"] == rows and r["implementation"] == kind
            ]
            assert len(values) == 5
            mean, sd = statistics.mean(values), statistics.stdev(values)
            x = j + (-0.18 if offset == 0 else 0.18)
            ax.bar(
                x,
                mean,
                width=width,
                color=color,
                alpha=0.15,
                edgecolor=color,
                linewidth=1.2,
                label=label if j == 0 else None,
                zorder=2,
            )
            # Horizontal offsets reveal each of the five nearly coincident peaks.
            ax.scatter(
                [x + (i - 2) * 0.027 for i in range(5)],
                values,
                s=22,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
            ax.errorbar(
                x, mean, yerr=sd, fmt="none", color=color, capsize=5, elinewidth=1.5, zorder=5
            )
            ax.text(
                x,
                mean + 29,
                f"{mean:,.3f}",
                ha="center",
                va="bottom",
                size=11,
                weight="bold",
                color=color,
            )
    ax.set_xticks(range(len(counts)), [f"{n:,} images" for n in counts])
    ax.set_ylim(0, 1180)
    ax.set_yticks(range(0, 1200, 200))
    ax.set_ylabel("Process peak resident memory (MiB)", labelpad=12)
    ax.grid(axis="y", color="#E0E8EC", linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#BCCBD4")
    ax.tick_params(axis="both", length=0, pad=9)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.005, 1.145),
        frameon=False,
        ncol=2,
        fontsize=11,
        handlelength=1.3,
        columnspacing=2.0,
    )

    largest = max(counts)
    pairs = {
        seed: {
            r["implementation"]: r["peak_rss_bytes"]
            for r in observations
            if r["rows"] == largest and r["seed"] == seed
        }
        for seed in {r["seed"] for r in observations}
    }
    reduction = statistics.mean([100 * (1 - p["indexed"] / p["baseline"]) for p in pairs.values()])
    fig.text(
        0.10,
        0.184,
        f"{reduction:.2f}% mean paired reduction at {largest:,} images",
        size=16,
        weight="bold",
        color="#087F83",
    )
    fig.text(
        0.10,
        0.145,
        "Bars: means. Dots: individual processes. Whiskers: sample SD (ddof=1); small at this scale.",
        size=10,
        color="#536C7D",
    )
    fig.text(
        0.10,
        0.104,
        "Includes runtime startup, input checks and one dataset pass. No training, prediction or camera input.",
        size=10,
        color="#536C7D",
    )
    fig.text(
        0.10,
        0.073,
        "One synthetic workload on one machine; this does not establish faster execution or higher accuracy.",
        size=10,
        color="#536C7D",
    )
    fig.text(
        0.10,
        0.036,
        "Measured 2026-10-03 · Source: docs/evidence/input-memory-results.json · No observations excluded",
        size=9,
        color="#6E8391",
    )
    fig.savefig(ROOT / "input-memory.png", facecolor="white", metadata={"Software": "Matplotlib"})
    fig.savefig(
        ROOT / "input-memory.svg",
        facecolor="white",
        metadata={"Date": None, "Creator": "Matplotlib"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
