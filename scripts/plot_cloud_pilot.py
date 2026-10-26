"""Export publication figures from the selected pilot cells (no inference)."""
import argparse
import csv
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.cells.open() as handle:
        rows = {row["cell"]: row for row in csv.DictReader(handle)}
    strategies = ["full_context", "prefix_cached_full_context", "retrieval", "compiled_context"]
    labels = ["Full context", "Prefix cached", "BM25 retrieval", "Compiled rules"]
    colors = ["#2868a9", "#d27b31"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "contextbench-pilot-v1"})
    args.output.mkdir(parents=True, exist_ok=True)

    def row(strategy, concurrency):
        return rows[f"pilot-enterprise-{strategy}-c{concurrency}"]

    def save(figure, name):
        figure.savefig(args.output / f"{name}.svg", metadata={"Date": None})
        figure.savefig(args.output / f"{name}.png", dpi=220)
        plt.close(figure)

    x = np.arange(len(strategies))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, concurrency in zip(axes, (1, 8)):
        for offset, field, label, color in (
            (-.18, "decision_correct", "Decision + limit", colors[0]),
            (.18, "strict_correct", "Including exact citations", colors[1]),
        ):
            bars = axis.bar(x + offset, [int(row(s, concurrency)[field]) for s in strategies],
                            width=.35, label=label, color=color)
            axis.bar_label(bars, padding=3)
        axis.set(title=f"Client concurrency {concurrency}", xticks=x, xticklabels=labels,
                 ylim=(0, 10.8), yticks=range(0, 11, 2))
        axis.tick_params(axis="x", rotation=15)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Correct responses out of 10")
    axes[0].legend(loc="upper left", frameon=False)
    figure.suptitle("Pilot answer quality: valid JSON did not ensure correct answers", fontsize=14)
    figure.text(.5, .02, "10 synthetic cases per cell; repeated cases across strategies and concurrency. "
                "No confidence intervals or population claims.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .06, 1, .94))
    save(figure, "enterprise_quality")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    tokens = [float(row(strategy, 1)["mean_input_tokens"]) for strategy in strategies]
    bars = axes[0].bar(x, tokens, color=colors[0])
    axes[0].bar_label(bars, labels=[f"{value:,.0f}" for value in tokens], padding=4)
    axes[0].set(yscale="log", ylim=(300, 80000), ylabel="Mean model input tokens (log scale)",
                title="Input size from server usage", xticks=x, xticklabels=labels)
    for concurrency, offset, color in ((1, -.18, colors[0]), (8, .18, colors[1])):
        values = [float(row(strategy, concurrency)["ttft_p95_s"]) for strategy in strategies]
        axes[1].bar(x + offset, values, width=.35, color=color,
                    label=f"Concurrency {concurrency}")
    axes[1].set(yscale="log", ylabel="Client TTFT p95, seconds (log scale)",
                title="Time to first content: 10 requests per cell", xticks=x, xticklabels=labels)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Enterprise pilot: observed input size and first-content latency", fontsize=14)
    figure.text(.5, .02, "Separate server per cell; prefix-cached cells have a configured warmup. "
                "Small-sample observations; no causal or equal-quality claim.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .06, 1, .94))
    save(figure, "enterprise_context_latency")


if __name__ == "__main__":
    main()
