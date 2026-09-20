"""Generate analysis tables and charts from completed run artifacts.

The script never fills unavailable measurements and never treats correlation as causation.
"""
import argparse
import csv
import json
from pathlib import Path


def values(summary, scenario=None):
    quality = summary.get("quality", {})
    metrics = summary.get("successful_request_metrics", {})
    telemetry = summary.get("telemetry", {})
    kv = []
    for series in (telemetry.get("kv_cache", {}).get("series") or {}).values():
        if series.get("p95") is not None:
            kv.append(series["p95"])
    gpu = []
    for device in (telemetry.get("local_gpu", {}).get("devices") or {}).values():
        used = device.get("memory_used_mib", {}).get("p95")
        if used is not None:
            gpu.append(used)
    return {
        "strategy": summary.get("strategy"),
        "context tokens": summary.get("average_context_tokens"),
        "concurrency": summary.get("concurrency"),
        "accuracy": quality.get("correct", summary.get("correctness")),
        "critical misses": quality.get("critical_rule_miss_rate"),
        "TTFT p95": metrics.get("ttft_s", {}).get("p95"),
        "latency p95": metrics.get("latency_s", {}).get("p95"),
        "throughput": summary.get("throughput", {}).get("output_tokens_per_s", {}).get("value"),
        "KV cache": sum(kv) / len(kv) if kv else None,
        "GPU memory": sum(gpu) / len(gpu) if gpu else None,
        "batch duration": summary.get("batch_duration_s"),
        "queue p95": metrics.get("client_queue_s", {}).get("p95"),
        "scenario": scenario,
    }


def load_rows(root: Path):
    rows = []
    for path in root.rglob("summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            scenario = "warm" if "warm" in path.as_posix() else (
                "cold" if "prefix-cache" in path.as_posix() else None)
            rows.extend(values(item, scenario) for item in value)
    return rows


def minimum_safe(rows, accuracy, critical_miss_rate):
    candidates = [row for row in rows if row["context tokens"] is not None
                  and row["accuracy"] is not None and row["critical misses"] is not None
                  and row["accuracy"] >= accuracy
                  and row["critical misses"] <= critical_miss_rate]
    return min(candidates, key=lambda row: row["context tokens"]) if candidates else None


def chart(rows, output: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    baselines = {row["concurrency"]: row["context tokens"] for row in rows
                 if row["strategy"] == "full_context"}
    series = [("context tokens vs TTFT", "context tokens", "TTFT p95"),
              ("context tokens vs throughput", "context tokens", "throughput"),
              ("context tokens vs KV-cache usage", "context tokens", "KV cache"),
              ("concurrency vs p95 latency", "concurrency", "latency p95"),
              ("context reduction vs correctness", "reduction %", "accuracy"),
              ("strategy vs critical-rule miss rate", "strategy", "critical misses"),
              ("total batch duration by strategy", "strategy", "batch duration"),
              ("cold vs warm prefix-cache comparison", "scenario", "TTFT p95")]
    for title, x_field, y_field in series:
        points = []
        for row in rows:
            if x_field == "reduction %":
                baseline = baselines.get(row["concurrency"])
                x_value = ((baseline - row["context tokens"]) / baseline * 100
                            if baseline and row["context tokens"] is not None else None)
            else:
                x_value = row.get(x_field)
            y_value = row.get(y_field)
            if x_value is not None and y_value is not None:
                if x_field == "scenario" and x_value not in {"cold", "warm"}:
                    continue
                points.append((x_value, y_value))
        if not points:
            continue
        x, y = zip(*points)
        plt.figure()
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.tight_layout()
        plt.savefig(output / (title.replace(" ", "_") + ".svg"))
        plt.close()


def write(rows, output: Path, accuracy: float, critical_miss_rate: float):
    output.mkdir(parents=True, exist_ok=True)
    fields = ["strategy", "context tokens", "concurrency", "accuracy", "critical misses",
              "TTFT p95", "throughput", "KV cache", "GPU memory", "batch duration"]
    with (output / "analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    safe = minimum_safe(rows, accuracy, critical_miss_rate)
    lines = ["# Experiment Results", "", "## Measured observations", "",
             "Raw request metrics, server usage, telemetry, failures, and warmup records remain "
             "in each run directory. The table below contains only observed aggregate fields.", "",
             "| strategy | context tokens | reduction % | accuracy | critical misses | TTFT p95 | throughput | KV cache | GPU memory | batch duration |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    baselines = {(r["concurrency"]): r["context tokens"] for r in rows if r["strategy"] == "full_context"}
    for row in rows:
        baseline = baselines.get(row["concurrency"])
        reduction = ((baseline - row["context tokens"]) / baseline * 100
                     if baseline and row["context tokens"] is not None else None)
        fields_row = [row["strategy"], row["context tokens"], reduction, row["accuracy"],
                      row["critical misses"], row["TTFT p95"], row["throughput"], row["KV cache"],
                      row["GPU memory"], row["batch duration"]]
        lines.append("| " + " | ".join("n/a" if value is None else str(value) for value in fields_row) + " |")
    lines += ["", "## Derived metrics", "",
              f"Minimum Safe Context uses accuracy >= {accuracy} and critical miss rate <= "
              f"{critical_miss_rate}.", "",
              f"**Minimum Safe Context:** {safe['context tokens'] if safe else 'not observed'}", "",
              "Reduction percentages, rates, percentiles, and Minimum Safe Context are derived "
              "from the preserved measurements; unavailable inputs remain unavailable.", "",
              "## Interpretation", "",
              "These results describe the observed benchmark workload. Differences are associations "
              "under the recorded hardware, server, ordering, and cache state; they do not by "
              "themselves establish causation or generalize to unseen workloads.", ""]
    (output / "EXPERIMENT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    chart(rows, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    parser.add_argument("--accuracy", type=float, default=.95)
    parser.add_argument("--critical-miss-rate", type=float, default=0.0)
    args = parser.parse_args()
    write(load_rows(args.results), args.output, args.accuracy, args.critical_miss_rate)
    print(args.output / "EXPERIMENT_RESULTS.md")


if __name__ == "__main__":
    main()