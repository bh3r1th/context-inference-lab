import math
from collections import defaultdict


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentiles(values: list[float]) -> dict:
    values = sorted(values)
    def quantile(q):
        if not values:
            return None
        i = (len(values) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)
    return {"count": len(values), **{f"p{p}": quantile(p / 100) for p in (50, 95, 99)}}


QUALITY_FIELDS = (
    "correct", "schema_valid", "answer_correct", "required_rule_coverage",
    "abstention_correct", "cross_document_reasoning_success", "buried_exception_success",
    "minimum_safe_context",
)


def _quality(rows: list[dict]) -> dict:
    result = {}
    for field in QUALITY_FIELDS:
        values = [r.get(field) for r in rows if r.get(field) is not None]
        result[field] = sum(values) / len(values) if values else None
    result["rule_recall"] = (sum(r["rule_recall"] for r in rows) / len(rows)
                              if rows else None)
    result["missed_critical_rules"] = sum(len(r.get("missed_critical_rules", []))
                                          for r in rows)
    result["incorrect_rule_inclusions"] = sum(len(r.get("incorrect_rule_inclusion", []))
                                               for r in rows)
    result["critical_rule_miss_rate"] = (
        sum(bool(r.get("missed_critical_rules", [])) for r in rows) / len(rows)
        if rows else None)
    result["requests"] = len(rows)
    return result


def _quality_groups(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("strategy"), row.get("case"), row.get("context_chars"),
               row.get("concurrency"))
        groups[key].append(row)
    return [{"strategy": strategy, "case": case, "context_size": context_size,
             "context_size_unit": "chars", "concurrency": concurrency, **_quality(group)}
            for (strategy, case, context_size, concurrency), group in sorted(
                groups.items(), key=lambda item: tuple("" if value is None else str(value)
                                                       for value in item[0]))]


def summarize(rows: list[dict], duration: float | None) -> dict:
    successful = [r for r in rows if r["error"] is None]
    fields = ("latency_s", "ttft_s", "client_queue_s", "context_build_s", "batch_response_s",
              "decode_tokens_per_s", "input_tokens", "output_tokens", "context_chars",
              "context_token_count",
              "prefill_s", "server_queue_s", "decode_time_s", "stream_generation_s",
              "server_ttft_s", "server_mean_itl_s", "server_output_tokens_per_s",
              "output_tokens_per_s", "cached_input_tokens", "cached_input_fraction")
    token_totals, throughput = {}, {}
    for field in ("input_tokens", "output_tokens"):
        observed = [r[field] for r in rows if r.get(field) is not None]
        token_totals[field] = {"observed_total": sum(observed) if observed else None,
                               "observed_mean": sum(observed) / len(observed) if observed else None,
                               "observed_requests": len(observed),
                               "missing_requests": len(rows) - len(observed),
                               "source": f"per-request server usage: {field}",
                               "reason": None if observed else "no server usage returned"}
        complete = bool(successful) and all(r.get(field) is not None for r in successful)
        throughput[field + "_per_s"] = {
            "value": sum(r[field] for r in successful) / duration
            if complete and duration is not None and duration > 0 else None,
            "source": f"successful-request server {field} / measured batch duration",
            "reason": None if complete and duration is not None and duration > 0 else
            "requires usage for every successful request and positive batch duration"}
    metrics = {}
    for field in fields:
        values = [r[field] for r in successful if r.get(field) is not None]
        metrics[field] = {**percentiles(values), "reason": None if values else
                          "no available successful-request measurements; see per-request provenance"}
    return {
        "requests": len(rows), "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "failure_rate": (len(rows) - len(successful)) / len(rows) if rows else None,
        "failure_rate_reason": None if rows else "no measured request attempts",
        "average_input_tokens": token_totals["input_tokens"]["observed_mean"],
        "average_output_tokens": token_totals["output_tokens"]["observed_mean"],
        "token_average_scope": "requests with observed server usage, including failed requests",
        "average_context_tokens": _mean([r.get("context_token_count") for r in rows
                          if r.get("context_token_count") is not None]),
        "context_token_observed_requests": sum(r.get("context_token_count") is not None
                            for r in rows),
        "batch_duration_s": duration,
        "requests_per_s": len(successful) / duration if duration else None,
        "attempts_per_s": len(rows) / duration if duration else None,
        "rate_source": "successful/attempted request counts divided by measured batch wall time",
        "rate_unavailable_reason": None if duration is not None and duration > 0 else
        "batch did not start or has nonpositive duration",
        "throughput": throughput,
        "correctness": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        "schema_validity": sum(r["schema_valid"] for r in rows) / len(rows) if rows else None,
        "mean_rule_recall": sum(r["rule_recall"] for r in rows) / len(rows) if rows else None,
        "missed_critical_rules": sum(len(r["missed_critical_rules"]) for r in rows),
        "quality": _quality(rows),
        "quality_by_strategy_case_context_size_concurrency": _quality_groups(rows),
        "successful_request_metrics": metrics,
        "token_totals": token_totals,
    }
