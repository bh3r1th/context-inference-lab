"""Sequential single-GPU phase normalization for frozen benchmark configs."""
from pathlib import Path

from contextbench.experiments.config import ExperimentConfig, load_config


UNCACHED_STRATEGIES = {"full_context", "retrieval", "compiled_context"}


def phase_config(path: Path, phase: str, output_root: Path | None = None) -> ExperimentConfig:
    if phase not in {"uncached", "cached", "all"}:
        raise ValueError(f"unknown phase: {phase}")
    config = load_config(path)
    if phase == "all":
        return config
    values = config.model_dump()
    values["phase"] = phase
    server = config.server.model_dump()
    server["base_url"] = "http://localhost:8000/v1"
    server["metrics_url"] = "http://localhost:8000/metrics"
    server["prefix_caching"] = phase == "cached"
    values["server"] = server
    values["strategy_servers"] = {}
    if phase == "uncached":
        values["strategies"] = [name for name in config.strategies if name in UNCACHED_STRATEGIES]
        values["cache_scenario"] = None
    else:
        values["strategies"] = ["prefix_cached_full_context"]
        values["cache_scenario"] = config.cache_scenario or "warm"
    if not values["strategies"]:
        raise ValueError(f"config {path} has no strategies for phase {phase}")
    if output_root is not None:
        values["output_dir"] = str(output_root.resolve())
    return ExperimentConfig.model_validate(values)