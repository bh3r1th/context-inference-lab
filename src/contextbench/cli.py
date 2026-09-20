import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import get_args

from contextbench.context import ContextConfig, make_strategy
from contextbench.context.config import StrategyName, load_context_config
from contextbench.experiments.config import load_config
from contextbench.experiments.orchestrator import run
from contextbench.models import Decision


def main():
    parser = argparse.ArgumentParser(description="Enterprise context-design experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
    commands.add_parser("schema")
    inspect = commands.add_parser("inspect-context", help="Build context without an inference server")
    inspect.add_argument("--dataset", type=Path, default=Path("data/enterprise"))
    inspect.add_argument("--strategy", choices=get_args(StrategyName), default="full_context")
    inspect.add_argument("--config", type=Path, help="Context-only YAML configuration")
    event = inspect.add_mutually_exclusive_group(required=True)
    event.add_argument("--event-id", help="Select exactly one event from dataset/events.json")
    event.add_argument("--event-file", type=Path, help="JSON file containing one input event object")
    inspect.add_argument("--top-k", type=int, help="Override configured retrieval top-k")
    inspect.add_argument("--retrieval-method", choices=["bm25", "embedding"])
    inspect.add_argument("--format", choices=["json", "prompt"], default="json")
    inspect.add_argument("--output", type=Path, help="Write UTF-8 output instead of stdout")
    args = parser.parse_args()
    if args.command == "schema":
        print(json.dumps(Decision.model_json_schema(), indent=2))
        return
    if args.command == "inspect-context":
        try:
            options = load_context_config(args.config) if args.config else ContextConfig()
            values = options.model_dump()
            if args.top_k is not None:
                values["retrieval"]["top_k"] = args.top_k
            if args.retrieval_method:
                values["retrieval"]["method"] = args.retrieval_method
            options = ContextConfig.model_validate(values)
            if args.event_file:
                event = json.loads(args.event_file.read_text(encoding="utf-8"))
            else:
                events = json.loads((args.dataset / "events.json").read_text(encoding="utf-8"))
                matches = [e for e in events if e.get("event_id") == args.event_id]
                if len(matches) != 1:
                    raise ValueError(f"event ID {args.event_id!r} matched {len(matches)} records")
                event = matches[0]
            if not isinstance(event, dict):
                raise ValueError("event must be a JSON object")
            result = make_strategy(args.strategy, args.dataset, config=options).build(event)
            data = {**asdict(result), "selected_source_ids": list(result.selected_source_ids)}
            output = result.prompt if args.format == "prompt" else json.dumps(
                data, indent=2, allow_nan=False)
            if args.output:
                args.output.write_text(output + "\n", encoding="utf-8")
            else:
                print(output)
        except (ValueError, OSError, ImportError) as exc:
            parser.error(str(exc))
        return
    config = load_config(args.config)
    if args.command == "validate":
        print(config.model_dump_json(indent=2))
    else:
        print(asyncio.run(run(config)))


if __name__ == "__main__":
    main()
