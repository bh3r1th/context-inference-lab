"""Create equivalent workloads with controlled rendered full-context sizes."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from contextbench.context.prompt import render_messages, render_prompt
from contextbench.context.sources import load_sources


TARGETS = {"2k": 2_000, "8k": 8_000, "16k": 16_000, "32k": 32_000, "64k": 64_000}
PAD = ("Archived operational reference {index}: this historical note is intentionally "
       "non-normative and exists only to control input context size. ")


def token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def core_document(text: str) -> str:
    """Retain one stable heading and normative RULE records; remove archive bulk."""
    heading = next((line for line in text.splitlines() if line.startswith("#")), "# Policy")
    lines = [heading, *(line for line in text.splitlines() if line.startswith("RULE "))]
    return "\n\n".join(lines) + "\n"


def rendered_prompt_tokens(root: Path, event: dict) -> tuple[int, int]:
    documents, _ = load_sources(root)
    text = "\n\n".join(f"SOURCE {document.id}\n{document.text}" for document in documents)
    messages = render_messages(text, event)
    return token_count(render_prompt(messages)), token_count(messages[0]["content"])


def refresh_manifest(root: Path):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest["sha256"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")


def prepare_template(source: Path, output: Path):
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name in ("events.json", "expected.json", "lookup.json", "sources.json", "records.json",
                 "output.schema.json", "manifest.json"):
        shutil.copy2(source / name, output / name)
    documents = output / "documents"
    documents.mkdir()
    for document in (source / "documents").glob("*.md"):
        documents.joinpath(document.name).write_text(
            core_document(document.read_text(encoding="utf-8")), encoding="utf-8")


def generate(source: Path, output: Path):
    template = output / "template"
    prepare_template(source, template)
    events = json.loads((template / "events.json").read_text(encoding="utf-8"))
    if not events:
        raise ValueError("scaling source must contain at least one event")
    for label, target in TARGETS.items():
        target_root = output / label
        shutil.copytree(template, target_root)
        policy = target_root / "documents" / "policy.md"
        text = policy.read_text(encoding="utf-8")
        index = 0
        observed, static = rendered_prompt_tokens(target_root, events[0])
        while observed < target:
            text += "\n" + PAD.format(index=index)
            index += 1
            policy.write_text(text, encoding="utf-8")
            observed, static = rendered_prompt_tokens(target_root, events[0])
        metadata = {
            "target_input_tokens": target,
            "observed_prompt_tokens_first_event": observed,
            "observed_static_prefix_tokens_first_event": static,
            "tokenizer": "regex-words-and-punctuation-v1",
            "target_definition": "rendered full_context prompt including first event",
            "task_structure": "identical events, expected answers, and normative rules",
            "padding": "non-normative archive text in policy.md",
            "output_tokens": "shared server max_tokens from context-scaling config",
        }
        (target_root / "scaling_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        refresh_manifest(target_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/enterprise"))
    parser.add_argument("--output", type=Path, default=Path("results/context-scaling"))
    args = parser.parse_args()
    generate(args.source, args.output)
    print(f"Generated {len(TARGETS)} workloads under {args.output}")


if __name__ == "__main__":
    main()
