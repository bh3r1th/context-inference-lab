"""Read source documents and lookup data only; never read evaluation artifacts."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from contextbench.models import StrictModel


class RuleRecord(StrictModel):
    id: str = Field(min_length=1)
    when: dict[str, Any] = Field(default_factory=dict)
    fact: str = Field(min_length=1)
    document: str | None = None
    section: str | None = None


@dataclass(frozen=True)
class Reference:
    document: str
    section: str | None
    line: int


@dataclass
class Rule:
    id: str
    when: dict[str, Any]
    fact: str
    references: list[Reference]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    rule_spans: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    start: int
    end: int
    text: str
    rule_ids: tuple[str, ...]


def load_sources(root: Path) -> tuple[list[Document], list[Rule]]:
    documents, rules = [], {}

    def add_rule(record: RuleRecord, reference: Reference):
        if any(isinstance(v, (dict, list)) for v in record.when.values()):
            raise ValueError(f"rule {record.id}: predicates must use scalar equality values")
        if record.id in rules:
            previous = rules[record.id]
            if (json.dumps(previous.when, sort_keys=True) != json.dumps(record.when, sort_keys=True)
                    or previous.fact != record.fact):
                raise ValueError(f"conflicting definitions for source ID {record.id}")
            previous.references.append(reference)
        else:
            rules[record.id] = Rule(record.id, record.when, record.fact, [reference])

    for path in sorted((root / "documents").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
            continue
        name = path.relative_to(root / "documents").as_posix()
        text = path.read_text(encoding="utf-8")
        spans, offset, section = [], 0, None
        for number, line in enumerate(text.splitlines(keepends=True), 1):
            if line.startswith("#"):
                section = line.lstrip("#").strip()
            if line.startswith("RULE "):
                try:
                    record = RuleRecord.model_validate_json(line[5:])
                except ValueError as exc:
                    raise ValueError(f"invalid RULE in {name}:{number}: {exc}") from exc
                add_rule(record, Reference(name, record.section or section, number))
                spans.append((offset, offset + len(line.rstrip("\r\n")), record.id))
            offset += len(line)
        documents.append(Document(name, text, tuple(spans)))
    lookup = root / "lookup.json"
    if lookup.exists():
        rows = json.loads(lookup.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("lookup.json must contain a list of identified lookup rows")
        rendered, spans, offset = [], [], 0
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError("lookup rows must be objects")
            if "fact" in row:
                record = RuleRecord.model_validate(row)
            elif (isinstance(row.get("city"), str) and type(row.get("limit_usd")) is int
                  and row["limit_usd"] >= 0):
                record = RuleRecord(id=row.get("id"), when={"city": row["city"]},
                                    fact=f"{row['city']} travel limit is {row['limit_usd']} USD.")
            else:
                raise ValueError("lookup rows require id/when/fact or id/city/limit_usd")
            add_rule(record, Reference("lookup.json", record.section, number))
            text = json.dumps(row, sort_keys=True, allow_nan=False)
            rendered.append(text)
            spans.append((offset, offset + len(text), record.id))
            offset += len(text) + 2
        documents.append(Document("lookup.json", "\n\n".join(rendered), tuple(spans)))
    if not documents or not any(d.text.strip() for d in documents):
        raise ValueError("no nonempty source documents or lookup data found")
    return documents, list(rules.values())


def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    chunks = []
    for document in documents:
        for paragraph in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", document.text):
            start, stop = paragraph.span()
            while start < stop:
                end = min(start + size, stop)
                ids = tuple(rid for lo, hi, rid in document.rule_spans if lo < end and hi > start)
                chunks.append(Chunk(f"{document.id}:{start}-{end}", document.id, start, end,
                                    document.text[start:end], ids))
                if end == stop:
                    break
                start = end - overlap
    return chunks


def covered_rules(documents: list[Document], chunks: list[Chunk]) -> list[str]:
    """Count a rule only when selected chunk spans cover its entire annotation."""
    covered = set()
    for document in documents:
        spans = sorted((c.start, c.end) for c in chunks if c.document == document.id)
        for start, end, rule_id in document.rule_spans:
            cursor = start
            for lo, hi in spans:
                if lo > cursor:
                    break
                cursor = max(cursor, hi)
                if cursor >= end:
                    covered.add(rule_id)
                    break
    return sorted(covered)
