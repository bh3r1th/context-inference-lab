import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from contextbench.cli import main
from contextbench.context import (
    CompiledContextStrategy,
    ContextConfig,
    FullContextStrategy,
    RetrievalConfig,
    RetrievalContextStrategy,
    TokenizerConfig,
    make_strategy,
)
from contextbench.context.adapters import HuggingFaceTokenCounter, SentenceTransformerEncoder
from contextbench.context.config import load_context_config
from contextbench.context.sources import Document, chunk_documents, covered_rules
from contextbench.context.strategies import normalized
from contextbench.experiments.config import ExperimentConfig
from contextbench.workload import DatasetConfig, generate


@pytest.fixture
def enterprise(tmp_path):
    generate(tmp_path, DatasetConfig(events=24, reference_sections=8, lookup_rows=4))
    return tmp_path


class CharacterCounter:
    metadata = {"method": "test-character-counter", "is_estimate": True}

    def count(self, text):
        return len(text)


@pytest.mark.parametrize("strategy", ["full_context", "retrieval", "compiled_context"])
def test_complete_typed_result_and_no_answer_reads(enterprise, monkeypatch, strategy):
    original = Path.read_text

    def source_only(path, *args, **kwargs):
        assert path.name not in {"expected.json", "records.json", "sources.json", "manifest.json"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", source_only)
    event = {"event_id": "CHECK", "city": "Harbor-0001", "category": "travel"}
    result = make_strategy(strategy, enterprise, token_counter=CharacterCounter()).build(event)
    assert result.prompt and result.text and result.selected_source_ids == result.source_ids
    assert result.token_count == len(result.prompt)
    assert result.metadata["strategy"] == strategy
    assert result.metadata["tokenizer"]["method"] == "test-character-counter"
    assert json.loads(result.messages[-1]["content"]) == event
    assert result.prompt.endswith(result.messages[-1]["content"])
    assert "CHECK" not in result.text
    assert "CHECK" in result.prompt


def test_full_includes_every_document_and_lookup_with_stable_prefix(enterprise):
    strategy = FullContextStrategy(enterprise)
    one, two = strategy.build({"event_id": "A"}), strategy.build({"event_id": "B"})
    assert one.text == two.text and one.messages[0] == two.messages[0]
    assert one.prompt != two.prompt
    assert "lookup.json" in one.source_ids
    for path in (enterprise / "documents").glob("*.md"):
        assert path.read_text(encoding="utf-8") in one.text
    for row in json.loads((enterprise / "lookup.json").read_text()):
        assert json.dumps(row, sort_keys=True) in one.text
    assert one.metadata["tokenizer"]["is_estimate"] is True


def test_compiled_includes_required_sources_without_oracle_selection(enterprise):
    strategy = CompiledContextStrategy(enterprise)
    records = json.loads((enterprise / "records.json").read_text())
    for record in records:
        result = strategy.build(record["event"])
        assert set(record["ground_truth"]["required_source_ids"]) <= set(result.source_ids)
        for rule in json.loads(result.text):
            assert rule["fact"] and rule["references"]
            assert all(ref["document"] and ref["line"] > 0 for ref in rule["references"])
        lookup_ids = [rid for rid in result.source_ids if rid.startswith("L_")]
        assert len(lookup_ids) <= 1
        if lookup_ids:
            refs = result.metadata["references"][lookup_ids[0]]
            assert {r["document"] for r in refs} == {"lookup.json", "lookup.md"}
    assert "R_EMERGENCY" not in strategy.build({"emergency": 1}).source_ids
    assert "R_EMERGENCY" in strategy.build({"emergency": True}).source_ids
    assert "R_VENDOR" not in strategy.build({"vendor_status": "clear"}).source_ids


def test_chunking_bounded_stable_and_complete():
    text = "# Heading\n\n" + "office expense " * 50 + "\n\nLast paragraph."
    doc = Document("policy.md", text, ())
    chunks = chunk_documents([doc], 80, 13)
    assert chunks == chunk_documents([doc], 80, 13)
    assert len({c.id for c in chunks}) == len(chunks)
    covered = set()
    for chunk in chunks:
        assert len(chunk.text) <= 80
        assert chunk.text == text[chunk.start:chunk.end]
        covered.update(range(chunk.start, chunk.end))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    assert any(a.end - b.start == 13 for a, b in zip(chunks, chunks[1:]))


def test_partial_rule_chunks_do_not_claim_complete_coverage():
    text = "RULE " + "x" * 140
    document = Document("policy.md", text, ((0, len(text), "R_LONG"),))
    chunks = chunk_documents([document], 80, 20)
    assert all(c.rule_ids == ("R_LONG",) for c in chunks)
    assert covered_rules([document], chunks[:1]) == []
    assert covered_rules([document], chunks) == ["R_LONG"]


@pytest.fixture
def prose(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.md").write_text("Medical emergency reimbursement.\n\n"
                                    "Office furniture purchase.\n\n"
                                    "Employee laptop replacement.", encoding="utf-8")
    return tmp_path


def test_bm25_ranking_scores_ties_and_topk(prose):
    strategy = RetrievalContextStrategy(prose, 1)
    result = strategy.build({"category": "furniture"})
    assert "furniture" in result.text
    score = result.metadata["chunks"][0]["score"]
    assert score == pytest.approx(math.log(1 + 2.5 / 1.5))
    assert len(result.source_ids) == 1
    tied = strategy.build({"not_in_documents": "nonsense"})
    assert tied.source_ids == (strategy.chunks[0].id,)
    all_chunks = RetrievalContextStrategy(prose, 100).build({})
    assert len(all_chunks.source_ids) == 3
    assert all_chunks == RetrievalContextStrategy(prose, 100).build({})


class SemanticEncoder:
    """Controlled vectors prove cosine ranking; they are not quality evidence."""
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0] if ("Medical" in t or "ambulance" in t) else [0.0, 5.0]
                for t in texts]


def test_embedding_cosine_ranking_and_index_reuse(prose):
    encoder = SemanticEncoder()
    config = ContextConfig(retrieval=RetrievalConfig(method="embedding", top_k=1))
    strategy = RetrievalContextStrategy(prose, config=config, encoder=encoder)
    assert len(encoder.calls) == 1 and len(encoder.calls[0]) == 3
    result = strategy.build({"description": "ambulance"})
    assert "Medical emergency" in result.text
    assert result.metadata["chunks"][0]["score"] == pytest.approx(1.0)
    assert result.metadata["retrieval"]["method"] == "embedding"
    strategy.build({"description": "ambulance"})
    assert [len(c) for c in encoder.calls] == [3, 1, 1]


@pytest.mark.parametrize("vectors,count,dimension", [
    ([], 1, None), ([[0, 0]], 1, None), ([[float("nan")]], 1, None),
    ([[float("inf")]], 1, None), ([[1, 2], [1]], 2, None), ([[1, 2]], 1, 3),
])
def test_invalid_embeddings_fail_explicitly(vectors, count, dimension):
    with pytest.raises(ValueError):
        normalized(vectors, count, dimension)


def test_source_errors_and_no_silent_compilation(prose):
    with pytest.raises(ValueError, match="no annotated"):
        CompiledContextStrategy(prose)
    path = prose / "documents/policy.md"
    path.write_text('RULE {"id":"R_X","fact":"one"}\n'
                    'RULE {"id":"R_X","fact":"two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting definitions"):
        CompiledContextStrategy(prose)
    path.write_text('RULE {"id":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="policy.md:1"):
        CompiledContextStrategy(prose)


def test_missing_and_empty_sources(tmp_path):
    with pytest.raises(ValueError, match="no nonempty"):
        FullContextStrategy(tmp_path)
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents/empty.md").write_text(" \n\n")
    with pytest.raises(ValueError, match="no nonempty"):
        RetrievalContextStrategy(tmp_path)


@pytest.mark.parametrize("values", [
    {"retrieval": {"top_k": 0}}, {"retrieval": {"method": "unknown"}},
    {"retrieval": {"chunk_chars": 10, "overlap_chars": 10}},
    {"retrieval": {"bm25_b": 1.1}}, {"retrieval": {"bm25_k1": 0.0}},
    {"tokenizer": {"method": "huggingface"}}, {"unknown": True},
])
def test_invalid_context_configuration(values):
    with pytest.raises(ValidationError):
        ContextConfig.model_validate(values)


def test_context_examples_and_experiment_configuration():
    for path in (Path(__file__).parents[1] / "configs/context").glob("*.yaml"):
        assert isinstance(load_context_config(path), ContextConfig)
    server = {"model_revision": "abc", "vllm_version": "test", "launch_command": "test"}
    config = ExperimentConfig.model_validate({"server": server,
                                              "context": {"retrieval": {"top_k": 2}}})
    assert config.context.retrieval.top_k == 2
    legacy = ExperimentConfig.model_validate({"server": server, "retrieval_top_k": 3})
    assert legacy.context.retrieval.top_k == 3
    assert ExperimentConfig.model_validate(legacy.model_dump()) == legacy
    assert ExperimentConfig.model_validate(config.model_dump()) == config
    with pytest.raises(ValueError, match="conflicts"):
        ExperimentConfig.model_validate({"server": server, "retrieval_top_k": 3,
                                         "context": {"retrieval": {"top_k": 2}}})


@pytest.mark.parametrize("strategy", ["full_context", "retrieval", "compiled_context"])
def test_inspection_cli(enterprise, monkeypatch, capsys, strategy):
    monkeypatch.setattr(sys, "argv", ["contextbench", "inspect-context", "--dataset",
                                     str(enterprise), "--event-id", "EV-000000", "--strategy",
                                     strategy, "--top-k", "2"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["token_count"] > 0 and result["selected_source_ids"]
    assert result["metadata"]["strategy"] == strategy
    assert json.loads(result["messages"][-1]["content"])["event_id"] == "EV-000000"


def test_inspection_external_event_and_prompt_file(prose, monkeypatch, capsys):
    event = prose / "event.json"
    event.write_text('{"event_id":"external","description":"furniture"}')
    output = prose / "prompt.txt"
    config = prose / "context.yaml"
    config.write_text("retrieval:\n  top_k: 1\n")
    monkeypatch.setattr(sys, "argv", ["contextbench", "inspect-context", "--dataset", str(prose),
                                     "--event-file", str(event), "--strategy", "retrieval",
                                     "--config", str(config), "--format", "prompt",
                                     "--output", str(output)])
    main()
    assert "external" in output.read_text() and "furniture" in output.read_text()
    assert capsys.readouterr().out == ""


def test_inspection_unknown_event_errors(enterprise, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["contextbench", "inspect-context", "--dataset",
                                     str(enterprise), "--event-id", "DOES-NOT-EXIST"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "matched 0 records" in capsys.readouterr().err


def test_optional_huggingface_adapter(monkeypatch):
    calls = []

    class Tokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append((model, kwargs))
            return Tokenizer()

        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return [3, 4, 5]

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer))
    counter = HuggingFaceTokenCounter(TokenizerConfig(method="huggingface", model="local-test"))
    assert counter.count("some text") == 3
    assert counter.metadata["is_estimate"] is False
    assert calls[0][1]["local_files_only"] and not calls[0][1]["trust_remote_code"]


def test_optional_sentence_transformer_adapter(monkeypatch):
    calls = []

    class Model:
        max_seq_length = 5
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))

        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] and kwargs["convert_to_numpy"]
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    encoder = SentenceTransformerEncoder(RetrievalConfig(method="embedding"))
    assert encoder.encode(["short"]) == [[1.0, 0.0]]
    assert calls[0][1]["local_files_only"] and not calls[0][1]["trust_remote_code"]
    with pytest.raises(ValueError, match="exceeds model token limit"):
        encoder.encode(["too long"])


def test_missing_optional_dependencies_report_install_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"contextbench\[embeddings\]"):
        SentenceTransformerEncoder(RetrievalConfig(method="embedding"))
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match=r"contextbench\[tokenizers\]"):
        HuggingFaceTokenCounter(TokenizerConfig(method="huggingface", model="local"))
