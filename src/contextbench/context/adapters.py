"""Optional local embedding/tokenizer adapters, loaded only when requested."""
import re
from typing import Any, Protocol, Sequence

from contextbench.context.config import RetrievalConfig, TokenizerConfig


class TokenCounter(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def count(self, text: str) -> int: ...


class EmbeddingEncoder(Protocol):
    """Encoders can be injected without an inference backend."""
    def encode(self, texts: list[str]) -> Sequence[Sequence[float]]: ...


class RegexTokenCounter:
    metadata = {"method": "regex", "name": "unicode-words-and-punctuation-v1",
                "is_estimate": True, "scope": "prompt_text_without_chat_template"}

    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class HuggingFaceTokenCounter:
    def __init__(self, config: TokenizerConfig):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install contextbench[tokenizers] for Hugging Face token counts") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, revision=config.revision, local_files_only=config.local_files_only,
            trust_remote_code=False)
        self.metadata = {"method": "huggingface", "name": config.model,
                         "revision": config.revision, "is_estimate": False,
                         "scope": "prompt_text_without_chat_template"}

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))


def make_token_counter(config: TokenizerConfig) -> TokenCounter:
    if config.method == "regex":
        return RegexTokenCounter()
    return HuggingFaceTokenCounter(config)


class SentenceTransformerEncoder:
    def __init__(self, config: RetrievalConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install contextbench[embeddings] for embedding retrieval") from exc
        self.model = SentenceTransformer(
            config.embedding_model, revision=config.embedding_revision,
            device=config.embedding_device, local_files_only=config.local_files_only,
            trust_remote_code=False)
        self.batch_size = config.embedding_batch_size

    def encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        # Reject silent truncation: provenance must refer to content actually embedded.
        for text in texts:
            length = len(self.model.tokenizer.encode(text, truncation=False))
            if length > self.model.max_seq_length:
                raise ValueError("embedding input exceeds model token limit; reduce chunk_chars "
                                 "or shorten the event/use a longer-context embedding model")
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True).tolist()
