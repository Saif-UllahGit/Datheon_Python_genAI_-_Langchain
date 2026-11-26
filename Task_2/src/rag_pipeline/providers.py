"""Embedding and LLM provider factories (Gemini, OpenAI, and deterministic offline mock)."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ProviderName = Literal["gemini", "openai", "mock"]

DEFAULT_CHAT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "mock": "mock-grounded-v1",
}

DEFAULT_EMBED_MODELS: dict[str, str] = {
    "gemini": "models/text-embedding-004",
    "openai": "text-embedding-3-small",
    "mock": "mock-hash-embed-256d",
}


class DeterministicHashEmbeddings(Embeddings):
    """Offline deterministic embeddings using hashed unigram + character trigram vectors."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        normalized = text.lower()
        tokens = re.findall(r"[a-z0-9]+", normalized)
        features = list(tokens)
        for token in tokens:
            if len(token) >= 3:
                features.extend(token[i : i + 3] for i in range(len(token) - 2))

        if not features:
            return vec

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GroundedMockChatModel(BaseChatModel):
    """Offline mock LLM that strictly answers from retrieved [Source: ..., p. ...] chunks."""

    model_name: str = "mock-grounded-v1"

    @property
    def _llm_type(self) -> str:
        return "mock-grounded-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = "\n".join(str(m.content) for m in messages)

        context_match = re.search(
            r"CONTEXT:\s*(.*?)\s*QUESTION:\s*(.*?)(?:\nANSWER:|\Z)",
            prompt_text,
            flags=re.DOTALL,
        )
        if not context_match:
            reply = "I cannot answer this based on the provided documents."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        context_block = context_match.group(1).strip()
        question = context_match.group(2).strip()
        q_tokens = {
            t
            for t in re.findall(r"[a-z0-9]+", question.lower())
            if len(t) > 2 and t not in {"what", "when", "where", "which", "how", "does", "the", "are", "for"}
        }

        chunk_pattern = re.compile(
            r"(\[Source:\s*[^,\]]+,\s*p\.\s*\d+\])\s*(.*?)(?=\[Source:|\Z)",
            flags=re.DOTALL,
        )
        scored_sentences: list[tuple[int, str, str]] = []
        for citation, body in chunk_pattern.findall(context_block):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", body.strip()):
                clean = sentence.strip()
                if not clean:
                    continue
                s_tokens = set(re.findall(r"[a-z0-9]+", clean.lower()))
                overlap = len(q_tokens & s_tokens)
                if overlap > 0:
                    scored_sentences.append((overlap, clean, citation))

        if not scored_sentences:
            reply = "I cannot answer this based on the provided documents."
        else:
            scored_sentences.sort(key=lambda item: item[0], reverse=True)
            top = scored_sentences[:3]
            parts = [f"{text} {cit}" for _, text, cit in top]
            reply = " ".join(parts)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


def create_embeddings(
    provider: ProviderName = "gemini",
    model: str | None = None,
    api_key: str | None = None,
) -> Embeddings:
    """Instantiate an Embeddings backend for the chosen provider."""
    if provider == "mock":
        return DeterministicHashEmbeddings()

    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini embeddings.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model or os.getenv("GEMINI_EMBED_MODEL") or DEFAULT_EMBED_MODELS["gemini"],
            google_api_key=key,
        )

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing OPENAI_API_KEY for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model or os.getenv("OPENAI_EMBED_MODEL") or DEFAULT_EMBED_MODELS["openai"],
            api_key=key,
        )

    raise ValueError(f"Unsupported embedding provider: {provider!r}")


def create_llm(
    provider: ProviderName = "gemini",
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Instantiate a chat LLM backend for grounded RetrievalQA."""
    if provider == "mock":
        return GroundedMockChatModel()

    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini chat model.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_CHAT_MODEL") or DEFAULT_CHAT_MODELS["gemini"],
            google_api_key=key,
            temperature=temperature,
        )

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing OPENAI_API_KEY for OpenAI chat model.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or os.getenv("OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODELS["openai"],
            api_key=key,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
