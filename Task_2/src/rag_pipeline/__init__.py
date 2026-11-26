"""Grounded PDF/Text RAG pipeline using LangChain, RecursiveCharacterTextSplitter, and ChromaDB/FAISS."""

from .chain import Citation, RAGAnswer, RAGQueryEngine
from .ingestion import (
    ChunkConfig,
    VectorStoreBackend,
    build_or_load_vectorstore,
    chunk_documents,
    load_documents,
)
from .providers import create_embeddings, create_llm

__all__ = [
    "ChunkConfig",
    "Citation",
    "RAGAnswer",
    "RAGQueryEngine",
    "VectorStoreBackend",
    "build_or_load_vectorstore",
    "chunk_documents",
    "create_embeddings",
    "create_llm",
    "load_documents",
]
