"""Document loading, recursive chunking, and ChromaDB/FAISS vector store indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Literal, Sequence

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - legacy LangChain fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

VectorStoreBackend = Literal["chroma", "faiss"]
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for RecursiveCharacterTextSplitter."""

    chunk_size: int = 800
    chunk_overlap: int = 150
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")


def _collect_files(paths: Sequence[str | Path]) -> list[Path]:
    collected: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Document path does not exist: {path}")
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    collected.append(child)
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type {path.suffix!r} for {path}. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
            collected.append(path)
    if not collected:
        raise ValueError("No supported PDF or text documents found in the provided paths.")
    return collected


def load_documents(paths: Sequence[str | Path]) -> list[Document]:
    """Load PDF and text documents with normalized 1-based page citation metadata."""
    files = _collect_files(paths)
    loaded: list[Document] = []

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                raw_page = doc.metadata.get("page", 0)
                page_number = int(raw_page) + 1 if isinstance(raw_page, (int, float)) else 1
                doc.metadata = {
                    **doc.metadata,
                    "source": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "page": page_number,
                    "doc_type": "pdf",
                }
                if doc.page_content.strip():
                    loaded.append(doc)
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata = {
                    **doc.metadata,
                    "source": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "page": 1,
                    "doc_type": "text",
                }
                if doc.page_content.strip():
                    loaded.append(doc)

    if not loaded:
        raise ValueError("Loaded documents contain no extractable text.")
    return loaded


def chunk_documents(
    documents: Sequence[Document],
    config: ChunkConfig | None = None,
) -> list[Document]:
    """Split documents recursively while preserving source and page citation metadata."""
    cfg = config or ChunkConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=list(cfg.separators),
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))
    for idx, chunk in enumerate(chunks):
        file_name = chunk.metadata.get("file_name", "unknown")
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{file_name}:p{page}:c{idx}"
    return chunks


def _import_chroma():
    try:
        from langchain_chroma import Chroma

        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma  # type: ignore

        return Chroma


def build_or_load_vectorstore(
    chunks: Sequence[Document] | None,
    embeddings: Embeddings,
    backend: VectorStoreBackend = "chroma",
    persist_dir: str | Path | None = None,
    collection_name: str = "rag_documents",
    reset: bool = False,
) -> VectorStore:
    """Index document chunks into ChromaDB or FAISS, or load an existing index."""
    persist_path = Path(persist_dir) if persist_dir is not None else None
    if reset and persist_path is not None and persist_path.exists():
        shutil.rmtree(persist_path)

    if backend == "chroma":
        Chroma = _import_chroma()
        kwargs: dict[str, object] = {
            "collection_name": collection_name,
            "embedding_function": embeddings,
        }
        if persist_path is not None:
            persist_path.mkdir(parents=True, exist_ok=True)
            kwargs["persist_directory"] = str(persist_path)

        if chunks:
            return Chroma.from_documents(
                documents=list(chunks),
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(persist_path) if persist_path is not None else None,
            )
        if persist_path is None or not persist_path.exists():
            raise ValueError("Cannot load Chroma store: no chunks provided and persist_dir missing.")
        return Chroma(**kwargs)  # type: ignore[arg-type]

    if backend == "faiss":
        from langchain_community.vectorstores import FAISS

        if chunks:
            store = FAISS.from_documents(documents=list(chunks), embedding=embeddings)
            if persist_path is not None:
                persist_path.mkdir(parents=True, exist_ok=True)
                store.save_local(str(persist_path))
            return store

        if persist_path is None or not persist_path.exists():
            raise ValueError("Cannot load FAISS store: no chunks provided and persist_dir missing.")
        return FAISS.load_local(
            str(persist_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    raise ValueError(f"Unsupported vector store backend: {backend!r}")
