#!/usr/bin/env python3
"""Executable CLI for the PDF/Text RAG pipeline using LangChain and ChromaDB/FAISS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from rag_pipeline.chain import RAGQueryEngine
from rag_pipeline.ingestion import (
    ChunkConfig,
    build_or_load_vectorstore,
    chunk_documents,
    load_documents,
)
from rag_pipeline.providers import create_embeddings, create_llm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions grounded strictly in custom PDF/text documents with page citations."
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=None,
        help="Paths to PDF/text files or directories to ingest.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load an existing persisted vector store from --persist-dir without re-ingesting.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Question to ask against the indexed documents.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive Q&A loop.",
    )
    parser.add_argument(
        "--vectorstore",
        default="chroma",
        choices=["chroma", "faiss"],
        help="Vector store backend (default: chroma).",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Directory to persist/load ChromaDB or FAISS index.",
    )
    parser.add_argument(
        "--collection-name",
        default="rag_documents",
        help="ChromaDB collection name (default: rag_documents).",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete existing --persist-dir before indexing.",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "mock"],
        help="LLM provider (default: gemini).",
    )
    parser.add_argument(
        "--embed-provider",
        default=None,
        choices=["gemini", "openai", "mock"],
        help="Optional override for embeddings provider (defaults to --provider).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional chat model override.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Optional embedding model override.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="RecursiveCharacterTextSplitter chunk size (default: 800).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="RecursiveCharacterTextSplitter overlap (default: 150).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve per query (default: 4).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output answer and source page citations as JSON.",
    )
    return parser.parse_args(argv)


def _print_result(result, as_json: bool) -> None:
    if as_json:
        payload = {
            "question": result.question,
            "answer": result.answer,
            "citations": [
                {
                    "file_name": c.file_name,
                    "page": c.page,
                    "source": c.source,
                    "chunk_id": c.chunk_id,
                    "snippet": c.snippet,
                }
                for c in result.citations
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(result.format_display())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.load_only and not args.docs:
        print("Error: provide --docs <paths...> or use --load-only with --persist-dir.", file=sys.stderr)
        return 1
    if not args.question and not args.interactive:
        print("Error: provide --question '...' or pass --interactive.", file=sys.stderr)
        return 1

    try:
        embed_provider = args.embed_provider or args.provider
        embeddings = create_embeddings(provider=embed_provider, model=args.embed_model)
        llm = create_llm(provider=args.provider, model=args.model, temperature=0.0)

        chunks = None
        if not args.load_only:
            raw_docs = load_documents(args.docs)
            chunks = chunk_documents(
                raw_docs,
                ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
            )
            print(
                f"Indexed {len(raw_docs)} document page(s) into {len(chunks)} chunk(s) "
                f"using {args.vectorstore}.",
                file=sys.stderr,
            )

        vectorstore = build_or_load_vectorstore(
            chunks=chunks,
            embeddings=embeddings,
            backend=args.vectorstore,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            reset=args.reset_index,
        )
        engine = RAGQueryEngine(vectorstore=vectorstore, llm=llm, top_k=args.top_k)

        if args.question:
            result = engine.ask(args.question)
            _print_result(result, as_json=args.json)

        if args.interactive:
            print("Interactive RAG mode. Type 'exit' or 'quit' to stop.", file=sys.stderr)
            while True:
                try:
                    user_q = input("\nQuestion> ").strip()
                except EOFError:
                    break
                if not user_q or user_q.lower() in {"exit", "quit"}:
                    break
                result = engine.ask(user_q)
                _print_result(result, as_json=args.json)
                print("-" * 60)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
