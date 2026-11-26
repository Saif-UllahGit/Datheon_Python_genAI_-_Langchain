"""End-to-end tests for PDF/text loading, recursive chunking, ChromaDB/FAISS, and RetrievalQA."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import rag_cli  # noqa: E402
from rag_pipeline.chain import RAGQueryEngine  # noqa: E402
from rag_pipeline.ingestion import (  # noqa: E402
    ChunkConfig,
    build_or_load_vectorstore,
    chunk_documents,
    load_documents,
)
from rag_pipeline.providers import (  # noqa: E402
    DeterministicHashEmbeddings,
    GroundedMockChatModel,
)

SAMPLE_PDF = PROJECT_ROOT / "samples" / "company_policy_handbook.pdf"
SAMPLE_TXT = PROJECT_ROOT / "samples" / "architecture_notes.txt"


class IngestionTests(unittest.TestCase):
    def test_loads_multipage_pdf_and_text_with_one_based_pages(self) -> None:
        docs = load_documents([SAMPLE_PDF, SAMPLE_TXT])
        pdf_pages = sorted(d.metadata["page"] for d in docs if d.metadata["doc_type"] == "pdf")
        txt_pages = [d.metadata["page"] for d in docs if d.metadata["doc_type"] == "text"]

        self.assertEqual(pdf_pages, [1, 2, 3])
        self.assertEqual(txt_pages, [1])
        self.assertTrue(all(d.metadata["file_name"] for d in docs))

    def test_recursive_chunking_preserves_citation_metadata(self) -> None:
        docs = load_documents([SAMPLE_PDF])
        chunks = chunk_documents(docs, ChunkConfig(chunk_size=220, chunk_overlap=40))

        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["file_name"], "company_policy_handbook.pdf")
            self.assertIn(chunk.metadata["page"], (1, 2, 3))
            self.assertIn("chunk_id", chunk.metadata)


class VectorStoreAndRetrievalQATests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = PROJECT_ROOT / ".rag_test_tmp"
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = DeterministicHashEmbeddings()
        self.llm = GroundedMockChatModel()
        docs = load_documents([SAMPLE_PDF, SAMPLE_TXT])
        self.chunks = chunk_documents(docs, ChunkConfig(chunk_size=350, chunk_overlap=50))

    def tearDown(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_chroma_persistence_and_page_citations(self) -> None:
        chroma_dir = self.tmp_dir / "chroma_db"
        store = build_or_load_vectorstore(
            chunks=self.chunks,
            embeddings=self.embeddings,
            backend="chroma",
            persist_dir=chroma_dir,
            collection_name="test_handbook",
            reset=True,
        )
        reloaded = build_or_load_vectorstore(
            chunks=None,
            embeddings=self.embeddings,
            backend="chroma",
            persist_dir=chroma_dir,
            collection_name="test_handbook",
        )
        engine = RAGQueryEngine(vectorstore=reloaded, llm=self.llm, top_k=3)

        result = engine.ask("How many weeks of paid parental leave do caregivers receive?")
        self.assertIn("16 weeks", result.answer)
        self.assertTrue(
            any(c.file_name == "company_policy_handbook.pdf" and c.page == 2 for c in result.citations)
        )
        del store

    def test_faiss_save_load_and_strict_out_of_scope_refusal(self) -> None:
        faiss_dir = self.tmp_dir / "faiss_index"
        build_or_load_vectorstore(
            chunks=self.chunks,
            embeddings=self.embeddings,
            backend="faiss",
            persist_dir=faiss_dir,
            reset=True,
        )
        reloaded = build_or_load_vectorstore(
            chunks=None,
            embeddings=self.embeddings,
            backend="faiss",
            persist_dir=faiss_dir,
        )
        engine = RAGQueryEngine(vectorstore=reloaded, llm=self.llm, top_k=3)

        in_scope = engine.ask("What is the travel meals cap per day?")
        self.assertIn("USD 75", in_scope.answer)
        self.assertTrue(any(c.page == 3 for c in in_scope.citations))

        out_of_scope = engine.ask("What is the price of crude oil on Mars?")
        self.assertEqual(out_of_scope.answer, "I cannot answer this based on the provided documents.")

    def test_cli_json_output_end_to_end(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = rag_cli.main(
                [
                    "--docs",
                    str(SAMPLE_PDF),
                    "--question",
                    "What is the home office ergonomic equipment stipend?",
                    "--provider",
                    "mock",
                    "--vectorstore",
                    "faiss",
                    "--json",
                ]
            )
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("1,200", payload["answer"])
        self.assertTrue(any(c["page"] == 1 for c in payload["citations"]))


if __name__ == "__main__":
    unittest.main()
