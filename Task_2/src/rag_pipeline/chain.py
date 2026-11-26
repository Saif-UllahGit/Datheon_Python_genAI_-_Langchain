"""Grounded RetrievalQA chain with mandatory source page citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

STRICT_QA_TEMPLATE = """You are a strict document question-answering assistant.
Answer the user's question using ONLY the retrieved context chunks provided below.

STRICT RULES:
1. Do NOT use outside knowledge, assumptions, or unstated facts.
2. If the answer cannot be found in the provided context, reply exactly:
   "I cannot answer this based on the provided documents."
3. Cite the exact source document and page number inline for every claim using the format [Source: <file_name>, p. <page>].
4. Be concise, accurate, and faithful to numbers, dates, and policy terms in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

DOCUMENT_CHUNK_TEMPLATE = "[Source: {file_name}, p. {page}]\n{page_content}"

STRICT_QA_PROMPT = PromptTemplate(
    template=STRICT_QA_TEMPLATE,
    input_variables=["context", "question"],
)

DOCUMENT_CHUNK_PROMPT = PromptTemplate(
    template=DOCUMENT_CHUNK_TEMPLATE,
    input_variables=["page_content", "file_name", "page"],
)


@dataclass(frozen=True)
class Citation:
    """Structured citation metadata for a retrieved source chunk."""

    file_name: str
    page: int
    source: str
    chunk_id: str
    snippet: str


@dataclass(frozen=True)
class RAGAnswer:
    """Final RAG answer paired with deduplicated page citations and raw chunks."""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    source_documents: tuple[Document, ...]

    def format_display(self) -> str:
        """Render a human-readable answer with a Sources section."""
        lines = [self.answer.strip()]
        if self.citations:
            lines.append("\nSources:")
            for idx, cit in enumerate(self.citations, start=1):
                lines.append(
                    f"  [{idx}] {cit.file_name} (p. {cit.page}) — {cit.snippet}"
                )
        return "\n".join(lines)


class _LCELRetrievalQAFallback:
    """Compatibility fallback mimicking RetrievalQA when langchain.chains is unavailable."""

    def __init__(self, retriever: Any, llm: BaseChatModel) -> None:
        self.retriever = retriever
        self.llm = llm

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs["query"]
        docs: list[Document] = list(self.retriever.invoke(query))
        formatted_chunks = [
            DOCUMENT_CHUNK_PROMPT.format(
                page_content=doc.page_content,
                file_name=doc.metadata.get("file_name", "unknown"),
                page=doc.metadata.get("page", 1),
            )
            for doc in docs
        ]
        prompt_str = STRICT_QA_PROMPT.format(
            context="\n\n".join(formatted_chunks),
            question=query,
        )
        response = self.llm.invoke(prompt_str)
        answer_text = response.content if hasattr(response, "content") else str(response)
        return {"result": answer_text, "source_documents": docs}


def _build_retrieval_qa_chain(
    llm: BaseChatModel,
    retriever: Any,
) -> Any:
    """Construct LangChain RetrievalQA chain with classic/LCEL fallback."""
    retrieval_qa_cls = None
    try:
        from langchain.chains import RetrievalQA

        retrieval_qa_cls = RetrievalQA
    except ImportError:
        try:
            from langchain_classic.chains import RetrievalQA  # type: ignore

            retrieval_qa_cls = RetrievalQA
        except ImportError:
            retrieval_qa_cls = None

    if retrieval_qa_cls is not None:
        return retrieval_qa_cls.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": STRICT_QA_PROMPT,
                "document_prompt": DOCUMENT_CHUNK_PROMPT,
            },
        )
    return _LCELRetrievalQAFallback(retriever=retriever, llm=llm)


class RAGQueryEngine:
    """Executes RetrievalQA queries and attaches deduplicated source page citations."""

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseChatModel,
        top_k: int = 4,
        search_type: str = "similarity",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": top_k},
        )
        self.qa_chain = _build_retrieval_qa_chain(llm=llm, retriever=self.retriever)

    def ask(self, question: str) -> RAGAnswer:
        """Query the vector store and return a strictly grounded, cited answer."""
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        raw = self.qa_chain.invoke({"query": question.strip()})
        answer_text = str(raw.get("result", "")).strip()
        source_docs: list[Document] = list(raw.get("source_documents", []))

        # Ensure metadata defaults exist on every retrieved document
        seen_keys: set[tuple[str, int]] = set()
        citations: list[Citation] = []
        for doc in source_docs:
            file_name = str(doc.metadata.get("file_name", "unknown"))
            page = int(doc.metadata.get("page", 1))
            source = str(doc.metadata.get("source", file_name))
            chunk_id = str(doc.metadata.get("chunk_id", f"{file_name}:p{page}"))
            key = (file_name, page)
            if key not in seen_keys:
                seen_keys.add(key)
                snippet = " ".join(doc.page_content.split())[:140]
                if len(" ".join(doc.page_content.split())) > 140:
                    snippet += "..."
                citations.append(
                    Citation(
                        file_name=file_name,
                        page=page,
                        source=source,
                        chunk_id=chunk_id,
                        snippet=snippet,
                    )
                )

        return RAGAnswer(
            question=question.strip(),
            answer=answer_text,
            citations=tuple(citations),
            source_documents=tuple(source_docs),
        )
