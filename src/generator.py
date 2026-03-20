from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retriever import RAGRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class RAGGenerator:
    """Generate grounded answers from retrieved chunks."""

    def __init__(
        self,
        persist_directory: str = "vector_db",
        collection_name: str = "rag_chunks",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        retrieval_k: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.retriever = RAGRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model,
            k=retrieval_k,
        )
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research assistant. Answer only using the provided context. "
                    "If the answer is not supported by the context, clearly say you do not have enough evidence.",
                ),
                (
                    "human",
                    "Question:\n{question}\n\n"
                    "Context:\n{context}\n\n"
                    "Return a concise answer grounded in the context.",
                ),
            ]
        )

    @staticmethod
    def _format_context(chunks: List[Dict[str, object]]) -> str:
        parts: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("source") or "unknown_source"
            page = chunk.get("page_number") or "unknown_page"
            content = str(chunk.get("content") or "").strip()
            parts.append(f"[Chunk {idx} | source={source} | page={page}]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_citations(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen = set()
        citations: List[Dict[str, object]] = []

        for chunk in chunks:
            source = chunk.get("source")
            page = chunk.get("page_number")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"source": source, "page_number": page})

        return citations

    def answer_query(self, question: str) -> Dict[str, object]:
        """Retrieve context and generate an answer with citations."""
        question = question.strip()
        if not question:
            return {
                "answer": "Please provide a non-empty question.",
                "citations": [],
                "used_chunks": [],
            }

        retrieved_chunks = self.retriever.retrieve_with_citations(question)
        if not retrieved_chunks:
            return {
                "answer": "I could not find enough evidence in the indexed documents.",
                "citations": [],
                "used_chunks": [],
            }

        context = self._format_context(retrieved_chunks)
        messages = self.prompt.format_messages(question=question, context=context)
        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "citations": self._build_citations(retrieved_chunks),
            "used_chunks": retrieved_chunks,
        }


def ask_question(
    question: str,
    persist_directory: str = "vector_db",
    collection_name: str = "rag_chunks",
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    retrieval_k: int = 5,
) -> Dict[str, object]:
    generator = RAGGenerator(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        retrieval_k=retrieval_k,
    )
    return generator.answer_query(question)