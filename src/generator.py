from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research assistant. Answer only using the provided context. "
                    "If the answer is not supported by the context, clearly say you do not have enough evidence.",
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\n"
                    "Question:\n{question}\n\n"
                    "Context:\n{context}\n\n"
                    "Return a concise answer grounded in the context.",
                ),
            ]
        )
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the user's latest message into a standalone retrieval query for research papers. "
                    "Use the conversation history only to resolve references (for example: it, this method, that paper). "
                    "Return only the rewritten query text.",
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\n"
                    "Latest user message:\n{question}",
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

    @staticmethod
    def _format_history(history: Sequence[Dict[str, str]]) -> str:
        if not history:
            return "(none)"

        lines: List[str] = []
        for turn in history:
            role = (turn.get("role") or "user").strip().lower()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")

        return "\n".join(lines) if lines else "(none)"

    def _rewrite_query(self, question: str, history: Sequence[Dict[str, str]]) -> str:
        if not history:
            return question

        messages = self.rewrite_prompt.format_messages(
            question=question,
            history=self._format_history(history),
        )
        response = self.llm.invoke(messages)
        rewritten = str(response.content or "").strip()
        return rewritten or question

    def answer_chat(
        self,
        question: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
        history_window: int = 8,
    ) -> Dict[str, object]:
        """Answer a user question using recent conversation context plus retrieved chunks."""
        question = question.strip()
        if not question:
            return {
                "answer": "Please provide a non-empty question.",
                "citations": [],
                "used_chunks": [],
                "standalone_query": "",
            }

        chat_history = list(history or [])
        if history_window > 0:
            recent_history = chat_history[-history_window:]
        else:
            recent_history = chat_history

        standalone_query = self._rewrite_query(question=question, history=recent_history)
        retrieved_chunks = self.retriever.retrieve_with_citations(standalone_query)
        if not retrieved_chunks:
            return {
                "answer": "I could not find enough evidence in the indexed documents.",
                "citations": [],
                "used_chunks": [],
                "standalone_query": standalone_query,
            }

        context = self._format_context(retrieved_chunks)
        messages = self.answer_prompt.format_messages(
            question=question,
            history=self._format_history(recent_history),
            context=context,
        )
        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "citations": self._build_citations(retrieved_chunks),
            "used_chunks": retrieved_chunks,
            "standalone_query": standalone_query,
        }

    def answer_query(self, question: str) -> Dict[str, object]:
        """Backward-compatible single-turn entry point."""
        return self.answer_chat(question=question, history=[])


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