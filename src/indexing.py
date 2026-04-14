from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class VectorIndexManager:
    """Create and manage a persistent Chroma index for split document chunks."""

    def __init__(
        self,
        persist_directory: str = "vector_db",
        collection_name: str = "rag_chunks",
        embedding_model: str = "text-embedding-3-small",
        ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in the project .env file at the repository root."
            )
        self.persist_path = Path(persist_directory)
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def _build_store(self) -> Chroma:
        self.persist_path.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

    def index_documents(
        self,
        chunks: Iterable[Document],
        batch_size: int = 100,
        reset_collection: bool = False,
    ) -> Chroma:
        """Embed chunk documents and store them in Chroma in batches."""
        docs = [doc for doc in chunks if doc.page_content and doc.page_content.strip()]
        store = self._build_store()

        if reset_collection:
            store.delete_collection()
            store = self._build_store()

        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            ids = [str(uuid4()) for _ in batch]
            store.add_documents(batch, ids=ids)

        return store

    def load_vector_store(self) -> Chroma:
        """Load an existing persistent vector store."""
        return self._build_store()

    def get_retriever(self, k: int = 5, search_type: str = "similarity"):
        """Return retriever ready for query pipeline usage."""
        store = self._build_store()
        return store.as_retriever(search_type=search_type, search_kwargs={"k": k})


def build_index_from_chunks(
    chunks: Iterable[Document],
    persist_directory: str = "vector_db",
    collection_name: str = "rag_chunks",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 100,
    reset_collection: bool = False,
) -> Chroma:
    """Convenience function to index chunks directly without class wiring."""
    manager = VectorIndexManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    return manager.index_documents(
        chunks=chunks,
        batch_size=batch_size,
        reset_collection=reset_collection,
    )


def count_indexed_documents(
    persist_directory: str = "vector_db",
    collection_name: str = "rag_chunks",
    embedding_model: str = "text-embedding-3-small",
) -> Optional[int]:
    """Return the current number of vectors in the collection."""
    manager = VectorIndexManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    store = manager.load_vector_store()
    try:
        return store._collection.count()
    except Exception:
        return None
