from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from indexing import VectorIndexManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class RAGRetriever:
	"""Retriever wrapper for querying indexed chunk documents."""

	def __init__(
		self,
		persist_directory: str = "vector_db",
		collection_name: str = "rag_chunks",
		embedding_model: str = "text-embedding-3-small",
		k: int = 5,
		search_type: str = "similarity",
	) -> None:
		self.manager = VectorIndexManager(
			persist_directory=persist_directory,
			collection_name=collection_name,
			embedding_model=embedding_model,
		)
		self.k = k
		self.search_type = search_type

	def _get_retriever(self):
		return self.manager.get_retriever(k=self.k, search_type=self.search_type)

	def retrieve(self, query: str) -> List[Document]:
		"""Return top matching chunk documents for the query."""
		query = query.strip()
		if not query:
			return []
		retriever = self._get_retriever()
		return retriever.invoke(query)

	def retrieve_with_citations(self, query: str) -> List[Dict[str, object]]:
		"""Return retrieval results in a citation-friendly structure."""
		documents = self.retrieve(query)
		results: List[Dict[str, object]] = []

		for doc in documents:
			metadata = dict(doc.metadata)
			results.append(
				{
					"content": doc.page_content,
					"source": metadata.get("source"),
					"file_path": metadata.get("file_path"),
					"page_number": metadata.get("page_number"),
					"metadata": metadata,
				}
			)

		return results


def get_relevant_chunks(
	query: str,
	k: int = 5,
	persist_directory: str = "vector_db",
	collection_name: str = "rag_chunks",
	embedding_model: str = "text-embedding-3-small",
) -> List[Document]:
	"""Convenience function for one-off retrieval."""
	retriever = RAGRetriever(
		persist_directory=persist_directory,
		collection_name=collection_name,
		embedding_model=embedding_model,
		k=k,
	)
	return retriever.retrieve(query)
