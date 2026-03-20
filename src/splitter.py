from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
	documents: List[Document],
	chunk_size: int = 1000,
	chunk_overlap: int = 150,
) -> List[Document]:
	"""Split documents using LangChain RecursiveCharacterTextSplitter."""
	if chunk_overlap >= chunk_size:
		raise ValueError("chunk_overlap must be smaller than chunk_size")

	valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
	if not valid_docs:
		return []

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],
	)

	# split_documents preserves metadata for each resulting chunk.
	return splitter.split_documents(valid_docs)


if __name__ == "__main__":
	from loader import load_pdfs_to_documents

	result = split_documents(
		load_pdfs_to_documents(r"D:\RAG_assistant\attention_paper.pdf"),
		chunk_size=1000,
		chunk_overlap=150,
	)

	print(f"Total chunks: {len(result)}")
	# if result:
	# 	print("Sample metadata:", result[1].metadata)
	# 	print("Sample chunk preview:")
	# 	print(result[1].page_content)
	print(result[0])