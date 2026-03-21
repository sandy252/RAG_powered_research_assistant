from pathlib import Path
# from src import loader, splitter, retriever, generator, indexing
import shutil
import sys
import tempfile

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.loader import load_documents
from src.splitter import split_documents
from src.indexing import build_index_from_chunks, count_indexed_documents
from src.generator import RAGGenerator


st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("RAG Powered Research Assistant")
st.caption("Upload research papers, build index, and ask grounded questions.")

if "collection_name" not in st.session_state:
    st.session_state.collection_name = "rag_chunks_ui"
if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size", min_value=300, max_value=2500, value=1000, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=150, step=10)
    retrieval_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    reset_index = st.checkbox("Reset index before indexing", value=True)

uploaded_files = st.file_uploader(
    "Upload one or more PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

url_input = st.text_area(
    "Or add paper URLs (one per line)",
    placeholder="https://arxiv.org/pdf/1706.03762.pdf\nhttps://example.com/paper-page",
)

urls = [line.strip() for line in url_input.splitlines() if line.strip()]
has_sources = bool(uploaded_files) or bool(urls)

if st.button("Build / Update Index", type="primary", disabled=not has_sources):
    with st.spinner("Processing files/URLs and building vector index..."):
        temp_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
        saved_paths = []

        try:
            for uploaded in uploaded_files or []:
                save_path = temp_dir / uploaded.name
                save_path.write_bytes(uploaded.getbuffer())
                saved_paths.append(str(save_path))

            source_items = [*saved_paths, *urls]
            documents = load_documents(source_items)
            if not documents:
                raise ValueError("No readable content found in provided files/URLs.")

            chunks = split_documents(
                documents,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
            )

            build_index_from_chunks(
                chunks=chunks,
                persist_directory="vector_db",
                collection_name=st.session_state.collection_name,
                batch_size=100,
                reset_collection=reset_index,
            )

            total_indexed = count_indexed_documents(
                persist_directory="vector_db",
                collection_name=st.session_state.collection_name,
            )

            st.session_state.indexed = True
            st.success(
                f"Index ready. Loaded pages: {len(documents)} | Chunks: {len(chunks)} | "
                f"Vectors in collection: {total_indexed}"
            )
        except Exception as exc:
            st.session_state.indexed = False
            st.error(f"Indexing failed: {exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

st.divider()

question = st.text_input("Ask a question about your uploaded papers")

if st.button("Get Answer", disabled=not st.session_state.indexed):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                generator = RAGGenerator(
                    persist_directory="vector_db",
                    collection_name=st.session_state.collection_name,
                    retrieval_k=int(retrieval_k),
                )
                result = generator.answer_query(question)

                st.subheader("Answer")
                st.write(result.get("answer", "No answer generated."))

                citations = result.get("citations", [])
                st.subheader("Citations")
                if citations:
                    for item in citations:
                        source = item.get("source") or "unknown_source"
                        page = item.get("page_number") or "unknown_page"
                        st.write(f"- {source} (page {page})")
                else:
                    st.write("No citations available.")

                with st.expander("Retrieved Chunks (Debug)"):
                    for i, chunk in enumerate(result.get("used_chunks", []), start=1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(f"Source: {chunk.get('source')} | Page: {chunk.get('page_number')}")
                        st.write((chunk.get("content") or "")[:800])
                        st.divider()
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
