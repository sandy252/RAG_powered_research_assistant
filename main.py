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
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "citation_visibility" not in st.session_state:
    st.session_state.citation_visibility = {}

with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size", min_value=300, max_value=2500, value=1000, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=150, step=10)
    retrieval_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    reset_index = st.checkbox("Reset index before indexing", value=True)
    if st.button("Clear Chat"):
        st.session_state.chat_messages = []
        st.session_state.citation_visibility = {}
        st.success("Chat history cleared.")

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
            st.session_state.chat_messages = []
            st.session_state.citation_visibility = {}
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

st.subheader("Chat with Your Papers")

for idx, msg in enumerate(st.session_state.chat_messages):
    with st.chat_message(msg.get("role", "assistant")):
        st.write(msg.get("content", ""))

        if msg.get("role") == "assistant":
            citations = msg.get("citations", [])
            if citations:
                is_visible = st.session_state.citation_visibility.get(idx, False)
                button_label = "Hide citations" if is_visible else "Show citations"
                if st.button(button_label, key=f"toggle_citations_{idx}"):
                    st.session_state.citation_visibility[idx] = not is_visible
                    is_visible = st.session_state.citation_visibility[idx]

                if is_visible:
                    st.caption("Citations")
                    for item in citations:
                        source = item.get("source") or "unknown_source"
                        page = item.get("page_number") or "unknown_page"
                        st.write(f"- {source} (page {page})")

            used_chunks = msg.get("used_chunks", [])
            if used_chunks:
                with st.expander(f"Retrieved Chunks (Debug) - Turn {idx + 1}"):
                    standalone_query = msg.get("standalone_query")
                    if standalone_query:
                        st.write(f"Standalone retrieval query: {standalone_query}")
                        st.divider()

                    for i, chunk in enumerate(used_chunks, start=1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(f"Source: {chunk.get('source')} | Page: {chunk.get('page_number')}")
                        st.write((chunk.get("content") or "")[:800])
                        st.divider()

if not st.session_state.indexed:
    st.info("Build the index first, then start chatting with your papers.")

user_prompt = st.chat_input(
    "Ask a question about your indexed papers",
    disabled=not st.session_state.indexed,
)

if user_prompt is not None and user_prompt.strip():
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt.strip()})

    with st.spinner("Retrieving and generating answer..."):
        try:
            generator = RAGGenerator(
                persist_directory="vector_db",
                collection_name=st.session_state.collection_name,
                retrieval_k=int(retrieval_k),
            )
            history = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in st.session_state.chat_messages[:-1]
            ]
            result = generator.answer_chat(question=user_prompt, history=history, history_window=8)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": result.get("answer", "No answer generated."),
                    "citations": result.get("citations", []),
                    "used_chunks": result.get("used_chunks", []),
                    "standalone_query": result.get("standalone_query", ""),
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
