# RAG-powered Research Assistant

A research assistant that uses Retrieval-Augmented Generation (RAG) to answer questions grounded in a local corpus of documents (PDF, DOCX, or HTML pages).

This project provides a simple pipeline and Streamlit UI to:

- Ingest documents (local PDFs/DOCX or remote URLs)
- Split documents into manageable chunks
- Create and persist embeddings in a Chroma-backed vector store
- Retrieve relevant chunks and generate answers with an LLM constrained to retrieved context

## Key Capabilities

- Ingest PDFs (per-page extraction) and DOCX files (requires `python-docx`) and web pages (requires `beautifulsoup4`).
- Split long text into chunks with configurable `chunk_size` and `chunk_overlap`.
- Persist embeddings using OpenAI embeddings and Chroma in the `vector_db/` folder.
- Perform retrieval with configurable Top-K and return citation-friendly metadata (source, page).
- Rewrite conversational questions into standalone retrieval queries to handle follow-ups.
- Streamlit UI (`main.py`) to build/update index and chat with your indexed papers interactively.

## Requirements

- Python 3.10+
- An OpenAI API key placed in a `.env` file at the repository root with `OPENAI_API_KEY=...`.
- Install required Python packages in `requirements.txt`. Optional extras:
	- `python-docx` to ingest DOCX files
	- `beautifulsoup4` to ingest HTML pages

## Installation

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

```text
OPENAI_API_KEY=sk-...
```

## Running the App (Streamlit UI)

Launch the Streamlit interface (recommended):

```powershell
streamlit run main.py
```

What you can do in the UI:

- Upload one or more PDF/DOCX files or paste paper URLs (one per line).
- Configure `Chunk size`, `Chunk overlap`, and `Top K` retrieval.
- Build/update the vector index (optionally reset the existing collection).
- Ask questions in a chat interface that returns answers grounded in retrieved chunks and shows citations.

## Programmatic Usage

The code exposes small, focused modules you can import in scripts.

- `src/loader.py` — `load_documents(sources)` returns a list of `langchain_core.documents.Document` objects from file paths or URLs.
- `src/splitter.py` — `split_documents(documents, chunk_size, chunk_overlap)` returns chunked `Document`s.
- `src/indexing.py` — `build_index_from_chunks(...)` and `VectorIndexManager` to create or load a persistent Chroma index. Requires `OPENAI_API_KEY`.
- `src/retriever.py` — `RAGRetriever` wrapper and `get_relevant_chunks()` convenience function to get top-k chunks.
- `src/generator.py` — `RAGGenerator` building retrieval + `ChatOpenAI` LLM pipeline. Methods: `answer_chat()`, `answer_query()`, and convenience `ask_question()`.



## Configuration

- Embedding model: default is `text-embedding-3-small` (set in `src/indexing.py`).
- LLM model: default is `gpt-4o-mini` (set in `src/generator.py`).
- Adjust `chunk_size`, `chunk_overlap`, and `retrieval_k` in the UI or by passing different args to the functions/classes.


## Files and Responsibilities

- `main.py` — Streamlit UI: upload sources, build index, chat with papers, configure chunking and retrieval.
- `src/loader.py` — PDF/DOCX/URL ingestion to LangChain `Document` objects (includes per-page PDF extraction and metadata like `source` and `page_number`).
- `src/splitter.py` — Splits documents into chunks using `RecursiveCharacterTextSplitter`.
- `src/indexing.py` — Manages embedding model, persists vectors to Chroma, and exposes retriever creation.
- `src/retriever.py` — Retrieves chunks and formats citation-friendly metadata.
- `src/generator.py` — Rewrites conversational queries, retrieves context, constructs a prompt that instructs the LLM to answer only from the provided context, and returns citations.

## Contributing

Contributions are welcome. Please open issues for feature requests or bug reports, and submit pull requests with tests when appropriate.

## License

Add a license file if you plan to publish or share the code. Currently the repository has no explicit license.
