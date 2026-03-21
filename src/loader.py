from pathlib import Path
from importlib import import_module
from io import BytesIO
from typing import Iterable, List, Sequence, Union
from urllib.parse import urlparse
import warnings
from langchain_core.documents import Document
from PyPDF2 import PdfReader
import requests


def load_pdfs_to_documents(pdf_source: Union[str, Sequence[str]]) -> List[Document]:
    documents: List[Document] = []

    pdf_paths: List[Path] = []
    if isinstance(pdf_source, str):
        source_path = Path(pdf_source)
        if source_path.is_dir():
            pdf_paths = sorted(source_path.glob("*.pdf"))
        else:
            pdf_paths = [source_path]
    else:
        pdf_paths = [Path(path) for path in pdf_source]

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            warnings.warn(f"Skipping missing file: {pdf_path}")
            continue

        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            warnings.warn(f"Skipping non-PDF path: {pdf_path}")
            continue

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            warnings.warn(f"Skipping unreadable PDF {pdf_path}: {exc}")
            continue

        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_number": page_index,
                        "source": pdf_path.name,
                        "file_path": str(pdf_path),
                    },
                )
            )

    return documents


def _extract_pdf_documents_from_reader(reader: PdfReader, source_label: str, file_path: str = "") -> List[Document]:
    documents: List[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        metadata = {
            "page_number": page_index,
            "source": source_label,
            "source_type": "pdf",
        }
        if file_path:
            metadata["file_path"] = file_path

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )
    return documents


def _load_pdf_file(pdf_path: Path) -> List[Document]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        warnings.warn(f"Skipping unreadable PDF {pdf_path}: {exc}")
        return []

    return _extract_pdf_documents_from_reader(
        reader=reader,
        source_label=pdf_path.name,
        file_path=str(pdf_path),
    )


def _load_docx_file(docx_path: Path) -> List[Document]:
    try:
        docx_module = import_module("docx")
    except Exception:
        warnings.warn("python-docx is not installed; DOCX files will be skipped.")
        return []

    DocxDocument = getattr(docx_module, "Document", None)
    if DocxDocument is None:
        warnings.warn("python-docx is unavailable; DOCX files will be skipped.")
        return []

    try:
        doc = DocxDocument(str(docx_path))
    except Exception as exc:
        warnings.warn(f"Skipping unreadable DOCX {docx_path}: {exc}")
        return []

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    if not paragraphs:
        return []

    content = "\n\n".join(paragraphs)
    return [
        Document(
            page_content=content,
            metadata={
                "page_number": 1,
                "source": docx_path.name,
                "file_path": str(docx_path),
                "source_type": "docx",
            },
        )
    ]


def _is_probably_pdf_url(url: str, content_type: str) -> bool:
    if urlparse(url).path.lower().endswith(".pdf"):
        return True
    return "application/pdf" in (content_type or "").lower()


def _load_url_source(url: str, timeout: int = 30) -> List[Document]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        warnings.warn(f"Skipping unsupported URL scheme: {url}")
        return []

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        warnings.warn(f"Skipping unreachable URL {url}: {exc}")
        return []

    content_type = response.headers.get("Content-Type", "")
    if _is_probably_pdf_url(url, content_type):
        try:
            reader = PdfReader(BytesIO(response.content))
        except Exception as exc:
            warnings.warn(f"Skipping unreadable PDF from URL {url}: {exc}")
            return []

        return _extract_pdf_documents_from_reader(
            reader=reader,
            source_label=url,
            file_path=url,
        )

    try:
        bs4_module = import_module("bs4")
    except Exception:
        warnings.warn("beautifulsoup4 is not installed; HTML URL ingestion will be skipped.")
        return []

    BeautifulSoup = getattr(bs4_module, "BeautifulSoup", None)
    if BeautifulSoup is None:
        warnings.warn("beautifulsoup4 is unavailable; HTML URL ingestion will be skipped.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = "\n".join(line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip())
    if not text:
        warnings.warn(f"Skipping empty web content: {url}")
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "page_number": 1,
                "source": url,
                "file_path": url,
                "source_type": "url",
            },
        )
    ]

def load_documents(sources: Union[str, Sequence[str]]) -> List[Document]:
    """Load documents from local PDF/DOCX files or URLs into LangChain Documents."""
    if isinstance(sources, str):
        source_items: Iterable[str] = [sources]
    else:
        source_items = sources

    documents: List[Document] = []
    for source in source_items:
        if isinstance(source, str) and source.lower().startswith(("http://", "https://")):
            documents.extend(_load_url_source(source))
            continue

        source_path = Path(source)
        if not source_path.exists():
            warnings.warn(f"Skipping missing file: {source_path}")
            continue

        if source_path.is_dir():
            for candidate in sorted(source_path.glob("*")):
                if candidate.suffix.lower() == ".pdf":
                    documents.extend(_load_pdf_file(candidate))
                elif candidate.suffix.lower() == ".docx":
                    documents.extend(_load_docx_file(candidate))
            continue

        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(_load_pdf_file(source_path))
        elif suffix == ".docx":
            documents.extend(_load_docx_file(source_path))
        else:
            warnings.warn(f"Skipping unsupported file type: {source_path}")

    return documents
