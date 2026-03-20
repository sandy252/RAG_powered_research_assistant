from pathlib import Path
from typing import List, Sequence, Union
import warnings
from langchain_core.documents import Document
from PyPDF2 import PdfReader


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
