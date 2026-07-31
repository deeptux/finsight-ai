"""PDF parsing, table extraction, metadata enrichment, and vector storage."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    ensure_directories,
    get_gemini_api_key,
)


def _table_to_markdown(table: List[List[str | None]]) -> str:
    """Convert a pdfplumber table (list of rows) to Markdown table syntax."""
    if not table:
        return ""

    normalized: List[List[str]] = []
    for row in table:
        if row is None:
            continue
        normalized.append([(cell if cell is not None else "").strip() for cell in row])

    if not normalized:
        return ""

    col_count = max(len(r) for r in normalized)
    for row in normalized:
        while len(row) < col_count:
            row.append("")

    header = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_page_with_pdfplumber(page) -> Tuple[str, bool]:
    """
    Extract page content via pdfplumber.
    Returns (text, has_tables).
    """
    tables = page.extract_tables() or []
    meaningful_tables = [t for t in tables if t and any(any(cell for cell in row) for row in t)]
    if meaningful_tables:
        parts = [_table_to_markdown(t) for t in meaningful_tables]
        # Also keep non-table text on table pages for context.
        narrative = (page.extract_text() or "").strip()
        if narrative:
            parts.insert(0, narrative)
        return "\n\n".join(p for p in parts if p), True
    text = (page.extract_text() or "").strip()
    return text, False


def _extract_page_with_pypdf(reader: PdfReader, page_index: int) -> str:
    """Fallback narrative extraction via pypdf for a single page."""
    try:
        return (reader.pages[page_index].extract_text() or "").strip()
    except Exception:
        return ""


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extract per-page text. Use pdfplumber tables when present;
    otherwise fall back to pypdf narrative text.
    Returns list of (page_num_1_indexed, page_text).
    """
    pages: List[Tuple[int, str]] = []
    try:
        pypdf_reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF with pypdf: {exc}") from exc

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text, has_tables = _extract_page_with_pdfplumber(page)
                if not has_tables or not text:
                    fallback = _extract_page_with_pypdf(pypdf_reader, i)
                    if fallback:
                        text = fallback if not text else text
                if text:
                    pages.append((page_num, text))
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF with pdfplumber: {exc}") from exc

    return pages


def _build_documents(pdf_path: Path, pages: List[Tuple[int, str]]) -> List[Document]:
    """Chunk pages and prepend metadata context headers before embedding."""
    filename = pdf_path.name
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents: List[Document] = []
    for page_num, page_text in pages:
        page_chunks = splitter.split_text(page_text)
        for chunk in page_chunks:
            enriched = f"[DOCUMENT: {filename} | PAGE: {page_num}]\n\n{chunk}"
            documents.append(
                Document(
                    page_content=enriched,
                    metadata={
                        "source": filename,
                        "page": page_num,
                    },
                )
            )
    return documents


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Google free-tier embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_gemini_api_key(),
    )


def get_vectorstore() -> Chroma:
    """Open or create the persistent Chroma vector store."""
    ensure_directories()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def ingest_pdf(path: str | Path) -> int:
    """
    Index a financial PDF into ChromaDB.
    Returns the number of chunks indexed.
    Raises ValueError with user-facing messages on failure.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise ValueError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path.name}")

    try:
        pages = extract_pdf_pages(pdf_path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Corrupt or unreadable PDF ({pdf_path.name}): {exc}") from exc

    if not pages:
        raise ValueError(f"No extractable text or tables found in {pdf_path.name}.")

    documents = _build_documents(pdf_path, pages)
    if not documents:
        raise ValueError(f"Chunking produced no documents for {pdf_path.name}.")

    try:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to embed/store documents: {exc}") from exc

    return len(documents)


def delete_indexed_source(filename: str) -> int:
    """
    Delete all Chroma chunks for a given source filename.
    Returns the number of deleted ids (best-effort).
    """
    store = get_vectorstore()
    collection = store._collection
    existing = collection.get(where={"source": filename}, include=[])
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def clear_all_indexed_chunks() -> None:
    """Delete every chunk in the active collection."""
    store = get_vectorstore()
    collection = store._collection
    existing = collection.get(include=[])
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
