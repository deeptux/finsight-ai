"""PDF parsing, table extraction, metadata enrichment, and vector storage."""

from __future__ import annotations

from functools import lru_cache
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
    EMBED_BATCH_SIZE,
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


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Google free-tier embedding model (cached; avoid a new HTTP client per search)."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_gemini_api_key(),
        request_options={"timeout": 25},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Open or create the persistent Chroma vector store."""
    ensure_directories()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _as_row(value: object) -> list:
    """Unwrap Chroma query batches: [[a, b]] -> [a, b]."""
    if value is None:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return list(value[0])
    if isinstance(value, list):
        return list(value)
    return []


def query_similar_documents(query: str, k: int) -> List[Document]:
    """
    Similarity search that skips Chroma 1.x ghost neighbors.

    chromadb 1.5 can return ANN ids that are not real records (document/metadata
    are None). LangChain then raises Document.page_content validation errors.
    """
    store = get_vectorstore()
    collection = store._collection
    embedding_fn = store._embedding_function
    if embedding_fn is None:
        return []

    total = int(collection.count() or 0)
    if total < 1:
        return []

    query_embedding = embedding_fn.embed_query(query)
    n_results = min(total, max(k * 2, 8))
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    ids = _as_row((raw or {}).get("ids"))
    texts = _as_row((raw or {}).get("documents"))
    metas = _as_row((raw or {}).get("metadatas"))
    # Pad so zip doesn't drop ids when a column is shorter
    while len(texts) < len(ids):
        texts.append(None)
    while len(metas) < len(ids):
        metas.append(None)

    missing_ids = [i for i, text in zip(ids, texts) if i and not text]
    recovered: dict[str, tuple[str, dict]] = {}
    if missing_ids:
        try:
            got = collection.get(ids=missing_ids, include=["documents", "metadatas"])
            for rid, text, meta in zip(
                got.get("ids") or [],
                got.get("documents") or [],
                got.get("metadatas") or [],
            ):
                if rid and text:
                    recovered[str(rid)] = (text, meta if isinstance(meta, dict) else {})
        except Exception:
            recovered = {}

    docs: List[Document] = []
    seen: set[str] = set()
    for rid, text, meta in zip(ids, texts, metas):
        if not text and rid:
            packed = recovered.get(str(rid))
            if packed:
                text, meta = packed
        if not isinstance(text, str) or not text.strip():
            continue
        meta_dict = meta if isinstance(meta, dict) else {}
        key = f"{meta_dict.get('source')}|{meta_dict.get('page')}|{text[:160]}"
        if key in seen:
            continue
        seen.add(key)
        docs.append(Document(page_content=text, metadata=meta_dict))
        if len(docs) >= k:
            break
    return docs


_CHUNK_CACHE: List[Document] | None = None


def invalidate_store_caches() -> None:
    """Drop cached chunks after ingest/delete so search sees fresh data."""
    global _CHUNK_CACHE
    _CHUNK_CACHE = None


def iter_stored_chunks() -> List[Document]:
    """Load all stored chunks (skip empty rows) for substring fallback."""
    global _CHUNK_CACHE
    if _CHUNK_CACHE is not None:
        return _CHUNK_CACHE
    store = get_vectorstore()
    collection = store._collection
    raw = collection.get(include=["documents", "metadatas"])
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    docs: List[Document] = []
    for text, meta in zip(texts, metas):
        if not isinstance(text, str) or not text.strip():
            continue
        docs.append(Document(page_content=text, metadata=meta if isinstance(meta, dict) else {}))
    _CHUNK_CACHE = docs
    return docs


def ingest_pdf(path: str | Path, *, progress_filename: str | None = None) -> int:
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

    total = len(documents)
    if progress_filename:
        from src.index_jobs import report_index_progress

        report_index_progress(progress_filename, embedded=0, total_chunks=total)

    try:
        vectorstore = get_vectorstore()
        batch_size = max(1, EMBED_BATCH_SIZE)
        for start in range(0, total, batch_size):
            batch = documents[start : start + batch_size]
            vectorstore.add_documents(batch)
            if progress_filename:
                from src.index_jobs import report_index_progress

                report_index_progress(
                    progress_filename,
                    embedded=min(start + len(batch), total),
                    total_chunks=total,
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to embed/store documents: {exc}") from exc

    invalidate_store_caches()
    return total


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
    invalidate_store_caches()
    return len(ids)


def clear_all_indexed_chunks() -> None:
    """Delete every chunk in the active collection."""
    store = get_vectorstore()
    collection = store._collection
    existing = collection.get(include=[])
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    invalidate_store_caches()
