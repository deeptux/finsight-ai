"""PDF page rendering with optional text highlighting for the viewer panel."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import DATA_DIR, PROJECT_ROOT


def resolve_pdf_path(filename: str) -> Path | None:
    """Locate an indexed PDF under data/ or _financials/."""
    name = Path(filename).name
    for base in (DATA_DIR, PROJECT_ROOT / "_financials"):
        candidate = base / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _highlight_candidates(query: str | None) -> list[str]:
    """Build search strings from citation context for page highlighting."""
    if not query:
        return []
    text = re.sub(r"\s+", " ", query).strip()
    if not text:
        return []

    candidates: list[str] = []
    # Prefer longer distinctive spans first
    if len(text) >= 24:
        candidates.append(text[:120])
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text)
    for n in (10, 8, 6, 4):
        if len(words) >= n:
            candidates.append(" ".join(words[-n:]))
            candidates.append(" ".join(words[:n]))
    # Acronyms
    for acr in re.findall(r"\b[A-Z]{2,10}\b", text):
        candidates.append(acr)

    out: list[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in out:
            out.append(c)
    return out[:8]


def render_pdf_page_png(
    filename: str,
    page_num: int,
    highlight_query: str | None = None,
    zoom: float = 1.6,
) -> tuple[bytes | None, int, str | None]:
    """
    Render a 1-based page to PNG bytes.
    Returns (png_bytes, page_count, error_message).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, 0, "PyMuPDF is not installed. Run: pip install pymupdf"

    path = resolve_pdf_path(filename)
    if path is None:
        return None, 0, f"PDF not found on disk: {filename}"

    try:
        doc = fitz.open(path)
    except Exception as exc:
        return None, 0, f"Failed to open PDF: {exc}"

    try:
        page_count = doc.page_count
        if page_count < 1:
            return None, 0, "PDF has no pages."
        idx = max(1, min(int(page_num), page_count)) - 1
        page = doc[idx]

        for term in _highlight_candidates(highlight_query):
            try:
                rects = page.search_for(term)
            except Exception:
                continue
            for rect in rects[:12]:
                annot = page.add_highlight_annot(rect)
                if annot is not None:
                    annot.set_colors(stroke=(1, 0.92, 0.2))
                    annot.update()
            if rects:
                break

        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png"), page_count, None
    except Exception as exc:
        return None, 0, f"Failed to render page: {exc}"
    finally:
        doc.close()


def pdf_page_count(filename: str) -> int:
    path = resolve_pdf_path(filename)
    if path is None:
        return 0
    try:
        import fitz

        doc = fitz.open(path)
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    except Exception:
        return 0
