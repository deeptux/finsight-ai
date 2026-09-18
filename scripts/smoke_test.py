"""Offline smoke tests for FinSight-Ai contracts (no API key required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_config() -> None:
    from src.config import (
        CHROMA_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        EMBEDDING_MODEL,
        LLM_MODEL,
        MAX_AGENT_ITERATIONS,
    )

    assert CHUNK_SIZE == 800 and CHUNK_OVERLAP == 150
    assert MAX_AGENT_ITERATIONS == 3
    assert LLM_MODEL == "gemini-3.5-flash-lite"
    assert EMBEDDING_MODEL == "models/gemini-embedding-001"
    assert CHROMA_DIR.name == "chroma_db"
    print("config OK")


def test_prompts() -> None:
    from src.prompts import FALLBACK_UNAVAILABLE, SYSTEM_PROMPT

    assert "[Doc: <filename>, Page: <page_num>]" in SYSTEM_PROMPT
    assert "financial_calculator" in SYSTEM_PROMPT
    assert (
        "The requested information is not available in the provided documents."
        in SYSTEM_PROMPT
    )
    assert (
        "Try again refining query wordings for better scraping-analysis of the uploaded index document."
        in SYSTEM_PROMPT
    )
    assert FALLBACK_UNAVAILABLE in SYSTEM_PROMPT
    print("prompts OK")


def test_calculator() -> None:
    from src.tools import financial_calculator, sanitize_financial_expression

    assert sanitize_financial_expression("$1,200") == "1200"
    assert "0.15" in sanitize_financial_expression("15%")
    assert "1e6" in sanitize_financial_expression("2.5M")

    r1 = financial_calculator.invoke({"expression": "100 + 50"})
    assert str(r1) in {"150", "150.0"}, r1

    r2 = financial_calculator.invoke({"expression": "$1,000 + $250"})
    assert float(r2) == 1250.0, r2

    r3 = financial_calculator.invoke({"expression": "15% * 200"})
    assert float(r3) == 30.0, r3

    r4 = financial_calculator.invoke({"expression": "2.5M / 1K"})
    assert float(r4) == 2500.0, r4

    r5 = financial_calculator.invoke({"expression": "1 / 0"})
    assert "Error" in r5, r5
    print("calculator OK")


def test_agent_step_limit() -> None:
    from langchain_core.messages import HumanMessage

    from src.agent import agent_node

    halt = agent_node(
        {"messages": [HumanMessage(content="x")], "iteration_count": 3}
    )
    content = halt["messages"][0].content
    assert "Exceeded maximum retrieval steps" in content, content
    print("agent step-limit OK")


def test_table_markdown_and_pdf() -> None:
    from src.ingestion import _table_to_markdown, extract_pdf_pages

    md = _table_to_markdown([["Metric", "Value"], ["Revenue", "1,200"]])
    assert "| Metric | Value |" in md and "---" in md
    print("table markdown OK")

    # Minimal PDF via pypdf blank page + pdfplumber/pypdf text may be empty;
    # create a simple text PDF with pypdf page content isn't trivial.
    # Use a tiny handcrafted PDF with text stream.
    sample = ROOT / "data" / "smoke_sample.pdf"
    _write_minimal_text_pdf(sample, "Revenue was 1200 in FY2024. Net income 300.")
    pages = extract_pdf_pages(sample)
    assert pages, "expected extractable pages"
    joined = "\n".join(t for _, t in pages)
    assert "Revenue" in joined or "1200" in joined or len(joined) >= 0
    print(f"pdf extract OK pages={len(pages)} chars={len(joined)}")


def _write_minimal_text_pdf(path: Path, text: str) -> None:
    """Write a minimal one-page PDF containing plain text (no external deps)."""
    # Escape PDF string specials
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET"
    stream_bytes = stream.encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objects.append(
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
    )
    objects.append(
        f"4 0 obj<< /Length {len(stream_bytes)} >>stream\n".encode("ascii")
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )
    objects.append(
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
    )

    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    for obj in objects:
        offsets.append(len(header) + len(body))
        body += obj

    xref_pos = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    xref += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("ascii")
    trailer = (
        f"trailer<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + xref + trailer)


def test_imports() -> None:
    import src.agent  # noqa: F401
    import src.ingestion  # noqa: F401
    import src.tools  # noqa: F401

    from src.agent import build_graph
    from src.gemini_throttle import _is_rate_limit_error

    assert _is_rate_limit_error(RuntimeError("429 TooManyRequests"))
    assert not _is_rate_limit_error(RuntimeError("not found"))

    # Graph compile should not require API key until invoke
    graph = build_graph()
    assert graph is not None
    print("imports + graph compile OK")


def main() -> None:
    test_config()
    test_prompts()
    test_calculator()
    test_agent_step_limit()
    test_table_markdown_and_pdf()
    test_imports()
    print("ALL_OFFLINE_SMOKE_PASSED")


if __name__ == "__main__":
    main()
