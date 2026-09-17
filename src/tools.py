"""Vector retriever and regex-sanitized financial math tool."""

from __future__ import annotations

import re

import numexpr
from langchain_core.documents import Document
from langchain_core.tools import tool

from src.config import RETRIEVER_K
from src.ingestion import iter_stored_chunks, query_similar_documents


def sanitize_financial_expression(expression: str) -> str:
    """
    Pre-parse financial expressions before numexpr evaluation:
    - Remove $, commas, and extra whitespace
    - Convert percentages (15% -> 0.15)
    - Convert magnitude suffixes K/M/B
    """
    expr = expression.strip()
    expr = expr.replace("$", "").replace(",", "")
    expr = re.sub(r"\s+", " ", expr).strip()

    expr = re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        lambda m: str(float(m.group(1)) / 100.0),
        expr,
    )

    def _magnitude(match: re.Match[str]) -> str:
        number = match.group(1)
        suffix = match.group(2).upper()
        scale = {"K": "1e3", "M": "1e6", "B": "1e9"}[suffix]
        return f"({number} * {scale})"

    expr = re.sub(
        r"(\d+(?:\.\d+)?)\s*([KMBkmb])\b",
        _magnitude,
        expr,
    )

    expr = re.sub(r"\s+", "", expr)
    return expr


@tool
def financial_calculator(expression: str) -> str:
    """
    Evaluate a sanitized arithmetic expression for financial math.
    Use for ratios, growth rates, percentages and any numeric calculation.
    Supports $, commas, % suffixes and K/M/B magnitude suffixes.
    """
    try:
        sanitized = sanitize_financial_expression(expression)
        if not sanitized:
            return "Error: empty expression after sanitization."

        result = numexpr.evaluate(sanitized)
        try:
            value = result.item()
        except AttributeError:
            value = float(result)
        return str(value)
    except Exception as exc:
        return f"Error evaluating expression '{expression}': {exc}"


_QUESTION_PREFIX = re.compile(
    r"^(what|who|where|when|how|why|which|whom|do you call|is there|are there|"
    r"can you|could you|tell me|explain|define|i just saw|from the docs)"
    r"[\w\s,'\"-]*?\b",
    re.IGNORECASE,
)


def _strip_question_framing(query: str) -> str:
    """Remove leading question boilerplate so document phrases can match."""
    q = (query or "").strip().strip('"').strip("'").strip()
    q = re.sub(r"\?+$", "", q).strip()
    # Peel a couple of question-prefix layers max
    for _ in range(2):
        nxt = _QUESTION_PREFIX.sub("", q).strip(" ,:-")
        if nxt == q:
            break
        q = nxt
    return q


def _search_query_variants(query: str) -> list[str]:
    """Build query variants to reduce embedding flakiness."""
    variants: list[str] = []
    base = (query or "").strip()
    if base:
        variants.append(base)

    stripped = base.strip().strip('"').strip("'").strip()
    if stripped and stripped not in variants:
        variants.append(stripped)

    no_punct = re.sub(r"[?\"']+$", "", stripped).strip()
    if no_punct and no_punct not in variants:
        variants.append(no_punct)

    content = _strip_question_framing(stripped)
    if content and content not in variants:
        variants.append(content)

    # Content phrases as additional embedding queries
    for phrase in _content_phrases(stripped)[:3]:
        if phrase not in variants:
            variants.append(phrase)

    for acr in re.findall(r"\b[A-Z]{2,10}\b", stripped):
        if acr not in variants:
            variants.append(acr)

    for token in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9._-]{2,}\.pdf\b", stripped, flags=re.I):
        stem = token[:-4]
        if stem not in variants:
            variants.append(stem)
    for token in re.findall(r"\bUNH[-_]?Q?\d*\b", stripped, flags=re.I):
        if token not in variants:
            variants.append(token)

    keywords = []
    low = stripped.lower()
    if "net income" in low:
        keywords.append("net income")
    if "growth" in low or "percentage" in low:
        keywords.append("percentage growth")
    if "revenue" in low or "net revenue" in low:
        keywords.append("net revenues")
    if "erisa" in low or "employee retirement" in low or (
        "1974" in low and "employer-sponsored" in low
    ):
        keywords.extend(
            ["ERISA", "Employee Retirement Income Security Act of 1974"]
        )
    if keywords:
        kw_query = " ".join(dict.fromkeys(keywords))
        if kw_query not in variants:
            variants.append(kw_query)

    return variants[:8]


def _content_phrases(query: str) -> list[str]:
    """
    Extract document-like phrases from a natural-language question.
    Turns:
      'what Act of 1974 regulates how our services are provided ... plans?'
    into phrases that exist in the PDF body.
    """
    content = _strip_question_framing(query)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", content)
    phrases: list[str] = []
    if len(content) >= 12:
        phrases.append(content.lower())

    for n in (10, 8, 6, 5):
        if len(words) < n:
            continue
        for i in range(0, len(words) - n + 1):
            phrase = " ".join(words[i : i + n]).lower()
            if len(phrase) >= 18:
                phrases.append(phrase)

    # Year + "act" pattern often appears in 10-K legal sections
    for m in re.finditer(r"\bact of \d{4}\b", content, flags=re.I):
        phrases.append(m.group(0).lower())

    out: list[str] = []
    for p in phrases:
        p = p.strip()
        if p and p not in out:
            out.append(p)
    return out[:12]


def _significant_needles(query: str) -> list[str]:
    """Phrases/terms to use for substring fallback search."""
    needles: list[str] = []
    needles.extend(_content_phrases(query))
    stripped = (query or "").strip().strip('"').strip("'")
    for acr in re.findall(r"\b[A-Z]{2,10}\b", stripped):
        needles.append(acr.lower())
    for acr in re.findall(r"\b[a-z]{2,10}\b", stripped.lower()):
        if acr in {"erisa", "hipaa", "glba", "dol"}:
            needles.append(acr)
    out: list[str] = []
    for n in needles:
        n = n.strip()
        if n and n not in out:
            out.append(n)
    return out[:12]


def _substring_fallback(query: str, limit: int = 6) -> list[Document]:
    """Scan indexed chunks for phrase matches (cheap at PoC scale)."""
    needles = _significant_needles(query)
    if not needles:
        return []

    scored: list[tuple[int, Document]] = []
    seen: set[str] = set()
    for doc in iter_stored_chunks():
        text = doc.page_content or ""
        if not text:
            continue
        hay = text.lower()
        score = sum(1 for n in needles if len(n) >= 12 and n in hay)
        if score <= 0:
            continue
        meta = doc.metadata or {}
        key = f"{meta.get('source')}|{meta.get('page')}|{text[:120]}"
        if key in seen:
            continue
        seen.add(key)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        text = doc.page_content or ""
        if not text.strip():
            continue
        source = (doc.metadata or {}).get("source", "unknown")
        page = (doc.metadata or {}).get("page", "?")
        parts.append(
            f"--- Chunk {i} (source={source}, page={page}) ---\n{text}"
        )
    return "\n\n".join(parts)


@tool
def search_financial_docs(query: str) -> str:
    """
    Search indexed financial PDF chunks for metrics, tables, definitions,
    regulations, and narrative facts. Returns grounded excerpts tagged with
    DOCUMENT/PAGE headers for citations.
    """
    try:
        seen: set[str] = set()
        docs: list[Document] = []

        # Phrase scan first (no Gemini embed). Quoted 10-K sentences match instantly.
        for doc in _substring_fallback(query, limit=6):
            text = doc.page_content or ""
            if not text:
                continue
            key = (
                f"{(doc.metadata or {}).get('source')}|"
                f"{(doc.metadata or {}).get('page')}|{text[:160]}"
            )
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)

        if len(docs) < RETRIEVER_K:
            for variant in _search_query_variants(query)[:2]:
                for doc in query_similar_documents(variant, RETRIEVER_K):
                    text = doc.page_content or ""
                    if not text:
                        continue
                    key = (
                        f"{(doc.metadata or {}).get('source')}|"
                        f"{(doc.metadata or {}).get('page')}|{text[:160]}"
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    docs.append(doc)
                if len(docs) >= RETRIEVER_K * 2:
                    break

        docs = docs[: max(RETRIEVER_K, 8)]
        if not docs:
            return (
                "No matching document chunks found. "
                "The requested information may not be available in the provided documents. "
                "Try again refining query wordings for better scraping-analysis of the uploaded index document."
            )
        return _format_docs(docs)
    except Exception as exc:
        return f"Error searching financial documents: {exc}"


TOOLS = [search_financial_docs, financial_calculator]
