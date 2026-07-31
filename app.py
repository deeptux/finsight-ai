"""Streamlit UI with live thought-chain trace and PDF upload."""

from __future__ import annotations

from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.agent import build_graph
from src.config import DATA_DIR, MAX_INDEXED_PDFS, ensure_directories, get_gemini_api_key
from src.index_jobs import (
    INDEXING_DISCLAIMER,
    MANIFEST_PATH,
    get_active_indexing,
    get_active_removing,
    get_indexed_sources,
    get_jobs_snapshot,
    is_clearing,
    is_indexing,
    is_removing,
    slots_remaining,
    start_clear_all_indexed_pdfs,
    start_indexing,
    start_remove_indexed_pdf,
    sync_manifest_from_chroma,
)
from src.prompts import FALLBACK_UNAVAILABLE

st.set_page_config(
    page_title="FinSight-Ai",
    page_icon="FA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal face / robot marks (red user, orange assistant) to match revised UI.
USER_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='10' fill='%23c62828'/>"
    "<circle cx='32' cy='24' r='10' fill='%23ffebee'/>"
    "<path d='M14 52c4-12 32-12 36 0' fill='%23ffebee'/>"
    "</svg>"
)
ASSISTANT_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='10' fill='%23e67e22'/>"
    "<rect x='16' y='20' width='32' height='28' rx='6' fill='%23fff3e0'/>"
    "<circle cx='26' cy='34' r='3' fill='%23e67e22'/>"
    "<circle cx='38' cy='34' r='3' fill='%23e67e22'/>"
    "<rect x='28' y='10' width='8' height='8' fill='%23fff3e0'/>"
    "<rect x='30' y='4' width='4' height='8' fill='%23ffe0b2'/>"
    "</svg>"
)

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    .stApp {
        background: linear-gradient(165deg, #0b1220 0%, #121a2b 45%, #0e1624 100%);
        color: #e8eef7;
        font-family: "IBM Plex Sans", "DM Sans", sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #0a101a;
        border-right: 1px solid #243047;
    }
    [data-testid="stSidebar"] * {
        color: #d7e0ee;
    }
    .block-container {
        padding-top: 1.25rem;
        max-width: 920px;
    }
    .finsight-hero {
        margin: 0 0 1.25rem 0;
    }
    .finsight-hero h1 {
        font-family: "DM Sans", "IBM Plex Sans", sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.15rem !important;
        letter-spacing: -0.02em;
        color: #f3f7ff !important;
        margin: 0 0 0.2rem 0 !important;
        padding: 0 !important;
        border: none !important;
    }
    .finsight-hero .subtitle {
        color: #9aabc4;
        font-size: 1.05rem;
        font-weight: 500;
        margin: 0 0 0.35rem 0;
    }
    .finsight-hero .tagline {
        color: #7f91ab;
        font-size: 0.92rem;
        margin: 0;
    }
    /* Shared chat bubble shell */
    div[data-testid="stChatMessage"] {
        background: rgba(16, 25, 39, 0.88);
        border: 1px solid #2a3a55;
        border-radius: 14px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.7rem;
        max-width: min(820px, 88%);
        width: fit-content;
    }
    div[data-testid="stChatMessage"] p {
        color: #d7e2f2;
        line-height: 1.55;
    }

    /* Ensure chat rows are flex so alignment + row-reverse work */
    div[data-testid="stChatMessage"] {
        display: flex !important;
        align-items: flex-start;
        gap: 0.65rem;
    }

    /* AI: left-aligned (avatar left of text) */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has(img[src*="e67e22"]) {
        margin-right: auto !important;
        margin-left: 0 !important;
        justify-content: flex-start;
        flex-direction: row !important;
    }

    /* User: right-aligned (avatar right of text) */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"]:has(img[src*="c62828"]) {
        margin-left: auto !important;
        margin-right: 0 !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start;
        background: rgba(36, 52, 78, 0.92);
        border-color: #3a5275;
    }

    div[data-testid="stExpander"] {
        background: #101927;
        border: 1px solid #2b3c58;
        border-radius: 10px;
        margin-top: 0.35rem;
    }
    [data-testid="stChatInput"] {
        border-radius: 12px;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    @keyframes finsight-spin {
        to { transform: rotate(360deg); }
    }
    .finsight-spin {
        display: inline-block;
        animation: finsight-spin 0.8s linear infinite;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

def _pretty_tool_name(name: str) -> str:
    return (name or "tool").replace("_", " ").strip().title()

def _ensure_api_key() -> bool:
    try:
        get_gemini_api_key()
        return True
    except ValueError as exc:
        st.error(str(exc))
        st.info("Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.")
        return False


def _format_tool_trace(messages: list[BaseMessage]) -> str:
    """Build a readable thought-process trace from tool and AI messages."""
    sections: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = call.get("name", "tool")
                args = call.get("args", {})
                name = _pretty_tool_name(name)
                sections.append(f"**Tool call:** via `{name}`\n\n```\n{args}\n```")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            preview = content if len(content) <= 4000 else content[:4000] + "\n…(truncated)"
            name = _pretty_tool_name(name)
            sections.append(f"**Tool result:** via `{name}`\n\n```\n{preview}\n```")

    if not sections:
        return "_No tool calls for this turn._"

    return "\n\n".join(sections)


def _message_text(content: Any) -> str:
    """Normalize Gemini/LangChain message content (str, list parts, or dict) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content") or ""
        return str(text).strip() if text else str(content).strip()
    return str(content).strip()


def _final_answer(messages: list[BaseMessage]) -> str:
    """
    Prefer the last non-tool-call AI text answer.
    If the model returns an empty final message after tools, fall back to
    calculator output, then a short extract from search hits.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        text = _message_text(msg.content)
        tool_calls = getattr(msg, "tool_calls", None) or []
        if text and not tool_calls:
            return text

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "financial_calculator":
            result = _message_text(msg.content)
            if result and not result.startswith("Error"):
                return result

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "search_financial_docs":
            result = _message_text(msg.content)
            if result and "No matching document chunks" not in result and "Error" not in result[:40]:
                # Last-resort grounded snippet so UI never blanks after a hit.
                preview = result if len(result) <= 1200 else result[:1200] + "…"
                return (
                    "Retrieved document context (model returned an empty final message):\n\n"
                    f"{preview}"
                )

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _message_text(msg.content)
            if text:
                return text

    return "No response generated."


def _with_indexing_disclaimer(answer: str, indexing_active: bool) -> str:
    """Append live-index disclaimer when PDF ingest is still running."""
    if not indexing_active:
        return answer
    if INDEXING_DISCLAIMER in answer:
        return answer
    return f"{answer.rstrip()}\n\n*{INDEXING_DISCLAIMER}*"


def _render_answer(answer: str) -> None:
    """Render answer; emphasize the query-refinement tip when present."""
    tip = (
        "Try again refining query wordings for better scraping-analysis "
        "of the uploaded index document."
    )
    body = answer
    if tip in body:
        body = body.replace(tip, f"**{tip}**")
    st.markdown(body)


def _get_graph():
    """
    Build graph per call (not cache_resource).
    Caching kept serving stale ToolNode/search code after hot-reloads, which
    made paraphrase questions miss ERISA while 'what is erisa?' still worked
    on a freshly imported module.
    """
    return build_graph()


def _render_remove_button(name: str) -> None:
    """Remove control; each active remove shows its own loader (parallel queue)."""
    if is_removing(name):
        st.markdown(
            '<div style="text-align:center;padding:0.35rem 0;">'
            '<span class="finsight-spin" style="font-size:1.25rem;">⟳</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("Remove", key=f"remove_{name}", use_container_width=True):
        start_remove_indexed_pdf(name)
        # Instant loader paint; Chroma delete continues in background.
        st.rerun()


def _ellipse_filename(name: str, head: int = 22, tail: int = 12) -> str:
    """
    Middle-ellipsis for long PDF names.
    e.g. Fiscal Year 2024 Annual Report on Form 10-K.FINAL_.pdf
      -> Fiscal Year 2024 Annual Rep...K.FINAL_.pdf
    """
    if not name or len(name) <= head + tail + 3:
        return name
    return f"{name[:head]}...{name[-tail:]}"


def _reset_uploader_widget() -> None:
    """Remount file_uploader so it keeps the empty Upload button look."""
    st.session_state.upload_widget_key = st.session_state.get("upload_widget_key", 0) + 1


def _queue_uploads_and_reset(uploaded_files: list) -> None:
    """
    Auto-start indexing for selected PDFs, then reset the uploader widget
    so it keeps the empty 'Upload' look for the next selection.
    """
    if not uploaded_files:
        return

    # Always reset widget after a selection event to avoid sticky files / rerun loops.
    if not _ensure_api_key():
        _reset_uploader_widget()
        st.rerun()
        return

    if slots_remaining() <= 0:
        st.error(
            f"Cap reached: only {MAX_INDEXED_PDFS} PDFs may be indexed. "
            "Remove a PDF first to free a slot."
        )
        _reset_uploader_widget()
        st.rerun()
        return

    batch = [(f.name, f.getvalue()) for f in uploaded_files]
    _started, skipped = start_indexing(batch)
    for msg in skipped:
        st.warning(msg)

    _reset_uploader_widget()
    st.rerun()


@st.fragment(run_every=2)
def _documents_panel() -> None:
    """
    Live upload + index status. Kept in one fragment so the Upload control /
    'hidden' banner stay in sync with slot counts after Clear / Remove.
    """
    from src.index_jobs import get_inflight_indexing

    active = get_active_indexing()
    inflight = get_inflight_indexing()
    removing = get_active_removing()
    jobs = get_jobs_snapshot()
    indexed = get_indexed_sources()
    slots = slots_remaining()

    st.write(f"Upload up to **{MAX_INDEXED_PDFS}** financial PDFs.")

    # Live capacity check (ready + actively indexing only; cancelling frees slots).
    if slots <= 0:
        st.warning(
            f"Upload hidden* **{MAX_INDEXED_PDFS}** PDF slot(s) in use "
            f"({len(indexed)} ready · {len(inflight)} indexing). "
            "Remove or clear to upload again."
        )
    else:
        uploaded = st.file_uploader(
            "PDF upload",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_uploader_{st.session_state.upload_widget_key}",
            label_visibility="collapsed",
        )
        if uploaded:
            _queue_uploads_and_reset(list(uploaded))

    st.markdown("**Indexed PDF(s)**")
    st.caption(
        f"{len(indexed)}/{MAX_INDEXED_PDFS} ready · "
        f"{len(inflight)} indexing · "
        f"{len(removing)} removing · "
        f"{slots} slot(s) left"
    )

    for name in active:
        job = jobs.get(name, {})
        status = job.get("status", "running")
        short = _ellipse_filename(name)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"`{short}`", help=name)
        with c2:
            st.markdown(
                f'<div style="text-align:center;padding:0.2rem 0;">'
                f'<span class="finsight-spin">⟳</span> '
                f'<span style="font-size:0.8rem;color:#9aabc4;">{status}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    for name in indexed:
        short = _ellipse_filename(name)
        c1, c2 = st.columns([3, 1])
        with c1:
            if is_removing(name):
                st.markdown(f"`{short}` · removing…", help=name)
            else:
                chunks = jobs.get(name, {}).get("chunks")
                label = f"`{short}`"
                if chunks is not None:
                    label += f" · {chunks} chunks"
                st.markdown(label, help=name)
        with c2:
            _render_remove_button(name)

    for name, job in jobs.items():
        if job.get("status") == "error" and name not in indexed and name not in active:
            st.error(f"`{_ellipse_filename(name)}` failed: {job.get('error', 'unknown error')}")

    if not indexed and not active and not removing:
        st.caption("No PDFs yet! Upload to start indexing.")

    # Show Clear only when there is something ready or actively indexing.
    # Hide when the list is only "cancelling" leftovers (or empty).
    if indexed or inflight or is_clearing():
        if is_clearing():
            st.markdown(
                '<div style="text-align:center;padding:0.55rem 0;border:1px solid #2b3c58;'
                'border-radius:8px;margin-top:0.35rem;">'
                '<span class="finsight-spin" style="font-size:1.25rem;">⟳</span> '
                '<span style="color:#9aabc4;">Clearing…</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif indexed or inflight:
            if st.button("Clear all indexed PDFs", use_container_width=True):
                start_clear_all_indexed_pdfs()
                st.rerun()


def main() -> None:
    ensure_directories()
    if not MANIFEST_PATH.exists():
        sync_manifest_from_chroma()

    if "upload_widget_key" not in st.session_state:
        st.session_state.upload_widget_key = 0

    st.markdown(
        """
        <div class="finsight-hero">
            <h1>FinSight-Ai</h1>
            <p class="subtitle">Financial PDF Agentic RAG (powered by Gemini 2.5 flash-lite)</p>
            <p class="tagline">[ grounded answers, strict citations ]</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict[str, Any]] = []

    with st.sidebar:
        st.header("Documents")
        _documents_panel()

        st.divider()
        st.markdown(
            f"**Data folder:** `{DATA_DIR}`  \n"
            "Chunks persist in `./chroma_db`."
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            _render_answer(turn["answer"])
            with st.expander("Thought Process & Tool Calls", expanded=False):
                st.markdown(turn["trace"])

    prompt = st.chat_input("Ask a financial question grounded in your PDFs…")
    if prompt:
        if not _ensure_api_key():
            st.stop()

        indexing_active = is_indexing()

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Analyzing with retrieval + calculator guardrails…"):
                try:
                    graph = _get_graph()
                    result = graph.invoke(
                        {
                            "messages": [HumanMessage(content=prompt)],
                            "iteration_count": 0,
                        },
                        config={"recursion_limit": 12},
                    )
                    messages: list[BaseMessage] = list(result.get("messages", []))
                    answer = _final_answer(messages)
                    trace = _format_tool_trace(messages)
                except Exception as exc:
                    answer = f"Agent error: {exc}"
                    trace = "_Agent failed before producing a tool trace._"

            # Re-check at render time in case indexing finished mid-request.
            answer = _with_indexing_disclaimer(answer, is_indexing() or indexing_active)
            _render_answer(answer)
            with st.expander("Thought Process & Tool Calls", expanded=False):
                st.markdown(trace)

        st.session_state.chat_history.append(
            {"question": prompt, "answer": answer, "trace": trace}
        )


if __name__ == "__main__":
    main()
