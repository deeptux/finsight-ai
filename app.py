"""Streamlit UI with live thought-chain trace and PDF upload."""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components
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
from src.dual_sidebar import inject_dual_sidebar_sync, sync_right_sidebar_query
from src.pdf_viewer import render_pdf_page_png
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

    /* Mobile-first scale; ~1024px desktop is the reference comfort zone */
    :root {
        --finsight-sidebar-width: min(100vw, 20rem);
        --fs-base: clamp(0.8125rem, 0.72rem + 0.45vw, 1rem);
        --fs-hero: clamp(1.35rem, 0.85rem + 2.4vw, 2.15rem);
        --fs-subtitle: clamp(0.875rem, 0.78rem + 0.5vw, 1.05rem);
        --fs-tagline: clamp(0.78rem, 0.72rem + 0.35vw, 0.92rem);
        --fs-sidebar-h: clamp(1.05rem, 0.95rem + 0.55vw, 1.25rem);
        --fs-sidebar-body: clamp(0.8125rem, 0.74rem + 0.4vw, 0.9375rem);
        --fs-caption: clamp(0.72rem, 0.66rem + 0.3vw, 0.8125rem);
        --pad-main-x: clamp(0.5rem, 0.35rem + 1.2vw, 1.25rem);
    }

    @media (min-width: 1024px) {
        :root {
            --finsight-sidebar-width: 21rem;
        }
    }

    @media (min-width: 1920px) {
        :root {
            --finsight-sidebar-width: 587px;
            --sidebar-width: 587px;
        }
    }

    @media (min-width: 1024px) and (max-width: 1919px) {
        :root {
            --sidebar-width: 21rem;
        }
    }

    .stApp {
        background: linear-gradient(165deg, #0b1220 0%, #121a2b 45%, #0e1624 100%);
        color: #e8eef7;
        font-family: "IBM Plex Sans", "DM Sans", sans-serif;
        font-size: var(--fs-base);
    }

    /* Left Documents sidebar — fixed width when open only (not when collapsed) */
    [data-testid="stSidebar"][aria-expanded="true"],
    section[data-testid="stSidebar"][aria-expanded="true"] {
        background: #0a101a;
        border-right: 1px solid #243047;
        width: var(--finsight-sidebar-width) !important;
        min-width: var(--finsight-sidebar-width) !important;
        max-width: var(--finsight-sidebar-width) !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"],
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 100% !important;
        max-width: 100% !important;
    }
    [data-testid="stSidebar"] * {
        color: #d7e0ee;
    }
    [data-testid="stSidebarResizeHandle"],
    [data-testid="collapsedControl"] + div [data-testid="stSidebarResizeHandle"] {
        display: none !important;
        pointer-events: none !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: var(--fs-sidebar-h) !important;
        line-height: 1.25 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        font-size: var(--fs-sidebar-body);
        line-height: 1.45;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        font-size: var(--fs-caption) !important;
    }
    [data-testid="stSidebar"] code {
        font-size: var(--fs-caption) !important;
        word-break: break-all;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
    [data-testid="stSidebar"] .stButton > button {
        font-size: var(--fs-sidebar-body) !important;
        padding: 0.35rem 0.5rem !important;
    }

    /* Compact sidebar chrome so Documents content sits higher */
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        min-height: 0 !important;
        height: auto !important;
        max-height: 2.25rem !important;
        padding: 0.15rem 0.35rem 0.1rem 0.35rem !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] > div {
        min-height: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.35rem !important;
        padding-bottom: 0.75rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] [data-testid="stHeader"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    @media (min-width: 1920px) {
        [data-testid="stSidebar"][aria-expanded="true"],
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 587px !important;
            min-width: 587px !important;
            max-width: 587px !important;
        }
    }

    /* Narrow phones: stack filename / Remove rows in Documents panel */
    @media (max-width: 480px) {
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.25rem !important;
        }
        [data-testid="stSidebar"] [data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 0 !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button {
            width: 100%;
        }
    }

    /* Streamlit main shell: beat emotion-cache padding-top: 6rem (use longhand, not shorthand) */
    html body .stApp section[data-testid="stMain"] div[data-testid="stMainBlockContainer"],
    html body .stApp section.main div[data-testid="stMainBlockContainer"],
    html body .stApp [data-testid="stMainBlockContainer"] {
        padding-top: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
    }

    section.main .block-container {
        padding-top: 0 !important;
        padding-left: var(--pad-main-x);
        padding-right: var(--pad-main-x);
        max-width: min(920px, 100%);
        margin-left: auto;
        margin-right: auto;
        transition: max-width 0.35s ease, margin 0.35s ease;
    }
    section.main .block-container > div:first-child {
        padding-top: 0 !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        height: auto !important;
        min-height: 0 !important;
        max-height: 2.5rem !important;
    }
    [data-testid="stToolbar"] {
        max-height: 2.5rem !important;
    }
    [data-testid="stAppViewContainer"] > section.main {
        padding-top: 0 !important;
    }

    /* Sidebar open: chat column centered in the remaining main area */
    section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main .block-container {
        max-width: min(920px, calc(100vw - var(--finsight-sidebar-width) - 2.5rem));
    }

    /* Sidebar collapsed: chat + hero visually centered on full viewport */
    section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main .block-container {
        max-width: min(920px, 96vw);
        margin-left: auto !important;
        margin-right: auto !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main .finsight-hero {
        text-align: center;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main [data-testid="stChatInput"] {
        max-width: min(920px, 96vw);
        margin-left: auto !important;
        margin-right: auto !important;
    }

    @media (min-width: 1920px) {
        section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main .block-container {
            max-width: min(920px, calc(100vw - 587px - 3rem));
        }
    }

    .finsight-hero {
        margin: 0 0 clamp(0.5rem, 0.35rem + 0.6vw, 0.85rem) 0;
        padding-top: 0 !important;
    }
    .finsight-hero h1 {
        font-family: "DM Sans", "IBM Plex Sans", sans-serif !important;
        font-weight: 700 !important;
        font-size: var(--fs-hero) !important;
        letter-spacing: -0.02em;
        color: #f3f7ff !important;
        margin: 0 0 0.2rem 0 !important;
        padding: 0 !important;
        border: none !important;
        line-height: 1.15 !important;
    }
    .finsight-hero .subtitle {
        color: #9aabc4;
        font-size: var(--fs-subtitle);
        font-weight: 500;
        margin: 0 0 0.35rem 0;
        line-height: 1.35;
    }
    .finsight-hero .tagline {
        color: #7f91ab;
        font-size: var(--fs-tagline);
        margin: 0;
        line-height: 1.35;
    }

    /* Shared chat bubble shell */
    div[data-testid="stChatMessage"] {
        background: rgba(16, 25, 39, 0.88);
        border: 1px solid #2a3a55;
        border-radius: clamp(10px, 2vw, 14px);
        padding: clamp(0.45rem, 0.35rem + 0.5vw, 0.55rem) clamp(0.55rem, 0.4rem + 0.6vw, 0.75rem);
        margin-bottom: 0.7rem;
        max-width: min(820px, 92vw);
        width: fit-content;
    }
    div[data-testid="stChatMessage"] p {
        color: #d7e2f2;
        line-height: 1.55;
        font-size: var(--fs-base);
    }

    div[data-testid="stChatMessage"] {
        display: flex !important;
        align-items: flex-start;
        gap: clamp(0.4rem, 0.3rem + 0.4vw, 0.65rem);
    }

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has(img[src*="e67e22"]) {
        margin-right: auto !important;
        margin-left: 0 !important;
        justify-content: flex-start;
        flex-direction: row !important;
    }

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
        font-size: var(--fs-base);
    }
    body.finsight-right-sidebar-open section.main .block-container {
        max-width: 100% !important;
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-left: var(--pad-main-x) !important;
        padding-right: var(--pad-main-x) !important;
    }
    /* Chat column grows; PDF column keeps fixed width */
    body.finsight-right-sidebar-open [data-testid="stColumn"]:has(#finsight-chat-main-column),
    body.finsight-right-sidebar-open [data-testid="column"]:has(#finsight-chat-main-column) {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        max-width: none !important;
        width: auto !important;
    }
    body.finsight-right-sidebar-open [data-testid="stHorizontalBlock"]:has(#finsight-right-sidebar-marker) {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Pin chat input to viewport bottom (outside narrow column flow) */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        font-size: var(--fs-base) !important;
        z-index: 999900;
    }
    section.main [data-testid="stVerticalBlock"]:has(> div [data-testid="stChatInput"]),
    section.main [data-testid="stElementContainer"]:has([data-testid="stChatInput"]),
    section.main [data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        z-index: 999900 !important;
        padding: 0.65rem 1rem 0.85rem 1rem !important;
        margin: 0 !important;
        background: linear-gradient(180deg, rgba(11, 18, 32, 0) 0%, rgba(11, 18, 32, 0.92) 35%, #0b1220 100%) !important;
        pointer-events: none;
    }
    section.main [data-testid="stVerticalBlock"]:has(> div [data-testid="stChatInput"]) [data-testid="stChatInput"],
    section.main [data-testid="stElementContainer"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"],
    section.main [data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] {
        pointer-events: auto;
        max-width: min(920px, 96vw) !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main [data-testid="stVerticalBlock"]:has([data-testid="stChatInput"]),
    section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main [data-testid="stElementContainer"]:has([data-testid="stChatInput"]),
    section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main [data-testid="stBottomBlockContainer"] {
        left: var(--finsight-sidebar-width) !important;
    }

    body.finsight-right-sidebar-open section.main [data-testid="stVerticalBlock"]:has([data-testid="stChatInput"]),
    body.finsight-right-sidebar-open section.main [data-testid="stElementContainer"]:has([data-testid="stChatInput"]),
    body.finsight-right-sidebar-open section.main [data-testid="stBottomBlockContainer"] {
        right: var(--finsight-sidebar-width) !important;
    }
    body.finsight-right-sidebar-open section.main [data-testid="stVerticalBlock"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"],
    body.finsight-right-sidebar-open section.main [data-testid="stElementContainer"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] {
        max-width: min(920px, calc(100vw - var(--finsight-sidebar-width) - 3rem)) !important;
    }

    body.finsight-right-sidebar-open section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main [data-testid="stElementContainer"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"],
    body.finsight-right-sidebar-open section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main [data-testid="stVerticalBlock"]:has([data-testid="stChatInput"]) [data-testid="stChatInput"] {
        max-width: min(920px, calc(100vw - 2 * var(--finsight-sidebar-width) - 4rem)) !important;
    }

    [data-testid="stChatInput"] textarea {
        font-size: var(--fs-base) !important;
    }

    html body .stApp [data-testid="stMainBlockContainer"] {
        padding-bottom: 5.75rem !important;
    }

    @keyframes finsight-spin {
        to { transform: rotate(360deg); }
    }
    .finsight-spin {
        display: inline-block;
        animation: finsight-spin 0.8s linear infinite;
    }

    /* Right PDF column — native st.columns split (no JS dock / off-screen transform) */
    [data-testid="stHorizontalBlock"]:has(#finsight-right-sidebar-marker) {
        align-items: flex-start !important;
        width: 100% !important;
    }
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker),
    [data-testid="column"]:has(#finsight-right-sidebar-marker) {
        flex: 0 0 var(--finsight-sidebar-width) !important;
        width: var(--finsight-sidebar-width) !important;
        min-width: var(--finsight-sidebar-width) !important;
        max-width: var(--finsight-sidebar-width) !important;
        background: #0a101a !important;
        border-left: 1px solid #243047 !important;
        padding: 0.15rem 0.65rem 1rem 0.65rem !important;
        box-sizing: border-box !important;
        position: sticky !important;
        top: 0 !important;
        align-self: flex-start !important;
        max-height: 100vh !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        box-shadow: -8px 0 24px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) *,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) * {
        color: #d7e0ee;
    }
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) h1,
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) h2,
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) h3,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) h1,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) h2,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) h3 {
        font-size: var(--fs-sidebar-h) !important;
        line-height: 1.25 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) p,
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) label,
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) .stMarkdown,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) p,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) label,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) .stMarkdown {
        font-size: var(--fs-sidebar-body);
        line-height: 1.45;
    }
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) .stButton > button,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) .stButton > button {
        font-size: var(--fs-sidebar-body) !important;
        padding: 0.35rem 0.5rem !important;
    }
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) .finsight-right-sidebar-header,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) .finsight-right-sidebar-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 2.25rem;
        max-height: 2.25rem;
        padding: 0.15rem 0.1rem 0.1rem 0.1rem;
        margin-bottom: 0.35rem;
        border-bottom: 1px solid #243047;
    }
    [data-testid="stColumn"]:has(#finsight-right-sidebar-marker) .finsight-right-sidebar-header span,
    [data-testid="column"]:has(#finsight-right-sidebar-marker) .finsight-right-sidebar-header span {
        font-size: var(--fs-sidebar-h);
        font-weight: 600;
        color: #e8eef7;
    }

    @media (min-width: 1920px) {
        [data-testid="stColumn"]:has(#finsight-right-sidebar-marker),
        [data-testid="column"]:has(#finsight-right-sidebar-marker) {
            flex: 0 0 587px !important;
            width: 587px !important;
            min-width: 587px !important;
            max-width: 587px !important;
        }
    }

    /* Right-edge open chevron when PDF panel is closed */
    [data-testid="stHorizontalBlock"]:has(#finsight-right-rail-marker) {
        position: fixed !important;
        right: 0 !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: auto !important;
        height: auto !important;
        min-height: 0 !important;
        z-index: 999992 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        overflow: visible !important;
    }
    body.finsight-right-sidebar-open [data-testid="stHorizontalBlock"]:has(#finsight-right-rail-marker) {
        display: none !important;
    }
    [data-testid="stColumn"]:has(#finsight-right-rail-marker) .stButton > button,
    [data-testid="column"]:has(#finsight-right-rail-marker) .stButton > button {
        background: #0a101a !important;
        border: 1px solid #243047 !important;
        border-right: none !important;
        border-radius: 8px 0 0 8px !important;
        color: #d7e0ee !important;
        min-width: 1.75rem !important;
        width: 1.75rem !important;
        padding: 0.65rem 0.35rem !important;
        box-shadow: -4px 0 12px rgba(0, 0, 0, 0.25);
        font-size: 1.1rem !important;
        line-height: 1 !important;
    }
    [data-testid="stColumn"]:has(#finsight-right-rail-marker) .stButton > button:hover,
    [data-testid="column"]:has(#finsight-right-rail-marker) .stButton > button:hover {
        background: #121a28 !important;
        color: #f3f7ff !important;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

MAIN_BLOCK_PADDING_FIX = """
<style id="finsight-main-padding-fix-late">
html body .stApp [data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 5.75rem !important;
    padding-left: 2rem !important;
}
</style>
"""


def _inject_main_block_padding_fix() -> None:
    """
    Streamlit emotion styles often set padding-top: 6rem after our first CSS pass.
    Re-apply via late stylesheet + inline styles on the main block container.
    """
    st.markdown(MAIN_BLOCK_PADDING_FIX, unsafe_allow_html=True)
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          function apply() {
            const el = doc.querySelector('[data-testid="stMainBlockContainer"]');
            if (!el) return;
            const rightOpen = doc.body.classList.contains("finsight-right-sidebar-open");
            el.style.setProperty("padding-top", "1rem", "important");
            el.style.setProperty("padding-bottom", "1rem", "important");
            el.style.setProperty("padding-left", "2rem", "important");
            el.style.setProperty("padding-right", "1rem", "important");
            el.style.setProperty("padding-bottom", "5.75rem", "important");
          }
          apply();
          setTimeout(apply, 100);
          setTimeout(apply, 400);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


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


def _ellipse_filename(name: str, head: int = 18, tail: int = 9) -> str:
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
        c1, c2 = st.columns([2, 1])
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
        c1, c2 = st.columns([2, 1])
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


def _init_dual_sidebar_state() -> None:
    if "right_sidebar_open" not in st.session_state:
        st.session_state.right_sidebar_open = False
    if "pdf_view_doc" not in st.session_state:
        st.session_state.pdf_view_doc = ""
    if "pdf_view_page" not in st.session_state:
        st.session_state.pdf_view_page = 1
    if "pdf_view_highlight" not in st.session_state:
        st.session_state.pdf_view_highlight = ""


def _render_right_sidebar_rail() -> None:
    """Fixed right-edge chevron (styled via CSS on row containing #finsight-right-rail-marker)."""
    _rail_col, = st.columns([1])
    with _rail_col:
        st.markdown(
            '<div id="finsight-right-rail-marker" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "‹",
            key="finsight_open_right_sidebar",
            help="Open PDF viewer",
        ):
            st.session_state.right_sidebar_open = True
            st.session_state.collapse_left_for_right = True
            st.rerun()


def _render_right_sidebar_panel(indexed: list[str]) -> None:
    """PDF viewer column content (placed in right st.columns sibling)."""
    if not indexed:
        return

    st.markdown(
        '<div id="finsight-right-sidebar-marker" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    _hdr, _close = st.columns([5, 1])
    with _hdr:
        st.markdown(
            '<div class="finsight-right-sidebar-header"><span>PDF Viewer</span></div>',
            unsafe_allow_html=True,
        )
    with _close:
        if st.button(
            "›",
            key="finsight_close_right_sidebar",
            help="Close PDF viewer",
        ):
            st.session_state.right_sidebar_open = False
            st.rerun()

    names = sorted(indexed)
    default_doc = st.session_state.pdf_view_doc
    if default_doc not in names:
        default_doc = names[0]
        st.session_state.pdf_view_doc = default_doc

    doc_idx = names.index(st.session_state.pdf_view_doc)
    picked = st.selectbox(
        "Document",
        names,
        index=doc_idx,
        key="finsight_pdf_select",
        label_visibility="collapsed",
    )
    if picked != st.session_state.pdf_view_doc:
        st.session_state.pdf_view_doc = picked
        st.session_state.pdf_view_page = 1

    page = int(st.session_state.pdf_view_page or 1)
    highlight = (st.session_state.pdf_view_highlight or "").strip() or None

    png, page_count, err = render_pdf_page_png(
        st.session_state.pdf_view_doc,
        page,
        highlight_query=highlight,
    )
    if page_count > 0 and page > page_count:
        page = page_count
        st.session_state.pdf_view_page = page
        png, page_count, err = render_pdf_page_png(
            st.session_state.pdf_view_doc,
            page,
            highlight_query=highlight,
        )

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("Prev", key="finsight_pdf_prev", disabled=page <= 1):
            st.session_state.pdf_view_page = max(1, page - 1)
            st.rerun()
    with nav2:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(page_count, 1),
            value=page,
            step=1,
            key="finsight_pdf_page_input",
            label_visibility="collapsed",
        )
        if int(new_page) != page:
            st.session_state.pdf_view_page = int(new_page)
            st.rerun()
        st.caption(f"of {page_count or '?'}" if page_count else "Page")
    with nav3:
        if st.button(
            "Next",
            key="finsight_pdf_next",
            disabled=page_count > 0 and page >= page_count,
        ):
            st.session_state.pdf_view_page = page + 1
            st.rerun()

    if err:
        st.warning(err)
    elif png:
        st.image(png, use_container_width=True)
    else:
        st.info("Select a document to preview pages.")


def main() -> None:
    ensure_directories()
    if not MANIFEST_PATH.exists():
        sync_manifest_from_chroma()

    _init_dual_sidebar_state()
    sync_right_sidebar_query()

    indexed = get_indexed_sources()
    has_indexed = bool(indexed)
    if not has_indexed:
        st.session_state.right_sidebar_open = False

    right_open = bool(st.session_state.get("right_sidebar_open")) and has_indexed
    collapse_left = bool(st.session_state.pop("collapse_left_for_right", False))
    inject_dual_sidebar_sync(right_open=right_open, collapse_left=collapse_left)

    if "upload_widget_key" not in st.session_state:
        st.session_state.upload_widget_key = 0

    if right_open and has_indexed:
        chat_col, pdf_col = st.columns([1, 1], gap="small")
    else:
        chat_col = st.container()
        pdf_col = None

    with chat_col:
        st.markdown(
            '<div id="finsight-chat-main-column" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        if has_indexed and not right_open:
            _render_right_sidebar_rail()

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

        for turn in st.session_state.chat_history:
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(turn["question"])
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                _render_answer(turn["answer"])
                with st.expander("Thought Process & Tool Calls", expanded=False):
                    st.markdown(turn["trace"])

    if pdf_col is not None:
        with pdf_col:
            _render_right_sidebar_panel(indexed)

    prompt = st.chat_input("Ask a financial question grounded in your PDFs…")
    if prompt:
        if not _ensure_api_key():
            st.stop()

        indexing_active = is_indexing()

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

        answer = _with_indexing_disclaimer(answer, is_indexing() or indexing_active)
        st.session_state.chat_history.append(
            {"question": prompt, "answer": answer, "trace": trace}
        )
        st.rerun()

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

    _inject_main_block_padding_fix()


if __name__ == "__main__":
    main()
