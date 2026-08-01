"""Streamlit UI with live thought-chain trace and PDF upload."""

from __future__ import annotations

import html
import io
import re
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
    recover_stale_index_jobs,
    slots_remaining,
    start_clear_all_indexed_pdfs,
    start_indexing,
    start_remove_indexed_pdf,
    sync_manifest_from_chroma,
)
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

    /*
     * Streamlit emotion class names (e.g. st-emotion-cache-tn0cau) rotate each run;
     * target stable testids/classes so the root main column flex never keeps gap: 1rem.
     */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stMainBlockContainer"] > div.stVerticalBlock[data-testid="stVerticalBlock"] {
        gap: 0 !important;
        row-gap: 0 !important;
        column-gap: 0 !important;
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

    /* Chat main — single column (PDF lives in left sidebar tabs) */
    section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main .block-container:has(#finsight-chat-main-column) {
        max-width: min(920px, calc(100vw - var(--finsight-sidebar-width) - 2.5rem));
    }

    section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main .block-container:has(#finsight-chat-main-column) {
        max-width: min(920px, 96vw);
        margin-left: auto !important;
        margin-right: auto !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main .block-container:has(#finsight-chat-main-column) .finsight-hero {
        text-align: center;
    }

    @media (min-width: 1920px) {
        section[data-testid="stSidebar"][aria-expanded="true"] ~ section.main .block-container:has(#finsight-chat-main-column) {
            max-width: min(920px, calc(100vw - 587px - 3rem));
        }
    }

    section.main .block-container:has(#finsight-chat-main-column) {
        display: flex !important;
        flex-direction: column !important;
        min-height: calc(100vh - 3.5rem) !important;
        max-height: calc(100vh - 3.5rem) !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
    }
    section.main .block-container:has(#finsight-chat-main-column)
        > [data-testid="stVerticalBlock"],
    section.main .block-container:has(#finsight-chat-main-column)
        > [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }
    [data-testid="stVerticalBlock"]:has(#finsight-chat-scroll),
    [data-testid="stVerticalBlockBorderWrapper"]:has(#finsight-chat-scroll) {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain;
        padding-right: 0.15rem;
    }
    section.main .block-container:has(#finsight-chat-main-column)
        [data-testid="stElementContainer"]:has([data-testid="stChatInput"]) {
        flex-shrink: 0 !important;
        margin-top: auto !important;
        padding: 0.5rem 0 0.25rem 0 !important;
        background: linear-gradient(180deg, rgba(11, 18, 32, 0) 0%, #0b1220 45%) !important;
        position: sticky !important;
        bottom: 0 !important;
        z-index: 5 !important;
    }
    section.main .block-container:has(#finsight-chat-main-column) [data-testid="stChatInput"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
    }

    [data-testid="stChatInput"] {
        border-radius: 12px;
        font-size: var(--fs-base) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: var(--fs-base) !important;
    }

    @keyframes finsight-spin {
        to { transform: rotate(360deg); }
    }
    .finsight-spin {
        display: inline-block;
        animation: finsight-spin 0.8s linear infinite;
    }

    .finsight-hero {
        margin: 0 0 clamp(0.5rem, 0.35rem + 0.6vw, 0.85rem) 0;
        padding-top: 0 !important;
    }
    .finsight-hero-header-row [data-testid="column"]:last-child {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: flex-start !important;
        padding-top: 0.35rem !important;
    }
    .finsight-hero-header-row [data-testid="column"]:last-child [data-testid="stButton"] {
        width: auto !important;
        margin-left: auto !important;
    }
    .finsight-hero-header-row [data-testid="column"]:last-child button {
        white-space: nowrap !important;
        font-size: var(--fs-caption) !important;
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
        box-sizing: border-box;
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
        width: min(820px, 92vw) !important;
        max-width: min(820px, 92vw) !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]):has(.finsight-answer-body-marker)
        [data-testid="stMarkdownContainer"],
    div[data-testid="stChatMessage"]:has(img[src*="e67e22"]):has(.finsight-answer-body-marker)
        [data-testid="stMarkdownContainer"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]):has(.finsight-answer-body-marker)
        [data-testid="stVerticalBlockBorderWrapper"] {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        [data-testid="stExpander"],
    div[data-testid="stChatMessage"]:has(img[src*="e67e22"]) [data-testid="stExpander"] {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"]:has(img[src*="c62828"]) {
        margin-left: auto !important;
        margin-right: 0 !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start;
        background: rgba(36, 52, 78, 0.92);
        border-color: #3a5275;
        max-width: min(820px, 92vw);
    }

    div[data-testid="stExpander"] {
        background: #101927;
        border: 1px solid #2b3c58;
        border-radius: 10px;
        margin-top: 0.35rem;
        font-size: var(--fs-base);
    }

    /* In-thread “Analyzing…” — single bubble (no nested box overlap) */
    div[data-testid="stChatMessage"]:has(.finsight-analyzing-status) {
        width: min(820px, 92vw) !important;
        max-width: min(820px, 92vw) !important;
        padding: 0.55rem 0.85rem !important;
        margin-bottom: 0.7rem !important;
        align-items: center !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-analyzing-status) [data-testid="stMarkdownContainer"],
    div[data-testid="stChatMessage"]:has(.finsight-analyzing-status) [data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-analyzing-status) .finsight-analyzing-status {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        margin: 0;
        color: #c5d4ea;
        font-size: var(--fs-base);
        line-height: 1.45;
        width: auto;
        max-width: 100%;
        box-sizing: border-box;
    }
    div[data-testid="stChatMessage"]:has(.finsight-analyzing-status) p {
        margin: 0 !important;
        padding: 0 !important;
        display: contents;
    }
    .finsight-analyzing-status .finsight-spin {
        color: #e67e22;
        font-size: 1.15rem;
        flex-shrink: 0;
        line-height: 1;
    }

    /* Citation magnifying-glass icons in assistant answers */
    .finsight-cite-icon {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: -0.15em;
        width: 1.35rem;
        height: 1.35rem;
        margin: 0 0 0 0.05rem;
        cursor: help;
        color: #9ec5f5;
        flex-shrink: 0;
        white-space: nowrap;
    }
    .finsight-cite-icon.finsight-cite-clickable {
        cursor: pointer;
    }
    .finsight-cite-icon:focus {
        outline: 1px solid #5a8fd4;
        outline-offset: 2px;
        border-radius: 4px;
    }
    .finsight-cite-lens {
        width: 1.25rem;
        height: 1.25rem;
        stroke: currentColor;
        fill: none;
        opacity: 0.95;
    }
    .finsight-cite-icon:hover .finsight-cite-lens,
    .finsight-cite-icon:focus .finsight-cite-lens {
        opacity: 1;
        color: #b8d4ff;
    }
    .finsight-cite-tip {
        display: none;
        position: absolute;
        left: 50%;
        bottom: calc(100% + 6px);
        transform: translateX(-50%);
        min-width: 10rem;
        max-width: min(22rem, 70vw);
        padding: 0.45rem 0.55rem;
        background: #0f1726;
        border: 1px solid #3a5275;
        border-radius: 8px;
        color: #dce6f5;
        font-size: var(--fs-caption);
        line-height: 1.35;
        white-space: normal;
        word-break: break-word;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        z-index: 20;
        pointer-events: none;
        text-align: left;
    }
    .finsight-cite-icon:hover .finsight-cite-tip,
    .finsight-cite-icon:focus .finsight-cite-tip {
        display: block;
    }
    .finsight-cite-icon.finsight-cite-clickable .finsight-cite-tip {
        pointer-events: auto;
        cursor: pointer;
    }
    .finsight-cite-icon.finsight-cite-clickable:hover .finsight-cite-lens,
    .finsight-cite-icon.finsight-cite-clickable:focus .finsight-cite-lens {
        color: #d4e8ff;
        stroke-width: 2.1;
    }
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker) p:not(.finsight-cite-bullet) {
        margin: 0 0 0.65rem 0;
    }
    /* Hidden answer boundary markers (not flex items) */
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stElementContainer"]:has(> .finsight-answer-body-marker),
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stElementContainer"]:has(> .finsight-answer-body-end) {
        display: none !important;
    }
    div[data-testid="stChatMessage"] .finsight-answer-body-marker,
    div[data-testid="stChatMessage"] .finsight-answer-body-end {
        display: none !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker) p.finsight-cite-bullet {
        margin: 0 0 1.5rem 0 !important;
        padding: 0 !important;
        line-height: 1.55 !important;
        color: #e8eef7;
        word-break: normal;
        overflow-wrap: break-word;
    }
    div[data-testid="stChatMessage"] .finsight-cite-bullet-mark {
        color: #8fa3bf;
        font-weight: 600;
        margin-right: 0.4rem;
        user-select: none;
    }
    /* Citation answers: no flex gap; bullet spacing on p.finsight-cite-bullet */
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stVerticalBlock"]:has(.finsight-cite-bullet-row),
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stVerticalBlockBorderWrapper"]:has(.finsight-cite-bullet-row)
        > [data-testid="stVerticalBlock"] {
        gap: 0 !important;
        row-gap: 0 !important;
        column-gap: 0 !important;
    }
    /* Streamlit pairs gap:1rem with margin-bottom:-1rem on children; gap:0 must reset both */
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stVerticalBlock"]:has(.finsight-cite-bullet-row)
        [data-testid="stElementContainer"],
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stVerticalBlockBorderWrapper"]:has(.finsight-cite-bullet-row)
        [data-testid="stElementContainer"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stElementContainer"]:has(.finsight-cite-bullet-row)
        [data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
        flex-wrap: nowrap !important;
        width: 100% !important;
        max-width: 100% !important;
        gap: 0.35rem !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stElementContainer"]:has(.finsight-cite-bullet-row)
        [data-testid="column"]:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker)
        [data-testid="stElementContainer"]:has(.finsight-cite-bullet-row)
        [data-testid="column"]:last-child {
        flex: 0 0 2rem !important;
        width: 2rem !important;
        min-width: 2rem !important;
        padding-top: 0.05rem !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker) [data-testid="stButton"] > button {
        min-height: 1.45rem !important;
        height: 1.45rem !important;
        padding: 0 0.35rem !important;
        margin: 0 !important;
        font-size: 0.95rem !important;
        line-height: 1 !important;
        color: #9ec5f5 !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stChatMessage"]:has(.finsight-answer-body-marker) [data-testid="stButton"] > button:hover {
        color: #d4e8ff !important;
        border-color: #3a5275 !important;
        background: rgba(30, 48, 78, 0.55) !important;
    }

    /* PDF viewer tab (left sidebar) */
    [data-testid="stSidebar"] .finsight-pdf-page-scroll-host {
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        overscroll-behavior: contain;
    }
    [data-testid="stSidebar"] .finsight-pdf-page-scroll-host [data-testid="stImage"] {
        width: 100% !important;
    }
    #finsight-pdf-page-preview {
        display: block !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    #finsight-pdf-nav-marker {
        display: block !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(#finsight-pdf-nav-marker)
        + [data-testid="stElementContainer"] > [data-testid="stHorizontalBlock"] {
        gap: 0.15rem !important;
        align-items: center !important;
        flex-wrap: nowrap !important;
        margin-bottom: 0.35rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(#finsight-pdf-nav-marker)
        + [data-testid="stElementContainer"] [data-testid="stNumberInput"] {
        max-width: 2.85rem !important;
        min-width: 2.85rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(#finsight-pdf-nav-marker)
        + [data-testid="stElementContainer"] [data-testid="stNumberInput"] input {
        text-align: center !important;
        padding: 0.2rem 0.15rem !important;
        font-size: var(--fs-sidebar-body) !important;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(#finsight-pdf-nav-marker)
        + [data-testid="stElementContainer"] [data-testid="stNumberInputStepDown"],
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(#finsight-pdf-nav-marker)
        + [data-testid="stElementContainer"] [data-testid="stNumberInputStepUp"] {
        display: none !important;
    }
    [data-testid="stSidebar"] .finsight-pdf-nav-of {
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        font-size: var(--fs-sidebar-body) !important;
        color: #9aa8bc !important;
    }
    [data-testid="stSidebar"] [data-testid="stTabs"] [data-testid="stVerticalBlock"] {
        padding-top: 0.25rem !important;
    }

</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

MAIN_BLOCK_PADDING_FIX = """
<style id="finsight-main-padding-fix-late">
html body .stApp [data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-right: 1.5rem !important;
    padding-bottom: 0 !important;
    padding-left: 2rem !important;
}
html body .stApp [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
html body .stApp [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] {
    gap: 0 !important;
    row-gap: 0 !important;
    column-gap: 0 !important;
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
          function fixCiteBulletSpacing() {
            doc.querySelectorAll('[data-testid="stChatMessage"]').forEach(function (msg) {
              if (!msg.querySelector('.finsight-answer-body-marker')) return;
              msg.querySelectorAll('.finsight-answer-body-marker, .finsight-answer-body-end').forEach(function (m) {
                var shell = m.closest('[data-testid="stElementContainer"]');
                if (shell) shell.style.setProperty('display', 'none', 'important');
              });
              msg.querySelectorAll('[data-testid="stVerticalBlock"]').forEach(function (vb) {
                if (!vb.querySelector('.finsight-cite-bullet-row')) return;
                vb.style.setProperty('gap', '0', 'important');
                vb.style.setProperty('row-gap', '0', 'important');
                vb.style.setProperty('column-gap', '0', 'important');
              });
              msg.querySelectorAll('[data-testid="stElementContainer"]').forEach(function (ec) {
                if (!ec.closest('[data-testid="stVerticalBlock"]') ||
                    !ec.closest('[data-testid="stVerticalBlock"]').querySelector('.finsight-cite-bullet-row')) {
                  return;
                }
                ec.style.setProperty('margin-top', '0', 'important');
                ec.style.setProperty('margin-bottom', '0', 'important');
              });
              msg.querySelectorAll('p.finsight-cite-bullet').forEach(function (p) {
                p.style.setProperty('margin-bottom', '1.5rem', 'important');
              });
            });
          }
          function apply() {
            const el = doc.querySelector('[data-testid="stMainBlockContainer"]');
            if (!el) return;
            el.style.setProperty("padding-top", "0", "important");
            el.style.setProperty("padding-bottom", "0", "important");
            el.style.setProperty("padding-left", "2rem", "important");
            el.style.setProperty("padding-right", "1.5rem", "important");
            el.querySelectorAll(':scope > [data-testid="stVerticalBlock"]').forEach(function (vb) {
              vb.style.setProperty("gap", "0", "important");
              vb.style.setProperty("row-gap", "0", "important");
              vb.style.setProperty("column-gap", "0", "important");
            });
            fixCiteBulletSpacing();
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


_CITATION_TAG_RE = re.compile(r"\[Doc:[^\]]+\]", re.IGNORECASE)
# Model often emits "[Doc: …]." or "[Doc: …]\n." — render as ".📖" not "📖" on its own line.
_CITATION_BEFORE_PERIOD_RE = re.compile(
    r"(\[Doc:[^\]]+\])\s*\.",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_citation_placement(text: str) -> str:
    """Place citation tags immediately after the sentence period (inline)."""
    normalized = _CITATION_BEFORE_PERIOD_RE.sub(r".\1", text)
    normalized = re.sub(r"\.\s+(\[Doc:)", r".\1", normalized, flags=re.IGNORECASE)
    return normalized


_CITATION_PARSE_RE = re.compile(
    r"\[Doc:\s*(?P<doc>.+?)\s*,\s*Page:\s*(?P<pages>[^\]]+)\]",
    re.IGNORECASE,
)


def _parse_citation_tag(citation: str) -> tuple[str, int] | None:
    match = _CITATION_PARSE_RE.match(citation.strip())
    if not match:
        return None
    doc = match.group("doc").strip()
    page_token = match.group("pages").strip().split(",")[0].strip()
    try:
        page = max(1, int(page_token))
    except ValueError:
        return None
    return doc, page


def _resolve_indexed_doc(doc: str, indexed: list[str]) -> str | None:
    """Match citation filename to an indexed PDF (exact or fuzzy)."""
    needle = doc.strip()
    if not needle or not indexed:
        return None
    if needle in indexed:
        return needle
    lower = needle.lower()
    for name in indexed:
        if name.lower() == lower:
            return name
    base = needle.replace("\\", "/").rsplit("/", 1)[-1]
    for name in indexed:
        if name == base or name.lower() == base.lower():
            return name
        if name.lower().endswith(base.lower()) or base.lower() in name.lower():
            return name
    return None


def _on_citation_button_click(doc: str, page: int) -> None:
    """Streamlit rerun (no browser reload) — applied in main via pending cite."""
    st.session_state.finsight_cite_pending = {"doc": doc, "page": int(page)}


def _consume_pending_citation(indexed: list[str]) -> None:
    pending = st.session_state.pop("finsight_cite_pending", None)
    if not pending:
        return
    _apply_citation_navigation(
        doc=str(pending.get("doc", "")),
        page=int(pending.get("page", 1)),
        indexed=indexed,
    )


def _apply_citation_navigation(*, doc: str, page: int, indexed: list[str]) -> None:
    """Jump PDF viewer to doc/page; reset widget keys so Streamlit picks up session state."""
    resolved = _resolve_indexed_doc(doc, indexed) or doc.strip()
    names = sorted(indexed)
    if names and resolved not in names:
        fuzzy = _resolve_indexed_doc(doc, indexed)
        resolved = fuzzy if fuzzy else names[0]
    st.session_state.pdf_view_doc = resolved
    st.session_state.pdf_view_page = max(1, page)
    st.session_state.pdf_view_highlight = ""
    st.session_state.finsight_focus_pdf_tab = True
    st.session_state["finsight_pdf_select"] = resolved
    st.session_state["finsight_pdf_page_input"] = max(1, int(page))


def _render_cite_button(*, part: str, msg_key: str, para_idx: int, seg_idx: int) -> None:
    parsed = _parse_citation_tag(part)
    tip = f"{part}\n\nClick to open in PDF viewer."
    if not parsed:
        st.markdown(part)
        return
    doc, page = parsed
    st.button(
        "🔍",
        key=f"finsight_cite_{msg_key}_{para_idx}_{seg_idx}",
        help=tip,
        type="tertiary",
        on_click=_on_citation_button_click,
        args=(doc, page),
    )


def _render_cite_bullet_row(
    text: str,
    cite_part: str,
    *,
    msg_key: str,
    para_idx: int,
    seg_idx: int,
    first_in_answer: bool,
    last_in_para: bool,
) -> None:
    """One bullet line (full width) with PDF cite control aligned on the right."""
    p_classes = ["finsight-cite-bullet", "finsight-cite-bullet-row"]
    if first_in_answer:
        p_classes.append("finsight-cite-bullet-row-first")
    if last_in_para:
        p_classes.append("finsight-cite-bullet-row-last")
    class_attr = html.escape(" ".join(p_classes), quote=True)

    text_col, btn_col = st.columns([1, 0.06], gap="small", vertical_alignment="top")
    with text_col:
        body = _markdown_inline_html(text.strip()) if text.strip() else "—"
        st.markdown(
            f'<p class="{class_attr}">'
            f'<span class="finsight-cite-bullet-mark">•</span>{body}</p>',
            unsafe_allow_html=True,
        )
    with btn_col:
        _render_cite_button(
            part=cite_part,
            msg_key=msg_key,
            para_idx=para_idx,
            seg_idx=seg_idx,
        )


def _markdown_inline_html(text: str) -> str:
    """Inline markdown (** / *) for citation row text chunks."""
    if not text:
        return ""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped, flags=re.DOTALL)
    escaped = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        r"<em>\1</em>",
        escaped,
        flags=re.DOTALL,
    )
    return escaped.replace("\n", " ")


def _render_answer_paragraph(
    para: str,
    *,
    msg_key: str,
    para_idx: int,
    first_cite_in_answer: bool,
) -> bool:
    """Returns True after the first citation row has been rendered."""
    parts = re.split(r"(\[Doc:[^\]]+\])", para, flags=re.IGNORECASE)
    has_cite = any(_CITATION_TAG_RE.fullmatch(p or "") for p in parts)
    if not has_cite:
        st.markdown(_markdown_light_to_html(para), unsafe_allow_html=True)
        return first_cite_in_answer

    seg_idx = 0
    pending_text = ""
    cite_count = sum(1 for p in parts if p and _CITATION_TAG_RE.fullmatch(p))
    cite_seen = 0
    for part in parts:
        if not part:
            continue
        if _CITATION_TAG_RE.fullmatch(part):
            cite_seen += 1
            _render_cite_bullet_row(
                pending_text,
                part,
                msg_key=msg_key,
                para_idx=para_idx,
                seg_idx=seg_idx,
                first_in_answer=first_cite_in_answer,
                last_in_para=cite_seen == cite_count,
            )
            first_cite_in_answer = False
            pending_text = ""
            seg_idx += 1
        else:
            pending_text += part
    if pending_text.strip():
        st.markdown(_markdown_light_to_html(pending_text), unsafe_allow_html=True)
    return first_cite_in_answer


def _inject_sidebar_pdf_tab_focus() -> None:
    """After citation navigation: expand sidebar and select PDF viewer tab."""
    components.html(
        """
        <script>
        (function () {
          const topWin = window.top;
          const doc = topWin.document;

          function ensureSidebarOpen() {
            try {
              const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
              if (!sidebar || sidebar.getAttribute("aria-expanded") === "true") return;
              const openBtn =
                doc.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                doc.querySelector('[data-testid="collapsedControl"]');
              if (openBtn) openBtn.click();
            } catch (e) {}
          }

          function clickPdfViewerTab() {
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return false;
            const tabs = sidebar.querySelectorAll(
              '[data-testid="stTabs"] button, button[data-baseweb="tab"], [role="tab"]'
            );
            for (const tab of tabs) {
              const label = (tab.innerText || tab.textContent || "").trim();
              if (label === "PDF viewer") {
                tab.click();
                return true;
              }
            }
            return false;
          }

          function focusPdfTab() {
            ensureSidebarOpen();
            clickPdfViewerTab();
          }

          focusPdfTab();
          setTimeout(focusPdfTab, 120);
          setTimeout(focusPdfTab, 450);
          setTimeout(focusPdfTab, 900);
          setTimeout(focusPdfTab, 1600);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _markdown_light_to_html(text: str) -> str:
    """Minimal inline markdown (** / *) on HTML-escaped text."""
    if not text:
        return ""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped, flags=re.DOTALL)
    escaped = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        r"<em>\1</em>",
        escaped,
        flags=re.DOTALL,
    )
    parts = escaped.split("\n\n")
    blocks: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        blocks.append(f"<p>{part.replace(chr(10), '<br>')}</p>")
    return "".join(blocks) if blocks else ""


def _render_analyzing_status() -> None:
    """Styled in-bubble status while the agent runs (not plain st.spinner text)."""
    st.markdown(
        """
        <div class="finsight-analyzing-status" role="status" aria-live="polite">
            <span class="finsight-spin" aria-hidden="true">⟳</span>
            <span>Analyzing with retrieval + calculator guardrails…</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_agent_turn(question: str) -> tuple[str, str]:
    """Invoke LangGraph for one user question; return (answer, trace)."""
    try:
        graph = _get_graph()
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
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
    answer = _with_indexing_disclaimer(answer, is_indexing())
    return answer, trace


def _render_answer(answer: str, *, msg_key: str) -> None:
    """Render answer with inline citation buttons (Streamlit rerun — no browser reload)."""
    tip = (
        "Try again refining query wordings for better scraping-analysis "
        "of the uploaded index document."
    )
    body = answer or ""
    if tip in body:
        body = body.replace(tip, f"**{tip}**")
    body = _normalize_citation_placement(body)
    st.markdown(
        '<div class="finsight-answer-body-marker" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    first_cite = True
    for para_idx, para in enumerate(body.split("\n\n")):
        if para.strip():
            first_cite = _render_answer_paragraph(
                para,
                msg_key=msg_key,
                para_idx=para_idx,
                first_cite_in_answer=first_cite,
            )
    st.markdown(
        '<div class="finsight-answer-body-end" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )


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
            f"There are **{MAX_INDEXED_PDFS}** PDF slot(s) in use "
            f"({len(indexed)} ready · {len(inflight)} indexing). "
            "Remove/Clear to upload again."
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
            prog = ""
            emb = job.get("embedded")
            tot = job.get("total_chunks")
            if emb is not None and tot:
                prog = f" · {emb}/{tot} chunks"
            st.markdown(
                f'<div style="text-align:center;padding:0.2rem 0;">'
                f'<span class="finsight-spin">⟳</span> '
                f'<span style="font-size:0.8rem;color:#9aabc4;">{status}{prog}</span>'
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


def _init_pdf_view_state() -> None:
    if "pdf_view_doc" not in st.session_state:
        st.session_state.pdf_view_doc = ""
    if "pdf_view_page" not in st.session_state:
        st.session_state.pdf_view_page = 1
    if "pdf_view_highlight" not in st.session_state:
        st.session_state.pdf_view_highlight = ""


def _render_pdf_page_preview(*, png: bytes | None, err: str | None) -> None:
    """
    PDF page via st.image (Streamlit expand/fullscreen) inside a bounded scroll host.
    A zero-height parent script sets max-height on the image element container.
    """
    if err:
        st.warning(err)
        return
    if not png:
        st.info("Select a document to preview pages.")
        return

    st.markdown(
        '<div id="finsight-pdf-page-preview" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    st.image(io.BytesIO(png), use_container_width=True)
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;

          function pdfSidebarRoot() {
            const marker = doc.getElementById("finsight-pdf-sidebar-panel");
            if (!marker) return null;
            return marker.closest('[data-testid="stSidebar"]');
          }

          function scrollHost() {
            const root = pdfSidebarRoot();
            if (!root) return null;
            const img = root.querySelector('[data-testid="stImage"]');
            if (!img) return null;
            return img.closest('[data-testid="stElementContainer"]');
          }

          function applyScrollBounds() {
            const root = pdfSidebarRoot();
            const host = scrollHost();
            if (!root || !host) return;
            host.classList.add("finsight-pdf-page-scroll-host");
            const rootRect = root.getBoundingClientRect();
            const hostRect = host.getBoundingClientRect();
            const maxH = Math.max(120, Math.floor(rootRect.bottom - hostRect.top - 8));
            host.style.setProperty("max-height", maxH + "px", "important");
            host.style.setProperty("overflow-y", "auto", "important");
            host.style.setProperty("overflow-x", "hidden", "important");
            host.style.setProperty("min-height", "0", "important");
          }

          applyScrollBounds();
          setTimeout(applyScrollBounds, 80);
          setTimeout(applyScrollBounds, 400);
          if (window.parent && !window.parent.__finsightPdfScrollBound) {
            window.parent.__finsightPdfScrollBound = true;
            window.parent.addEventListener("resize", applyScrollBounds);
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _render_pdf_page_nav(*, page: int, page_count: int) -> None:
    """Single-row controls: Prev | page | − | + | of N | Next."""
    st.markdown(
        '<div id="finsight-pdf-nav-marker" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    max_page = max(page_count, 1)
    page = min(max(1, page), max_page)
    st.session_state.pdf_view_page = page
    st.session_state["finsight_pdf_page_input"] = page

    nav_prev, nav_page, nav_minus, nav_plus, nav_total, nav_next = st.columns(
        [0.95, 0.55, 0.35, 0.35, 0.75, 0.95],
        gap="small",
        vertical_alignment="center",
    )
    with nav_prev:
        if st.button(
            "Prev",
            key="finsight_pdf_prev",
            disabled=page <= 1,
            type="secondary",
        ):
            st.session_state.pdf_view_page = max(1, page - 1)
            st.rerun()
    with nav_page:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            step=1,
            key="finsight_pdf_page_input",
            label_visibility="collapsed",
        )
        if int(new_page) != st.session_state.pdf_view_page:
            st.session_state.pdf_view_page = int(new_page)
            st.rerun()
    with nav_minus:
        if st.button(
            "−",
            key="finsight_pdf_minus",
            disabled=page <= 1,
            help="Previous page",
            type="secondary",
        ):
            st.session_state.pdf_view_page = max(1, page - 1)
            st.rerun()
    with nav_plus:
        at_end = page_count > 0 and page >= page_count
        if st.button(
            "+",
            key="finsight_pdf_plus",
            disabled=at_end,
            help="Next page",
            type="secondary",
        ):
            st.session_state.pdf_view_page = min(page_count or page + 1, page + 1)
            st.rerun()
    with nav_total:
        total_label = str(page_count) if page_count else "?"
        st.markdown(
            f'<p class="finsight-pdf-nav-of">of {total_label}</p>',
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button(
            "Next",
            key="finsight_pdf_next",
            disabled=page_count > 0 and page >= page_count,
            type="secondary",
        ):
            st.session_state.pdf_view_page = page + 1
            st.rerun()


def _render_pdf_viewer_panel(indexed: list[str]) -> None:
    """PDF viewer (Documents sidebar → PDF viewer tab)."""
    st.markdown(
        '<div id="finsight-pdf-sidebar-panel" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    if not indexed:
        st.info("Upload and index PDFs in the Documents tab.")
        return

    names = sorted(indexed)
    if st.session_state.pdf_view_doc not in names:
        st.session_state.pdf_view_doc = names[0]
    if st.session_state.get("finsight_pdf_select") not in names:
        st.session_state["finsight_pdf_select"] = st.session_state.pdf_view_doc

    picked = st.selectbox(
        "Document",
        names,
        key="finsight_pdf_select",
        label_visibility="collapsed",
    )
    if picked != st.session_state.pdf_view_doc:
        st.session_state.pdf_view_doc = picked
        st.session_state.pdf_view_page = 1
        st.session_state["finsight_pdf_page_input"] = 1
    else:
        st.session_state.pdf_view_doc = picked

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

    _render_pdf_page_nav(page=page, page_count=page_count or 0)

    _render_pdf_page_preview(png=png, err=err)


def _render_data_info_panel() -> None:
    """Sidebar tab: persisted data paths."""
    st.markdown(f"**Data folder:** `{DATA_DIR}`")
    st.markdown("Chunks persist in `./chroma_db`.")


def _render_sidebar(indexed: list[str]) -> None:
    """Left sidebar: Documents, PDF viewer, and data info tabs."""
    tab_docs, tab_pdf, tab_data = st.tabs(["Documents", "PDF viewer", "Data"])
    with tab_docs:
        _documents_panel()
    with tab_pdf:
        _render_pdf_viewer_panel(indexed)
    with tab_data:
        _render_data_info_panel()

    if st.session_state.pop("finsight_focus_pdf_tab", False):
        _inject_sidebar_pdf_tab_focus()


def _clear_chat_history() -> None:
    st.session_state.chat_history = []
    st.session_state.agent_busy = False
    st.session_state.pop("agent_display_pass", None)


def _render_hero_header() -> None:
    """Title block with Clear chat on the right."""
    st.markdown('<div class="finsight-hero-header-row">', unsafe_allow_html=True)
    hero_text, hero_actions = st.columns([1, 0.22], vertical_alignment="top")
    with hero_text:
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
    with hero_actions:
        if st.button("Clear chat", key="finsight_clear_chat", type="secondary"):
            _clear_chat_history()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_main() -> None:
    """Center main area: hero, chat thread, input (no nested PDF column)."""
    st.markdown(
        '<div id="finsight-chat-main-column" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict[str, Any]] = []
    if "agent_busy" not in st.session_state:
        st.session_state.agent_busy = False

    scroll = st.container()
    with scroll:
        st.markdown(
            '<div id="finsight-chat-scroll" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        _render_hero_header()

        for turn_idx, turn in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(turn["question"])
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                if turn.get("answer") is None:
                    _render_analyzing_status()
                else:
                    _render_answer(turn["answer"], msg_key=f"t{turn_idx}")
                    with st.expander("Thought Process & Tool Calls", expanded=False):
                        st.markdown(turn["trace"])

    prompt = st.chat_input("Ask a financial question grounded in your PDFs…")
    if prompt:
        if not _ensure_api_key():
            st.stop()
        st.session_state.chat_history.append(
            {"question": prompt, "answer": None, "trace": ""}
        )
        st.session_state.agent_busy = True
        st.rerun()

    if (
        st.session_state.agent_busy
        and st.session_state.chat_history
        and st.session_state.chat_history[-1].get("answer") is None
    ):
        if not st.session_state.get("agent_display_pass"):
            st.session_state.agent_display_pass = True
            st.rerun()

        question = st.session_state.chat_history[-1]["question"]
        answer, trace = _run_agent_turn(question)
        st.session_state.chat_history[-1] = {
            "question": question,
            "answer": answer,
            "trace": trace,
        }
        st.session_state.agent_busy = False
        st.session_state.pop("agent_display_pass", None)
        st.rerun()


def main() -> None:
    ensure_directories()
    recover_stale_index_jobs()
    if not MANIFEST_PATH.exists():
        sync_manifest_from_chroma()

    _init_pdf_view_state()

    indexed = get_indexed_sources()
    _consume_pending_citation(indexed)

    if "upload_widget_key" not in st.session_state:
        st.session_state.upload_widget_key = 0

    with st.sidebar:
        _render_sidebar(indexed)

    _render_chat_main()
    _inject_main_block_padding_fix()


if __name__ == "__main__":
    main()
