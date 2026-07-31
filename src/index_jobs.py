"""Background multi-PDF indexing with a hard cap and live status.

Ingest runs in a process pool so Streamlit's UI thread is not GIL-blocked
by embedding work (which made chat appear frozen during first index).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, MAX_INDEXED_PDFS, ensure_directories

MANIFEST_PATH = DATA_DIR / "indexed_manifest.json"
STATUS_PATH = DATA_DIR / "index_jobs_status.json"

INDEXING_DISCLAIMER = (
    "There is/are pdf(s) that is still being indexed so this response "
    "might not be up-to-date to your inquiry"
)

_lock = threading.Lock()
_chroma_write_lock = threading.Lock()
_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()
_remove_executor: ThreadPoolExecutor | None = None
_remove_executor_lock = threading.Lock()

# Filenames whose in-flight ingest should be discarded when finished (Clear all).
_discard_on_complete: set[str] = set()
_CLEAR_JOB_KEY = "__clear_all__"


def _read_json(path: Path, default: Any) -> Any:
    ensure_directories()
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    ensure_directories()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_manifest() -> list[str]:
    data = _read_json(MANIFEST_PATH, [])
    return [str(x) for x in data] if isinstance(data, list) else []


def _write_manifest(names: list[str]) -> None:
    _write_json(MANIFEST_PATH, sorted(set(names)))


def _read_status() -> dict[str, dict[str, Any]]:
    data = _read_json(STATUS_PATH, {})
    return data if isinstance(data, dict) else {}


def _write_status(jobs: dict[str, dict[str, Any]]) -> None:
    _write_json(STATUS_PATH, jobs)


def _update_job(filename: str, payload: dict[str, Any]) -> None:
    with _lock:
        jobs = _read_status()
        jobs[filename] = payload
        _write_status(jobs)


def sync_manifest_from_chroma() -> list[str]:
    """Bootstrap manifest from Chroma source metadata when empty."""
    current = _read_manifest()
    if current:
        return current
    try:
        from src.ingestion import get_vectorstore

        store = get_vectorstore()
        raw = store.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        sources = sorted(
            {
                str(m.get("source"))
                for m in metas
                if isinstance(m, dict) and m.get("source")
            }
        )
        if sources:
            _write_manifest(sources[:MAX_INDEXED_PDFS])
            return sources[:MAX_INDEXED_PDFS]
    except Exception:
        pass
    return current


def get_indexed_sources() -> list[str]:
    return list(_read_manifest())


def get_jobs_snapshot() -> dict[str, dict[str, Any]]:
    return _read_status()


def get_inflight_indexing() -> list[str]:
    """PDFs actively occupying a slot (not yet cancelled/discarded)."""
    return [
        name
        for name, job in _read_status().items()
        if job.get("status") in {"queued", "running"}
    ]


def get_active_indexing() -> list[str]:
    """In-flight + cancelling (for status display)."""
    return [
        name
        for name, job in _read_status().items()
        if job.get("status") in {"queued", "running", "cancelling"}
    ]


def get_active_removing() -> list[str]:
    return [
        name
        for name, job in _read_status().items()
        if job.get("status") == "removing"
    ]


def is_indexing() -> bool:
    return bool(get_inflight_indexing())


def is_removing(filename: str | None = None) -> bool:
    active = set(get_active_removing())
    if filename is None:
        return bool(active)
    return filename in active


def is_clearing() -> bool:
    job = _read_status().get(_CLEAR_JOB_KEY) or {}
    return job.get("status") == "clearing"


def slots_remaining() -> int:
    """Slots used = ready manifest + actively indexing (excludes cancelling)."""
    indexed = set(get_indexed_sources())
    inflight = set(get_inflight_indexing())
    return max(0, MAX_INDEXED_PDFS - len(indexed | inflight))


def _mark_indexed(filename: str) -> None:
    names = _read_manifest()
    if filename not in names:
        names.append(filename)
        _write_manifest(names)


def _discard_ingest_result(filename: str) -> None:
    """Drop a just-finished ingest that was cancelled by Clear all."""
    try:
        from src.ingestion import delete_indexed_source

        with _chroma_write_lock:
            delete_indexed_source(filename)
    except Exception:
        pass

    with _lock:
        names = [n for n in _read_manifest() if n != filename]
        _write_manifest(names)
        jobs = _read_status()
        jobs.pop(filename, None)
        _write_status(jobs)

    local = DATA_DIR / filename
    if local.exists():
        try:
            local.unlink()
        except Exception:
            pass


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            # 2 workers: parallel multi-PDF without hammering free-tier embedding RPM
            _executor = ProcessPoolExecutor(max_workers=2)
        return _executor


def _ingest_in_process(filename: str, path_str: str) -> dict[str, Any]:
    """Picklable top-level worker for ProcessPoolExecutor."""
    from src.ingestion import ingest_pdf

    try:
        chunks = ingest_pdf(path_str)
        return {"filename": filename, "status": "done", "chunks": chunks}
    except Exception as exc:
        return {"filename": filename, "status": "error", "error": str(exc)}


def _on_ingest_done(future) -> None:
    try:
        result = future.result()
    except Exception:
        return
    filename = result.get("filename", "unknown")
    status = result.get("status")

    with _lock:
        discard = filename in _discard_on_complete
        _discard_on_complete.discard(filename)

    if discard:
        _discard_ingest_result(filename)
        return

    if status == "done":
        _update_job(filename, {"status": "done", "chunks": result.get("chunks", 0)})
        _mark_indexed(filename)
    else:
        _update_job(
            filename,
            {"status": "error", "error": result.get("error", "unknown error")},
        )


def start_indexing(files: list[tuple[str, bytes]]) -> tuple[list[str], list[str]]:
    """
    Queue one or more PDFs for process-pool indexing (non-blocking for Streamlit).
    Returns (started_filenames, skipped_messages).
    """
    ensure_directories()
    # Prefer manifest file only — avoid opening Chroma on the UI thread.
    if not MANIFEST_PATH.exists():
        # Best-effort bootstrap; ignore failures so Index stays snappy.
        try:
            sync_manifest_from_chroma()
        except Exception:
            pass

    started: list[str] = []
    skipped: list[str] = []

    indexed = set(get_indexed_sources())
    inflight = set(get_inflight_indexing())

    unique_files: dict[str, bytes] = {}
    for name, raw in files:
        unique_files[name] = raw

    executor = _get_executor()

    for filename, raw in unique_files.items():
        if filename in indexed:
            skipped.append(f"`{filename}` is already indexed.")
            continue
        if filename in inflight:
            skipped.append(f"`{filename}` is already indexing.")
            continue

        used = len(indexed | inflight | set(started))
        if used >= MAX_INDEXED_PDFS:
            skipped.append(
                f"`{filename}` skipped: cap of {MAX_INDEXED_PDFS} indexed PDFs reached."
            )
            continue

        dest = DATA_DIR / filename
        dest.write_bytes(raw)
        _update_job(filename, {"status": "queued"})

        future = executor.submit(_ingest_in_process, filename, str(dest))
        _update_job(filename, {"status": "running"})
        future.add_done_callback(_on_ingest_done)

        started.append(filename)
        inflight.add(filename)

    return started, skipped


def _get_remove_executor() -> ThreadPoolExecutor:
    global _remove_executor
    with _remove_executor_lock:
        if _remove_executor is None:
            # Multiple workers; Chroma deletes serialize on _chroma_write_lock.
            _remove_executor = ThreadPoolExecutor(max_workers=4)
        return _remove_executor


def _remove_worker(filename: str) -> dict[str, Any]:
    """Background delete; safe to queue many at once (UI stays responsive)."""
    try:
        from src.ingestion import delete_indexed_source

        with _chroma_write_lock:
            deleted = delete_indexed_source(filename)

        with _lock:
            names = [n for n in _read_manifest() if n != filename]
            _write_manifest(names)
            jobs = _read_status()
            jobs.pop(filename, None)
            _write_status(jobs)

        local = DATA_DIR / filename
        if local.exists():
            try:
                local.unlink()
            except Exception:
                pass

        return {"filename": filename, "status": "removed", "deleted": deleted}
    except Exception as exc:
        _update_job(filename, {"status": "error", "error": f"Remove failed: {exc}"})
        return {"filename": filename, "status": "error", "error": str(exc)}


def _on_remove_done(future) -> None:
    try:
        future.result()
    except Exception:
        pass


def start_remove_indexed_pdf(filename: str) -> str:
    """
    Queue a non-blocking remove. Multiple PDFs can be queued; each shows
    a removing loader in the UI immediately.
    """
    if filename in get_active_indexing():
        return f"Cannot remove `{filename}` while it is still indexing."
    if filename in get_active_removing():
        return f"`{filename}` is already being removed."
    if filename not in get_indexed_sources():
        return f"`{filename}` is not in the indexed list."

    # Mark removing immediately so consecutive Remove clicks each get a loader.
    _update_job(filename, {"status": "removing"})
    future = _get_remove_executor().submit(_remove_worker, filename)
    future.add_done_callback(_on_remove_done)
    return f"Removing `{filename}`…"


def remove_indexed_pdf(filename: str) -> str:
    """Backward-compatible sync remove (prefer start_remove_indexed_pdf)."""
    return start_remove_indexed_pdf(filename)


def _clear_ready_worker() -> dict[str, Any]:
    """
    Background clear:
    - delete all PDFs already in the manifest (done indexing)
    - mark in-flight ingests to be discarded when they finish
    """
    try:
        from src.ingestion import delete_indexed_source

        ready = list(get_indexed_sources())
        active = list(get_active_indexing())

        with _lock:
            for name in active:
                _discard_on_complete.add(name)

        for filename in ready:
            try:
                with _chroma_write_lock:
                    delete_indexed_source(filename)
            except Exception:
                pass
            local = DATA_DIR / filename
            if local.exists():
                try:
                    local.unlink()
                except Exception:
                    pass

        with _lock:
            # Keep only in-flight jobs (still indexing / cancelling); drop ready ones.
            jobs = _read_status()
            kept: dict[str, dict[str, Any]] = {}
            for name, job in jobs.items():
                if name == _CLEAR_JOB_KEY:
                    continue
                if name in active or job.get("status") in {"queued", "running"}:
                    kept[name] = {**job, "status": "cancelling"}
            _write_manifest([])
            _write_status(kept)

        return {"status": "cleared", "removed": ready, "cancelling": active}
    except Exception as exc:
        _update_job(_CLEAR_JOB_KEY, {"status": "error", "error": str(exc)})
        return {"status": "error", "error": str(exc)}
    finally:
        with _lock:
            jobs = _read_status()
            jobs.pop(_CLEAR_JOB_KEY, None)
            _write_status(jobs)


def start_clear_all_indexed_pdfs() -> str:
    """
    Non-blocking clear of ready indexed PDFs (with loader).
    Works while other PDFs are still indexing — those are cancelled/discarded
    when their ingest finishes.
    """
    if is_clearing():
        return "Clear already in progress."

    ready = get_indexed_sources()
    active = get_active_indexing()
    if not ready and not active:
        return "Nothing to clear."

    _update_job(_CLEAR_JOB_KEY, {"status": "clearing"})
    future = _get_remove_executor().submit(_clear_ready_worker)
    future.add_done_callback(_on_remove_done)
    return (
        f"Clearing {len(ready)} ready PDF(s)"
        + (f"; cancelling {len(active)} in-progress…" if active else "…")
    )


def clear_all_indexed_pdfs() -> str:
    """Backward-compatible entrypoint — queues async clear."""
    return start_clear_all_indexed_pdfs()
