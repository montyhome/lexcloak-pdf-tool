"""lexcloak-pdf-tool subprocess CLI.

Reads length-prefixed JSON commands from stdin and writes length-prefixed
JSON responses to stdout. See ``docs/PROTOCOL.md`` for the wire contract.

Protocol versions
-----------------
* **v2** (Session 224 base): per-call ``pdf_b64`` -- subprocess re-parses
  the PDF on every op. Stateless.
* **v3** (Session 291): added ``all_page_sizes`` batch op. Otherwise
  stateless -- still per-call ``pdf_b64``.
* **v4** (Session 295): adds stateful handle protocol. ``open_doc`` parses
  once and returns a handle UUID; per-page ops take ``handle`` instead of
  ``pdf_b64``. ``close_doc`` releases the cached document. Subprocess holds
  a parsed ``pymupdf.Document`` per handle, capped at ``_DOC_CACHE_MAX_SIZE``
  via LRU eviction. Stateless v2/v3 ops remain available -- v4 is purely
  additive so older clients continue to work against a v4 subprocess.
"""
from __future__ import annotations

import base64
import io
import json
import os
import struct
import sys
import time
import uuid
from collections import OrderedDict
from typing import Any

# Absolute imports rather than relative -- PyInstaller's ``--onefile`` runs
# ``__main__.py`` without package context, which breaks ``from . import X``.
# Absolute imports work in both ``python -m lexcloak_pdf_tool`` (package
# present) and the frozen binary (PyInstaller follows the import graph from
# the entry script).
from lexcloak_pdf_tool import (
    all_page_sizes,
    apply_redactions,
    decrypt_pdf,
    encrypt,
    extract_text_dict,
    extract_text_native,
    extract_text_ocr,
    extract_text_plain,
    get_metadata,
    insert_cover_page,
    is_encrypted,
    page_count,
    page_size,
    pymupdf_version,
    reduce_size,
    render_page,
    search_for,
    set_metadata,
    strip_metadata,
)
from lexcloak_pdf_tool.coords import deserialize_chardata, serialize_chardata
from lexcloak_pdf_tool.cover_page import _insert_cover_page_doc
from lexcloak_pdf_tool.page_split import extract_pages, extract_pages_from_doc
# Doc-variant helpers for the v4 stateful handle protocol. These take an
# already-open ``pymupdf.Document`` and reuse the same body as the bytes
# wrappers (which open + close per call). Underscore-prefixed because the
# library API exposes the bytes form publicly; the doc form is internal
# to the CLI's stateful protocol.
from lexcloak_pdf_tool.extract import (
    _extract_text_dict_doc,
    _extract_text_native_doc,
    _extract_text_ocr_doc,
    _extract_text_plain_doc,
    _search_for_doc,
)
from lexcloak_pdf_tool.metadata import (
    _all_page_sizes_doc,
    _get_metadata_doc,
    _is_encrypted_doc,
    _page_count_doc,
    _page_size_doc,
    _set_metadata_doc,
)
from lexcloak_pdf_tool.redact import (
    _apply_redactions_doc,
    _save_encrypted,
    _strip_metadata_doc,
    open_pdf,
    open_pdf_path,
)
from lexcloak_pdf_tool.reduce_size import _apply_reductions, _validate_reduce_params
from lexcloak_pdf_tool.render import _render_page_doc

# MuPDF's C library writes error/warning lines directly to fd 1 (stdout),
# bypassing Python's sys.stdout. In a length-prefixed JSON IPC protocol
# that is fatal: the parent reads the leaked bytes as the next frame's
# length prefix and rejects the connection as oversized. Redirect MuPDF's
# C-side messages to stderr -- the parent already tails stderr for
# diagnostics, so observability is preserved. See tests/test_cli.py
# (test_render_does_not_contaminate_stdout_with_mupdf_warnings) for the
# end-to-end regression. Repro before the fix: a tagged PDF whose render
# emits ``"MuPDF error: ...\n\n"`` on fd 1 ahead of the proper length
# prefix; the parent reads ``M u P D`` (= 0x4D755044 = 1,299,533,892)
# as the frame size and aborts.
#
# Import ``pymupdf``, never the legacy ``fitz`` alias. Two reasons, and the
# first one bites before ``set_messages`` below can run:
#
# 1. ``import fitz`` writes to stdout at IMPORT time. PyMuPDF 1.28.2 made the
#    alias print ``"warning: The `fitz` API is deprecated ..."`` (99 bytes) on
#    fd 1 -- and an import-time write lands ahead of every mitigation this
#    module installs. The parent read ``w a r n`` (= 0x7761726E =
#    2,002,874,990) as a frame length and every PDF op died. Nothing in the
#    protocol can defend a channel that is already dirty at startup; only not
#    importing the alias can.
# 2. PyMuPDF says the alias "will be removed in future", at which point
#    ``import fitz`` is an ImportError and the subprocess cannot start at all.
#
# ``pymupdf`` is the same module object under both names -- ``fitz`` is a
# ``from pymupdf import *`` shim -- so this is a spelling change, not a
# behavior change. Verified on 1.27.2.3 and 1.28.2: every attribute this
# package uses resolves to the identical object. Guarded by
# tests/test_cli.py::test_importing_the_package_writes_nothing_to_stdout.
import pymupdf as _pymupdf

_pymupdf.set_messages(stream=sys.stderr)


# Protocol constants -- see docs/PROTOCOL.md.
# v3 (2026-05-09) added the ``all_page_sizes`` op. v4 (2026-05-10) added
# the stateful handle protocol (open_doc/close_doc + per-op _h variants).
# v5 (0.6.8) added ``render_clip`` + ``list_annotations``. v6 (0.7.0)
# adds ``extract_pages``/``extract_pages_h`` (page-range split with
# re-based bookmarks) and ``open_doc_path`` (handle from a filesystem
# path rather than inline bytes). Both landed under v6 before any v6
# release was cut, so no shipped binary ever advertised 6 with only a
# subset -- do NOT add a further op to 6 once 0.7.0 is released.
# Older versions stay supported so a newer
# subprocess can still serve older clients cleanly; once every shipping
# client speaks v4+, drop 2 + 3 from the set.
PROTOCOL_VERSION = 6
SUPPORTED_PROTOCOL_VERSIONS = {2, 3, 4, 5, 6}
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024  # 256 MiB per frame.
LENGTH_PREFIX_BYTES = 4
LENGTH_STRUCT = struct.Struct(">I")  # big-endian uint32.


# ── Stateful handle cache (v4) ───────────────────────────────────────
# UUID-keyed cache of parsed ``pymupdf.Document`` instances. ``OrderedDict``
# carries LRU eviction order: ``move_to_end`` on access, ``popitem(last=
# False)`` to evict the oldest. Bound the cache to a small N so a long-
# running subprocess that leaks handles (or holds many concurrently for a
# legitimate compare-docs use case) can't exhaust memory.
_DOC_CACHE_MAX_SIZE = 16
_DOC_CACHE: "OrderedDict[str, _pymupdf.Document]" = OrderedDict()


class HandleNotFound(Exception):
    """Raised when a handle-bearing op references a missing or closed handle."""


def _evict_lru_if_full() -> None:
    """If the cache is at capacity, close + drop the oldest entry."""
    while len(_DOC_CACHE) >= _DOC_CACHE_MAX_SIZE:
        old_handle, old_doc = _DOC_CACHE.popitem(last=False)
        try:
            old_doc.close()
        except Exception:
            pass


def _store_handle(doc: _pymupdf.Document) -> str:
    """Insert ``doc`` under a fresh UUID handle. Evicts LRU if at capacity."""
    _evict_lru_if_full()
    handle = str(uuid.uuid4())
    _DOC_CACHE[handle] = doc
    return handle


def _resolve_handle(handle: Any) -> _pymupdf.Document:
    """Return the cached ``pymupdf.Document`` for ``handle``, marking it MRU.

    Raises :class:`HandleNotFound` if the handle is missing, closed, or
    of the wrong type. The CLI dispatcher reflects the exception class
    name as ``error_type`` so clients can distinguish "subprocess crashed"
    from "handle stale" semantically.
    """
    if not isinstance(handle, str) or not handle:
        raise HandleNotFound(
            f"handle must be a non-empty string, got {type(handle).__name__}"
        )
    doc = _DOC_CACHE.get(handle)
    if doc is None:
        raise HandleNotFound(f"unknown or closed handle: {handle}")
    _DOC_CACHE.move_to_end(handle)
    return doc


def _drop_handle(handle: Any) -> bool:
    """Pop ``handle`` from the cache and close the document. Idempotent.

    Returns True if the handle existed; False if it was already gone or
    invalid. Never raises.
    """
    if not isinstance(handle, str) or not handle:
        return False
    doc = _DOC_CACHE.pop(handle, None)
    if doc is None:
        return False
    try:
        doc.close()
    except Exception:
        pass
    return True


def _stderr(line: str) -> None:
    """Write a line to stderr with a trailing newline + flush.

    Stays disciplined about flushing: subprocess parents poll stderr
    line-by-line for observability and a buffered write would defer
    diagnostic lines past the response that follows.
    """
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except OSError:
        # Stderr can be closed by a parent that walked away (e.g., test
        # teardown killed the harness). Don't crash the worker over a lost
        # diagnostic -- protocol stdout is the load-bearing channel.
        pass


def _read_exact(stream, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``stream`` or raise ``EOFError`` on
    short read.

    ``stream.read(n)`` may return fewer than ``n`` bytes on a partial pipe
    fill. The protocol can't recover from a half-frame, so accumulate
    until full or signal EOF.
    """
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(
                f"stdin closed mid-frame (expected {n} bytes, got "
                f"{n - remaining})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream) -> dict | None:
    """Read one length-prefixed JSON frame from ``stream``.

    Returns the parsed dict, or ``None`` on clean EOF. Raises:

    * ``EOFError`` if EOF arrives mid-frame.
    * ``OverflowError`` if the prefix declares a payload > MAX_PAYLOAD_BYTES.
    * ``json.JSONDecodeError`` on malformed JSON.
    * ``ValueError`` if the frame's top-level isn't a dict.
    """
    header = stream.read(LENGTH_PREFIX_BYTES)
    if not header:
        return None
    if len(header) < LENGTH_PREFIX_BYTES:
        raise EOFError(
            f"stdin closed mid-header (expected {LENGTH_PREFIX_BYTES} bytes, "
            f"got {len(header)})"
        )
    length = LENGTH_STRUCT.unpack(header)[0]
    if length == 0:
        return {}
    if length > MAX_PAYLOAD_BYTES:
        raise OverflowError(
            f"frame length {length} exceeds limit {MAX_PAYLOAD_BYTES}"
        )
    payload = _read_exact(stream, length)
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(
            f"protocol frame must be JSON object, got {type(obj).__name__}"
        )
    return obj


def _write_frame(stream, obj: Any) -> None:
    """Serialize ``obj`` to JSON, length-prefix, write + flush."""
    payload = json.dumps(obj, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise OverflowError(
            f"response payload {len(payload)} exceeds limit "
            f"{MAX_PAYLOAD_BYTES}"
        )
    stream.write(LENGTH_STRUCT.pack(len(payload)))
    stream.write(payload)
    stream.flush()


# ── Op dispatch ──────────────────────────────────────────────────────


def _ok(result: Any) -> dict:
    return {"ok": True, "result": result}


def _err(exc: BaseException) -> dict:
    return {
        "ok": False,
        "error": str(exc) or type(exc).__name__,
        "error_type": type(exc).__name__,
    }


def _decode_pdf(cmd: dict) -> bytes:
    """Pull and base64-decode the ``pdf_b64`` field. Raise on missing/bad."""
    if "pdf_b64" not in cmd:
        raise KeyError("op requires 'pdf_b64'")
    try:
        return base64.b64decode(cmd["pdf_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"pdf_b64 not valid base64: {exc}") from None


def _get_pdf_path(cmd: dict) -> str:
    """Pull and validate the ``pdf_path`` field. Raise on missing/bad.

    Mirrors :func:`_decode_pdf`'s contract for the path-based ops: ``KeyError``
    when the field is absent, ``ValueError`` when present but unusable. The
    existence and readability checks are here rather than left to PyMuPDF
    because PyMuPDF reports "missing file" and "unreadable file" with the
    same opaque message, and the caller needs to tell those apart.

    Deliberately does NOT echo the path into any error message. In this
    product the filename itself routinely carries PHI ("Master of <name> med
    records.pdf"), and these strings travel back over the wire and may be
    logged by the caller. The exception type names the fault; the path stays
    on the caller's side, which already knows it. Do not "improve" these
    messages by interpolating the path.
    """
    if "pdf_path" not in cmd:
        raise KeyError("op requires 'pdf_path'")
    pdf_path = cmd["pdf_path"]
    if not isinstance(pdf_path, str) or not pdf_path:
        raise ValueError(
            f"pdf_path must be a non-empty string, got "
            f"{type(pdf_path).__name__}"
        )
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError("pdf_path does not exist or is not a file")
    if not os.access(pdf_path, os.R_OK):
        raise PermissionError("pdf_path exists but is not readable")
    return pdf_path


def _get_handle(cmd: dict) -> str:
    """Pull the ``handle`` field. Raise HandleNotFound on missing/bad."""
    handle = cmd.get("handle")
    if not isinstance(handle, str) or not handle:
        raise HandleNotFound(
            f"op requires non-empty 'handle' string, got "
            f"{type(handle).__name__}"
        )
    return handle


# ── Stateless ops (v2/v3) ────────────────────────────────────────────


def _op_render(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    # Accept dpi as float -- callers that need sub-integer precision to
    # hit a target pixel width exactly (zoom=200/595 for an A4 page) must
    # round-trip without losing the fractional part.
    dpi = float(cmd.get("dpi", 150))
    png = render_page(pdf_bytes, page, dpi=dpi)
    return {"png_b64": base64.b64encode(png).decode("ascii")}


def _op_render_clip(cmd: dict) -> dict:
    """Render one clip of one page (v5+). See `render.render_clip`."""
    from .render import render_clip
    pdf_bytes = _decode_pdf(cmd)
    clip = cmd.get("clip")
    if not isinstance(clip, (list, tuple)) or len(clip) != 4:
        raise ValueError(
            "clip must be a 4-element [x0, y0, x1, y1] array, got "
            f"{type(clip).__name__}"
        )
    try:
        clip = tuple(float(v) for v in clip)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"clip coords must be numeric: {exc}") from exc
    png = render_clip(pdf_bytes, int(cmd.get("page", 0)), clip,
                      dpi=float(cmd.get("dpi", 150)),
                      gray=bool(cmd.get("gray", True)))
    return {"png_b64": base64.b64encode(png).decode("ascii")}


def _op_list_annotations(cmd: dict) -> dict:
    """Per-page annotation subtype counts (v5+). Never contents.

    See `annotations.list_annotations` for why the payload is this narrow.
    """
    from .annotations import list_annotations
    return {"pages": list_annotations(_decode_pdf(cmd))}


def _op_extract_native(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    return {"words": extract_text_native(pdf_bytes, page)}


def _op_extract_ocr(cmd: dict) -> dict | None:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    tessdata_path = cmd.get("tessdata_path")
    psm = int(cmd.get("psm", 3))
    result = extract_text_ocr(pdf_bytes, page,
                              tessdata_path=tessdata_path, psm=psm)
    if result is None:
        # Tesseract unavailable / page failed -- propagate the null
        # verbatim so callers can pick a fallback path.
        return None
    return {
        "text": result.get("text") or "",
        "chardata": serialize_chardata(result.get("chars") or []),
        "spans": result.get("spans") or [],
    }


def _op_search_for(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    needle = cmd.get("needle") or ""
    if not isinstance(needle, str):
        raise ValueError(f"needle must be a string, got {type(needle).__name__}")
    raw_chardata = cmd.get("ocr_chardata")
    chardata = deserialize_chardata(raw_chardata) if raw_chardata else None
    whole_word = bool(cmd.get("whole_word", False))
    split = bool(cmd.get("split", False))
    rects = search_for(pdf_bytes, page, needle, ocr_chardata=chardata,
                       whole_word=whole_word, split=split)
    return {
        "rects": [[float(r.x0), float(r.y0), float(r.x1), float(r.y1)]
                  for r in rects],
    }


def _op_apply_redactions(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    matches = cmd.get("matches") or []
    redact_label = cmd.get("redact_label", "")
    active_categories = cmd.get("active_categories")
    removed_pages = cmd.get("removed_pages")
    blackout_pages = cmd.get("blackout_pages")
    output_protection = cmd.get("output_protection")
    out_bytes, protection_applied = apply_redactions(
        pdf_bytes,
        matches,
        redact_label=redact_label,
        active_categories=active_categories,
        removed_pages=removed_pages,
        blackout_pages=blackout_pages,
        output_protection=output_protection,
    )
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "protection_applied": bool(protection_applied),
    }


def _op_strip_metadata(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    out_bytes = strip_metadata(pdf_bytes)
    return {"pdf_b64": base64.b64encode(out_bytes).decode("ascii")}


def _op_set_metadata(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    fields = cmd.get("fields")
    if fields is None:
        raise ValueError("op requires 'fields' dict")
    out_bytes = set_metadata(pdf_bytes, fields)
    return {"pdf_b64": base64.b64encode(out_bytes).decode("ascii")}


def _op_insert_cover_page(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    context = cmd.get("context")
    if context is None:
        raise ValueError("op requires 'context' dict")
    out_bytes = insert_cover_page(pdf_bytes, context)
    return {"pdf_b64": base64.b64encode(out_bytes).decode("ascii")}


def _op_reduce_size(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    dpi = cmd.get("dpi")
    quality = cmd.get("quality", 75)
    grayscale = bool(cmd.get("grayscale", False))
    # v0.6.6: opt-in metadata carry-across (the caller's post-redaction
    # marking, e.g. the Spec-13 notice, otherwise dies in the scrub).
    # Absent key -> None -> historical behavior; a JSON array arrives as a
    # list, which _validate_preserve_metadata accepts.
    preserve_metadata = cmd.get("preserve_metadata")
    out_bytes, info = reduce_size(
        pdf_bytes, dpi=dpi, quality=quality, grayscale=grayscale,
        preserve_metadata=preserve_metadata,
    )
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "info": info,
    }


def _op_extract_pages(cmd: dict) -> dict:
    """v6: one contiguous page range as a standalone PDF (see page_split)."""
    pdf_bytes = _decode_pdf(cmd)
    from_page = int(cmd.get("from_page", 0))
    to_page = int(cmd.get("to_page", -1))
    out_bytes = extract_pages(pdf_bytes, from_page, to_page)
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "page_count": to_page - from_page + 1,
    }


def _op_page_count(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    return {"count": page_count(pdf_bytes)}


def _op_page_size(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    width, height = page_size(pdf_bytes, page)
    return {"width": float(width), "height": float(height)}


def _op_all_page_sizes(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    sizes = all_page_sizes(pdf_bytes)
    return {"sizes": [[float(w), float(h)] for w, h in sizes]}


def _op_is_encrypted(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    return {"encrypted": bool(is_encrypted(pdf_bytes))}


def _op_extract_text_dict(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    return {"blocks": extract_text_dict(pdf_bytes, page)}


def _op_extract_text_plain(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    return {"text": extract_text_plain(pdf_bytes, page)}


def _op_get_metadata(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    return get_metadata(pdf_bytes)


def _op_decrypt(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    password = cmd.get("password", "")
    if not isinstance(password, str):
        raise ValueError(
            f"password must be a string, got {type(password).__name__}"
        )
    out_bytes, page_count_value = decrypt_pdf(pdf_bytes, password)
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "page_count": page_count_value,
    }


def _op_encrypt(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    password = cmd.get("password", "")
    # ``encrypt`` validates the password type (raises ValueError -> _err) and
    # rejects already-encrypted input; empty password is a no-op returning the
    # bytes unchanged with protection_applied=False.
    out_bytes, protection_applied = encrypt(pdf_bytes, password)
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "protection_applied": bool(protection_applied),
    }


# ── Stateful handle ops (v4) ─────────────────────────────────────────


def _op_open_doc(cmd: dict) -> dict:
    """Parse ``pdf_b64`` once, store the doc under a UUID handle.

    Returns ``{"handle": "<uuid>"}``. The handle stays valid until
    ``close_doc`` or LRU eviction (oldest dropped when cache is full).
    """
    pdf_bytes = _decode_pdf(cmd)
    doc = open_pdf(pdf_bytes)
    handle = _store_handle(doc)
    return {"handle": handle}


def _scrub_path_from_error(exc: Exception, secret_path: str) -> Exception:
    """Return ``exc`` re-made with ``secret_path`` stripped from its message.

    Our own validation never echoes the path, but PyMuPDF does: a non-PDF
    file comes back as ``FileDataError: Failed to open file '<full path>' as
    type pdf.`` That message crosses the wire to the caller, and in this
    product the filename routinely carries PHI ("Master of <name> med
    records.pdf"). Scrub it here, at the one place a path-opened document can
    fail, rather than trusting every upstream library not to interpolate it.

    The exception CLASS is preserved -- callers switch on ``error_type`` and
    must keep being able to tell a corrupt file from a missing one. Only the
    text changes. Both the full path and its basename are replaced, since a
    message may quote either.
    """
    msg = str(exc)
    if secret_path:
        msg = msg.replace(secret_path, "<pdf_path>")
        base = os.path.basename(secret_path)
        if base:
            msg = msg.replace(base, "<pdf_path>")
    try:
        return type(exc)(msg)
    except Exception:  # noqa: BLE001 -- exotic ctor; the scrub still matters
        return ValueError(msg)


def _op_open_doc_path(cmd: dict) -> dict:
    """Open the PDF at ``pdf_path`` once, store it under a UUID handle.

    The path-based sibling of :func:`_op_open_doc`, added in v6. Handle
    semantics are identical -- same UUID handle, same LRU eviction, same
    ``close_doc`` -- and so is the treatment of an encrypted document: it is
    opened and handed back like any other, and callers ask ``is_encrypted_h``.
    The only difference is where the bytes come from, which is the entire
    point: see :func:`~lexcloak_pdf_tool.redact.open_pdf_path` for why the
    path form costs one shared mmap instead of one private copy per reader.
    """
    pdf_path = _get_pdf_path(cmd)
    try:
        doc = open_pdf_path(pdf_path)
    except Exception as exc:  # noqa: BLE001 -- re-raised, scrubbed, below
        raise _scrub_path_from_error(exc, pdf_path) from None
    handle = _store_handle(doc)
    return {"handle": handle}


def _op_close_doc(cmd: dict) -> dict:
    """Pop the handle from the cache and close the doc. Idempotent.

    Returns ``{"closed": bool}`` -- ``True`` if the handle existed,
    ``False`` if it was already gone (mid-shutdown race or LRU eviction
    raced against the caller's cleanup). Never raises.
    """
    handle = cmd.get("handle")
    closed = _drop_handle(handle)
    return {"closed": closed}


def _op_page_count_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    return {"count": _page_count_doc(doc)}


def _op_page_size_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    page = int(cmd.get("page", 0))
    width, height = _page_size_doc(doc, page)
    return {"width": float(width), "height": float(height)}


def _op_all_page_sizes_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    sizes = _all_page_sizes_doc(doc)
    return {"sizes": [[float(w), float(h)] for w, h in sizes]}


def _op_is_encrypted_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    return {"encrypted": bool(_is_encrypted_doc(doc))}


def _op_get_metadata_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    return _get_metadata_doc(doc)


def _op_render_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    page = int(cmd.get("page", 0))
    dpi = float(cmd.get("dpi", 150))
    png = _render_page_doc(doc, page, dpi=dpi)
    return {"png_b64": base64.b64encode(png).decode("ascii")}


def _op_extract_native_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    page = int(cmd.get("page", 0))
    return {"words": _extract_text_native_doc(doc, page)}


def _op_extract_ocr_h(cmd: dict) -> dict | None:
    doc = _resolve_handle(_get_handle(cmd))
    page = int(cmd.get("page", 0))
    tessdata_path = cmd.get("tessdata_path")
    psm = int(cmd.get("psm", 3))
    result = _extract_text_ocr_doc(doc, page,
                                   tessdata_path=tessdata_path, psm=psm)
    if result is None:
        return None
    return {
        "text": result.get("text") or "",
        "chardata": serialize_chardata(result.get("chars") or []),
        "spans": result.get("spans") or [],
    }


def _op_extract_text_dict_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    page = int(cmd.get("page", 0))
    return {"blocks": _extract_text_dict_doc(doc, page)}


def _op_extract_text_plain_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    page = int(cmd.get("page", 0))
    return {"text": _extract_text_plain_doc(doc, page)}


def _op_search_for_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    page = int(cmd.get("page", 0))
    needle = cmd.get("needle") or ""
    if not isinstance(needle, str):
        raise ValueError(f"needle must be a string, got {type(needle).__name__}")
    raw_chardata = cmd.get("ocr_chardata")
    chardata = deserialize_chardata(raw_chardata) if raw_chardata else None
    whole_word = bool(cmd.get("whole_word", False))
    split = bool(cmd.get("split", False))
    rects = _search_for_doc(doc, page, needle, ocr_chardata=chardata,
                            whole_word=whole_word, split=split)
    return {
        "rects": [[float(r.x0), float(r.y0), float(r.x1), float(r.y1)]
                  for r in rects],
    }


def _op_apply_redactions_h(cmd: dict) -> dict:
    doc = _resolve_handle(_get_handle(cmd))
    matches = cmd.get("matches") or []
    redact_label = cmd.get("redact_label", "")
    active_categories = cmd.get("active_categories")
    removed_pages = cmd.get("removed_pages")
    blackout_pages = cmd.get("blackout_pages")
    output_protection = cmd.get("output_protection")
    out_bytes, protection_applied = _apply_redactions_doc(
        doc, matches,
        redact_label=redact_label,
        active_categories=active_categories,
        removed_pages=removed_pages,
        blackout_pages=blackout_pages,
        output_protection=output_protection,
    )
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "protection_applied": bool(protection_applied),
    }


def _op_strip_metadata_h(cmd: dict) -> dict:
    """Mutate the open doc + return its bytes.

    Note: this leaves the cached doc in a metadata-stripped state. Callers
    typically use this op as a save-and-finish step alongside ``close_doc``;
    if you need the original metadata back, reopen via ``open_doc``.
    """
    import io
    doc = _resolve_handle(_get_handle(cmd))
    _strip_metadata_doc(doc)
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, clean=True)
    return {"pdf_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def _op_set_metadata_h(cmd: dict) -> dict:
    """Merge ``fields`` into the cached doc + return its bytes.

    Mirrors ``_op_strip_metadata_h``: mutates the live doc, then saves
    a fresh byte stream. The cached doc retains the merged metadata for
    any subsequent ops on this handle.
    """
    import io
    doc = _resolve_handle(_get_handle(cmd))
    fields = cmd.get("fields")
    if fields is None:
        raise ValueError("op requires 'fields' dict")
    _set_metadata_doc(doc, fields)
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, clean=True)
    return {"pdf_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def _op_insert_cover_page_h(cmd: dict) -> dict:
    """Insert a Spec-14 cover page into the cached doc + return its bytes.

    Mutates the live doc: the cover lands at page index 0, original
    pages shift to index 1+. Subsequent ops on this handle see the
    page-shifted document; callers that need both views should reopen
    a fresh handle from the original bytes.
    """
    import io
    doc = _resolve_handle(_get_handle(cmd))
    context = cmd.get("context")
    if context is None:
        raise ValueError("op requires 'context' dict")
    _insert_cover_page_doc(doc, context)
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, clean=True)
    return {"pdf_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def _save_doc_bytes(doc) -> bytes:
    """Serialize a live ``pymupdf.Document`` with the standard save params."""
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, clean=True)
    return buf.getvalue()


def _op_reduce_size_h(cmd: dict) -> dict:
    """Shrink the cached doc in place + return its bytes + size info.

    The handle holds a parsed doc, not its original bytes, so the current
    state is serialized first for ``orig_size`` and the no-grow comparison.
    Like the other ``_h`` save ops this leaves the cached doc in its reduced
    state; callers use it as a save-and-finish step before ``close_doc``.
    """
    doc = _resolve_handle(_get_handle(cmd))
    dpi = cmd.get("dpi")
    quality = cmd.get("quality", 75)
    grayscale = bool(cmd.get("grayscale", False))
    preserve_metadata = cmd.get("preserve_metadata")
    _validate_reduce_params(dpi, quality)
    if doc.is_encrypted and doc.needs_pass:
        raise ValueError("reduce_size() input must be cleartext")
    orig_bytes = _save_doc_bytes(doc)
    applied_dpi = _apply_reductions(
        doc, dpi=dpi, quality=quality, grayscale=grayscale,
        preserve_metadata=preserve_metadata,
    )
    new_bytes = _save_doc_bytes(doc)
    if len(new_bytes) >= len(orig_bytes):
        out_bytes, applied_dpi = orig_bytes, None
    else:
        out_bytes = new_bytes
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "info": {
            "orig_size": len(orig_bytes),
            "new_size": len(out_bytes),
            "applied_dpi": applied_dpi,
        },
    }


def _op_encrypt_h(cmd: dict) -> dict:
    """AES-256 encrypt the cached doc + return its bytes.

    Unlike the other ``_h`` save ops (``set_metadata_h`` / ``reduce_size_h``)
    this does NOT mutate the cached document -- PyMuPDF applies encryption at
    save time, so the handle's in-memory doc stays cleartext and reusable.
    Empty password is a no-op returning cleartext bytes + ``protection_applied
    =False``; already-encrypted input is rejected (the pipeline only encrypts
    cleartext).
    """
    doc = _resolve_handle(_get_handle(cmd))
    password = cmd.get("password", "")
    if not isinstance(password, str):
        raise ValueError(
            f"password must be a string, got {type(password).__name__}"
        )
    if doc.is_encrypted and doc.needs_pass:
        raise ValueError(
            "encrypt() input must be cleartext; use decrypt_pdf() to "
            "authenticate first"
        )
    if not password:
        out_bytes, protection_applied = _save_doc_bytes(doc), False
    else:
        out_bytes, protection_applied = _save_encrypted(doc, password)
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "protection_applied": bool(protection_applied),
    }


def _op_extract_pages_h(cmd: dict) -> dict:
    """v6: page-range extraction from the cached doc.

    Read-only against the handle — the cached document is never mutated
    or closed, so the handle stays fully usable afterwards (unlike the
    mutate-and-save ``_h`` ops above).
    """
    doc = _resolve_handle(_get_handle(cmd))
    from_page = int(cmd.get("from_page", 0))
    to_page = int(cmd.get("to_page", -1))
    out_bytes = extract_pages_from_doc(doc, from_page, to_page)
    return {
        "pdf_b64": base64.b64encode(out_bytes).decode("ascii"),
        "page_count": to_page - from_page + 1,
    }


_OPS = {
    # v2/v3 stateless ops
    "render": _op_render,
    "render_clip": _op_render_clip,
    "list_annotations": _op_list_annotations,
    "extract_native": _op_extract_native,
    "extract_ocr": _op_extract_ocr,
    "extract_text_dict": _op_extract_text_dict,
    "extract_text_plain": _op_extract_text_plain,
    "search_for": _op_search_for,
    "apply_redactions": _op_apply_redactions,
    "strip_metadata": _op_strip_metadata,
    "set_metadata": _op_set_metadata,
    "insert_cover_page": _op_insert_cover_page,
    "reduce_size": _op_reduce_size,
    "page_count": _op_page_count,
    "page_size": _op_page_size,
    "all_page_sizes": _op_all_page_sizes,
    "is_encrypted": _op_is_encrypted,
    "get_metadata": _op_get_metadata,
    "decrypt": _op_decrypt,
    "encrypt": _op_encrypt,
    # v4 stateful handle ops
    "open_doc": _op_open_doc,
    "close_doc": _op_close_doc,
    "page_count_h": _op_page_count_h,
    "page_size_h": _op_page_size_h,
    "all_page_sizes_h": _op_all_page_sizes_h,
    "is_encrypted_h": _op_is_encrypted_h,
    "get_metadata_h": _op_get_metadata_h,
    "render_h": _op_render_h,
    "extract_native_h": _op_extract_native_h,
    "extract_ocr_h": _op_extract_ocr_h,
    "extract_text_dict_h": _op_extract_text_dict_h,
    "extract_text_plain_h": _op_extract_text_plain_h,
    "search_for_h": _op_search_for_h,
    "apply_redactions_h": _op_apply_redactions_h,
    "strip_metadata_h": _op_strip_metadata_h,
    "set_metadata_h": _op_set_metadata_h,
    "insert_cover_page_h": _op_insert_cover_page_h,
    "reduce_size_h": _op_reduce_size_h,
    "encrypt_h": _op_encrypt_h,
    # v6 page-range split
    "extract_pages": _op_extract_pages,
    "extract_pages_h": _op_extract_pages_h,
    "open_doc_path": _op_open_doc_path,
}


def _handle(cmd: dict) -> dict | None:
    """Dispatch one command. Returns the response dict or ``None`` on exit."""
    op = cmd.get("op")
    if op == "exit":
        return None

    client_version = cmd.get("protocol_version")
    if client_version is not None:
        try:
            cv_int = int(client_version)
        except (TypeError, ValueError):
            cv_int = None
        if cv_int not in SUPPORTED_PROTOCOL_VERSIONS:
            return {
                "ok": False,
                "error": (
                    f"protocol_version mismatch: client={client_version} "
                    f"server supports {sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
                ),
                "error_type": "ProtocolVersionMismatch",
            }

    handler = _OPS.get(op)
    if handler is None:
        return {
            "ok": False,
            "error": f"unknown op: {op!r}",
            "error_type": "UnknownOp",
        }
    try:
        result = handler(cmd)
        return _ok(result)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def _package_version() -> str:
    """Return the ``lexcloak-pdf-tool`` package version.

    Prefers the installed distribution metadata (``importlib.metadata``, which
    derives from ``pyproject.toml``); falls back to the package ``__version__``
    literal when no dist-info is present -- the frozen PyInstaller bundle
    collects the module code but not the ``.dist-info`` directory.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("lexcloak-pdf-tool")
        except PackageNotFoundError:
            pass
    except ImportError:  # pragma: no cover -- importlib.metadata is stdlib on 3.12
        pass
    from lexcloak_pdf_tool import __version__
    return __version__


def main() -> int:
    """Run the CLI loop: read frame -> dispatch -> write frame -> repeat."""
    # ``--version`` short-circuit: print the package version to STDOUT and exit
    # before the length-prefixed JSON loop. The closed app's
    # ``emit_compat_manifest.py`` probes this to confirm the bundled subprocess
    # matches the pinned tag (Session 354 + 342). STDOUT only, bare semver --
    # the stderr startup banner below carries pymupdf_version and must never be
    # parsed as the app version (regression-locked downstream).
    if "--version" in sys.argv[1:]:
        print(_package_version())
        return 0

    # Force binary mode on stdin/stdout so newline translation doesn't
    # corrupt PNG/PDF/length-prefix bytes on Windows.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    if hasattr(os, "set_blocking"):
        try:
            os.set_blocking(stdin.fileno(), True)
        except (OSError, ValueError):
            pass

    _stderr(
        f"lexcloak_pdf_tool starting protocol_version={PROTOCOL_VERSION} "
        f"pymupdf_version={pymupdf_version()}"
    )

    while True:
        try:
            cmd = _read_frame(stdin)
        except EOFError as exc:
            _stderr(f"lexcloak_pdf_tool stdin EOF mid-frame: {exc}")
            return 1
        except OverflowError as exc:
            _stderr(f"lexcloak_pdf_tool protocol error (oversized frame): {exc}")
            try:
                _write_frame(stdout, _err(exc))
            except OSError:
                pass
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            _stderr(f"lexcloak_pdf_tool protocol error (malformed frame): {exc}")
            try:
                _write_frame(stdout, _err(exc))
            except OSError:
                pass
            # Cannot recover -- frame boundary is lost once we read past a
            # bad header. Exit so the parent restarts a fresh subprocess.
            return 3

        if cmd is None:
            _stderr("lexcloak_pdf_tool stdin closed cleanly; exiting")
            return 0

        op = cmd.get("op", "?")
        start = time.monotonic()
        response = _handle(cmd)
        duration_ms = int((time.monotonic() - start) * 1000)

        if response is None:
            _stderr(f"exit ok=True duration_ms={duration_ms}")
            return 0

        ok = response.get("ok", False)
        _stderr(f"{op} ok={ok} duration_ms={duration_ms}")
        try:
            _write_frame(stdout, response)
        except OSError as exc:
            _stderr(f"lexcloak_pdf_tool stdout closed: {exc}")
            return 0


if __name__ == "__main__":
    sys.exit(main())
