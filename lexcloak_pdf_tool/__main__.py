"""lexcloak-pdf-tool subprocess CLI.

Reads length-prefixed JSON commands from stdin and writes length-prefixed
JSON responses to stdout. See ``docs/PROTOCOL.md`` for the wire contract.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import time
from typing import Any

# Absolute imports rather than relative -- PyInstaller's ``--onefile`` runs
# ``__main__.py`` without package context, which breaks ``from . import X``.
# Absolute imports work in both ``python -m lexcloak_pdf_tool`` (package
# present) and the frozen binary (PyInstaller follows the import graph from
# the entry script).
from lexcloak_pdf_tool import (
    apply_redactions,
    decrypt_pdf,
    extract_text_dict,
    extract_text_native,
    extract_text_ocr,
    extract_text_plain,
    get_metadata,
    is_encrypted,
    page_count,
    page_size,
    pymupdf_version,
    render_page,
    search_for,
    strip_metadata,
)
from lexcloak_pdf_tool.coords import deserialize_chardata, serialize_chardata


# Protocol constants -- see docs/PROTOCOL.md.
PROTOCOL_VERSION = 2
SUPPORTED_PROTOCOL_VERSIONS = {2}
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024  # 256 MiB per frame.
LENGTH_PREFIX_BYTES = 4
LENGTH_STRUCT = struct.Struct(">I")  # big-endian uint32.


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


def _op_render(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    # Accept dpi as float -- callers that need sub-integer precision to
    # hit a target pixel width exactly (zoom=200/595 for an A4 page) must
    # round-trip without losing the fractional part.
    dpi = float(cmd.get("dpi", 150))
    png = render_page(pdf_bytes, page, dpi=dpi)
    return {"png_b64": base64.b64encode(png).decode("ascii")}


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
    output_protection = cmd.get("output_protection")
    out_bytes, protection_applied = apply_redactions(
        pdf_bytes,
        matches,
        redact_label=redact_label,
        active_categories=active_categories,
        removed_pages=removed_pages,
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


def _op_page_count(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    return {"count": page_count(pdf_bytes)}


def _op_page_size(cmd: dict) -> dict:
    pdf_bytes = _decode_pdf(cmd)
    page = int(cmd.get("page", 0))
    width, height = page_size(pdf_bytes, page)
    return {"width": float(width), "height": float(height)}


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


_OPS = {
    "render": _op_render,
    "extract_native": _op_extract_native,
    "extract_ocr": _op_extract_ocr,
    "extract_text_dict": _op_extract_text_dict,
    "extract_text_plain": _op_extract_text_plain,
    "search_for": _op_search_for,
    "apply_redactions": _op_apply_redactions,
    "strip_metadata": _op_strip_metadata,
    "page_count": _op_page_count,
    "page_size": _op_page_size,
    "is_encrypted": _op_is_encrypted,
    "get_metadata": _op_get_metadata,
    "decrypt": _op_decrypt,
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


def main() -> int:
    """Run the CLI loop: read frame -> dispatch -> write frame -> repeat."""
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
