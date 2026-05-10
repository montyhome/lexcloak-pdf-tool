"""CLI subprocess tests for ``lexcloak_pdf_tool``.

Spawns the real subprocess (``python -m lexcloak_pdf_tool``), exercises every
op via the JSON protocol, and asserts response shape + correctness.

Resilience cases:
  * corrupt PDF input
  * oversized payload (length-prefix > MAX)
  * unknown op
  * mid-stream disconnect
  * malformed length prefix (truncated header)
  * garbage stdin bytes / malformed JSON between commands
"""
from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys

import fitz
import pytest


PROTOCOL_VERSION = 3
LENGTH_STRUCT = struct.Struct(">I")


# ── Subprocess helpers ──────────────────────────────────────────────


class CLISession:
    """Convenience wrapper for the request/response protocol.

    Used as a context manager so the subprocess is reliably terminated
    even when a test asserts inside the body.
    """

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "lexcloak_pdf_tool"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, payload: bytes) -> None:
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def write_frame(self, obj: dict) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.write(LENGTH_STRUCT.pack(len(payload)) + payload)

    def read_frame(self) -> dict | None:
        header = self.proc.stdout.read(4)
        if not header:
            return None
        length = LENGTH_STRUCT.unpack(header)[0]
        body = self.proc.stdout.read(length)
        return json.loads(body.decode("utf-8"))

    def call(self, op: str, **kwargs) -> dict:
        cmd = {"protocol_version": PROTOCOL_VERSION, "op": op}
        cmd.update(kwargs)
        self.write_frame(cmd)
        return self.read_frame()

    def close(self, timeout: float = 5) -> tuple[int, bytes]:
        try:
            self.write_frame({"op": "exit"})
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            stderr = self.proc.stderr.read()
        except (BrokenPipeError, OSError):
            stderr = b""
        self.proc.wait(timeout=timeout)
        return self.proc.returncode, stderr

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except Exception:
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass


def _make_pdf(text: str = "Patient SSN 123-45-6789",
              x: float = 50, y: float = 100,
              fontsize: float = 12, n_pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(x, y), text, fontsize=fontsize)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _make_pdf_triggering_mupdf_warning() -> bytes:
    """Build a minimal PDF whose render emits a MuPDF C-side warning.

    Corrupts a content-stream operator with an unknown keyword so MuPDF's
    interpreter writes ``"MuPDF error: syntax error: unknown keyword: ..."``
    to its message destination during page rendering. The render itself
    still succeeds (MuPDF skips the bad op), but the warning gets emitted.

    Used to regression-test that the warning routes to stderr rather than
    contaminating stdout's IPC channel.
    """
    import re
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(fitz.Point(50, 100), "Hello", fontsize=12)
    pdf = doc.tobytes()
    doc.close()
    m = re.search(rb"stream\n(.*?)\nendstream", pdf, re.DOTALL)
    assert m is not None, "expected at least one content stream in synthesized PDF"
    return pdf[: m.start(1)] + b"NOT_A_REAL_PDF_OP /x /y" + pdf[m.end(1):]


# ── Happy-path round-trips for every op ─────────────────────────────


def test_page_count_roundtrip():
    with CLISession() as s:
        resp = s.call("page_count", pdf_b64=_b64(_make_pdf(n_pages=4)))
    assert resp["ok"] is True
    assert resp["result"]["count"] == 4


def test_page_size_roundtrip():
    with CLISession() as s:
        resp = s.call("page_size", pdf_b64=_b64(_make_pdf()), page=0)
    assert resp["ok"] is True
    assert resp["result"]["width"] == pytest.approx(612.0)
    assert resp["result"]["height"] == pytest.approx(792.0)


def test_all_page_sizes_roundtrip():
    """Batch op returns one [width, height] entry per page."""
    pdf = _make_pdf(n_pages=3)
    with CLISession() as s:
        resp = s.call("all_page_sizes", pdf_b64=_b64(pdf))
    assert resp["ok"] is True
    sizes = resp["result"]["sizes"]
    assert len(sizes) == 3
    for entry in sizes:
        assert len(entry) == 2
        w, h = entry
        assert w == pytest.approx(612.0)
        assert h == pytest.approx(792.0)


def test_is_encrypted_roundtrip_plaintext():
    with CLISession() as s:
        resp = s.call("is_encrypted", pdf_b64=_b64(_make_pdf()))
    assert resp["result"]["encrypted"] is False


def test_is_encrypted_roundtrip_real_pw():
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="real-password",
        owner_pw="real-password",
    )
    doc.close()
    with CLISession() as s:
        resp = s.call("is_encrypted", pdf_b64=_b64(pdf))
    assert resp["result"]["encrypted"] is True


def test_render_roundtrip():
    with CLISession() as s:
        resp = s.call("render", pdf_b64=_b64(_make_pdf()), page=0, dpi=72)
    assert resp["ok"] is True
    png = base64.b64decode(resp["result"]["png_b64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_does_not_contaminate_stdout_with_mupdf_warnings():
    """Regression: MuPDF C-side warnings must route to stderr, not stdout.

    Before the fix, MuPDF's C library wrote warning lines directly to fd 1,
    ahead of the length-prefixed JSON response. The first 4 leaked bytes
    (typically ``M u P D`` from ``"MuPDF error: ..."``) were read as a
    bogus uint32 length prefix (1,299,533,892), exceeding ``MAX_PAYLOAD_BYTES``
    and breaking the IPC. ``__main__.py`` now calls ``fitz.set_messages``
    at startup to redirect MuPDF's C-side messages to stderr.
    """
    pdf = _make_pdf_triggering_mupdf_warning()
    with CLISession() as s:
        resp = s.call("render", pdf_b64=_b64(pdf), page=0, dpi=72)
    # A clean response decode is itself proof that stdout wasn't contaminated:
    # ``CLISession.call`` reads a length prefix and exact payload, and would
    # raise on a mismatch. Assert the response shape and PNG signature too.
    assert resp["ok"] is True
    png = base64.b64decode(resp["result"]["png_b64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_routes_mupdf_warning_text_to_stderr():
    """Positive observability check: the warning is preserved on stderr.

    Drives the subprocess directly (rather than via ``CLISession``) so we
    can read stderr at exit. Asserts the MuPDF warning text appears there,
    proving observability is preserved while stdout stays clean.
    """
    pdf = _make_pdf_triggering_mupdf_warning()
    proc = subprocess.Popen(
        [sys.executable, "-m", "lexcloak_pdf_tool"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    cmd = {"op": "render", "pdf_b64": _b64(pdf),
           "page": 0, "dpi": 72, "protocol_version": PROTOCOL_VERSION}
    payload = json.dumps(cmd).encode("utf-8")
    stdout, stderr = proc.communicate(
        input=LENGTH_STRUCT.pack(len(payload)) + payload, timeout=30,
    )
    # Stdout: exactly one valid frame, header matches body length.
    assert len(stdout) >= 4
    body_len = LENGTH_STRUCT.unpack(stdout[:4])[0]
    assert len(stdout) == 4 + body_len, (
        "stdout contaminated — leaked bytes ahead of the length prefix; "
        f"first 80 stdout bytes: {stdout[:80]!r}"
    )
    # Stderr: contains the MuPDF warning that the C library emitted.
    stderr_text = stderr.decode("utf-8", errors="replace")
    assert "MuPDF" in stderr_text, (
        f"expected MuPDF warning on stderr, got: {stderr_text!r}"
    )


def test_extract_native_roundtrip():
    with CLISession() as s:
        resp = s.call("extract_native",
                      pdf_b64=_b64(_make_pdf("Hello world")), page=0)
    words = [w["text"] for w in resp["result"]["words"]]
    assert "Hello" in words
    assert "world" in words


def test_search_for_roundtrip_native_path():
    with CLISession() as s:
        resp = s.call("search_for",
                      pdf_b64=_b64(_make_pdf("Find John here")),
                      page=0, needle="John")
    rects = resp["result"]["rects"]
    assert len(rects) == 1
    assert len(rects[0]) == 4


def test_search_for_roundtrip_chardata_path():
    chardata = [
        ["J", 10.0, 10.0, 15.0, 20.0],
        ["o", 15.0, 10.0, 20.0, 20.0],
        ["h", 20.0, 10.0, 25.0, 20.0],
        ["n", 25.0, 10.0, 30.0, 20.0],
    ]
    with CLISession() as s:
        resp = s.call("search_for",
                      pdf_b64=_b64(_make_pdf()), page=0,
                      needle="John", ocr_chardata=chardata)
    assert resp["ok"] is True
    assert len(resp["result"]["rects"]) == 1


def test_apply_redactions_roundtrip_no_matches():
    with CLISession() as s:
        resp = s.call("apply_redactions",
                      pdf_b64=_b64(_make_pdf()), matches=[])
    out = base64.b64decode(resp["result"]["pdf_b64"])
    assert out[:4] == b"%PDF"
    assert resp["result"]["protection_applied"] is True


def test_apply_redactions_roundtrip_with_match():
    pdf = _make_pdf("Patient SSN 123-45-6789")
    matches = [{
        "id": "x", "type": "SSN", "page": 0,
        "rect": {"x0": 30, "y0": 80, "x1": 300, "y1": 120},
        "enabled": True, "text": "123-45-6789",
    }]
    with CLISession() as s:
        resp = s.call("apply_redactions", pdf_b64=_b64(pdf),
                      matches=matches, redact_label="")
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = fitz.open(stream=out, filetype="pdf")
    redacted_text = out_doc[0].get_text()
    out_doc.close()
    assert "123-45-6789" not in redacted_text


def test_strip_metadata_roundtrip():
    doc = fitz.open()
    doc.set_metadata({"author": "Dr. Jane Doe"})
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes()
    doc.close()
    with CLISession() as s:
        resp = s.call("strip_metadata", pdf_b64=_b64(pdf))
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = fitz.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert (meta.get("author") or "") == ""


def test_extract_ocr_roundtrip_when_unavailable():
    """If Tesseract isn't installed locally, ``extract_ocr`` returns null.

    Callers are expected to pick a fallback path (native-text extraction).
    The test passes whether Tesseract is present or not -- the contract is
    "either valid result dict or null".
    """
    with CLISession() as s:
        resp = s.call("extract_ocr", pdf_b64=_b64(_make_pdf()), page=0)
    assert resp["ok"] is True
    result = resp["result"]
    if result is None:
        return
    assert isinstance(result, dict)
    assert "text" in result
    assert "chardata" in result
    assert "spans" in result


# ── Observability ───────────────────────────────────────────────────


def test_startup_line_emitted_on_stderr():
    s = CLISession()
    s.call("page_count", pdf_b64=_b64(_make_pdf()))
    code, stderr = s.close()
    text = stderr.decode("utf-8", errors="replace")
    assert "lexcloak_pdf_tool starting" in text
    assert "protocol_version=3" in text
    assert "pymupdf_version=" in text


def test_per_op_timing_lines_emitted():
    s = CLISession()
    s.call("page_count", pdf_b64=_b64(_make_pdf()))
    s.call("is_encrypted", pdf_b64=_b64(_make_pdf()))
    code, stderr = s.close()
    text = stderr.decode("utf-8", errors="replace")
    assert "page_count ok=True" in text
    assert "is_encrypted ok=True" in text
    assert "duration_ms=" in text


def test_per_op_timing_records_failures():
    s = CLISession()
    s.write_frame({"protocol_version": PROTOCOL_VERSION, "op": "page_count",
                   "pdf_b64": "!!!not-base64!!!"})
    s.read_frame()
    code, stderr = s.close()
    text = stderr.decode("utf-8", errors="replace")
    assert "page_count ok=False" in text


# ── Resilience cases ────────────────────────────────────────────────


def test_unknown_op_returns_structured_error():
    with CLISession() as s:
        resp = s.call("dance_a_jig", pdf_b64=_b64(_make_pdf()))
    assert resp["ok"] is False
    assert resp["error_type"] == "UnknownOp"
    assert "dance_a_jig" in resp["error"]


def test_protocol_version_v1_rejected():
    """v1-declared client frames are rejected: supported set is {2, 3}."""
    with CLISession() as s:
        s.write_frame({"protocol_version": 1, "op": "page_count",
                       "pdf_b64": _b64(_make_pdf())})
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "ProtocolVersionMismatch"


def test_protocol_version_v4_rejected():
    """Future versions outside the supported set are rejected."""
    with CLISession() as s:
        s.write_frame({"protocol_version": 4, "op": "page_count",
                       "pdf_b64": _b64(_make_pdf())})
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "ProtocolVersionMismatch"


def test_protocol_version_v2_still_accepted():
    """v2 clients keep working against a v3 subprocess (backward compat)."""
    with CLISession() as s:
        s.write_frame({"protocol_version": 2, "op": "page_count",
                       "pdf_b64": _b64(_make_pdf(n_pages=2))})
        resp = s.read_frame()
    assert resp["ok"] is True
    assert resp["result"]["count"] == 2


def test_corrupt_pdf_returns_structured_error_not_crash():
    with CLISession() as s:
        resp = s.call("page_count", pdf_b64=_b64(b"this is not a PDF"))
    assert resp["ok"] is False
    assert "error_type" in resp
    assert resp["error_type"] != ""


def test_malformed_base64_pdf_returns_structured_error():
    with CLISession() as s:
        s.write_frame({"protocol_version": PROTOCOL_VERSION, "op": "page_count",
                       "pdf_b64": "!!!definitely not base64!!!"})
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"


def test_missing_pdf_b64_returns_structured_error():
    with CLISession() as s:
        s.write_frame({"protocol_version": PROTOCOL_VERSION, "op": "page_count"})
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "KeyError"


def test_invalid_page_number_returns_structured_error():
    with CLISession() as s:
        resp = s.call("page_size",
                      pdf_b64=_b64(_make_pdf(n_pages=2)), page=99)
    assert resp["ok"] is False
    assert resp["error_type"] == "IndexError"


def test_invalid_match_payload_returns_structured_error():
    pdf = _make_pdf()
    matches = [{
        "id": "bad", "type": "SSN", "page": "abc",
        "rect": {"x0": 10, "y0": 10, "x1": 50, "y1": 30},
        "enabled": True,
    }]
    with CLISession() as s:
        resp = s.call("apply_redactions", pdf_b64=_b64(pdf), matches=matches)
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "non-integer" in resp["error"].lower() or "page" in resp["error"]


def test_oversized_length_prefix_terminates_subprocess():
    """A length prefix > MAX_PAYLOAD_BYTES is rejected without allocating."""
    s = CLISession()
    s.write(LENGTH_STRUCT.pack(1 << 30))  # 1 GiB -- well beyond 256 MiB.
    resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "OverflowError"
    s.proc.wait(timeout=5)
    assert s.proc.returncode != 0
    s.proc.stderr.read()


def test_truncated_length_prefix_clean_eof():
    """Closing stdin mid-prefix surfaces as EOFError in stderr + non-zero exit."""
    s = CLISession()
    s.write(b"\x00\x00")  # 2 bytes -- incomplete 4-byte prefix.
    s.proc.stdin.close()
    s.proc.wait(timeout=5)
    stderr = s.proc.stderr.read().decode("utf-8", errors="replace")
    assert "EOF" in stderr or "stdin closed" in stderr
    assert s.proc.returncode != 0


def test_clean_eof_returns_zero():
    """Closing stdin at frame boundary is a clean exit."""
    s = CLISession()
    s.proc.stdin.close()
    s.proc.wait(timeout=5)
    assert s.proc.returncode == 0
    stderr = s.proc.stderr.read().decode("utf-8", errors="replace")
    assert "stdin closed cleanly" in stderr


def test_garbage_json_payload_returns_diagnostic_then_exits():
    """Malformed JSON inside a valid-length frame reports an error and exits.
    Frame boundary is lost once the parser walks past a bad payload, so the
    parent must restart a fresh subprocess."""
    s = CLISession()
    bad_payload = b"this is not JSON, definitely not"
    s.write(LENGTH_STRUCT.pack(len(bad_payload)) + bad_payload)
    resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] in {"JSONDecodeError", "ValueError"}
    s.proc.wait(timeout=5)
    assert s.proc.returncode != 0
    s.proc.stderr.read()


def test_explicit_exit_op_terminates_cleanly():
    s = CLISession()
    s.write_frame({"protocol_version": PROTOCOL_VERSION, "op": "exit"})
    s.proc.stdin.close()
    s.proc.wait(timeout=5)
    assert s.proc.returncode == 0


def test_multiple_ops_sequential_no_state_leak():
    """Two calls to the same op with different inputs both return correctly --
    no carryover from the first op's PDF buffer."""
    pdf_a = _make_pdf("AAA", n_pages=1)
    pdf_b = _make_pdf("BBB", n_pages=3)
    with CLISession() as s:
        resp_a = s.call("page_count", pdf_b64=_b64(pdf_a))
        resp_b = s.call("page_count", pdf_b64=_b64(pdf_b))
    assert resp_a["result"]["count"] == 1
    assert resp_b["result"]["count"] == 3


# ── extract_text_dict / plain ────────────────────────────────────────


def test_extract_text_dict_roundtrip():
    """Returns the full block/line/span hierarchy."""
    pdf = _make_pdf("Hello world")
    with CLISession() as s:
        resp = s.call("extract_text_dict", pdf_b64=_b64(pdf), page=0)
    assert resp["ok"] is True
    blocks = resp["result"]["blocks"]
    text_blocks = [b for b in blocks if b.get("type") == 0]
    assert text_blocks, "expected at least one text block"
    spans = [span for line in text_blocks[0].get("lines", [])
             for span in line.get("spans", [])]
    span_text = " ".join(s.get("text", "") for s in spans)
    assert "Hello" in span_text
    assert "world" in span_text
    assert all("size" in s and "font" in s for s in spans)


def test_extract_text_dict_strips_image_binary():
    """``type=1`` (image) blocks have their ``image`` bytes stripped."""
    src = fitz.open()
    src_page = src.new_page(width=50, height=50)
    src_page.draw_rect(fitz.Rect(0, 0, 50, 50),
                       color=(0, 0, 0), fill=(0, 0, 0))
    png = src_page.get_pixmap().tobytes("png")
    src.close()

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(50, 50, 150, 150), stream=png)
    pdf_bytes = doc.tobytes()
    doc.close()
    with CLISession() as s:
        resp = s.call("extract_text_dict", pdf_b64=_b64(pdf_bytes), page=0)
    assert resp["ok"] is True
    blocks = resp["result"]["blocks"]
    image_blocks = [b for b in blocks if b.get("type") == 1]
    assert image_blocks, "expected at least one image block"
    for block in image_blocks:
        assert "image" not in block, (
            f"image binary leaked through wire: {list(block.keys())}"
        )


def test_extract_text_dict_invalid_page_returns_structured_error():
    with CLISession() as s:
        resp = s.call("extract_text_dict",
                      pdf_b64=_b64(_make_pdf(n_pages=2)), page=99)
    assert resp["ok"] is False
    assert resp["error_type"] == "IndexError"


def test_extract_text_plain_roundtrip():
    """Plain text via ``page.get_text()``."""
    with CLISession() as s:
        resp = s.call("extract_text_plain",
                      pdf_b64=_b64(_make_pdf("Plain text body")), page=0)
    assert resp["ok"] is True
    text = resp["result"]["text"]
    assert "Plain text body" in text


def test_extract_text_plain_invalid_page_returns_structured_error():
    with CLISession() as s:
        resp = s.call("extract_text_plain",
                      pdf_b64=_b64(_make_pdf(n_pages=1)), page=5)
    assert resp["ok"] is False
    assert resp["error_type"] == "IndexError"


# ── get_metadata ─────────────────────────────────────────────────────


def test_get_metadata_roundtrip_with_fields():
    """Nested ``{metadata, has_xmp}`` shape."""
    doc = fitz.open()
    doc.set_metadata({
        "author": "Dr. Jane Doe",
        "title": "Test Document",
        "subject": "",  # empty -- should be dropped from metadata dict
    })
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes()
    doc.close()
    with CLISession() as s:
        resp = s.call("get_metadata", pdf_b64=_b64(pdf))
    assert resp["ok"] is True
    result = resp["result"]
    assert set(result.keys()) == {"metadata", "has_xmp"}
    assert result["metadata"]["author"] == "Dr. Jane Doe"
    assert result["metadata"]["title"] == "Test Document"
    assert "subject" not in result["metadata"]
    assert isinstance(result["has_xmp"], bool)


def test_get_metadata_roundtrip_plain_pdf_no_xmp():
    """A plain PDF reports ``has_xmp=False``."""
    with CLISession() as s:
        resp = s.call("get_metadata", pdf_b64=_b64(_make_pdf()))
    assert resp["ok"] is True
    assert resp["result"]["has_xmp"] is False


# ── decrypt ──────────────────────────────────────────────────────────


def _make_encrypted_pdf(password: str = "secret",
                        text: str = "Confidential content") -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(fitz.Point(50, 100), text, fontsize=12)
    pdf = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw=password,
    )
    doc.close()
    return pdf


def test_decrypt_roundtrip_correct_password():
    """Correct password unlocks the PDF; bytes round-trip readable."""
    pdf = _make_encrypted_pdf("secret")
    with CLISession() as s:
        resp = s.call("decrypt", pdf_b64=_b64(pdf), password="secret")
    assert resp["ok"] is True
    out = base64.b64decode(resp["result"]["pdf_b64"])
    assert out[:4] == b"%PDF"
    assert resp["result"]["page_count"] == 1
    out_doc = fitz.open(stream=out, filetype="pdf")
    assert not out_doc.is_encrypted or not out_doc.needs_pass
    text = out_doc[0].get_text()
    out_doc.close()
    assert "Confidential content" in text


def test_decrypt_returns_wrong_password_error():
    """Wrong password is an op-level error (instance stays alive)."""
    pdf = _make_encrypted_pdf("secret")
    with CLISession() as s:
        resp = s.call("decrypt", pdf_b64=_b64(pdf), password="wrong")
        followup = s.call("page_count", pdf_b64=_b64(_make_pdf()))
    assert resp["ok"] is False
    assert resp["error_type"] == "WrongPasswordError"
    assert followup["ok"] is True
    assert followup["result"]["count"] == 1


def test_decrypt_unencrypted_pdf_passes_through():
    """Unencrypted input is re-saved cleanly (defensive path)."""
    pdf = _make_pdf("Plaintext", n_pages=2)
    with CLISession() as s:
        resp = s.call("decrypt", pdf_b64=_b64(pdf), password="ignored")
    assert resp["ok"] is True
    out = base64.b64decode(resp["result"]["pdf_b64"])
    assert out[:4] == b"%PDF"
    assert resp["result"]["page_count"] == 2


def test_decrypt_non_string_password_returns_value_error():
    """Password must be a string; non-string surfaces structured error."""
    pdf = _make_encrypted_pdf("secret")
    with CLISession() as s:
        s.write_frame({
            "protocol_version": PROTOCOL_VERSION,
            "op": "decrypt",
            "pdf_b64": _b64(pdf),
            "password": 12345,
        })
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "password" in resp["error"].lower()


# ── Tesseract-dependent (skipped when unavailable) ──────────────────


def test_serialize_chardata_round_trip_through_cli():
    """OCR chardata produced by extract_ocr (in flat form on the wire) is
    deserialize_chardata-able + usable by search_for via the CLI.

    Skipped when Tesseract isn't installed locally."""
    pdf = _make_pdf("Searchable text here")
    with CLISession() as s:
        ocr_resp = s.call("extract_ocr", pdf_b64=_b64(pdf), page=0)
        if ocr_resp["result"] is None:
            pytest.skip("Tesseract unavailable")
        chardata = ocr_resp["result"]["chardata"]
        search_resp = s.call("search_for",
                             pdf_b64=_b64(pdf), page=0,
                             needle="Searchable",
                             ocr_chardata=chardata)
    assert search_resp["ok"] is True
    assert isinstance(search_resp["result"]["rects"], list)
