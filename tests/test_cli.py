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
  * stdout cleanliness -- at runtime (MuPDF C-side warnings) and at import
    time (a dependency printing on fd 1 before any handler is installed)
"""
from __future__ import annotations

import base64
import json
import pathlib
import re
import struct
import subprocess
import sys

import pymupdf
import pytest


PROTOCOL_VERSION = 4
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
    doc = pymupdf.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(pymupdf.Point(x, y), text, fontsize=fontsize)
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
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(pymupdf.Point(50, 100), "Hello", fontsize=12)
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
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
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
    and breaking the IPC. ``__main__.py`` now calls ``pymupdf.set_messages``
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


def test_importing_the_package_writes_nothing_to_stdout():
    """Regression: nothing may touch stdout at IMPORT time.

    The two tests above cover *runtime* leaks, which ``set_messages`` can
    redirect. An import-time write is a strictly worse failure: it lands
    before any mitigation this package installs, so the very first frame
    the parent reads is already garbage.

    That is not hypothetical. PyMuPDF 1.28.2 made ``import fitz`` print
    ``"warning: The `fitz` API is deprecated ..."`` (99 bytes) to stdout,
    and the parent read ``w a r n`` (0x7761726E = 2,002,874,990) as a frame
    length -- breaking every PDF operation. The fix was to import
    ``pymupdf`` directly (0.6.7); this test is what keeps it fixed, for any
    future dependency that decides to greet the world on fd 1.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import lexcloak_pdf_tool, lexcloak_pdf_tool.__main__"],
        capture_output=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"import failed: {proc.stderr.decode('utf-8', errors='replace')!r}"
    )
    assert proc.stdout == b"", (
        "import-time write to stdout — this corrupts the frame channel "
        f"before any handler runs. Leaked: {proc.stdout[:200]!r}"
    )


def test_package_does_not_import_the_deprecated_fitz_alias():
    """Static guard: no module may reach PyMuPDF through ``fitz``.

    ``import fitz`` is the specific import-time stdout writer that broke
    the transport (see the test above), and PyMuPDF states the alias
    "will be removed in future" — at which point it is an ImportError at
    spawn. Both names are the same module, so ``pymupdf`` costs nothing.
    """
    pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "lexcloak_pdf_tool"
    offenders = [
        f"{path.name}:{lineno}"
        for path in sorted(pkg_dir.glob("*.py"))
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if re.match(r"\s*(import\s+fitz|from\s+fitz\s+import)\b", line)
    ]
    assert not offenders, f"deprecated `fitz` import(s): {offenders}"


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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    redacted_text = out_doc[0].get_text()
    out_doc.close()
    assert "123-45-6789" not in redacted_text


def test_strip_metadata_roundtrip():
    doc = pymupdf.open()
    doc.set_metadata({"author": "Dr. Jane Doe"})
    doc.new_page(width=612, height=792)
    pdf = doc.tobytes()
    doc.close()
    with CLISession() as s:
        resp = s.call("strip_metadata", pdf_b64=_b64(pdf))
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert (meta.get("author") or "") == ""


def test_set_metadata_roundtrip_spec_13_fields():
    """The launch payload: Spec-13 Subject + Producer + Keywords."""
    pdf = _make_pdf()
    fields = {
        "subject": "Auto-redacted by Lex Cloak. Review before distribution.",
        "producer": "Lex Cloak 1.7.8",
        "keywords": "auto-redacted, review-required",
    }
    with CLISession() as s:
        resp = s.call("set_metadata", pdf_b64=_b64(pdf), fields=fields)
    assert resp["ok"] is True
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    meta = out_doc.metadata or {}
    out_doc.close()
    assert meta["subject"] == fields["subject"]
    assert meta["producer"] == fields["producer"]
    assert meta["keywords"] == fields["keywords"]


def test_set_metadata_missing_fields_arg_returns_error():
    """Omitting the ``fields`` key surfaces a ValueError via the wire."""
    pdf = _make_pdf()
    with CLISession() as s:
        resp = s.call("set_metadata", pdf_b64=_b64(pdf))
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "fields" in resp["error"]


def test_set_metadata_unknown_key_returns_error():
    """Unknown key surfaces ValueError with the offending key named."""
    pdf = _make_pdf()
    with CLISession() as s:
        resp = s.call("set_metadata", pdf_b64=_b64(pdf),
                      fields={"bogus_field": "x"})
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "bogus_field" in resp["error"]


def test_set_metadata_empty_fields_returns_valid_pdf():
    """Empty fields round-trips a valid PDF (no metadata change)."""
    pdf = _make_pdf()
    with CLISession() as s:
        resp = s.call("set_metadata", pdf_b64=_b64(pdf), fields={})
    assert resp["ok"] is True
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    assert len(out_doc) == 1  # page count preserved
    out_doc.close()


# ── insert_cover_page ───────────────────────────────────────────────


def _cover_context(date="2026-05-17", n=12, p=5, version="1.7.8") -> dict:
    return {
        "date": date, "redacted_count": n, "page_count": p,
        "product_version": version,
    }


def test_insert_cover_page_roundtrip_adds_page():
    pdf = _make_pdf(n_pages=3)
    with CLISession() as s:
        resp = s.call("insert_cover_page", pdf_b64=_b64(pdf),
                      context=_cover_context(p=3))
    assert resp["ok"] is True
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    try:
        assert len(out_doc) == 4
        # The verbatim title (em-dash) must be searchable on page 0
        assert out_doc[0].search_for("Redacted document — review required")
    finally:
        out_doc.close()


def test_insert_cover_page_missing_context_returns_error():
    pdf = _make_pdf()
    with CLISession() as s:
        resp = s.call("insert_cover_page", pdf_b64=_b64(pdf))
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "context" in resp["error"]


def test_insert_cover_page_missing_required_key_returns_error():
    pdf = _make_pdf()
    with CLISession() as s:
        resp = s.call("insert_cover_page", pdf_b64=_b64(pdf),
                      context={"date": "2026-05-17", "redacted_count": 1})
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "page_count" in resp["error"]


def test_insert_cover_page_unknown_key_returns_error():
    pdf = _make_pdf()
    bad_ctx = {**_cover_context(), "bogus_key": "x"}
    with CLISession() as s:
        resp = s.call("insert_cover_page", pdf_b64=_b64(pdf), context=bad_ctx)
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "bogus_key" in resp["error"]


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
    assert "protocol_version=4" in text
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
    """v1-declared client frames are rejected: supported set is {2, 3, 4}."""
    with CLISession() as s:
        s.write_frame({"protocol_version": 1, "op": "page_count",
                       "pdf_b64": _b64(_make_pdf())})
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "ProtocolVersionMismatch"


def test_protocol_version_v5_rejected():
    """Future versions outside the supported set are rejected."""
    with CLISession() as s:
        s.write_frame({"protocol_version": 5, "op": "page_count",
                       "pdf_b64": _b64(_make_pdf())})
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "ProtocolVersionMismatch"


def test_protocol_version_v2_still_accepted():
    """v2 clients keep working against a v4 subprocess (backward compat)."""
    with CLISession() as s:
        s.write_frame({"protocol_version": 2, "op": "page_count",
                       "pdf_b64": _b64(_make_pdf(n_pages=2))})
        resp = s.read_frame()
    assert resp["ok"] is True
    assert resp["result"]["count"] == 2


def test_protocol_version_v3_still_accepted():
    """v3 clients (Session 291) keep working against a v4 subprocess."""
    with CLISession() as s:
        s.write_frame({"protocol_version": 3, "op": "page_count",
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
    src = pymupdf.open()
    src_page = src.new_page(width=50, height=50)
    src_page.draw_rect(pymupdf.Rect(0, 0, 50, 50),
                       color=(0, 0, 0), fill=(0, 0, 0))
    png = src_page.get_pixmap().tobytes("png")
    src.close()

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(50, 50, 150, 150), stream=png)
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
    doc = pymupdf.open()
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
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(pymupdf.Point(50, 100), text, fontsize=12)
    pdf = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
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
    out_doc = pymupdf.open(stream=out, filetype="pdf")
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


# ── encrypt (Session 342) ────────────────────────────────────────────


def test_encrypt_roundtrip_stateless():
    """The stateless encrypt op AES-256 protects a cleartext PDF; the output
    authenticates with the supplied password."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    with CLISession() as s:
        resp = s.call("encrypt", pdf_b64=_b64(pdf), password="pw-123")
    assert resp["ok"] is True
    assert resp["result"]["protection_applied"] is True
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    assert out_doc.is_encrypted
    assert out_doc.authenticate("pw-123") > 0
    out_doc.close()


def test_encrypt_empty_password_noop_returns_cleartext():
    """Empty password is a no-op: protection_applied False + openable output."""
    pdf = _make_pdf("Unprotected")
    with CLISession() as s:
        resp = s.call("encrypt", pdf_b64=_b64(pdf), password="")
    assert resp["ok"] is True
    assert resp["result"]["protection_applied"] is False
    out = base64.b64decode(resp["result"]["pdf_b64"])
    out_doc = pymupdf.open(stream=out, filetype="pdf")
    assert not (out_doc.is_encrypted and out_doc.needs_pass)
    out_doc.close()


def test_encrypt_rejects_already_encrypted_input():
    """Encrypted input surfaces a structured ValueError naming the cleartext
    invariant; the subprocess stays alive for the next op."""
    pdf = _make_encrypted_pdf("secret")
    with CLISession() as s:
        resp = s.call("encrypt", pdf_b64=_b64(pdf), password="new-pw")
        followup = s.call("page_count", pdf_b64=_b64(_make_pdf()))
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "cleartext" in resp["error"].lower()
    assert followup["ok"] is True


def test_encrypt_non_string_password_returns_value_error():
    """Password must be a string; a non-string surfaces a structured error."""
    with CLISession() as s:
        s.write_frame({
            "protocol_version": PROTOCOL_VERSION,
            "op": "encrypt",
            "pdf_b64": _b64(_make_pdf()),
            "password": 12345,
        })
        resp = s.read_frame()
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "password" in resp["error"].lower()


def test_encrypt_two_calls_same_session_serialize():
    """Wire-lock: the subprocess handles two encrypt calls back-to-back on one
    session (one frame at a time), staying alive across both."""
    pdf = _make_pdf("Body")
    with CLISession() as s:
        first = s.call("encrypt", pdf_b64=_b64(pdf), password="pw-a")
        second = s.call("encrypt", pdf_b64=_b64(pdf), password="pw-b")
    assert first["ok"] is True and second["ok"] is True
    d1 = pymupdf.open(stream=base64.b64decode(first["result"]["pdf_b64"]),
                   filetype="pdf")
    d2 = pymupdf.open(stream=base64.b64decode(second["result"]["pdf_b64"]),
                   filetype="pdf")
    assert d1.authenticate("pw-a") > 0 and d2.authenticate("pw-b") > 0
    d1.close()
    d2.close()


def test_encrypt_handle_roundtrip_leaves_cached_doc_cleartext():
    """The handle encrypt op returns encrypted bytes but does NOT mutate the
    cached doc (encryption is a save param) -- a follow-up op on the same
    handle still sees cleartext."""
    pdf = _make_pdf("Handle body", n_pages=2)
    with CLISession() as s:
        handle = _open(s, pdf)
        resp = s.call("encrypt_h", handle=handle, password="pw-h")
        # The cached doc is untouched: a subsequent op still works on cleartext.
        followup = s.call("page_count_h", handle=handle)
    assert resp["ok"] is True
    assert resp["result"]["protection_applied"] is True
    out_doc = pymupdf.open(stream=base64.b64decode(resp["result"]["pdf_b64"]),
                        filetype="pdf")
    assert out_doc.authenticate("pw-h") > 0
    out_doc.close()
    assert followup["ok"] is True
    assert followup["result"]["count"] == 2


# ── --version flag (Session 342 / 354 rider) ─────────────────────────
# emit_compat_manifest.py in the closed app runs ``[binary, "--version"]`` and
# parses the semver off STDOUT to verify the bundled subprocess matches the
# pinned tag. These lock that contract (stdout-only, bare semver, exit 0, no
# JSON frame, no startup banner).

# The exact regex emit_compat_manifest._parse_pdf_tool_version applies to stdout.
_COMPAT_MANIFEST_VERSION_RE = r"(?<![\w.])(\d+\.\d+\.\d+)"


def _run_version_flag() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "lexcloak_pdf_tool", "--version"],
        capture_output=True, timeout=30,
    )


def test_version_flag_prints_semver_to_stdout_and_exits_zero():
    import re
    proc = _run_version_flag()
    assert proc.returncode == 0
    stdout = proc.stdout.decode("utf-8").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", stdout), f"not a bare semver: {stdout!r}"
    # The closed-app probe's regex must match this stdout.
    assert re.search(_COMPAT_MANIFEST_VERSION_RE, stdout)


def test_version_flag_matches_package_version():
    from lexcloak_pdf_tool import __version__
    proc = _run_version_flag()
    assert proc.stdout.decode("utf-8").strip() == __version__


def test_version_flag_writes_no_json_frame_and_no_startup_banner():
    """--version short-circuits before the length-prefixed loop: stdout is the
    bare semver (no 4-byte length prefix), and the stderr startup banner that
    the normal path emits is absent."""
    proc = _run_version_flag()
    stdout = proc.stdout
    # A JSON frame would be a 4-byte big-endian length prefix + ``{`` (0x7b);
    # a bare semver line starts with an ASCII digit instead.
    assert stdout[:1].isdigit()
    stderr = proc.stderr.decode("utf-8")
    assert "starting protocol_version" not in stderr


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


# ── Stateful handle protocol (v4) ──────────────────────────────────


def _open(s: CLISession, pdf: bytes) -> str:
    """Open ``pdf`` in the subprocess and return the assigned handle."""
    resp = s.call("open_doc", pdf_b64=_b64(pdf))
    assert resp["ok"] is True, resp
    handle = resp["result"]["handle"]
    assert isinstance(handle, str) and len(handle) > 0
    return handle


def test_open_doc_returns_uuid_handle():
    with CLISession() as s:
        handle = _open(s, _make_pdf())
    # UUID4 string: 8-4-4-4-12 hex characters
    assert len(handle) == 36
    assert handle.count("-") == 4


def test_close_doc_returns_closed_true_for_known_handle():
    with CLISession() as s:
        handle = _open(s, _make_pdf())
        resp = s.call("close_doc", handle=handle)
    assert resp["ok"] is True
    assert resp["result"]["closed"] is True


def test_double_close_is_idempotent():
    """Second close on the same handle returns closed=False (already gone)."""
    with CLISession() as s:
        handle = _open(s, _make_pdf())
        first = s.call("close_doc", handle=handle)
        second = s.call("close_doc", handle=handle)
    assert first["ok"] is True
    assert first["result"]["closed"] is True
    assert second["ok"] is True
    assert second["result"]["closed"] is False


def test_close_unknown_handle_returns_closed_false():
    with CLISession() as s:
        resp = s.call("close_doc", handle="not-a-real-handle")
    assert resp["ok"] is True
    assert resp["result"]["closed"] is False


def test_handle_op_after_close_raises_handle_not_found():
    with CLISession() as s:
        handle = _open(s, _make_pdf())
        s.call("close_doc", handle=handle)
        resp = s.call("page_count_h", handle=handle)
    assert resp["ok"] is False
    assert resp["error_type"] == "HandleNotFound"


def test_handle_op_with_unknown_handle_raises():
    with CLISession() as s:
        resp = s.call("page_count_h", handle="00000000-0000-0000-0000-000000000000")
    assert resp["ok"] is False
    assert resp["error_type"] == "HandleNotFound"


def test_handle_op_with_missing_handle_field_raises():
    with CLISession() as s:
        resp = s.call("page_count_h")  # no handle kwarg
    assert resp["ok"] is False
    assert resp["error_type"] == "HandleNotFound"


def test_handle_op_with_non_string_handle_raises():
    with CLISession() as s:
        resp = s.call("page_count_h", handle=12345)
    assert resp["ok"] is False
    assert resp["error_type"] == "HandleNotFound"


def test_page_count_h_matches_stateless():
    pdf = _make_pdf(n_pages=5)
    with CLISession() as s:
        stateless = s.call("page_count", pdf_b64=_b64(pdf))
        handle = _open(s, pdf)
        handle_resp = s.call("page_count_h", handle=handle)
    assert stateless["result"]["count"] == handle_resp["result"]["count"] == 5


def test_page_size_h_matches_stateless():
    pdf = _make_pdf()
    with CLISession() as s:
        stateless = s.call("page_size", pdf_b64=_b64(pdf), page=0)
        handle = _open(s, pdf)
        handle_resp = s.call("page_size_h", handle=handle, page=0)
    assert stateless["result"] == handle_resp["result"]


def test_all_page_sizes_h_matches_stateless():
    pdf = _make_pdf(n_pages=3)
    with CLISession() as s:
        stateless = s.call("all_page_sizes", pdf_b64=_b64(pdf))
        handle = _open(s, pdf)
        handle_resp = s.call("all_page_sizes_h", handle=handle)
    assert stateless["result"] == handle_resp["result"]


def test_render_h_matches_stateless():
    pdf = _make_pdf()
    with CLISession() as s:
        stateless = s.call("render", pdf_b64=_b64(pdf), page=0, dpi=120)
        handle = _open(s, pdf)
        handle_resp = s.call("render_h", handle=handle, page=0, dpi=120)
    # Bytes are deterministic for a given PDF + DPI under PyMuPDF
    assert stateless["result"]["png_b64"] == handle_resp["result"]["png_b64"]


def test_extract_text_dict_h_matches_stateless():
    pdf = _make_pdf("Important text", n_pages=2)
    with CLISession() as s:
        stateless = s.call("extract_text_dict", pdf_b64=_b64(pdf), page=1)
        handle = _open(s, pdf)
        handle_resp = s.call("extract_text_dict_h", handle=handle, page=1)
    assert stateless["result"] == handle_resp["result"]


def test_extract_text_plain_h_matches_stateless():
    pdf = _make_pdf("Some text content")
    with CLISession() as s:
        stateless = s.call("extract_text_plain", pdf_b64=_b64(pdf), page=0)
        handle = _open(s, pdf)
        handle_resp = s.call("extract_text_plain_h", handle=handle, page=0)
    assert stateless["result"] == handle_resp["result"]


def test_extract_native_h_matches_stateless():
    pdf = _make_pdf("Patient SSN 123-45-6789")
    with CLISession() as s:
        stateless = s.call("extract_native", pdf_b64=_b64(pdf), page=0)
        handle = _open(s, pdf)
        handle_resp = s.call("extract_native_h", handle=handle, page=0)
    assert stateless["result"] == handle_resp["result"]


def test_search_for_h_matches_stateless():
    pdf = _make_pdf("Patient SSN 123-45-6789")
    with CLISession() as s:
        stateless = s.call("search_for", pdf_b64=_b64(pdf), page=0,
                           needle="SSN")
        handle = _open(s, pdf)
        handle_resp = s.call("search_for_h", handle=handle, page=0,
                             needle="SSN")
    assert stateless["result"]["rects"] == handle_resp["result"]["rects"]
    assert len(stateless["result"]["rects"]) >= 1


def test_get_metadata_h_matches_stateless():
    pdf = _make_pdf()
    with CLISession() as s:
        stateless = s.call("get_metadata", pdf_b64=_b64(pdf))
        handle = _open(s, pdf)
        handle_resp = s.call("get_metadata_h", handle=handle)
    assert stateless["result"] == handle_resp["result"]


def test_is_encrypted_h_returns_false_on_plaintext():
    with CLISession() as s:
        handle = _open(s, _make_pdf())
        resp = s.call("is_encrypted_h", handle=handle)
    assert resp["ok"] is True
    assert resp["result"]["encrypted"] is False


def test_is_encrypted_h_returns_true_on_real_password_pdf():
    """Opening a password-protected PDF should still produce a handle.

    PyMuPDF's ``pymupdf.open`` accepts encrypted docs without authenticating;
    operations on the doc fail until ``authenticate(password)`` runs. The
    handle protocol surfaces this state via ``is_encrypted_h`` so the
    closed app can prompt for password before attempting per-page reads.
    Reuses the existing ``_make_encrypted_pdf`` helper from the decrypt
    test section.
    """
    encrypted_pdf = _make_encrypted_pdf("secret123")
    with CLISession() as s:
        handle = _open(s, encrypted_pdf)
        resp = s.call("is_encrypted_h", handle=handle)
    assert resp["ok"] is True
    assert resp["result"]["encrypted"] is True


def test_apply_redactions_h_produces_redacted_output():
    """Handle-based redaction produces a valid PDF whose target rect is blanked."""
    pdf = _make_pdf("Patient SSN 123-45-6789")
    matches = [{
        "page": 0,
        "rect": {"x0": 50.0, "y0": 90.0, "x1": 200.0, "y1": 110.0},
        "enabled": True,
        "type": "SSN",
    }]
    with CLISession() as s:
        handle = _open(s, pdf)
        resp = s.call("apply_redactions_h",
                      handle=handle, matches=matches,
                      active_categories=["SSN"])
    assert resp["ok"] is True
    assert resp["result"]["protection_applied"] is True
    out_bytes = base64.b64decode(resp["result"]["pdf_b64"])
    # Re-open the redacted PDF and verify the SSN text is gone from page 0
    out_doc = pymupdf.open(stream=out_bytes, filetype="pdf")
    try:
        page_text = out_doc[0].get_text()
        assert "123-45-6789" not in page_text
    finally:
        out_doc.close()


def test_strip_metadata_h_produces_bytes():
    pdf = _make_pdf()
    with CLISession() as s:
        handle = _open(s, pdf)
        resp = s.call("strip_metadata_h", handle=handle)
    assert resp["ok"] is True
    out_b64 = resp["result"]["pdf_b64"]
    out_bytes = base64.b64decode(out_b64)
    # Re-open the stripped PDF and verify metadata is empty
    out_doc = pymupdf.open(stream=out_bytes, filetype="pdf")
    try:
        meta = out_doc.metadata or {}
        # Default fields stripped to empty strings; PyMuPDF may add format/encryption
        assert not meta.get("author")
        assert not meta.get("title")
    finally:
        out_doc.close()


def test_set_metadata_h_matches_stateless():
    """Handle-form output equals stateless-form output for same fields."""
    pdf = _make_pdf()
    fields = {"subject": "Auto-redacted", "producer": "Lex Cloak 1.7.8"}
    with CLISession() as s:
        stateless = s.call("set_metadata", pdf_b64=_b64(pdf), fields=fields)
        handle = _open(s, pdf)
        handle_resp = s.call("set_metadata_h", handle=handle, fields=fields)
    assert stateless["ok"] is True and handle_resp["ok"] is True

    # Compare by re-opening both and reading metadata back -- byte
    # equality is not guaranteed (PyMuPDF embeds timestamps + producer
    # autogen during save), but the metadata fields should match.
    sl_doc = pymupdf.open(stream=base64.b64decode(stateless["result"]["pdf_b64"]),
                       filetype="pdf")
    h_doc = pymupdf.open(stream=base64.b64decode(handle_resp["result"]["pdf_b64"]),
                      filetype="pdf")
    try:
        sl_meta = sl_doc.metadata or {}
        h_meta = h_doc.metadata or {}
        assert sl_meta["subject"] == h_meta["subject"] == "Auto-redacted"
        assert sl_meta["producer"] == h_meta["producer"] == "Lex Cloak 1.7.8"
    finally:
        sl_doc.close()
        h_doc.close()


def test_set_metadata_h_persists_on_cached_doc():
    """After set_metadata_h, subsequent get_metadata_h on the SAME handle
    returns the merged fields -- proves the mutation is live on the cached
    doc, not just baked into the returned bytes."""
    pdf = _make_pdf()
    with CLISession() as s:
        handle = _open(s, pdf)
        s.call("set_metadata_h", handle=handle,
               fields={"subject": "persisted"})
        get_resp = s.call("get_metadata_h", handle=handle)
    assert get_resp["ok"] is True
    assert get_resp["result"]["metadata"]["subject"] == "persisted"


def test_set_metadata_h_unknown_key_returns_error():
    pdf = _make_pdf()
    with CLISession() as s:
        handle = _open(s, pdf)
        resp = s.call("set_metadata_h", handle=handle,
                      fields={"bogus_field": "x"})
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "bogus_field" in resp["error"]


def test_insert_cover_page_h_matches_stateless():
    """Handle-form output shape matches stateless: +1 page, title on cover."""
    pdf = _make_pdf(n_pages=2)
    ctx = _cover_context(p=2)
    with CLISession() as s:
        stateless = s.call("insert_cover_page", pdf_b64=_b64(pdf),
                           context=ctx)
        handle = _open(s, pdf)
        handle_resp = s.call("insert_cover_page_h", handle=handle,
                             context=ctx)
    assert stateless["ok"] is True and handle_resp["ok"] is True
    for resp in (stateless, handle_resp):
        out_doc = pymupdf.open(
            stream=base64.b64decode(resp["result"]["pdf_b64"]),
            filetype="pdf")
        try:
            assert len(out_doc) == 3
            assert out_doc[0].search_for(
                "Redacted document — review required")
        finally:
            out_doc.close()


def test_insert_cover_page_h_persists_on_cached_doc():
    """After insert_cover_page_h, page_count_h on the SAME handle reports
    +1 -- proves the mutation is live on the cached doc."""
    pdf = _make_pdf(n_pages=2)
    with CLISession() as s:
        handle = _open(s, pdf)
        s.call("insert_cover_page_h", handle=handle,
               context=_cover_context(p=2))
        count_resp = s.call("page_count_h", handle=handle)
    assert count_resp["ok"] is True
    assert count_resp["result"]["count"] == 3


def test_insert_cover_page_h_missing_context_returns_error():
    pdf = _make_pdf()
    with CLISession() as s:
        handle = _open(s, pdf)
        resp = s.call("insert_cover_page_h", handle=handle)
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"
    assert "context" in resp["error"]


def test_multiple_handles_isolated():
    """Two handles into two PDFs return their own page counts."""
    pdf_a = _make_pdf(n_pages=3)
    pdf_b = _make_pdf(n_pages=7)
    with CLISession() as s:
        h_a = _open(s, pdf_a)
        h_b = _open(s, pdf_b)
        count_a = s.call("page_count_h", handle=h_a)
        count_b = s.call("page_count_h", handle=h_b)
        # Close one; the other still works
        s.call("close_doc", handle=h_a)
        count_a_after = s.call("page_count_h", handle=h_a)
        count_b_after = s.call("page_count_h", handle=h_b)
    assert count_a["result"]["count"] == 3
    assert count_b["result"]["count"] == 7
    assert count_a_after["ok"] is False  # h_a closed
    assert count_a_after["error_type"] == "HandleNotFound"
    assert count_b_after["result"]["count"] == 7  # h_b unaffected


def test_lru_eviction_oldest_dropped_when_cache_full():
    """Open more than _DOC_CACHE_MAX_SIZE handles; the oldest is evicted."""
    from lexcloak_pdf_tool.__main__ import _DOC_CACHE_MAX_SIZE
    pdfs = [_make_pdf(f"doc-{i}") for i in range(_DOC_CACHE_MAX_SIZE + 2)]
    with CLISession() as s:
        handles = [_open(s, p) for p in pdfs]
        # First handle should have been evicted (LRU)
        oldest_resp = s.call("page_count_h", handle=handles[0])
        # Most recent handles still resolve
        newest_resp = s.call("page_count_h", handle=handles[-1])
    assert oldest_resp["ok"] is False
    assert oldest_resp["error_type"] == "HandleNotFound"
    assert newest_resp["ok"] is True
    assert newest_resp["result"]["count"] == 1


def test_handle_protocol_uses_smaller_payload_than_stateless():
    """The whole point of v4: handle ops carry tiny per-call payloads.

    Doesn't measure latency (flaky); asserts the structural property that
    a handle op's request frame is dramatically smaller than a stateless
    op's request frame for the same logical call. Eyeballable from the
    payload size alone.
    """
    pdf = _make_pdf(n_pages=10)
    pdf_b64 = _b64(pdf)
    stateless_payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "op": "page_size", "pdf_b64": pdf_b64, "page": 0,
    }).encode()
    handle_payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "op": "page_size_h", "handle": "x" * 36, "page": 0,
    }).encode()
    # Handle payload should be a small constant; stateless grows with PDF size.
    assert len(handle_payload) < 200
    assert len(stateless_payload) > 5 * len(handle_payload)


# ── reduce_size preserve_metadata across the wire (v0.6.6) ─────────
#
# The scrub inside reduce_size wipes all metadata, including a marking the
# CALLER applied after redacting (Lex Cloak's Spec-13 "Auto-redacted"
# notice lives in `subject`). `preserve_metadata` carries named keys across
# it. Both wire ops must honour it: the stateless `reduce_size` AND the v4
# `reduce_size_h`, which bypasses the reduce_size() wrapper entirely and
# calls _apply_reductions directly.

_SPEC_13 = "Auto-redacted by Lex Cloak. Review before distribution."


def _make_stamped_pdf(**meta) -> bytes:
    """Text PDF with caller-applied metadata, big enough that the no-grow
    guard cannot short-circuit the scrub and preserve the stamp trivially."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for i in range(40):
        page.insert_text(pymupdf.Point(50, 80 + i * 16),
                         "Reference 553-01-8842 filed 2026-03-04.", fontsize=11)
    doc.set_metadata(meta)
    out = doc.tobytes(garbage=0, deflate=False)
    doc.close()
    return out


def _subject_of(pdf_b64: str) -> str:
    doc = pymupdf.open(stream=base64.b64decode(pdf_b64), filetype="pdf")
    try:
        return doc.metadata.get("subject") or ""
    finally:
        doc.close()


def test_reduce_size_wire_default_still_strips_subject():
    src = _make_stamped_pdf(subject=_SPEC_13, producer="SomeTool 9")
    with CLISession() as s:
        resp = s.call("reduce_size", pdf_b64=_b64(src))
    assert resp["ok"] is True, resp
    assert _subject_of(resp["result"]["pdf_b64"]) == ""


def test_reduce_size_wire_preserves_named_subject():
    src = _make_stamped_pdf(subject=_SPEC_13, producer="SomeTool 9")
    with CLISession() as s:
        resp = s.call("reduce_size", pdf_b64=_b64(src),
                      preserve_metadata=["subject"])
    assert resp["ok"] is True, resp
    out_b64 = resp["result"]["pdf_b64"]
    assert _subject_of(out_b64) == _SPEC_13
    # ...and the fingerprint key did NOT come back with it.
    assert b"SomeTool 9" not in base64.b64decode(out_b64)


def test_reduce_size_h_wire_preserves_named_subject():
    """reduce_size_h bypasses reduce_size() and calls _apply_reductions
    directly -- the preservation must live deep enough to cover it."""
    src = _make_stamped_pdf(subject=_SPEC_13)
    with CLISession() as s:
        handle = _open(s, src)
        resp = s.call("reduce_size_h", handle=handle,
                      preserve_metadata=["subject"])
    assert resp["ok"] is True, resp
    assert _subject_of(resp["result"]["pdf_b64"]) == _SPEC_13


def test_reduce_size_h_wire_default_still_strips_subject():
    src = _make_stamped_pdf(subject=_SPEC_13)
    with CLISession() as s:
        handle = _open(s, src)
        resp = s.call("reduce_size_h", handle=handle)
    assert resp["ok"] is True, resp
    assert _subject_of(resp["result"]["pdf_b64"]) == ""


@pytest.mark.parametrize("op", ["reduce_size", "reduce_size_h"])
def test_reduce_size_wire_rejects_unknown_key(op):
    """A misspelled key must be refused on BOTH ops as a clean protocol
    error -- silently ignoring it drops the caller's marking, which is the
    failure this parameter exists to prevent."""
    src = _make_stamped_pdf(subject=_SPEC_13)
    with CLISession() as s:
        if op.endswith("_h"):
            resp = s.call(op, handle=_open(s, src),
                          preserve_metadata=["Subject"])
        else:
            resp = s.call(op, pdf_b64=_b64(src),
                          preserve_metadata=["Subject"])
    assert resp["ok"] is False, resp
    assert "unknown metadata key" in json.dumps(resp)


@pytest.mark.parametrize("op", ["reduce_size", "reduce_size_h"])
def test_reduce_size_wire_preserves_all_three_spec_13_fields(op):
    """subject + producer + keywords, the full Spec-13 marking, across both
    wire ops."""
    src = _make_stamped_pdf(subject=_SPEC_13, producer="Lex Cloak 1.8.19",
                            keywords="redacted, auto-redacted")
    keys = ["subject", "producer", "keywords"]
    with CLISession() as s:
        if op.endswith("_h"):
            resp = s.call(op, handle=_open(s, src), preserve_metadata=keys)
        else:
            resp = s.call(op, pdf_b64=_b64(src), preserve_metadata=keys)
    assert resp["ok"] is True, resp
    doc = pymupdf.open(stream=base64.b64decode(resp["result"]["pdf_b64"]),
                    filetype="pdf")
    try:
        meta = dict(doc.metadata)
    finally:
        doc.close()
    assert meta["subject"] == _SPEC_13
    assert meta["producer"] == "Lex Cloak 1.8.19"
    assert meta["keywords"] == "redacted, auto-redacted"


def test_reduce_size_wire_rejects_bare_string_preserve_metadata():
    """JSON gives us a str if the caller forgets the array -- it must be
    rejected, not iterated into single characters."""
    src = _make_stamped_pdf(subject=_SPEC_13)
    with CLISession() as s:
        resp = s.call("reduce_size", pdf_b64=_b64(src),
                      preserve_metadata="subject")
    assert resp["ok"] is False, resp
    assert "list or tuple" in json.dumps(resp)
