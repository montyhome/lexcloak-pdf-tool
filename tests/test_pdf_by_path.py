"""v0.12.0 (protocol 10): a PDF over the frame limit travels as a file.

A frame holds ``MAX_PAYLOAD_BYTES`` (256 MiB) and base64 makes a PDF a third
larger, so a PDF of about 200 MB could not cross the wire in either
direction. A burn can grow a scanned document several-fold (image redaction
re-encodes the pixels it blanks), so a 57 MB source can come back as 220 MB.
Before this release the response write raised ``OverflowError``, which
escaped ``main()``: the subprocess exited having written nothing, and the
client could only read that as a crash.

Protocol 10:

* every op that takes ``pdf_b64`` also takes ``pdf_path`` (read, never
  written);
* every op that returns a PDF writes it to the request's ``out_path`` when
  one is given, and answers ``{"pdf_path": out_path, "pdf_size": n}`` in
  place of ``pdf_b64``. The file is created exclusively and readable by its
  owner only, and no error message names a path;
* a response over the frame limit is answered with an ``OverflowError``
  error frame, and the subprocess keeps serving.

Every fixture is built in memory with invented text.
"""
from __future__ import annotations

import base64
import json
import os
import stat
import struct
import subprocess
import sys

import pymupdf
import pytest

from lexcloak_pdf_tool.__main__ import PROTOCOL_VERSION
from test_cli import CLISession, _b64, _cover_context, _make_pdf
from test_redact_extraction_goldens import _canonical_pdf

_MATCH = {"page": 0, "type": "SSN",
          "rect": {"x0": 150.0, "y0": 60.0, "x1": 260.0, "y1": 80.0}}


def _encrypted_pdf() -> bytes:
    doc = pymupdf.open(stream=_make_pdf())
    out = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                      user_pw="secret", owner_pw="secret")
    doc.close()
    return out


#: Every stateless op that returns a PDF, with what else it needs.
STATELESS = {
    "apply_redactions": {"matches": [_MATCH]},
    "strip_metadata": {},
    "set_metadata": {"fields": {"subject": "Invented subject"}},
    "insert_cover_page": {"context": _cover_context(p=1)},
    "reduce_size": {},
    "extract_pages": {"from_page": 0, "to_page": 0},
    "encrypt": {"password": "pw-123"},
    "decrypt": {"password": "secret"},
}
#: Every handle op that returns a PDF.
HANDLE = {
    "apply_redactions_h": {"matches": [_MATCH]},
    "strip_metadata_h": {},
    "set_metadata_h": {"fields": {"subject": "Invented subject"}},
    "insert_cover_page_h": {"context": _cover_context(p=1)},
    "reduce_size_h": {},
    "extract_pages_h": {"from_page": 0, "to_page": 0},
}


def _source(op: str) -> bytes:
    return _encrypted_pdf() if op == "decrypt" else _make_pdf()


def _file_bytes(path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def test_the_tables_cover_every_op_that_returns_a_pdf():
    """A PDF-returning op added later must be added here, so it cannot ship
    without ``out_path``. Found by reading which handlers return a PDF."""
    import inspect

    from lexcloak_pdf_tool import __main__ as M
    returning = {name for name, fn in M._OPS.items()
                 if "_pdf_result(" in inspect.getsource(fn)
                 or "_redaction_result(" in inspect.getsource(fn)}
    assert returning == set(STATELESS) | set(HANDLE) | {"encrypt_h"}


@pytest.mark.parametrize("op", sorted(STATELESS))
def test_a_stateless_pdf_result_goes_to_out_path(tmp_path, op):
    out = tmp_path / "out.pdf"
    with CLISession() as s:
        inline = s.call(op, pdf_b64=_b64(_source(op)), **STATELESS[op])
        by_file = s.call(op, pdf_b64=_b64(_source(op)), out_path=str(out),
                         **STATELESS[op])
    assert inline["ok"] is True, inline
    assert by_file["ok"] is True, by_file
    result = by_file["result"]
    assert "pdf_b64" not in result
    assert result["pdf_path"] == str(out)
    assert result["pdf_size"] == out.stat().st_size
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    if op not in ("encrypt", "decrypt"):   # both draw fresh keys on each save
        assert (_canonical_pdf(_file_bytes(out))
                == _canonical_pdf(base64.b64decode(inline["result"]["pdf_b64"])))
    extra = {k: v for k, v in inline["result"].items() if k != "pdf_b64"}
    assert {k: v for k, v in result.items()
            if k not in ("pdf_path", "pdf_size")} == extra


@pytest.mark.parametrize("op", sorted(HANDLE) + ["encrypt_h"])
def test_a_handle_pdf_result_goes_to_out_path(tmp_path, op):
    out = tmp_path / "out.pdf"
    args = HANDLE.get(op, {"password": "pw-h"})
    with CLISession() as s:
        handle = s.call("open_doc", pdf_b64=_b64(_make_pdf()))["result"]["handle"]
        resp = s.call(op, handle=handle, out_path=str(out), **args)
    assert resp["ok"] is True, resp
    assert resp["result"]["pdf_path"] == str(out)
    assert pymupdf.open(str(out)).page_count >= 1 or op == "encrypt_h"


def test_an_op_reads_its_pdf_from_pdf_path(tmp_path):
    src = tmp_path / "src.pdf"
    src.write_bytes(_make_pdf(n_pages=3))
    before = src.read_bytes()
    with CLISession() as s:
        count = s.call("page_count", pdf_path=str(src))
        burned = s.call("apply_redactions", pdf_path=str(src), matches=[_MATCH])
        inline = s.call("apply_redactions", pdf_b64=_b64(src.read_bytes()),
                        matches=[_MATCH])
        handle = s.call("open_doc", pdf_path=str(src))
    assert count["result"]["count"] == 3
    assert (_canonical_pdf(base64.b64decode(burned["result"]["pdf_b64"]))
            == _canonical_pdf(base64.b64decode(inline["result"]["pdf_b64"])))
    assert handle["ok"] is True and handle["result"]["handle"]
    assert src.read_bytes() == before          # the caller's file is only read


def test_out_path_is_never_overwritten_and_the_error_names_no_path(tmp_path):
    out = tmp_path / "Invented Person records.pdf"
    out.write_bytes(b"already here")
    with CLISession() as s:
        resp = s.call("strip_metadata", pdf_b64=_b64(_make_pdf()), out_path=str(out))
    assert resp["ok"] is False
    assert resp["error_type"] == "FileExistsError"
    assert "Invented" not in resp["error"] and str(tmp_path) not in resp["error"]
    assert out.read_bytes() == b"already here"


def test_out_path_is_never_followed_through_a_link(tmp_path):
    target = tmp_path / "target.pdf"
    target.write_bytes(b"keep me")
    link = tmp_path / "link.pdf"
    link.symlink_to(target)
    with CLISession() as s:
        resp = s.call("strip_metadata", pdf_b64=_b64(_make_pdf()), out_path=str(link))
    assert resp["ok"] is False
    assert target.read_bytes() == b"keep me"


@pytest.mark.parametrize("bad", ["relative/out.pdf", 7, ""])
def test_out_path_must_be_an_absolute_path(bad):
    with CLISession() as s:
        resp = s.call("strip_metadata", pdf_b64=_b64(_make_pdf()), out_path=bad)
    assert resp["ok"] is False
    assert resp["error_type"] == "ValueError"


def test_a_missing_pdf_path_names_no_path(tmp_path):
    missing = tmp_path / "Invented Person chart.pdf"
    with CLISession() as s:
        resp = s.call("page_count", pdf_path=str(missing))
    assert resp["ok"] is False
    assert resp["error_type"] == "FileNotFoundError"
    assert "Invented" not in resp["error"] and str(tmp_path) not in resp["error"]


def test_pdf_b64_wins_when_both_are_sent(tmp_path):
    """The inline field is the historical contract; a path beside it is
    ignored, so a client that always sends both is never surprised."""
    src = tmp_path / "src.pdf"
    src.write_bytes(_make_pdf(n_pages=2))
    with CLISession() as s:
        resp = s.call("page_count", pdf_b64=_b64(_make_pdf(n_pages=5)),
                      pdf_path=str(src))
    assert resp["result"]["count"] == 5


# ── A response over the frame limit ──────────────────────────────────


_CHILD = r'''
import lexcloak_pdf_tool.__main__ as M
M.MAX_PAYLOAD_BYTES = int(__import__("sys").argv[1])
raise SystemExit(M.main())
'''


def _frame(obj: dict) -> bytes:
    body = json.dumps(obj).encode()
    return struct.pack(">I", len(body)) + body


def _frames(data: bytes) -> list[dict]:
    out, i = [], 0
    while i + 4 <= len(data):
        (n,) = struct.unpack(">I", data[i:i + 4])
        out.append(json.loads(data[i + 4:i + 4 + n]))
        i += 4 + n
    return out


def _grow(pdf: bytes, **extra) -> dict:
    """A request whose answer is a page larger than itself (a cover page),
    so a limit just above the request is under the response: a burn that
    grows a document past the frame, in miniature."""
    return {"protocol_version": PROTOCOL_VERSION, "op": "insert_cover_page",
            "pdf_b64": _b64(pdf), "context": _cover_context(p=1), **extra}


def test_a_response_over_the_frame_limit_is_an_error_and_the_process_lives_on():
    pdf = _make_pdf()
    burn = _grow(pdf)
    count = {"protocol_version": PROTOCOL_VERSION, "op": "page_count",
             "pdf_b64": _b64(pdf)}
    limit = len(json.dumps(burn)) + 64
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(limit)],
                          input=_frame(burn) + _frame(count), capture_output=True,
                          timeout=60)
    first, second = _frames(proc.stdout)
    assert proc.returncode == 0
    assert first["ok"] is False and first["error_type"] == "OverflowError"
    assert second["ok"] is True and second["result"]["count"] == 1


def test_the_same_response_over_the_limit_arrives_by_out_path(tmp_path):
    pdf = _make_pdf()
    out = tmp_path / "out.pdf"
    burn = _grow(pdf, out_path=str(out))
    limit = len(json.dumps(burn)) + 64
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(limit)],
                          input=_frame(burn), capture_output=True, timeout=60)
    (resp,) = _frames(proc.stdout)
    assert resp["ok"] is True
    assert pymupdf.open(str(out)).page_count == 2
    assert os.path.getsize(out) == resp["result"]["pdf_size"]
