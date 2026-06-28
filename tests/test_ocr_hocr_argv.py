"""Regression: Tesseract hOCR invocation must not use the ``hocr`` configfile.

Session 617 / 625: requesting hOCR via the bare ``hocr`` configfile argument
makes Tesseract read ``<datadir>/configs/hocr`` — a filesystem lookup the
bundled Windows tessdata layout cannot resolve. On failure Tesseract prints
``read_params_file: Can't open hocr``, EXITS 0, and emits an empty hOCR
skeleton, which slips past ``_run_tesseract``'s returncode + empty-stdout guards
and returns an empty OCR page SILENTLY (the AGPL subprocess ``extract_text_ocr``
cold path used by structured-text / Export-Text redaction + diagnostics).

The fix routes the invocation through ``_tesseract_hocr_argv``, which requests
hOCR via explicit ``-c tessedit_create_hocr=1 -c hocr_font_info=0`` (the two
lines of the stock ``configs/hocr`` file) — byte-identical output where the
configfile resolves, no filesystem dependency. These tests pin the argv shape
(platform-independent) and, when Tesseract is present, that a real PNG OCRs to
non-empty hOCR.
"""
from __future__ import annotations

import fitz
import pytest

from lexcloak_pdf_tool.ocr import (
    OCR_DPI,
    TESSDATA_PATH,
    TESSERACT_BINARY,
    _run_tesseract,
    _tesseract_hocr_argv,
)


def test_hocr_argv_uses_c_params_not_configfile():
    """The regression guard: the argv requests hOCR via ``-c`` params and never
    via the bare ``hocr`` configfile token."""
    argv = _tesseract_hocr_argv("/usr/bin/tesseract", "/tess/data", psm=3)
    # The bare configfile token (the bug) must be absent.
    assert "hocr" not in argv, (
        "bare 'hocr' configfile arg reintroduced — silently empties OCR on "
        "bundled Windows (Session 617)")
    # hOCR must be requested via the explicit -c params instead.
    assert "tessedit_create_hocr=1" in argv
    assert "hocr_font_info=0" in argv
    # ...and each must be introduced by its own ``-c`` flag (consecutive pair).
    for param in ("tessedit_create_hocr=1", "hocr_font_info=0"):
        i = argv.index(param)
        assert argv[i - 1] == "-c", f"{param} not preceded by -c"


def test_hocr_argv_threads_binary_psm_and_tessdata():
    """Structure: the builder threads the binary, psm, and tessdata dir into the
    canonical stdin->stdout invocation."""
    argv = _tesseract_hocr_argv("/opt/tess", "/my/tessdata", psm=6)
    assert argv[0] == "/opt/tess"
    assert argv[1:3] == ["stdin", "stdout"]
    assert argv[argv.index("--psm") + 1] == "6"
    assert argv[argv.index("--tessdata-dir") + 1] == "/my/tessdata"
    assert argv[argv.index("-l") + 1] == "eng"


@pytest.mark.skipif(
    TESSERACT_BINARY is None or TESSDATA_PATH is None,
    reason="Tesseract / tessdata unavailable")
def test_run_tesseract_real_roundtrip_reads_text():
    """End-to-end: a rendered text PNG OCRs to NON-EMPTY hOCR containing the
    seeded word — proving the ``-c`` invocation actually produces output (the
    bug returned an empty skeleton)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(fitz.Point(72, 200), "HELLOWORLD", fontsize=32)
    pix = page.get_pixmap(dpi=OCR_DPI)
    png = pix.tobytes("png")
    doc.close()

    hocr = _run_tesseract(png, psm=3, tessdata_path=str(TESSDATA_PATH))
    assert hocr is not None and len(hocr) > 0, "hOCR was empty (the silent bug)"
    text = hocr.decode("utf-8", errors="replace")
    assert "HELLOWORLD" in text.replace(" ", ""), "seeded word not recognized"
