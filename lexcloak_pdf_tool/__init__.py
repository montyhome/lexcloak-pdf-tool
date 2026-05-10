"""lexcloak-pdf-tool -- AGPL-licensed PyMuPDF I/O wrapper.

A subprocess CLI built around the specific PyMuPDF call patterns Lex Cloak
needs. The package can also be used as a library (``import lexcloak_pdf_tool``)
in any AGPL-compatible Python project.

See ``docs/PROTOCOL.md`` for the wire protocol the CLI speaks on
stdin/stdout.

Public API
----------
13 IPC-clean functions, all taking ``pdf_bytes`` (or nothing) and returning
JSON-friendly primitives. The CLI dispatch table maps each one to a wire op
of the same name.

* ``render_page``           -- render a page to PNG bytes.
* ``extract_text_native``   -- native-text words + bboxes.
* ``extract_text_ocr``      -- Tesseract OCR + character coordinates.
* ``extract_text_dict``     -- PyMuPDF block/line/span hierarchy.
* ``extract_text_plain``    -- plain text only.
* ``search_for``            -- substring / whole-word / split search.
* ``apply_redactions``      -- black-box redactions, optional re-encrypt.
* ``strip_metadata``        -- remove document metadata + XMP.
* ``page_count``            -- number of pages.
* ``page_size``             -- page dimensions in points.
* ``all_page_sizes``        -- batch: dimensions for every page.
* ``is_encrypted``          -- distinguish empty-pw from real-pw protection.
* ``get_metadata``          -- metadata dict + XMP-present flag.
* ``decrypt_pdf``           -- authenticate password-protected PDFs.

Plus ``pymupdf_version()`` (probe) and CharData serialization helpers in
``lexcloak_pdf_tool.coords``.
"""
from __future__ import annotations

import fitz as _fitz


version = _fitz.version


def pymupdf_version() -> str:
    """Return PyMuPDF's version string (``fitz.version[0]``)."""
    return _fitz.version[0]


from .render import render_page  # noqa: E402
from .extract import (  # noqa: E402
    extract_text_native,
    extract_text_ocr,
    extract_text_dict,
    extract_text_plain,
    search_for,
)
from .redact import apply_redactions, strip_metadata  # noqa: E402
from .encryption import decrypt_pdf, WrongPasswordError  # noqa: E402
from .metadata import (  # noqa: E402
    page_count,
    page_size,
    all_page_sizes,
    is_encrypted,
    get_metadata,
)
from .coords import (  # noqa: E402
    search_in_chars,
    search_whole_word_in_chars,
    split_search_in_chars,
    serialize_chardata,
    deserialize_chardata,
)


__all__ = [
    "version",
    "pymupdf_version",
    # IPC-clean ops (all wired to the CLI dispatch)
    "render_page",
    "extract_text_native",
    "extract_text_ocr",
    "extract_text_dict",
    "extract_text_plain",
    "search_for",
    "apply_redactions",
    "strip_metadata",
    "page_count",
    "page_size",
    "all_page_sizes",
    "is_encrypted",
    "get_metadata",
    "decrypt_pdf",
    "WrongPasswordError",
    # Library-only CharData helpers
    "search_in_chars",
    "search_whole_word_in_chars",
    "split_search_in_chars",
    "serialize_chardata",
    "deserialize_chardata",
]
