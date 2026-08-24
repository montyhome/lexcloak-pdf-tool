"""lexcloak-pdf-tool -- AGPL-licensed PyMuPDF I/O wrapper.

A subprocess CLI built around the specific PyMuPDF call patterns Lex Cloak
needs. The package can also be used as a library (``import lexcloak_pdf_tool``)
in any AGPL-compatible Python project.

See ``docs/PROTOCOL.md`` for the wire protocol the CLI speaks on
stdin/stdout.

Public API
----------
IPC-clean functions, all taking ``pdf_bytes`` (or nothing) and returning
JSON-friendly primitives. The CLI dispatch table maps each one to a wire op
of the same name.

* ``render_page``           -- render a page to PNG bytes.
* ``extract_text_native``   -- native-text words + bboxes.
* ``extract_text_ocr``      -- Tesseract OCR + character coordinates.
* ``extract_text_dict``     -- PyMuPDF block/line/span hierarchy.
* ``extract_text_plain``    -- plain text only.
* ``search_for``            -- substring / whole-word / split search.
* ``apply_redactions``      -- black-box redactions, out-of-content residue
                               scrub (annotations / attachments / document
                               JavaScript / thumbnails), optional re-encrypt.
* ``strip_metadata``        -- remove document metadata + XMP.
* ``page_count``            -- number of pages.
* ``page_size``             -- page dimensions in points.
* ``all_page_sizes``        -- batch: dimensions for every page.
* ``is_encrypted``          -- distinguish empty-pw from real-pw protection.
* ``get_metadata``          -- metadata dict + XMP-present flag.
* ``set_metadata``          -- merge metadata fields, preserving the rest.
* ``insert_cover_page``     -- insert a "review required" cover page at index 0.
* ``decrypt_pdf``           -- authenticate password-protected PDFs.
* ``encrypt``               -- AES-256 encrypt cleartext bytes under a password
                               (the encrypt-on-exit half of the decrypt/encrypt
                               pipeline bracket).
* ``reduce_size``           -- local size reduction: scrub + font subset +
                               optional DPI image downsample (opt-in, lossy)
                               + opt-in ``preserve_metadata`` carry-across.

Plus ``pymupdf_version()`` (probe) and CharData serialization helpers in
``lexcloak_pdf_tool.coords``.
"""
from __future__ import annotations

import pymupdf as _pymupdf


version = _pymupdf.version

#: Package version -- the single source of truth. ``pyproject.toml`` derives
#: its version from this attribute (setuptools dynamic ``version = {attr =
#: "lexcloak_pdf_tool.__version__"}``), and the CLI ``--version`` flag falls
#: back to it when ``importlib.metadata`` has no dist-info to read (e.g. inside
#: the frozen PyInstaller bundle). Keep it a plain string literal so
#: setuptools can extract it statically without importing this module.
__version__ = "0.7.0"


def pymupdf_version() -> str:
    """Return PyMuPDF's version string (``pymupdf.version[0]``)."""
    return _pymupdf.version[0]


from .render import render_page  # noqa: E402
from .extract import (  # noqa: E402
    extract_text_native,
    extract_text_ocr,
    extract_text_dict,
    extract_text_plain,
    search_for,
)
from .redact import apply_redactions, strip_metadata  # noqa: E402
from .cover_page import insert_cover_page  # noqa: E402
from .encryption import decrypt_pdf, encrypt, WrongPasswordError  # noqa: E402
from .page_split import extract_pages  # noqa: E402
from .reduce_size import reduce_size  # noqa: E402
from .metadata import (  # noqa: E402
    page_count,
    page_size,
    all_page_sizes,
    is_encrypted,
    get_metadata,
    set_metadata,
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
    "__version__",
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
    "set_metadata",
    "insert_cover_page",
    "decrypt_pdf",
    "encrypt",
    "WrongPasswordError",
    "reduce_size",
    "extract_pages",
    # Library-only CharData helpers
    "search_in_chars",
    "search_whole_word_in_chars",
    "split_search_in_chars",
    "serialize_chardata",
    "deserialize_chardata",
]
