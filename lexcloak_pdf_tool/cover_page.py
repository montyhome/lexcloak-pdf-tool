"""Insert a "review required" cover page at index 0 of a PDF.

Spec 14 from Lex Cloak's accuracy-disclaimer drafts: opt-in default-off
notice that the document was auto-redacted and recipients should review
before further distribution. Body text ships verbatim per drafts doc;
no marketing color or logo -- this is a chain-of-custody notice.

Renders via ``Page.insert_htmlbox`` (PyMuPDF Story engine) so the em-dash
in the title + footer survives extraction. The built-in PostScript
Helvetica that ``insert_textbox`` defaults to has no glyph for U+2014;
the Story engine ships a sans-serif fallback that does.

Each public ``<func>(pdf_bytes, context)`` opens a fresh
``fitz.Document``, calls the matching ``_<func>_doc(doc, context)``
helper, then closes. The ``_doc`` helper is reused by the CLI's
stateful handle protocol (v4) so the doc stays cached.
"""
from __future__ import annotations

import html as _html
import io

import fitz as _fitz

from .redact import open_pdf


# Verbatim copy from ~/redact/.research/accuracy-disclaimer-drafts.md
# lines 326-332. Do NOT paraphrase -- the prompt's Constraints section
# names "verbatim Spec-13/14 strings" as a launch invariant.
_COVER_PAGE_TITLE = "Redacted document — review required"
_COVER_PAGE_BODY_TEMPLATE = (
    "This PDF was processed with Lex Cloak auto-redaction on {date}. "
    "{n} items were redacted across {p} pages. Lex Cloak identifies "
    "common patterns of sensitive information but does not guarantee "
    "that every instance has been detected. The sender of this document "
    "has reviewed it; downstream recipients should apply their own "
    "review before further distribution."
)
_COVER_PAGE_FOOTER = "Lex Cloak — local-first PDF redaction. lexcloak.com"

# Required + optional keys in the context dict the caller supplies.
# Unknown keys raise ValueError so callers don't silently pass typos.
_REQUIRED_CONTEXT_KEYS = frozenset({"date", "redacted_count", "page_count"})
_OPTIONAL_CONTEXT_KEYS = frozenset({"product_version"})
_ALLOWED_CONTEXT_KEYS = _REQUIRED_CONTEXT_KEYS | _OPTIONAL_CONTEXT_KEYS

# Letter-size fallback when the input PDF has zero pages -- can't sniff
# a page size from an empty doc, and the Spec-14 use case (cover page
# on a redacted export) virtually always has at least one page anyway.
_DEFAULT_PAGE_WIDTH = 612.0   # 8.5"
_DEFAULT_PAGE_HEIGHT = 792.0  # 11"

# Layout in PDF point-space (72 DPI). 1" margins.
_MARGIN = 72.0


def _validate_context(context) -> dict:
    """Reject non-dict, missing required keys, unknown keys. Return the dict.

    ValueError surfaces as a named-key error so the wire layer's error
    message is debuggable.
    """
    if not isinstance(context, dict):
        raise ValueError(
            f"context must be a dict, got {type(context).__name__}"
        )
    unknown = set(context.keys()) - _ALLOWED_CONTEXT_KEYS
    if unknown:
        raise ValueError(
            f"unknown context key(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_CONTEXT_KEYS)}"
        )
    missing = _REQUIRED_CONTEXT_KEYS - set(context.keys())
    if missing:
        raise ValueError(
            f"missing required context key(s): {sorted(missing)}"
        )
    return context


def _format_body(context: dict) -> str:
    """Render the template with caller context. No pluralization -- the
    drafts doc text ships verbatim even at N=1 / P=1."""
    return _COVER_PAGE_BODY_TEMPLATE.format(
        date=str(context["date"]),
        n=int(context["redacted_count"]),
        p=int(context["page_count"]),
    )


def _pick_page_size(doc) -> tuple[float, float]:
    """Match the cover page to the first existing page's dimensions.

    Without this, a Letter cover on an A4 redacted doc (or vice versa)
    looks broken. Falls back to Letter on zero-page input.
    """
    if len(doc) == 0:
        return _DEFAULT_PAGE_WIDTH, _DEFAULT_PAGE_HEIGHT
    rect = doc[0].rect
    return float(rect.width), float(rect.height)


def _build_cover_html(body_text: str) -> str:
    """Return the HTML the Story engine renders into the cover page.

    Title + body + footer in a single flow. Sans-serif via the engine's
    bundled Unicode-capable fallback font. No color, no logo; minimum
    typography needed to convey a chain-of-custody notice.
    """
    return (
        '<div style="font-family: sans-serif; color: #000000;">'
        f'<h2 style="font-size: 16pt; font-weight: bold; margin: 0 0 24pt 0;">'
        f'{_html.escape(_COVER_PAGE_TITLE)}</h2>'
        f'<p style="font-size: 11pt; line-height: 1.45; margin: 0 0 32pt 0;">'
        f'{_html.escape(body_text)}</p>'
        f'<p style="font-size: 9pt; margin: 0;">'
        f'{_html.escape(_COVER_PAGE_FOOTER)}</p>'
        '</div>'
    )


def _insert_cover_page_doc(doc, context: dict) -> None:
    """Insert a Spec-14 cover page at page index 0. Mutates doc in place.

    Caller owns ``doc`` lifecycle; this helper does NOT close it.
    """
    _validate_context(context)
    width, height = _pick_page_size(doc)
    page = doc.new_page(pno=0, width=width, height=height)

    content_rect = _fitz.Rect(
        _MARGIN, _MARGIN,
        width - _MARGIN, height - _MARGIN,
    )
    page.insert_htmlbox(content_rect, _build_cover_html(_format_body(context)))


def insert_cover_page(pdf_bytes: bytes, context: dict) -> bytes:
    """Insert a Spec-14 cover page at index 0, return new PDF bytes.

    ``context`` keys:
      * ``date`` (str, e.g., "2026-05-16") -- when the redaction ran.
      * ``redacted_count`` (int) -- how many items were redacted.
      * ``page_count`` (int) -- pages in the redacted source (i.e., the
        page count BEFORE the cover page is inserted -- the body text
        describes the document the recipient is about to read).
      * ``product_version`` (str, optional) -- accepted for forward
        compatibility; not used in the current body text.

    Missing required keys / unknown keys / non-dict context raise
    ``ValueError`` before opening the PDF so bad payloads fail fast.
    """
    doc = open_pdf(pdf_bytes)
    try:
        _insert_cover_page_doc(doc, context)
        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()
    return buf.getvalue()
