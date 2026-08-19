"""Annotation enumeration -- subtype names and counts, nothing else.

This module exists because annotation *subtypes* cannot be recovered from
a rendered page or from extracted text: a ``/Text`` sticky note's content
is not returned by ``get_text()`` at all, and a ``/FreeText`` annot's text
IS returned while living in the annot's own appearance stream rather than
the page content stream. A caller that needs to prove "no text-bearing
annotation survived this export" therefore has to open the document and
enumerate, which is what this does.

**Deliberately narrow output.** The op returns per-page ``{subtype: count}``
and nothing else -- never annotation contents, never rects, never the
author, never dates. That is a privacy boundary, not an API-shape
preference: the caller feeds this into a PHI-free diagnostics payload, and
an op that *could* return annotation text would make that payload's
PHI-free claim depend on the caller's restraint rather than on this
module's inability to leak.
"""
from __future__ import annotations

from .redact import open_pdf


def list_annotations(pdf_bytes: bytes) -> list[dict]:
    """Per-page annotation subtype counts for ``pdf_bytes``.

    Returns one entry per page, in page order::

        [{"page": 0, "subtypes": {"Text": 2, "Link": 1}},
         {"page": 1, "subtypes": {}}]

    ``subtypes`` keys are PyMuPDF's subtype names with the leading slash
    stripped (``annot.type[1]`` -- e.g. ``Text``, ``FreeText``, ``Widget``,
    ``Link``). A page with no annotations yields an empty dict rather than
    being omitted, so the caller can distinguish "page has none" from
    "page not reported".

    **Raises rather than reporting a partial result.** A page whose
    annotation list cannot be walked aborts the whole call. The tempting
    alternative -- report that page as ``{"subtypes": {}, "error": ...}``
    and carry on -- turns a hard failure into a soft one that a caller can
    ignore, and the caller here is proving a NEGATIVE ("no annotation
    survived this export"). An empty dict from a page nobody could read is
    indistinguishable from an empty dict from a clean page unless every
    caller remembers to check a side-channel key. Raising removes the
    chance to forget.
    """
    doc = open_pdf(pdf_bytes)
    try:
        if doc.needs_pass:
            # An encrypted document opens fine and then reports ZERO
            # annotations, because nothing can be walked without the
            # password. Returning that empty result would let a caller
            # proving "no annotation survived" conclude it from a document
            # it never actually read -- the exact fail-open this op exists
            # to prevent. Refuse instead; a caller holding the password
            # decrypts first and calls again.
            raise ValueError(
                "cannot enumerate annotations of an encrypted document "
                "(needs_pass); decrypt before calling list_annotations")
        out: list[dict] = []
        for page_num in range(len(doc)):
            counts: dict[str, int] = {}
            for annot in doc[page_num].annots():
                subtype = annot.type[1]
                counts[subtype] = counts.get(subtype, 0) + 1
            out.append({"page": page_num, "subtypes": counts})
        return out
    finally:
        doc.close()
