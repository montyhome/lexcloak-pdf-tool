"""Redaction: black-box matches + strip metadata + optional re-encryption."""
from __future__ import annotations

import io
import logging

import fitz as _fitz

PDF_ENCRYPT_AES_256 = _fitz.PDF_ENCRYPT_AES_256
PDF_PERM_ACCESSIBILITY = _fitz.PDF_PERM_ACCESSIBILITY
Rect = _fitz.Rect
TEXT_ALIGN_CENTER = _fitz.TEXT_ALIGN_CENTER


def open_pdf(pdf_bytes: bytes) -> _fitz.Document:
    """Open a PDF from bytes -- module-local helper."""
    return _fitz.open(stream=pdf_bytes, filetype="pdf")


def _validate_redaction_payload(matches: list[dict],
                                removed_pages: list[int] | None) -> None:
    """Coerce + sanity-check the wire payload before opening the PDF.

    Mutates each match in place to coerce ``page`` to int and rect coords
    to float so downstream code can rely on the typed shape. Raises
    ``ValueError`` with a named-field message on the first malformed
    entry; the caller surfaces this as a 400-class error.

    Without this guard, a bad payload (e.g., ``{"page": "abc"}``,
    ``{"rect": {"x0": "??"}}``, swapped x0/x1) would surface as an opaque
    PyMuPDF error deep inside ``page.add_redact_annot``.
    """
    if removed_pages is not None:
        try:
            removed_pages[:] = [int(p) for p in removed_pages]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed redaction payload: removed_pages "
                f"contains non-integer entry -- {type(exc).__name__}"
            ) from None

    for i, m in enumerate(matches):
        if not isinstance(m, dict):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}] is not a dict"
            )
        try:
            m["page"] = int(m["page"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].page "
                f"missing or non-integer"
            )

        rect = m.get("rect")
        if not isinstance(rect, dict):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].rect "
                f"missing or not a dict"
            )
        try:
            x0 = float(rect["x0"])
            y0 = float(rect["y0"])
            x1 = float(rect["x1"])
            y1 = float(rect["y1"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].rect "
                f"has missing or non-numeric x0/y0/x1/y1"
            )
        if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
            raise ValueError(
                f"Malformed redaction payload: matches[{i}].rect "
                f"has invalid bounds x0={x0} y0={y0} x1={x1} y1={y1}"
            )
        rect["x0"] = x0
        rect["y0"] = y0
        rect["x1"] = x1
        rect["y1"] = y1


def _strip_metadata_doc(doc) -> None:
    """Remove all document metadata + XMP from a live ``fitz.Document``.

    Strips author, title, subject, creator, producer, creation/mod dates,
    keywords, and the full XMP metadata stream.
    """
    doc.set_metadata({
        "author": "",
        "title": "",
        "subject": "",
        "creator": "",
        "producer": "",
        "creationDate": "",
        "modDate": "",
        "keywords": "",
    })
    doc.del_xml_metadata()


def strip_metadata(pdf_bytes_or_doc):
    """Strip document metadata + XMP. Accepts bytes (IPC form) or Document.

    * ``bytes`` -> returns ``bytes`` with metadata stripped (IPC-clean form).
    * ``fitz.Document`` -> mutates in place, returns ``None``.
    """
    if isinstance(pdf_bytes_or_doc, (bytes, bytearray)):
        doc = open_pdf(bytes(pdf_bytes_or_doc))
        try:
            _strip_metadata_doc(doc)
            buf = io.BytesIO()
            doc.save(buf, garbage=4, deflate=True, clean=True)
        finally:
            doc.close()
        return buf.getvalue()
    _strip_metadata_doc(pdf_bytes_or_doc)
    return None


def _apply_redactions_doc(doc, matches: list[dict],
                          redact_label: str = "",
                          active_categories: list[str] | set[str] | None = None,
                          removed_pages: list[int] | None = None,
                          *,
                          output_protection: dict | None = None
                          ) -> tuple[bytes, bool]:
    """Apply redactions to an open ``fitz.Document`` and return (bytes, protected).

    Mutates ``doc`` in place: redaction annotations are applied, requested
    pages are deleted, metadata is stripped. Caller owns ``doc`` lifecycle —
    this helper does NOT close it. Used by both the bytes-IPC entry point
    and the v0.4.0 stateful handle protocol.
    """
    _validate_redaction_payload(matches, removed_pages)

    removed_set = set(removed_pages) if removed_pages else set()
    # `[]` means caller turned every category off — must NOT collapse to None.
    active_set = set(active_categories) if active_categories is not None else None
    by_page: dict[int, list[dict]] = {}
    for m in matches:
        if not m.get("enabled", True):
            continue
        if m["page"] in removed_set:
            continue
        if active_set is not None and m.get("type") not in active_set:
            if m.get("type") not in ("Manual Region", "Custom"):
                continue
        by_page.setdefault(m["page"], []).append(m)

    for pg_num, pg_matches in by_page.items():
        page = doc[pg_num]
        for m in pg_matches:
            r = m["rect"]
            rect = Rect(r["x0"], r["y0"], r["x1"], r["y1"])
            box_h = rect.height
            font_size = min(11, max(5, box_h * 0.7))
            if redact_label:
                page.add_redact_annot(
                    rect,
                    text=redact_label,
                    fontname="helv",
                    fontsize=font_size,
                    text_color=(1, 1, 1),
                    fill=(0, 0, 0),
                    align=TEXT_ALIGN_CENTER,
                )
            else:
                page.add_redact_annot(rect, fill=(0, 0, 0))
        page.apply_redactions()

    if removed_set:
        valid_removed = {p for p in removed_set if 0 <= p < len(doc)}
        if valid_removed and len(valid_removed) >= len(doc):
            raise ValueError("Cannot export: all pages have been removed")
        for pg_num in sorted(valid_removed, reverse=True):
            doc.delete_page(pg_num)

    _strip_metadata_doc(doc)

    buf = io.BytesIO()
    mode = (output_protection or {}).get("mode") if output_protection else None

    if output_protection is None or mode == "none":
        doc.save(buf, garbage=4, deflate=True, clean=True)
        protection_applied = True
    elif mode in ("same", "new"):
        password = output_protection.get("password") or ""
        if not password:
            doc.save(buf, garbage=4, deflate=True, clean=True)
            protection_applied = False
        else:
            try:
                doc.save(
                    buf,
                    garbage=4,
                    deflate=True,
                    clean=True,
                    encryption=PDF_ENCRYPT_AES_256,
                    user_pw=password,
                    owner_pw=password,
                    permissions=int(PDF_PERM_ACCESSIBILITY),
                )
                protection_applied = True
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Re-encryption failed; saving unprotected. exception_type=%s",
                    type(exc).__name__,
                )
                buf = io.BytesIO()
                doc.save(buf, garbage=4, deflate=True, clean=True)
                protection_applied = False
    else:
        logging.getLogger(__name__).warning(
            "apply_redactions: unknown output_protection mode; saving unprotected."
        )
        doc.save(buf, garbage=4, deflate=True, clean=True)
        protection_applied = False

    return buf.getvalue(), protection_applied


def apply_redactions(pdf_bytes: bytes, matches: list[dict],
                     redact_label: str = "",
                     active_categories: list[str] | set[str] | None = None,
                     removed_pages: list[int] | None = None,
                     *,
                     output_protection: dict | None = None
                     ) -> tuple[bytes, bool]:
    """Black-box redact enabled matches, return (new PDF bytes, protection_applied).

    Parameters
    ----------
    pdf_bytes
        Source PDF as bytes.
    matches
        List of ``{"page": int, "rect": {"x0": ..., "y0": ..., "x1": ...,
        "y1": ...}, "enabled": bool, "type": str}``. Disabled matches and
        matches whose ``type`` is not in ``active_categories`` are skipped
        (manual / custom regions are always included).
    redact_label
        Optional text to overlay on each black box ("REDACTED",
        "[b](6)", custom). Empty = plain black box.
    active_categories
        Optional set/list of category names. If provided, only matches
        whose ``type`` is in this set are redacted.
    removed_pages
        Optional list of page numbers to delete from the PDF.
    output_protection
        Optional ``{"mode": "same"|"new"|"none", "password": str?}``.
        Modes "same"/"new" require a non-empty password. Re-encryption
        failures fall back to unprotected output rather than blocking the
        download (``protection_applied=False`` signals this).

    Returns
    -------
    tuple[bytes, bool]
        ``(new_pdf_bytes, protection_applied)`` where ``protection_applied``
        means "the requested protection shipped":

        * ``output_protection=None``         -> always ``True``
        * ``{"mode": "none"}``               -> always ``True``
        * ``{"mode": "same"|"new", ...}``    -> ``True`` iff the encrypted
                                                save succeeded; ``False``
                                                on fallback.

    Raises
    ------
    ValueError
        On malformed wire payload (bad page index, non-numeric rect coords,
        inverted bounds).
    """
    doc = open_pdf(pdf_bytes)
    try:
        return _apply_redactions_doc(
            doc, matches,
            redact_label=redact_label,
            active_categories=active_categories,
            removed_pages=removed_pages,
            output_protection=output_protection,
        )
    finally:
        doc.close()
