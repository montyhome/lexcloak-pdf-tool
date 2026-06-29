"""Local PDF size reduction -- the finishing-step compressor.

A bytes-in / bytes-out op that shrinks a (typically already-redacted) PDF
without sending it anywhere. Every mainstream compressor (Smallpdf,
iLovePDF, Adobe online) *uploads* the file to the cloud; for a document
the user just redacted *because it is confidential*, that defeats the
point. ``reduce_size`` runs entirely in-process -- the same "nothing
leaves your machine" promise as the redaction itself.

Models on :mod:`lexcloak_pdf_tool.metadata` (``set_metadata``): open via
``open_pdf``, mutate the document, save with
``garbage=4, deflate=True, clean=True``.

Quality discipline (legal/medical fidelity matters -- fine print and
signatures must stay legible and admissible):

* **Lossless by default.** With ``dpi=None`` the op only scrubs orphan
  objects / metadata / thumbnails and subsets fonts. It never touches
  image resolution, so the rendered page is byte-faithful. The lossless
  gain is modest (redacted bytes are already saved garbage-collected and
  deflated upstream) but always safe.
* **Opt-in downsample.** ``rewrite_images`` runs only when a positive
  ``dpi`` is given, and only on images whose resolution *exceeds* the
  target -- already-small images are left alone.
* **No-grow guard.** PyMuPDF can occasionally emit a *larger* file
  (re-encode overhead on already-optimized streams; see PyMuPDF
  discussion #3645 "I got a bigger PDF"). If the candidate is not
  smaller than the input, the original bytes are returned unchanged --
  this op never makes a file bigger.
* **Cleartext only.** Recompressing image streams needs plaintext; the
  redaction route always compresses *before* any optional encryption.
  Encrypted/password-protected input is a programming error, rejected
  with ``ValueError`` rather than silently mangled.

The OCR/selectable text layer is preserved: ``scrub`` is called with
``hidden_text=False`` because a scanned redacted PDF carries its
searchable text as invisible (render-mode-3) text overlaid on the page
image -- ``scrub``'s default ``hidden_text=True`` would delete it.
"""
from __future__ import annotations

import io
import logging

from .redact import open_pdf

logger = logging.getLogger(__name__)


def _validate_reduce_params(dpi, quality) -> None:
    """Validate ``dpi`` / ``quality`` before touching the document.

    ``dpi`` is ``None`` (lossless) or a positive int. ``quality`` is an int
    in ``1..100``. Bools are rejected explicitly (``True``/``False`` are
    ints in Python and a wire payload could carry one). Raises ``ValueError``
    with a named-field message so the wire layer surfaces a clean error.
    """
    if dpi is not None:
        if isinstance(dpi, bool) or not isinstance(dpi, int):
            raise ValueError(
                f"dpi must be a positive int or None, got {type(dpi).__name__}"
            )
        if dpi <= 0:
            raise ValueError(f"dpi must be a positive int, got {dpi}")
    if isinstance(quality, bool) or not isinstance(quality, int):
        raise ValueError(
            f"quality must be an int in 1..100, got {type(quality).__name__}"
        )
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in 1..100, got {quality}")


# Lossless scrub flag-set: strip metadata / XMP / thumbnails / attachments /
# JavaScript / orphan objects, but change NOTHING the eye or a text extractor
# can observe. Two overrides are load-bearing for Lex Cloak:
#   * hidden_text=False -- the OCR text layer on a scanned redacted PDF is
#     invisible text (render mode 3) over the page image. scrub's default
#     (hidden_text=True) would delete it, destroying text selection and
#     extraction on exactly the documents this op targets.
#   * redactions=False  -- redactions are already burned in upstream by
#     apply_redactions; this op must never re-run redaction machinery on
#     finalized bytes. (scrub also forbids redactions=True / hidden_text=True
#     unless clean_pages=True; we deliberately want none of the three.)
# remove_links / reset_fields / reset_responses stay off so the op is
# strictly size-affecting, not content-affecting.
def _scrub_lossless(doc) -> None:
    doc.scrub(
        attached_files=True,
        embedded_files=True,
        javascript=True,
        metadata=True,
        xml_metadata=True,
        thumbnails=True,
        hidden_text=False,
        clean_pages=False,
        redactions=False,
        remove_links=False,
        reset_fields=False,
        reset_responses=False,
    )


def _apply_reductions(doc, *, dpi=None, quality=75, grayscale=False) -> int | None:
    """Mutate ``doc`` in place; return the DPI actually applied (or ``None``).

    Runs the lossless steps (scrub + ``subset_fonts``) always, then -- when
    ``dpi`` is given -- an image downsample. The two enhancement steps
    (``subset_fonts``, ``rewrite_images``) are best-effort: a failure in
    either is logged and skipped rather than failing the whole op, so a
    weird font or image can't cost the caller the lossless gain. A genuine
    ``save`` failure (handled by the caller) is the only hard error.

    ``rewrite_images`` uses ``dpi_threshold = dpi + 1`` because PyMuPDF
    requires ``dpi_target`` strictly below ``dpi_threshold``; the net effect
    is "cap every image at ``dpi``" while leaving images already at or below
    the target untouched.
    """
    _scrub_lossless(doc)
    try:
        doc.subset_fonts()
    except Exception as exc:  # noqa: BLE001 -- best-effort enhancement
        logger.warning("reduce_size: subset_fonts failed (non-fatal): %s", exc)
    if dpi is None:
        return None
    try:
        doc.rewrite_images(
            dpi_threshold=dpi + 1,
            dpi_target=dpi,
            quality=quality,
            set_to_gray=grayscale,
        )
    except Exception as exc:  # noqa: BLE001 -- fall back to lossless-only
        logger.warning(
            "reduce_size: image downsample failed, lossless only: %s", exc
        )
        return None
    return dpi


def reduce_size(
    pdf_bytes: bytes, *, dpi: int | None = None, quality: int = 75,
    grayscale: bool = False,
) -> tuple[bytes, dict]:
    """Compress ``pdf_bytes`` locally; return ``(out_bytes, info)``.

    ``info`` is ``{"orig_size", "new_size", "applied_dpi"}`` with truthful
    sizes. ``applied_dpi`` is the DPI that actually shaped the returned
    bytes: ``None`` for the lossless path, for a downsample that failed and
    fell back to lossless, or when the no-grow guard returned the original.

    Raises ``ValueError`` on a bad ``dpi``/``quality`` or on encrypted
    (password-protected) input.
    """
    _validate_reduce_params(dpi, quality)
    doc = open_pdf(pdf_bytes)
    try:
        if doc.is_encrypted and doc.needs_pass:
            raise ValueError("reduce_size() input must be cleartext")
        applied_dpi = _apply_reductions(
            doc, dpi=dpi, quality=quality, grayscale=grayscale
        )
        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True, clean=True)
        candidate = buf.getvalue()
    finally:
        doc.close()

    # No-grow guard -- never hand back something larger than we were given.
    if len(candidate) >= len(pdf_bytes):
        return pdf_bytes, {
            "orig_size": len(pdf_bytes),
            "new_size": len(pdf_bytes),
            "applied_dpi": None,
        }
    return candidate, {
        "orig_size": len(pdf_bytes),
        "new_size": len(candidate),
        "applied_dpi": applied_dpi,
    }
