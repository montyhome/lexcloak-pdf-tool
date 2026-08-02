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

from .redact import null_page_thumbnails, open_pdf

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


# The writable PDF metadata keys -- the legal values for
# ``preserve_metadata``. This is a KNOWN-KEY check, NOT a safety allowlist.
# Which keys are safe to keep is the CALLER's decision and this package
# cannot make it: ``producer`` holding "Lex Cloak 1.8.19" is a deliberate
# post-redaction marking, ``producer`` holding "HP Scanner 4.2" is a
# fingerprint leak -- same key, same type, opposite meanings, and only the
# caller knows which one the document is carrying. What the check DOES buy
# is typo rejection ("Subject", "subj"), because a misspelled key is a
# silent no-op and a silently-dropped marking is the exact failure this
# parameter exists to prevent. The safe default does the real work here and
# is unchanged: preserve nothing unless explicitly asked.
# ``format`` / ``encryption`` are excluded -- derived, not settable.
_PRESERVABLE_METADATA_KEYS = frozenset({
    "title", "author", "subject", "keywords", "creator", "producer",
    "creationDate", "modDate",
})


def _validate_preserve_metadata(preserve_metadata) -> frozenset[str]:
    """Normalize + validate ``preserve_metadata`` into a set of keys.

    ``None`` (the default) yields the empty set -- historical behavior, no
    metadata survives the scrub. Anything else must be a list/tuple of keys
    drawn from :data:`_PRESERVABLE_METADATA_KEYS`; a str is rejected outright
    rather than silently iterated into characters.
    """
    if preserve_metadata is None:
        return frozenset()
    if isinstance(preserve_metadata, str) or not isinstance(
        preserve_metadata, (list, tuple)
    ):
        raise ValueError(
            "preserve_metadata must be a list or tuple of keys, got "
            f"{type(preserve_metadata).__name__}"
        )
    keys = frozenset(preserve_metadata)
    unknown = keys - _PRESERVABLE_METADATA_KEYS
    if unknown:
        raise ValueError(
            "preserve_metadata contains unknown metadata key(s): "
            f"{sorted(unknown)}; known: {sorted(_PRESERVABLE_METADATA_KEYS)}"
        )
    return keys


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
#
# ``thumbnails=True`` is passed for intent but does NOT work on this
# flag-set: scrub's page loop early-``continue``s on
# ``not (clean_pages or hidden_text)`` before reaching its thumbnail branch,
# so this op silently never stripped thumbnails despite the module docstring
# saying it did (measured 2026-08-02, v0.6.6). ``null_page_thumbnails``
# below does it for real.
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
    null_page_thumbnails(doc)


def _apply_reductions(doc, *, dpi=None, quality=75, grayscale=False,
                      preserve_metadata=None) -> int | None:
    """Mutate ``doc`` in place; return the DPI actually applied (or ``None``).

    ``preserve_metadata`` lives HERE rather than in :func:`reduce_size` on
    purpose: the v4 stateful-handle op (``reduce_size_h``) bypasses
    ``reduce_size`` entirely and calls this helper directly, so preserving
    the stamp one level up would silently not apply on that path. Anything
    that scrubs goes through this function.

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
    keep_keys = _validate_preserve_metadata(preserve_metadata)
    # Snapshot BEFORE the scrub -- _scrub_lossless wipes the whole dict.
    preserved = {
        k: v for k, v in (doc.metadata or {}).items() if k in keep_keys and v
    } if keep_keys else {}

    _scrub_lossless(doc)

    if preserved:
        doc.set_metadata(preserved)
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
    preserve_metadata: tuple[str, ...] | list[str] | None = None,
) -> tuple[bytes, dict]:
    """Compress ``pdf_bytes`` locally; return ``(out_bytes, info)``.

    ``info`` is ``{"orig_size", "new_size", "applied_dpi"}`` with truthful
    sizes. ``applied_dpi`` is the DPI that actually shaped the returned
    bytes: ``None`` for the lossless path, for a downsample that failed and
    fell back to lossless, or when the no-grow guard returned the original.

    ``preserve_metadata`` (v0.6.6, keyword-only, default ``None`` = historical
    behavior) names metadata keys to carry across the scrub. The lossless
    scrub sets ``metadata=True``, which wipes the whole dict -- including any
    marking the CALLER applied after redacting. Lex Cloak's Spec-13
    "Auto-redacted" notice lives in ``subject`` and was silently erased by
    every successful compression before this flag existed; the caller now
    passes ``preserve_metadata=("subject",)`` to keep it.

    Opt-in rather than default-on because ``reduce_size`` is a general op:
    flipping the default would start preserving arbitrary third-party
    document metadata that callers currently rely on this op to strip. Same
    shape as v0.6.5's ``numeric_token_boundary`` -- and for the same reason
    v0.6.4 was superseded, which applied its change unconditionally.

    Keys are validated against :data:`_PRESERVABLE_METADATA_KEYS`, which is
    a known-key check and not a safety allowlist -- see the comment there.
    **Whether a given key is safe to keep is the caller's decision**: this
    op cannot distinguish a deliberate post-redaction marking from the
    original document's fingerprint, because they are the same key holding
    different strings. Preserve the narrowest set that carries your marking.

    Raises ``ValueError`` on a bad ``dpi``/``quality``, an unknown
    ``preserve_metadata`` key, or on encrypted (password-protected) input.
    """
    _validate_reduce_params(dpi, quality)
    # Validate early so a bad key fails before any document work; the
    # authoritative snapshot/restore lives in _apply_reductions.
    _validate_preserve_metadata(preserve_metadata)
    doc = open_pdf(pdf_bytes)
    try:
        if doc.is_encrypted and doc.needs_pass:
            raise ValueError("reduce_size() input must be cleartext")
        applied_dpi = _apply_reductions(
            doc, dpi=dpi, quality=quality, grayscale=grayscale,
            preserve_metadata=preserve_metadata,
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
