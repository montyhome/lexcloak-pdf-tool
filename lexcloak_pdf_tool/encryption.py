"""Password-protected PDF decryption + encryption.

Symmetric pair: ``decrypt_pdf`` authenticates a protected PDF and returns
cleartext bytes for in-memory storage; ``encrypt`` takes cleartext bytes and
returns AES-256 password-protected bytes. Together they bracket the Lex Cloak
pipeline -- decrypt on entry, encrypt on exit -- keeping every intermediate
step (redaction, Spec 13/14 metadata + cover page, size reduction) operating on
cleartext (Session 342).

Wrong-password on decrypt is an op-level error -- the caller can retry with a
different password without restarting the process.
"""
from __future__ import annotations

import io

from .redact import _save_encrypted, open_pdf


class WrongPasswordError(Exception):
    """Raised when ``decrypt_pdf`` cannot authenticate the supplied password.

    The CLI dispatch loop reflects ``type(exc).__name__`` into the
    response's ``error_type`` field -- clients pattern-match on
    ``"WrongPasswordError"`` to surface a retry UI.
    """


def decrypt_pdf(pdf_bytes: bytes, password: str) -> tuple[bytes, int]:
    """Authenticate ``password`` and return ``(cleartext_pdf_bytes, page_count)``.

    Raises :class:`WrongPasswordError` if the password fails. On success,
    re-saves with ``garbage=4, deflate=True, clean=True`` to produce a
    clean unencrypted copy. If the input was never encrypted, the password
    is ignored and a clean re-save is returned.
    """
    doc = open_pdf(pdf_bytes)
    try:
        if doc.is_encrypted:
            # PyMuPDF returns 0 on auth failure, 1 (user_pw), 2 (owner_pw),
            # or 4 (no password needed). Treat 0 alone as the wrong-password
            # case; everything else means the doc is now unlocked.
            auth_result = doc.authenticate(password)
            if auth_result == 0:
                raise WrongPasswordError(
                    "Authentication failed: the supplied password is incorrect."
                )
        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True, clean=True)
        return buf.getvalue(), len(doc)
    finally:
        doc.close()


def encrypt(pdf_bytes: bytes, password: str) -> tuple[bytes, bool]:
    """AES-256 encrypt cleartext ``pdf_bytes`` under ``password``.

    Returns ``(out_bytes, protection_applied)``:

    * **Empty / missing password** -> ``(pdf_bytes, False)``. A deliberate
      no-op that returns the input *unchanged* and signals "protection was
      skipped" -- mirrors ``apply_redactions``'s empty-password branch. The
      caller surfaces ``protection_applied=False`` to the user.
    * **Valid password** -> ``(encrypted_bytes, True)`` on success, or
      ``(unencrypted_bytes, False)`` if the PyMuPDF encrypted save fails (the
      fallback is shared with ``apply_redactions`` via ``_save_encrypted`` so
      the two paths cannot drift). A failed encryption never blocks the
      download.

    Raises
    ------
    ValueError
        If ``password`` is not a string, or if the input is itself encrypted
        (``is_encrypted and needs_pass``). The Lex Cloak pipeline only ever
        feeds cleartext into ``encrypt`` -- an already-encrypted input is a
        programming error, not a re-encryption request. Use ``decrypt_pdf``
        to authenticate first. (Mirrors ``reduce_size``'s cleartext-only
        invariant.)
    """
    if not isinstance(password, str):
        raise ValueError(
            f"password must be a string, got {type(password).__name__}"
        )
    # Fast no-op: return the caller's exact bytes, byte-for-byte. Routing an
    # empty password through a re-save would needlessly rewrite the stream.
    if not password:
        return pdf_bytes, False
    doc = open_pdf(pdf_bytes)
    try:
        if doc.is_encrypted and doc.needs_pass:
            raise ValueError(
                "encrypt() input must be cleartext; use decrypt_pdf() to "
                "authenticate first"
            )
        return _save_encrypted(doc, password)
    finally:
        doc.close()
