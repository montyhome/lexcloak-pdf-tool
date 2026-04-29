"""Password-protected PDF decryption.

Authenticates a password-protected PDF and returns cleartext bytes for
in-memory storage. Wrong-password is an op-level error -- the caller can
retry with a different password without restarting the process.
"""
from __future__ import annotations

import io

from .redact import open_pdf


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
