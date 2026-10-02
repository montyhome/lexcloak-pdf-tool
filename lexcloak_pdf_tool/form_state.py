"""Keep a form XObject's text state through MuPDF's content filter (v0.14.1).

``Page.apply_redactions`` and ``Document.save(clean=True)`` both rewrite page
content through MuPDF's content filter, which writes a graphics-state
operator only when it is needed. Two of its shortcuts lose state a form
XObject depends on (pymupdf 1.28.2; MuPDF's ``pdf-op-filter.c`` is
unchanged on its main branch):

* **Text state set just before a form is never written.** The filter sends
  pending text state (``Tc Tw Tz Tr Ts``) only inside ``BT``, when text is
  shown. A ``0 Tr`` between an ``ET`` and a form's ``Do`` is dropped, so the
  form runs with whatever the page last sent.
* **A form is filtered as if it started from the default text state.** So a
  form's own ``0 Tr`` (or ``0 Tc``, ``100 Tz`` ...) reads as already in
  effect and is dropped too.

Either alone would be harmless; together they change what a form draws.
The common case: a page that draws invisible text (``3 Tr``, a searchable
scan's text layer) and then a visible stamp or footer in a form that resets
``0 Tr``. After a redaction on that page, or a clean save of the document,
the form inherits ``3 Tr`` and its text, still in the file, draws nothing.
Nothing is removed, so a check that compares text by position and wording
sees no change.

The repair makes the filter's assumption true. After the filter has run,
every ``Do`` of a form it filtered is wrapped as
``q <default text state> ... Do Q``, so the form starts from exactly the
state it was written for, and the ``Q`` restores the page's own state
afterwards. That is right only when no form shows text with text state it
inherits and does not set (a form that relies on its caller's ``3 Tr``, say):
the filter wrote such a form with nothing to restore. So each page is read
before the filter runs, and a page where any form shows text with inherited
non-default state is left exactly as MuPDF writes it, as before this module.
So is a page whose content cannot be read.

Two entry points: :func:`apply_page_redactions` for a page's redactions,
and :func:`save_document` for a save with ``clean=True``. A page that never
sets a non-default text state cannot be affected, and a document with no
such page saves exactly as it did.
"""
from __future__ import annotations

import io
import re

#: The text state MuPDF's filter assumes at the start of a form.
DEFAULTS = {"Tc": 0.0, "Tw": 0.0, "Tz": 100.0, "Tr": 0.0, "Ts": 0.0}
#: Sets every parameter in ``DEFAULTS``, inside a ``q`` the wrap closes.
PREFIX = b"q 0 Tc 0 Tw 100 Tz 0 Tr 0 Ts "
#: Forms nested deeper than this are not followed: the page is left alone.
MAX_DEPTH = 16

_TOKEN = re.compile(rb"""
    (?P<ws>[\x00\t\n\x0c\r ]+)
  | (?P<comment>%[^\r\n]*)
  | (?P<dict><<|>>)
  | (?P<hex><[^<>]*>)
  | (?P<arr>[\[\]{}])
  | (?P<name>/[^\x00\t\n\x0c\r ()<>\[\]{}/%]*)
  | (?P<str>\()
  | (?P<word>[^\x00\t\n\x0c\r ()<>\[\]{}/%]+)
""", re.X)
_NUMBER = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)")
_EI = re.compile(rb"[\x00\t\n\x0c\r ]EI(?=[\x00\t\n\x0c\r ]|$)")
_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
_REF = re.compile(rb"/([^\x00\t\n\x0c\r /<>\[\]()]+)\s+(\d+)\s+\d+\s+R")

_SETS = {b"Tc": "Tc", b"Tw": "Tw", b"Tz": "Tz", b"Tr": "Tr", b"Ts": "Ts"}
_SHOWS = (b"Tj", b"TJ", b"'", b'"')


class ContentError(ValueError):
    """A content stream that cannot be read. Carries no content."""


# ── Reading content ────────────────────────────────────────────────────────


def _string_end(data: bytes, i: int) -> int:
    """The index just past the literal string that opens at ``data[i]``."""
    depth, n = 0, len(data)
    while i < n:
        c = data[i]
        if c == 0x5C:                       # backslash: skip the next byte
            i += 2
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ContentError("unterminated string")


def operators(data: bytes):
    """``(operator, operands, start, end)`` for each operator in ``data``.

    ``operands`` are the ``(kind, bytes)`` tokens before the operator, flat
    (a dictionary operand is its delimiters, keys and values in order);
    ``start``/``end`` span the operator with its operands. An inline image
    (``BI ... ID ... EI``) is one ``BI`` operator. Raises ``ContentError``
    on bytes that are not a content stream.
    """
    i, n = 0, len(data)
    operands: list = []
    start = None
    while i < n:
        m = _TOKEN.match(data, i)
        if m is None:
            raise ContentError("unreadable token")
        kind = m.lastgroup
        if kind in ("ws", "comment"):
            i = m.end()
            continue
        if start is None:
            start = i
        if kind == "str":
            j = _string_end(data, i)
            operands.append(("str", data[i:j]))
            i = j
            continue
        j = m.end()
        tok = data[i:j]
        if kind == "word" and not _NUMBER.fullmatch(tok) and tok not in (
                b"true", b"false", b"null"):
            if tok == b"BI":
                k = data.find(b"ID", j)
                e = _EI.search(data, k + 2) if k >= 0 else None
                if e is None:
                    raise ContentError("unterminated inline image")
                j = e.end()
                yield b"BI", [], start, j
            else:
                yield tok, operands, start, j
            operands, start = [], None
            i = j
            continue
        operands.append((kind, tok))
        i = j


def _name(token: bytes) -> str:
    return _ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]),
                       token[1:]).decode("latin-1")


def _page_xobjects(page) -> dict[str, int]:
    return {name: xref for xref, name, invoker, _ in page.get_xobjects()
            if invoker == 0}


def _form_xobjects(doc, xref: int) -> dict[str, int]:
    kind, value = doc.xref_get_key(xref, "Resources/XObject")
    if kind == "xref":
        value = doc.xref_object(int(value.split()[0]), compressed=True)
    elif kind != "dict":
        return {}
    return {_name(b"/" + name): int(ref)
            for name, ref in _REF.findall(value.encode("latin-1"))}


def _is_form(doc, xref) -> bool:
    return (xref is not None and 0 < xref < doc.xref_length()
            and doc.xref_is_stream(xref)
            and doc.xref_get_key(xref, "Subtype")[1] == "/Form")


def _number(operand) -> float:
    if operand[0] != "word":
        raise ContentError("operand is not a number")
    return float(operand[1])


def _page_content(doc, page) -> bytes:
    return b"\n".join(doc.xref_stream(x) or b"" for x in page.get_contents())


# ── Before the filter: may this page be restored, and does it need to be? ──


class _Reading:
    """What one page's content (with every form it reaches) shows."""

    def __init__(self, doc):
        self.doc = doc
        self.safe = True        # no form shows text with inherited non-default state
        self.at_risk = False    # some stream sets a non-default text state
        self.forms: set[int] = set()

    def read(self, data: bytes, names: dict[str, int], state: dict,
             inherited: set, depth: int) -> None:
        saved = []
        for op, args, _, _ in operators(data):
            if op == b"q":
                saved.append((dict(state), set(inherited)))
            elif op == b"Q":
                if saved:
                    state, inherited = saved.pop()
            elif op in _SETS and args:
                self._set(state, inherited, _SETS[op], _number(args[-1]))
            elif op in _SHOWS:
                if op == b'"' and len(args) >= 3:
                    self._set(state, inherited, "Tw", _number(args[0]))
                    self._set(state, inherited, "Tc", _number(args[1]))
                if any(state[p] != DEFAULTS[p] for p in inherited):
                    self.safe = False
            elif op == b"Do" and args and args[-1][0] == "name":
                xref = names.get(_name(args[-1][1]))
                if not _is_form(self.doc, xref):
                    continue
                if depth >= MAX_DEPTH:
                    raise ContentError("forms nested too deep")
                self.forms.add(xref)
                self.read(self.doc.xref_stream(xref) or b"",
                          _form_xobjects(self.doc, xref), dict(state),
                          set(DEFAULTS), depth + 1)

    def _set(self, state: dict, inherited: set, param: str, value: float) -> None:
        state[param] = value
        inherited.discard(param)
        if value != DEFAULTS[param]:
            self.at_risk = True


def _has_forms(page) -> bool:
    doc = page.parent
    return any(_is_form(doc, xref) for xref, _, _, _ in page.get_xobjects())


def _read_page(doc, page) -> _Reading | None:
    """The page's reading, or None when it cannot be read (left alone)."""
    reading = _Reading(doc)
    try:
        reading.read(_page_content(doc, page), _page_xobjects(page),
                     dict(DEFAULTS), set(), 0)
    except (ContentError, ValueError, RuntimeError):
        return None
    return reading


# ── After the filter: wrap each filtered form's Do ─────────────────────────


def _wrap(doc, data: bytes, names: dict[str, int], filtered) -> bytes | None:
    """``data`` with every form ``Do`` wrapped in the default text state, or
    None when it invokes a form ``filtered(xref)`` refuses, or a ``Do`` sits
    inside ``BT``, where ``q`` is not allowed."""
    out, last, in_text = [], 0, False
    for op, args, start, end in operators(data):
        if op == b"BT":
            in_text = True
        elif op == b"ET":
            in_text = False
        elif op == b"Do" and args and args[-1][0] == "name":
            xref = names.get(_name(args[-1][1]))
            if not _is_form(doc, xref):
                continue
            if in_text or not filtered(xref):
                return None
            out += [data[last:start], PREFIX, data[start:end], b" Q"]
            last = end
    if not out:
        return data
    out.append(data[last:])
    return b"".join(out)


def _restore(doc, page, forms, filtered) -> bool:
    """Wrap the form ``Do``s in ``page``'s content and in ``forms``. Writes
    nothing unless every stream could be wrapped; True when it wrote."""
    try:
        page_data = _wrap(doc, _page_content(doc, page), _page_xobjects(page),
                          filtered)
        if page_data is None:
            return False
        wrapped = {}
        for xref in forms:
            data = _wrap(doc, doc.xref_stream(xref) or b"",
                         _form_xobjects(doc, xref), filtered)
            if data is None:
                return False
            wrapped[xref] = data
    except (ContentError, ValueError, RuntimeError):
        return False
    changed = False
    if page_data != _page_content(doc, page):
        contents = page.get_contents()
        doc.update_stream(contents[0], page_data)
        for xref in contents[1:]:
            doc.update_stream(xref, b"")
        changed = True
    for xref, data in wrapped.items():
        if data != (doc.xref_stream(xref) or b""):
            doc.update_stream(xref, data)
            changed = True
    return changed


def apply_page_redactions(page, **kwargs) -> bool:
    """``page.apply_redactions(**kwargs)``, then restore form text state.

    The redaction filters every form the page shows into a new instance, so
    exactly those (and the forms inside them) are wrapped; a ``Do`` of any
    other form leaves the page as MuPDF wrote it. Returns True when the
    page was wrapped. The redaction is applied either way, and whatever
    ``Page.apply_redactions`` raises is raised.
    """
    if not _has_forms(page):
        page.apply_redactions(**kwargs)
        return False
    doc = page.parent
    reading = _read_page(doc, page)
    first_new = doc.xref_length()
    page.apply_redactions(**kwargs)
    if reading is None or not reading.safe or not reading.at_risk:
        return False
    instances = {x for x, _, _, _ in page.get_xobjects()
                 if x >= first_new and _is_form(doc, x)}
    return _restore(doc, page, instances, lambda x: x >= first_new)


def save_document(doc, **kwargs) -> bytes:
    """``doc.save(**kwargs)`` to bytes, keeping form text state through a
    ``clean=True`` save.

    The clean save filters every page's content and forms in place in
    ``doc`` as it writes (and may renumber its objects). When a page needed
    restoring, its form ``Do``s are wrapped in ``doc`` afterwards and the
    document is saved again with the same arguments but ``clean=False``, so
    nothing is filtered twice. A form that a page which cannot be restored
    also shows keeps its own content as MuPDF wrote it.
    """
    restore: set[int] = set()
    blocked: set[int] = set()
    if kwargs.get("clean"):
        for page in doc:
            if not _has_forms(page):
                continue
            reading = _read_page(doc, page)
            if reading is None or not reading.safe:
                blocked.add(page.number)
            elif reading.at_risk:
                restore.add(page.number)
    buf = io.BytesIO()
    doc.save(buf, **kwargs)
    if not restore:
        return buf.getvalue()
    shown_by: dict[int, set[int]] = {}
    for page in doc:
        for xref, _, _, _ in page.get_xobjects():
            if _is_form(doc, xref):
                shown_by.setdefault(xref, set()).add(page.number)
    done: set[int] = set()
    changed = False
    for pno in sorted(restore):
        page = doc[pno]
        forms = {x for x, _, _, _ in page.get_xobjects()
                 if _is_form(doc, x) and x not in done
                 and not shown_by.get(x, set()) & blocked}
        if _restore(doc, page, forms, lambda x: True):
            changed = True
        done |= forms
    if not changed:
        return buf.getvalue()
    buf = io.BytesIO()
    doc.save(buf, **{**kwargs, "clean": False})
    return buf.getvalue()
