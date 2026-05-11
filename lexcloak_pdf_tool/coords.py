"""CharData coordinate search + serialization.

CharData shapes
---------------
The package uses two CharData representations:

* **In-memory tuple form** -- ``list[(char_str, bbox|None)]`` where ``bbox``
  is ``(x0, y0, x1, y1)`` or ``None`` to mark a line break. This is what
  the Tesseract subprocess pipeline produces (see ``ocr.py``).

* **JSON-friendly flat form** -- ``list[(char_str, x0|None, y0|None,
  x1|None, y1|None)]``. Used over the IPC protocol so JSON serialization
  is trivial (no nested null tuples). Line breaks are still
  ``(char, None, None, None, None)``.

:func:`serialize_chardata` and :func:`deserialize_chardata` convert between
them. The CLI speaks the flat form on the wire.

The chardata search functions (:func:`search_in_chars`,
:func:`search_whole_word_in_chars`, :func:`split_search_in_chars`) return
:class:`_RectTuple` objects -- a fitz-free shape that mirrors
``fitz.Rect``'s accessor API (``x0/y0/x1/y1/width/height``).
"""
from __future__ import annotations

import re


# Maximum rect height -- caps oversized rects to prevent multi-line spans
# from creating giant redaction bars.
_MAX_OCR_LINE_H = 30.0


class _RectTuple:
    """Tuple-shaped rectangle compatible with ``fitz.Rect`` accessors.

    Used as the return shape of :func:`search_in_chars`,
    :func:`search_whole_word_in_chars`, and :func:`split_search_in_chars`.
    Callers that did ``r.x0`` / ``r.y0`` / ``r.x1`` / ``r.y1`` /
    ``r.width`` / ``r.height`` keep working unchanged. Iteration yields
    ``(x0, y0, x1, y1)`` for tuple unpacking.
    """

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def __iter__(self):
        return iter((self.x0, self.y0, self.x1, self.y1))

    def __eq__(self, other):
        if isinstance(other, _RectTuple):
            return (self.x0, self.y0, self.x1, self.y1) == (
                other.x0, other.y0, other.x1, other.y1)
        if isinstance(other, tuple) and len(other) == 4:
            return (self.x0, self.y0, self.x1, self.y1) == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"_RectTuple({self.x0}, {self.y0}, {self.x1}, {self.y1})"


# ── CharData search ──────────────────────────────────────────────────


def search_in_chars(needle: str, chars: list) -> list:
    """Search for all occurrences of ``needle`` in character-position data.

    Replicates PyMuPDF's ``page.search_for()`` behavior using pre-extracted
    character data. Returns capped :class:`_RectTuple` objects (height
    clamped to ``_MAX_OCR_LINE_H``).

    Whitespace normalization: line-break markers (bbox=None) are matched as
    single spaces, mirroring PyMuPDF's behavior.
    """
    if not chars or not needle:
        return []

    text_parts: list[str] = []
    char_map: list[tuple[str, tuple | None]] = []
    for c, bbox in chars:
        if bbox is None:
            text_parts.append(" ")
            char_map.append((" ", None))
        else:
            text_parts.append(c)
            char_map.append((c, bbox))

    full_text = "".join(text_parts)
    needle_lower = needle.lower()
    text_lower = full_text.lower()

    rects: list = []
    start = 0
    while True:
        idx = text_lower.find(needle_lower, start)
        if idx == -1:
            break

        match_bboxes = []
        for i in range(idx, min(idx + len(needle), len(char_map))):
            _, bbox = char_map[i]
            if bbox is not None:
                match_bboxes.append(bbox)

        if match_bboxes:
            line_groups = _group_by_line(match_bboxes)
            for group in line_groups:
                x0 = min(bb[0] for bb in group)
                y0 = min(bb[1] for bb in group)
                x1 = max(bb[2] for bb in group)
                y1 = max(bb[3] for bb in group)
                r = _RectTuple(x0, y0, x1, y1)
                if r.height > _MAX_OCR_LINE_H:
                    r = _RectTuple(r.x0, r.y0, r.x1,
                                   r.y0 + _MAX_OCR_LINE_H)
                rects.append(r)

        start = idx + 1

    return rects


def search_whole_word_in_chars(needle: str, chars: list) -> list:
    """Whole-word search in character data -- filters out substring hits.

    Equivalent to a word-boundary-respecting search using pre-extracted
    chars. Word-boundary semantics match the ``\\w`` regex class --
    rejects matches whose preceding or following character is
    alphanumeric or underscore.
    """
    if not chars or not needle:
        return []

    text_parts: list[str] = []
    char_map: list[tuple[str, tuple | None]] = []
    for c, bbox in chars:
        if bbox is None:
            text_parts.append(" ")
            char_map.append((" ", None))
        else:
            text_parts.append(c)
            char_map.append((c, bbox))

    full_text = "".join(text_parts)
    needle_lower = needle.lower()
    text_lower = full_text.lower()
    needle_len = len(needle_lower)

    rects: list = []
    start = 0
    while True:
        idx = text_lower.find(needle_lower, start)
        if idx == -1:
            break

        # Verification reads ``full_text`` directly so OCR spaces (which
        # arrive as ``bbox=None`` line-break markers, rendered as " "
        # above) show up between adjacent words. A bbox-proximity approach
        # concatenates adjacent-word chars without a separator and
        # falsely rejects valid matches like "Susan" in "Name: Susan R."
        # because the surrounding string reads "...name:susanr..." with
        # no space between needle and next-word.
        before_ok = idx == 0 or not (
            text_lower[idx - 1].isalnum() or text_lower[idx - 1] == "_"
        )
        after_idx = idx + needle_len
        after_ok = after_idx >= len(text_lower) or not (
            text_lower[after_idx].isalnum() or text_lower[after_idx] == "_"
        )
        if not (before_ok and after_ok):
            start = idx + 1
            continue

        match_bboxes = []
        for i in range(idx, min(idx + needle_len, len(char_map))):
            _, bbox = char_map[i]
            if bbox is not None:
                match_bboxes.append(bbox)

        if match_bboxes:
            line_groups = _group_by_line(match_bboxes)
            for group in line_groups:
                x0 = min(bb[0] for bb in group)
                y0 = min(bb[1] for bb in group)
                x1 = max(bb[2] for bb in group)
                y1 = max(bb[3] for bb in group)
                r = _RectTuple(x0, y0, x1, y1)
                if r.height > _MAX_OCR_LINE_H:
                    r = _RectTuple(r.x0, r.y0, r.x1,
                                   r.y0 + _MAX_OCR_LINE_H)
                rects.append(r)

        start = idx + 1

    return rects


def split_search_in_chars(text: str, chars: list) -> list:
    """Split-search for long strings -- head + tail union.

    Splits the input on whitespace, ``@``, and ``,``, takes the first and
    last meaningful tokens (>=3 chars), searches each, and combines hits
    on the same baseline. Useful for composite strings where punctuation
    or whitespace might be OCR-noisy ("Smith, Jane, MD").
    """
    parts = re.split(r'[\s@,]+', text)
    parts = [p for p in parts if len(p) >= 3]
    if not parts:
        return []

    head = parts[0]
    tail = parts[-1] if len(parts) > 1 else None

    head_rects = search_in_chars(head, chars)
    if not head_rects:
        return []

    if tail and tail != head:
        tail_rects = search_in_chars(tail, chars)
        if tail_rects:
            combined: list = []
            for hr in head_rects:
                for tr in tail_rects:
                    h_mid_y = (hr.y0 + hr.y1) / 2
                    t_mid_y = (tr.y0 + tr.y1) / 2
                    if abs(h_mid_y - t_mid_y) < 5 and tr.x1 > hr.x0:
                        combined.append(_RectTuple(
                            min(hr.x0, tr.x0),
                            min(hr.y0, tr.y0),
                            max(hr.x1, tr.x1),
                            max(hr.y1, tr.y1),
                        ))
                        break
            if combined:
                return combined

    return head_rects


def _group_by_line(bboxes: list, tolerance: float = 5.0) -> list[list]:
    """Group bounding boxes by approximate y-position (line)."""
    if not bboxes:
        return []
    sorted_bbs = sorted(bboxes, key=lambda bb: bb[1])
    groups = [[sorted_bbs[0]]]
    for bb in sorted_bbs[1:]:
        if abs(bb[1] - groups[-1][0][1]) < tolerance:
            groups[-1].append(bb)
        else:
            groups.append([bb])
    return groups


# ── CharData serialization ───────────────────────────────────────────


def serialize_chardata(chardata: list) -> list:
    """Normalize CharData to JSON-friendly flat 5-tuple form.

    Accepts:

    * **In-memory 2-tuple form** -- ``list[(char, bbox|None)]`` where
      ``bbox`` is ``(x0, y0, x1, y1)`` or ``None``.
    * **Flat 5-tuple form** -- already serialized; passes through unchanged
      (idempotent).

    Returns ``list[(char, x0|None, y0|None, x1|None, y1|None)]``. Line
    breaks are encoded as ``(char, None, None, None, None)``.

    The IPC protocol carries this form -- no nested nulls means trivial
    JSON round-trip.
    """
    if not chardata:
        return []
    out: list = []
    for entry in chardata:
        if len(entry) == 2:
            char, bbox = entry
            if bbox is None:
                out.append((char, None, None, None, None))
            else:
                out.append((char, bbox[0], bbox[1], bbox[2], bbox[3]))
        elif len(entry) == 5:
            out.append(tuple(entry))
        else:
            raise ValueError(
                f"serialize_chardata: unexpected entry shape (len={len(entry)})"
            )
    return out


def deserialize_chardata(chardata: list) -> list:
    """Convert flat 5-tuple form back to in-memory 2-tuple ``(char, bbox|None)``.

    Idempotent on the 2-tuple form.
    """
    if not chardata:
        return []
    out: list = []
    for entry in chardata:
        if len(entry) == 2:
            out.append((entry[0], entry[1]))
        elif len(entry) == 5:
            char, x0, y0, x1, y1 = entry
            if x0 is None or y0 is None or x1 is None or y1 is None:
                out.append((char, None))
            else:
                out.append((char, (x0, y0, x1, y1)))
        else:
            raise ValueError(
                f"deserialize_chardata: unexpected entry shape (len={len(entry)})"
            )
    return out
