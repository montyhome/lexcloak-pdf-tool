"""Read and rewrite strings a PDF carries outside its page content (v0.13.0).

A reader shows or follows these without any of them being drawn on a page:

* **outline** -- bookmark titles;
* **destination** -- named destinations (the ``/Names`` ``/Dests`` tree and the
  older catalog ``/Dests`` dictionary);
* **page_label** -- page-label prefixes;
* **layer** -- optional-content group names;
* **link** -- the address of a URI link, on a page or on a bookmark;
* **tag** -- ``/Alt`` and ``/ActualText`` on structure elements.

A redaction that burns a value from the page leaves every copy of it in these
places. This module lists them for a caller to judge and rewrites the ones the
caller names. It never decides what a string means.

Identifiers. Every entry carries an ``id`` that is stable for one document's
bytes: :func:`list_side_text` and :func:`rewrite_side_text` must see the same
source. ``apply_redactions`` applies a rewrite before any page operation and
before the save that renumbers objects, so an id never has to outlive a burn.
A caller checking a delivered file lists it again; it does not reuse ids.

Fail-safe in the same sense as ``sanitise``: a structure that cannot be read
is counted in ``unreadable`` and left as it was, and a rewrite that cannot be
applied is reported in ``skipped`` and changes nothing. The walk never calls
``Document.resolve_names``, which was measured crashing the process on a real
document; the name tree is walked object by object instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pymupdf import mupdf as _m

KINDS = ("outline", "destination", "page_label", "layer", "link", "tag")

#: Kinds whose text can be replaced, and kinds that can be removed outright.
TEXT_KINDS = frozenset({"outline", "destination", "page_label", "layer", "tag"})
REMOVE_KINDS = frozenset({"link", "tag"})

#: Structure keys that hold alternate text.
TAG_KEYS = ("Alt", "ActualText")

#: Bounds on any one walk, so a hostile file cannot make a list unbounded.
MAX_ENTRIES = 200_000
MAX_DEPTH = 256


# ── low-level helpers ───────────────────────────────────────────────────


def _present(obj) -> bool:
    return bool(obj.m_internal) and not _m.pdf_is_null(obj)


def _get(obj, key: str):
    return _m.pdf_dict_gets(obj, key)


def _num(obj) -> int:
    return _m.pdf_to_num(obj) if _m.pdf_is_indirect(obj) else 0


def _is_name(obj, name: str) -> bool:
    return _m.pdf_is_name(obj) and _m.pdf_to_name(obj) == name


def _text(obj) -> str:
    return _m.pdf_to_text_string(obj)


@dataclass
class _Walk:
    """What one listing found: entries, where each one lives, unread parts."""
    entries: list[dict] = field(default_factory=list)
    where: dict[str, tuple] = field(default_factory=dict)
    unreadable: int = 0

    def add(self, entry_id: str, kind: str, text: str, page, locator: tuple):
        if len(self.entries) >= MAX_ENTRIES:
            self.unreadable += 1
            return
        self.entries.append({"id": entry_id, "kind": kind, "text": text,
                             "page": page})
        self.where[entry_id] = locator


def _page_index(doc) -> dict[int, int]:
    return {doc[i].xref: i for i in range(len(doc))}


def _dest_page(dest, pages: dict[int, int]):
    """The page an explicit destination array points at, else ``None``."""
    if _m.pdf_is_array(dest) and _m.pdf_array_len(dest) > 0:
        return pages.get(_num(_m.pdf_array_get(dest, 0)))
    return None


def _uri_of(action):
    """The URI string object of a URI action dictionary, else ``None``."""
    if not _m.pdf_is_dict(action) or not _is_name(_get(action, "S"), "URI"):
        return None
    uri = _get(action, "URI")
    return uri if _m.pdf_is_string(uri) else None


# ── the walks ───────────────────────────────────────────────────────────


def _walk_outline(root, pages, walk: _Walk) -> None:
    outlines = _get(root, "Outlines")
    if not _m.pdf_is_dict(outlines):
        return
    stack, seen = [_get(outlines, "First")], set()
    while stack:
        item = stack.pop()
        if not _present(item):
            continue
        num = _num(item)
        if not num or not _m.pdf_is_dict(item):
            walk.unreadable += 1            # outline items are indirect dicts
            continue
        if num in seen:
            continue                        # a cycle: already listed
        seen.add(num)
        title = _get(item, "Title")
        dest = _get(item, "Dest")
        action = _get(item, "A")
        if not _present(dest) and _m.pdf_is_dict(action) \
                and _is_name(_get(action, "S"), "GoTo"):
            dest = _get(action, "D")
        if _m.pdf_is_string(title):
            walk.add(f"outline:{num}", "outline", _text(title),
                     _dest_page(dest, pages), ("outline", item))
        elif _present(title):
            walk.unreadable += 1
        uri = _uri_of(action)
        if uri is not None:
            walk.add(f"outline-link:{num}", "link", _text(uri),
                     _dest_page(dest, pages), ("outline-link", item))
        stack.append(_get(item, "Next"))
        stack.append(_get(item, "First"))


def _walk_name_tree(node, kind: str, prefix: str, walk: _Walk, pages,
                    key_array: str, on_value=None) -> None:
    """Walk a name or number tree, listing each leaf entry ``on_value`` names.

    ``key_array`` is ``Names`` for a name tree and ``Nums`` for a number tree.
    """
    stack, seen = [(node, 0)], set()
    while stack:
        cur, depth = stack.pop()
        if not _m.pdf_is_dict(cur) or depth > MAX_DEPTH:
            if _present(cur):
                walk.unreadable += 1
            continue
        num = _num(cur)
        if num:
            if num in seen:
                continue
            seen.add(num)
        kids = _get(cur, "Kids")
        if _m.pdf_is_array(kids):
            for i in reversed(range(_m.pdf_array_len(kids))):
                stack.append((_m.pdf_array_get(kids, i), depth + 1))
        items = _get(cur, key_array)
        if not _present(items):
            continue
        if not _m.pdf_is_array(items) or _m.pdf_array_len(items) % 2:
            walk.unreadable += 1
            continue
        holder = num or "r"
        for i in range(0, _m.pdf_array_len(items), 2):
            on_value(f"{prefix}:{holder}:{i}", items, i)


def _walk_destinations(root, pages, walk: _Walk) -> None:
    names = _get(root, "Names")
    tree = _get(names, "Dests") if _m.pdf_is_dict(names) else None

    def on_value(entry_id, items, i):
        key = _m.pdf_array_get(items, i)
        if not _m.pdf_is_string(key):
            walk.unreadable += 1
            return
        value = _m.pdf_array_get(items, i + 1)
        if _m.pdf_is_dict(value):
            value = _get(value, "D")
        walk.add(entry_id, "destination", _text(key), _dest_page(value, pages),
                 ("dest-tree", items, i))

    if tree is not None and _present(tree):
        _walk_name_tree(tree, "destination", "destination", walk, pages,
                        "Names", on_value)
    legacy = _get(root, "Dests")
    if _m.pdf_is_dict(legacy):
        for i in range(_m.pdf_dict_len(legacy)):
            key = _m.pdf_dict_get_key(legacy, i)
            value = _m.pdf_dict_get_val(legacy, i)
            if _m.pdf_is_dict(value):
                value = _get(value, "D")
            walk.add(f"destination:dict:{i}", "destination", _m.pdf_to_name(key),
                     _dest_page(value, pages), ("dest-dict", legacy, i))


def _walk_page_labels(root, pages, walk: _Walk) -> None:
    tree = _get(root, "PageLabels")

    def on_value(entry_id, items, i):
        start = _m.pdf_array_get(items, i)
        label = _m.pdf_array_get(items, i + 1)
        if not _m.pdf_is_dict(label):
            walk.unreadable += 1
            return
        prefix = _get(label, "P")
        if _m.pdf_is_string(prefix):
            page = _m.pdf_to_int(start) if _m.pdf_is_int(start) else None
            walk.add(entry_id, "page_label", _text(prefix), page,
                     ("label", label))
        elif _present(prefix):
            walk.unreadable += 1

    if _m.pdf_is_dict(tree):
        _walk_name_tree(tree, "page_label", "page_label", walk, pages, "Nums",
                        on_value)


def _walk_layers(root, walk: _Walk) -> None:
    props = _get(root, "OCProperties")
    ocgs = _get(props, "OCGs") if _m.pdf_is_dict(props) else None
    if ocgs is None or not _m.pdf_is_array(ocgs):
        return
    seen = set()
    for i in range(_m.pdf_array_len(ocgs)):
        ocg = _m.pdf_array_get(ocgs, i)
        num = _num(ocg)
        if not num or num in seen or not _m.pdf_is_dict(ocg):
            if not num and _present(ocg):
                walk.unreadable += 1
            continue
        seen.add(num)
        name = _get(ocg, "Name")
        if _m.pdf_is_string(name):
            walk.add(f"layer:{num}", "layer", _text(name), None, ("layer", ocg))
        elif _present(name):
            walk.unreadable += 1


def _walk_links(doc, walk: _Walk) -> None:
    pdf = _m.pdf_specifics(doc.this)
    for pno in range(len(doc)):
        page_obj = _m.pdf_lookup_page_obj(pdf, pno)
        annots = _get(page_obj, "Annots")
        if not _m.pdf_is_array(annots):
            continue
        for i in range(_m.pdf_array_len(annots)):
            annot = _m.pdf_array_get(annots, i)
            if not _m.pdf_is_dict(annot) \
                    or not _is_name(_get(annot, "Subtype"), "Link"):
                continue
            uri = _uri_of(_get(annot, "A"))
            num = _num(annot)
            if uri is None:
                continue
            if not num:
                walk.unreadable += 1        # a link must be removable by number
                continue
            walk.add(f"link:{num}", "link", _text(uri), pno, ("link", pno, num))


def _walk_tags(root, pages, walk: _Walk) -> None:
    tree = _get(root, "StructTreeRoot")
    if not _m.pdf_is_dict(tree):
        return
    stack, seen, direct = [(_get(tree, "K"), None, 0)], set(), 0
    while stack:
        node, page, depth = stack.pop()
        if depth > MAX_DEPTH:
            walk.unreadable += 1
            continue
        if _m.pdf_is_array(node):
            for i in reversed(range(_m.pdf_array_len(node))):
                stack.append((_m.pdf_array_get(node, i), page, depth + 1))
            continue
        if not _m.pdf_is_dict(node):
            continue                            # a marked-content id
        num = _num(node)
        if num:
            if num in seen:
                continue
            seen.add(num)
            label = str(num)
        else:
            direct += 1
            label = f"d{direct}"
        pg = _get(node, "Pg")
        if _m.pdf_is_indirect(pg):
            page = pages.get(_num(pg), page)
        if _present(_get(node, "S")):           # a structure element
            for key in TAG_KEYS:
                value = _get(node, key)
                if _m.pdf_is_string(value):
                    walk.add(f"tag:{label}:{key}", "tag", _text(value), page,
                             ("tag", node, key))
                elif _present(value):
                    walk.unreadable += 1
        kids = _get(node, "K")
        if _present(kids):
            stack.append((kids, page, depth + 1))


def _walk(doc) -> _Walk:
    walk = _Walk()
    pdf = _m.pdf_specifics(doc.this)
    if not pdf.m_internal:
        return walk
    root = _get(_m.pdf_trailer(pdf), "Root")
    if not _m.pdf_is_dict(root):
        walk.unreadable += 1
        return walk
    pages = _page_index(doc)
    for step in (lambda: _walk_outline(root, pages, walk),
                 lambda: _walk_destinations(root, pages, walk),
                 lambda: _walk_page_labels(root, pages, walk),
                 lambda: _walk_layers(root, walk),
                 lambda: _walk_links(doc, walk),
                 lambda: _walk_tags(root, pages, walk)):
        try:
            step()
        except Exception:  # noqa: BLE001 -- one unreadable structure, not all
            walk.unreadable += 1
    return walk


# ── public entry points ─────────────────────────────────────────────────


def list_side_text(doc) -> dict:
    """Every string listed above, as ``{"entries": [...], "unreadable": n}``.

    Each entry is ``{"id", "kind", "text", "page"}``; ``page`` is the 0-based
    page the string belongs to or points at, or ``None``. ``unreadable``
    counts structures that exist but could not be read, so a caller can tell
    "nothing here" from "could not look".
    """
    walk = _walk(doc)
    return {"entries": walk.entries, "unreadable": walk.unreadable}


def list_side_text_pdf(pdf_bytes: bytes) -> dict:
    """:func:`list_side_text` over PDF bytes (the IPC-clean form)."""
    import pymupdf
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ValueError("list_side_text needs an unencrypted document")
        return list_side_text(doc)
    finally:
        doc.close()


def validate_side_text_edits(edits) -> list[dict]:
    """The wire shape of a rewrite request, checked. Raises ``ValueError``.

    A list of ``{"id": str, "text": str}`` or ``{"id": str, "remove": true}``.
    """
    if not isinstance(edits, list):
        raise ValueError("side_text must be a list")
    out = []
    for edit in edits:
        if not isinstance(edit, dict) or not isinstance(edit.get("id"), str):
            raise ValueError("each side_text edit needs a string id")
        has_text = "text" in edit
        remove = edit.get("remove", False)
        if not isinstance(remove, bool):
            raise ValueError("side_text remove must be a boolean")
        if has_text == remove:
            raise ValueError("each side_text edit needs exactly one of text or remove")
        if has_text and not isinstance(edit["text"], str):
            raise ValueError("side_text text must be a string")
        out.append({"id": edit["id"], "text": edit.get("text"), "remove": remove})
    return out


def rewrite_side_text(doc, edits: list[dict]) -> dict:
    """Apply validated ``edits`` to ``doc`` in place.

    Returns ``{"applied": n, "skipped": [id, ...]}``. An id that does not
    resolve, an action its kind cannot take, and a destination rename that
    would leave two destinations with one name are skipped, never guessed at.
    Destination renames re-point every reference inside the file; a link from
    another file to the old name no longer resolves.
    """
    walk = _walk(doc)
    kinds = {e["id"]: e["kind"] for e in walk.entries}
    texts = {e["id"]: e["text"] for e in walk.entries}
    applied, skipped = 0, []
    renames: dict[str, str] = {}
    rename_ids: list[str] = []
    for edit in edits:
        entry_id = edit["id"]
        kind = kinds.get(entry_id)
        if kind is None or (edit["remove"] and kind not in REMOVE_KINDS) \
                or (not edit["remove"] and kind not in TEXT_KINDS):
            skipped.append(entry_id)
            continue
        if kind == "destination":
            renames[texts[entry_id]] = edit["text"]
            rename_ids.append(entry_id)
            continue
        if _apply_one(doc, walk.where[entry_id], edit):
            applied += 1
        else:
            skipped.append(entry_id)
    if rename_ids:
        if _rename_destinations(doc, walk, rename_ids, renames):
            applied += len(rename_ids)
        else:
            skipped.extend(rename_ids)
    return {"applied": applied, "skipped": skipped}


def _apply_one(doc, locator: tuple, edit: dict) -> bool:
    what = locator[0]
    try:
        if what == "link":
            _, pno, num = locator
            doc[pno].delete_link({"xref": num})
            return True
        if what == "outline-link":
            _m.pdf_dict_dels(locator[1], "A")
            return True
        if what == "tag" and edit["remove"]:
            _m.pdf_dict_dels(locator[1], locator[2])
            return True
        if edit["text"] is None:
            # A removal for a kind that cannot be removed. ``rewrite_side_text``
            # already skips it; a null text reaching MuPDF crashes the process.
            return False
        key = {"outline": "Title", "label": "P", "layer": "Name",
               "tag": locator[2] if what == "tag" else None}[what]
        _m.pdf_dict_puts(locator[1], key, _m.pdf_new_text_string(edit["text"]))
        return True
    except Exception:  # noqa: BLE001 -- leave this one as it was
        return False


# ── destination renames ─────────────────────────────────────────────────


def _rename_destinations(doc, walk: _Walk, ids: list[str],
                         renames: dict[str, str]) -> bool:
    """Rename destinations and every in-file reference to them.

    All or nothing: a rename that would give two destinations one name, or
    a tree that cannot be rebuilt, leaves every destination as it was.
    """
    names = {e["text"] for e in walk.entries if e["kind"] == "destination"}
    targets = list(renames.values())
    kept = names - set(renames)
    if any(not n for n in targets) or len(set(targets)) != len(targets) \
            or kept & set(targets):
        return False
    pdf = _m.pdf_specifics(doc.this)
    root = _get(_m.pdf_trailer(pdf), "Root")
    try:
        tree_ids = [e["id"] for e in walk.entries if e["kind"] == "destination"
                    and walk.where[e["id"]][0] == "dest-tree"]
        if any(walk.where[i][0] == "dest-tree" for i in ids):
            _rebuild_dest_tree(pdf, root, walk, tree_ids, renames)
        for i in ids:
            locator = walk.where[i]
            if locator[0] == "dest-dict":
                _, legacy, index = locator
                key = _m.pdf_dict_get_key(legacy, index)
                old = _m.pdf_to_name(key)
                value = _m.pdf_dict_get_val(legacy, index)
                _m.pdf_dict_puts(legacy, renames[old], value)
                _m.pdf_dict_dels(legacy, old)
        _repoint_references(pdf, renames)
    except Exception:  # noqa: BLE001
        return False
    return True


def _rebuild_dest_tree(pdf, root, walk: _Walk, tree_ids: list[str],
                       renames: dict[str, str]) -> None:
    """Replace the destination name tree with one sorted leaf."""
    leaves = {id(walk.where[i][1]): walk.where[i][1] for i in tree_ids}
    held = sum(_m.pdf_array_len(a) // 2 for a in leaves.values())
    if held != len(tree_ids):
        raise ValueError("a destination entry could not be read")
    pairs = []
    for entry_id in tree_ids:
        _, items, i = walk.where[entry_id]
        old = _text(_m.pdf_array_get(items, i))
        pairs.append((renames.get(old, old), _m.pdf_array_get(items, i + 1)))
    pairs.sort(key=lambda p: p[0])
    array = _m.pdf_new_array(pdf, 2 * len(pairs))
    for name, value in pairs:
        _m.pdf_array_push(array, _m.pdf_new_text_string(name))
        _m.pdf_array_push(array, value)
    leaf = _m.pdf_new_dict(pdf, 1)
    _m.pdf_dict_puts(leaf, "Names", array)
    _m.pdf_dict_puts(_get(root, "Names"), "Dests", _m.pdf_add_object(pdf, leaf))


def _repoint_references(pdf, renames: dict[str, str]) -> None:
    """Point every ``/Dest`` and go-to ``/D`` naming an old destination at its
    new name, keeping its type (a string stays a string, a name a name)."""
    def fix(holder, key):
        value = _get(holder, key)
        if _m.pdf_is_string(value) and _text(value) in renames:
            _m.pdf_dict_puts(holder, key,
                             _m.pdf_new_text_string(renames[_text(value)]))
        elif _m.pdf_is_name(value) and _m.pdf_to_name(value) in renames:
            _m.pdf_dict_puts(holder, key,
                             _m.pdf_new_name(renames[_m.pdf_to_name(value)]))

    for num in range(1, _m.pdf_xref_len(pdf)):
        try:
            obj = _m.pdf_load_object(pdf, num)
        except Exception:  # noqa: BLE001 -- a free or broken object
            continue
        if not _m.pdf_is_dict(obj):
            continue
        fix(obj, "Dest")
        for key in ("A", "OpenAction"):
            action = _get(obj, key)
            if _m.pdf_is_dict(action) and not _m.pdf_is_indirect(action) \
                    and _is_name(_get(action, "S"), "GoTo"):
                fix(action, "D")
        if _is_name(_get(obj, "S"), "GoTo"):
            fix(obj, "D")
