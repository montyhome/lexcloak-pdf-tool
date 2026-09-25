# Wire Protocol

`lexcloak-pdf-tool` is invoked as a subprocess and communicates with its
parent via length-prefixed JSON frames over stdin/stdout. **v0.6.8 ships
protocol version 5.** v2 + v3 + v4 stay in the supported set so a v5
subprocess serves older clients cleanly during rolling closed-app
upgrades.

There are two protocol modes (the subprocess speaks both simultaneously):

* **Stateless ops** (v2/v3 baseline) — every op carries `pdf_b64`. The
  subprocess parses the PDF on each call. Simple, no shared state.
* **Stateful handle ops** (v4+) — `open_doc(pdf_b64)` parses once and
  returns a UUID handle. Per-page ops take `handle` instead of `pdf_b64`.
  `close_doc(handle)` releases the parsed document. Subprocess holds
  parsed `pymupdf.Document` instances keyed by handle, capped at 16 entries
  via LRU eviction (oldest-evicted-first). Designed for callers that
  perform many per-page reads on the same PDF — eliminates the
  per-call IPC payload re-shoveling cost.

## Frame format

Every frame on either direction:

```
<4-byte length, big-endian uint32><JSON payload (UTF-8)>
```

The 4-byte prefix lets the protocol carry binary blobs (PDF bytes, PNG
bytes) as base64-encoded strings inside the JSON payload without delimiter
ambiguity. Either side reads exactly N bytes after the prefix -- no
scanning, no escape sequences. EOF on stdin at a frame boundary is treated
as a clean exit.

## Limits

- **Max payload size:** 256 MiB per frame. Larger payloads return an
  `OverflowError` response. Guards against malformed length prefixes that
  would otherwise allocate gigabytes.

## Request schema

```json
{
  "protocol_version": 4,
  "op": "render" | "extract_native" | "extract_ocr"
       | "extract_text_dict" | "extract_text_plain"
       | "search_for" | "apply_redactions" | "strip_metadata"
       | "set_metadata" | "insert_cover_page" | "reduce_size"
       | "page_count" | "page_size" | "all_page_sizes"
       | "is_encrypted" | "get_metadata" | "decrypt" | "encrypt"
       | "open_doc" | "close_doc" | "open_doc_path"
       | "extract_pages" | "extract_pages_h"
       | "trace_text" | "trace_text_h"
       | "residue_report"
       | "render_removed" | "render_removed_h"
       | "render_h" | "extract_native_h" | "extract_ocr_h"
       | "extract_text_dict_h" | "extract_text_plain_h"
       | "search_for_h" | "apply_redactions_h" | "strip_metadata_h"
       | "set_metadata_h" | "insert_cover_page_h" | "reduce_size_h"
       | "page_count_h" | "page_size_h" | "all_page_sizes_h"
       | "is_encrypted_h" | "get_metadata_h" | "encrypt_h"
       | "exit",
  ...op-specific fields (see below)
}
```

`protocol_version` values outside the supported set (v0.4.0: `{2, 3, 4}`)
are rejected with `error_type: "ProtocolVersionMismatch"`. v2 + v3
acceptance is intentional backward compat for the rolling closed-app
upgrade — the subprocess advertises 4 in its handshake but accepts 2/3
on the wire. `decrypt` deliberately has no `_h` variant: decryption
always operates on raw bytes (returns cleartext), and the caller's
handle-based workflow opens a fresh handle on the cleartext result.
`encrypt` (its symmetric counterpart) *does* have an `encrypt_h` — it is
a terminal save step, so encrypting an already-open handle is natural.

## Response schema

```json
{
  "ok": true | false,
  "result": {...} | null,
  "error": "<human-readable>",       // only when ok=false
  "error_type": "<exception class>"  // only when ok=false
}
```

`exit` produces no response; the subprocess terminates cleanly.

## Per-op contracts

All ops accept a base64-encoded `pdf_b64` field unless noted. Page numbers
are zero-indexed. Out-of-range pages return `error_type: "IndexError"`.

### `render`

Render a PDF page to PNG bytes.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |
| `dpi` | float | `150` |

**Result:** `{"png_b64": str}`.

### `render_clip`  *(v5+)*

Render **only a clip** of a page to PNG bytes.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |
| `clip` | `[x0, y0, x1, y1]` floats, PDF points | required |
| `dpi` | float | `150` |
| `gray` | bool | `true` |

**Result:** `{"png_b64": str}`.

This is **not** equivalent to `render` followed by a crop. MuPDF aligns the
output pixel grid to the clip's own — possibly fractional — origin, so the
two rasters agree only when the clip lands on an integer pixel boundary.
Measured on a 300-DPI page over 40 randomized fractional clips: a page-render
crop matches the clip render's geometry only under `irect` rounding (floor
the top-left, ceil the bottom-right) and is byte-identical in 27 of 40; the
rest differ by 1–9 intensity levels on ≤2.73% of pixels. A caller verifying a
region against what a clip render produced must use this op.

Refuses rather than returning a misleading image: `IndexError` for a page
index out of range, `ValueError` for a clip that is degenerate or clamps away
to nothing against the page rect, and `ValueError` for an encrypted document
(which would otherwise render blank).

### `list_annotations`  *(v5+)*

Per-page annotation **subtype names and counts**.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |

**Result:** `{"pages": [{"page": int, "subtypes": {str: int}}, ...]}` — one
entry per page in page order, with an empty `subtypes` dict for a page
carrying none (pages are never omitted).

**The payload is deliberately this narrow and must stay so.** Never
annotation contents, text, rects, author or dates. The intended caller folds
this into a diagnostics record documented as PHI-free; an op that *could*
return annotation text would make that claim depend on caller restraint
rather than on this op's inability to leak.

Two behaviours worth knowing before relying on it:

* **`/Link` annotations are not reported.** PyMuPDF's `page.annots()` does
  not yield them — they are reached through `page.links()` — so a document
  full of hyperlinks reports no annotations here.
* **Encrypted documents are refused, not reported as empty.** Such a
  document opens cleanly and reports a page count; the failure appears only
  when something walks the annots. Since the caller is proving a *negative*,
  returning "no annotations" for a document nothing could read is the exact
  wrong answer, so the op raises `ValueError` naming the cause instead.

### `extract_native`

Native PDF text words plus bounding boxes.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |

**Result:**
```json
{
  "words": [
    {"text": str, "x0": float, "y0": float, "x1": float, "y1": float,
     "block": int, "line": int, "word": int},
    ...
  ]
}
```

Whitespace-only tokens are filtered.

### `extract_ocr`

Tesseract OCR plus character coordinates and per-line spans.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |
| `tessdata_path` | string \| null | auto |
| `psm` | int | `3` (auto page segmentation) |

**Result:**
```json
{
  "text": str,
  "chardata": [[char, x0, y0, x1, y1] or [char, null, null, null, null], ...],
  "spans": [{"text": str, "bbox": [x0,y0,x1,y1], "size": float}, ...]
}
```

Returns `null` (not an error) when Tesseract is unavailable or OCR fails.
Callers should fall back to native-text extraction.

### `residue_report`  *(v8+)*

Counts of what a document still carries outside its visible page content.
Read-only, and the counts never include any text from the document. A file
`apply_redactions` produced is clean when every count is zero.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required (unencrypted) |
| `pages` | list of int | `null`: tagged text is not counted |

`pages` names the page indices a redaction touched, in the delivered
document's numbering. `/ActualText` and `/Alt` are counted only there, since
that is the only place `apply_redactions` removes them.

**Result:**
```json
{"report": {
  "active_content": int,
  "extra_info_keys": int,
  "page_metadata": int,
  "piece_info": int,
  "unknown_keys": int,
  "associated_files": int,
  "image_metadata": int,
  "tagged_text": int
}}
```

`active_content` counts actions other than go-to, URI and named (and, for
`/OpenAction`, anything but a go-to), and every `/AA` dictionary.
`extra_info_keys`, `page_metadata`, `piece_info` and `unknown_keys` count
metadata beyond the eight standard `/Info` keys. `image_metadata` counts
plain-DCT images still carrying a comment, EXIF or XMP segment.

### `trace_text`  *(v7+)*

Every word the page's content carries, with how it is drawn. The opposite
question to `extract_text_plain`: that op reports the text a viewer sees and
therefore leaves out text outside the page, text in optional-content groups
that are switched off, and glyphs removed by a clipping path. This op reports
all of it, with the facts a caller needs to compare each word against a
render of the page. It makes no judgement about visibility itself.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |

**Result:**
```json
{
  "rect": [x0, y0, x1, y1],
  "rotation": int,
  "image_cover": float,
  "words": [
    {"text": str, "bbox": [x0, y0, x1, y1], "size": float,
     "mode": int, "opacity": float, "layer": str, "layer_off": bool,
     "clipped": bool, "covered_by": "image" | "path" | null, "span": int,
     "chars": [[x0, y0, x1, y1], ...]}           // v9+, only on some words
  ],
  "covers": [{"seq": int, "kind": str, "box": [x0, y0, x1, y1]}]   // v9+
}
```

- `rect` and every `bbox` are in the **rotated** page frame (`page.rect`), the
  frame a render of the page uses. `rotation` is the page's `/Rotate`.
- `mode` is the text render mode (0 fill, 1 stroke, 2 fill+stroke,
  3 invisible, 4–7 the same with clipping).
- `opacity` is the fill alpha in effect for the word.
- `layer` is the optional-content group the word belongs to (`""` for none),
  and `layer_off` says whether that group is off in the document's current
  configuration. Words in switched-off groups are included; the op switches
  such groups on through the UI configuration while it reads, and restores
  the configuration before returning, so a handle's later renders are
  unchanged.
- `clipped` is true when no character of the word appears in MuPDF's
  clip-respecting extraction: it lies outside a clipping path or outside the
  page. Characters are matched by character and origin.
- `covered_by` is `"path"` or `"image"` when a fill or image drawn **later**
  covers at least 80% of the word's span box, else `null`. Whether the cover
  is opaque, and whether it depicts the word (as a page scan does over its own
  text layer), is for the caller to judge from a render.
- `span` is the drawing sequence number of the text span the word came from;
  words sharing it were drawn by one text-showing run, in any page rotation.
- `image_cover` is the share of the page area covered by image draws, summed
  and capped at 1.0 — about 1.0 on a typical scanned page.
- `chars` *(v9+)* is each character's box, in the order of `text` and in the
  same rotated frame as `bbox`. It is present only on a word that a fill,
  shading or image drawn **later** reaches into by any area, because a drawn
  box rarely stops at a word boundary: it can end before a trailing comma or
  in the middle of a letter, and a caller deciding which characters it hides
  needs them one by one.
- `covers` *(v9+)* lists each such later draw once: its drawing sequence
  number (comparable with `span`), its kind (`fill-path`, `fill-shade`,
  `fill-image` or `fill-imgmask`) and its rotated box. Whether a cover is
  opaque is not reported, and neither are blend mode or soft mask: a render
  of the page answers all of them together, and `render_removed` below gives
  the caller the render to compare with.

Words are split on whitespace within each span, in content-stream order.
`IndexError` for a page out of range.

### `render_removed`  *(v9+)*

A page rendered as it would look with some text removed, for comparison with
the page's own render: a character whose removal changes nothing inside its
box is not visible on the page. Nothing is delivered from this op, so the
removal runs once, with the thinnest band and no verify.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |
| `dpi` | float | `150` |
| `remove` | list of word dicts | required |

Each `remove` entry has the `remove_text` shape above without `page`
(`box`, `text`, `mode`, `opacity`, `layer`, optional `chars`).

**Result:**
```json
{"png_b64": str, "lost": [[x0, y0, x1, y1], ...], "kept": [[x0, y0, x1, y1], ...],
 "missed": [int, ...]}
```

- The PNG is rendered exactly as `render` renders at the same `dpi`, and
  with nothing to remove it is byte-identical to it.
- `lost` is the rotated box of every **other** character the removal took
  with it (a band through a covered letter can cross a label drawn over the
  same place), so a caller can tell a character's own effect from a
  neighbour's.
- Each run gets the thinnest band, and any named character still there gets
  the wider `remove_text` bands in turn, so the render shows what the
  removal would really take. `kept` is the box of each named character that
  even the widest band left: its render proves nothing about whether it shows.
- `missed` is the index of each entry whose word was not found.
- The removal runs on a copy of the whole document: a one-page copy would
  lose the document's optional-content configuration and draw a
  switched-off layer. The document or handle is not changed.

`ValueError` for a missing or malformed `remove`, `IndexError` for a page out
of range.

### `extract_text_dict`

PyMuPDF's `page.get_text("dict")` block hierarchy.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |

**Result:**
```json
{
  "blocks": [
    {"type": int, "bbox": [x0,y0,x1,y1], "lines": [
      {"bbox": [...], "spans": [
        {"text": str, "bbox": [...], "size": float, "font": str,
         "color": int, "flags": int}
      ], "wmode": int, "dir": [dx,dy]}
    ]}, ...
  ]
}
```

`type=0` is text, `type=1` is image. Image-block `image: bytes` field is
stripped before serialization to keep image-heavy PDFs under the
256 MiB frame budget.

### `extract_text_plain`

PyMuPDF's `page.get_text()` plain-text output.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |

**Result:** `{"text": str}`.

### `search_for`

Search for text on a page (substring, whole-word, or split modes).

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |
| `needle` | string | required |
| `ocr_chardata` | flat 5-tuple list \| null | null |
| `whole_word` | bool | `false` |
| `split` | bool | `false` |

`split=true` takes precedence over `whole_word=true` when both are set.

**Result:** `{"rects": [[x0, y0, x1, y1], ...]}`.

If `ocr_chardata` is provided, search runs in CharData-space (the OCR
output of `extract_ocr`); otherwise live-page semantics.

### `apply_redactions`

Black-box redactions, optional metadata strip, optional re-encryption.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `matches` | list of match dicts | `[]` |
| `redact_label` | string | `""` |
| `active_categories` | list of strings \| null | null |
| `removed_pages` | list of ints \| null | null |
| `output_protection` | dict \| null | null |
| `remove_text` *(v7+)* | list of word dicts \| null | null |
| `keep_uncovered_lines` *(0.11.0+)* | bool | `false` |

Match-dict shape:
```json
{
  "page": int,
  "rect": {"x0": float, "y0": float, "x1": float, "y1": float},
  "type": str,
  "enabled": bool,
  "redact_label": str
}
```

`redact_label` on a match is **optional** (since 0.6.4) and labels that box
alone, overriding the document-level `redact_label` above. Absent or empty
falls back to the document-level value, so a payload carrying no per-match
labels behaves exactly as it did before 0.6.4. Labelled and unlabelled
matches burn in the same pass.

Unknown match-dict keys are **ignored, not rejected** — this is deliberate
and load-bearing. A newer closed app may be paired with an older frozen
bundled CLI, so an unrecognized additive field must degrade (the older CLI
stamps the document-level label) rather than fail the export.

`output_protection` shape: `{"mode": "same"|"new"|"none", "password": str?}`.
Modes `"same"` and `"new"` require a non-empty password; the caller is
responsible for substituting the source password for `"same"` before
reaching this op.

`remove_text` (v7+) removes the text of the named words **without drawing
anything**: no fill, and images and vector graphics under them are left
alone. It runs before the matches are burned, and skips removed and
blacked-out pages. Each entry names a word the way `trace_text` reports it:

```json
{"page": int, "box": [x0, y0, x1, y1], "text": str, "mode": int,
 "opacity": float, "layer": str}
```

`box` (as-rendered frame) is required; the other fields are optional but
should be sent, because the word is found again in the page's own trace by
all of them and its characters are keyed by position *and* drawing
properties. That is what keeps visible text drawn in the same place (another
optional-content layer, say) from counting as the target. Removal goes
through a thin band across the middle of the word, is first tried on a
one-page copy, and is applied only if every target character went and every
other character stayed; failing that, each word is retried alone with wider
bands. Text outside the page, in switched-off layers, clipped away or in
render mode 3 is all reached.

*(v9+)* An entry may also carry `"chars": [int, ...]`: the positions, in
`text`, of the characters to remove, for a word only part of which is to go.
The entry must still name the whole word as `trace_text` reported it; a
position past its end means the word is not found. Each run of named
characters gets its own band across just those glyphs, and the check is the
same one, so the word's other characters must still be there afterwards. If
the set fails, the part is halved until each half is removed cleanly or is a
single character no band separates. That character is left and the entry is
reported `kept`, while what could be separated is removed. An older
subprocess ignores the field and removes the whole word.

`keep_uncovered_lines` (0.11.0+) keeps the text of a line a box does not
reach. MuPDF removes a glyph when a box reaches a tenth of the way into the
glyph's box, and that box runs from the font's ascender to its descender, so
a box over one line can take letters from the next line without touching
their ink: a heading just below a redacted name, or the next line of
single-spaced text under a value with a descender. With the flag set, a
glyph is kept when every box that reaches into its glyph box lies wholly
above or wholly below the ink of the glyphs that box touches on its line
(measured from the glyph outlines). Only plainly drawn text is kept: render
mode 0, 1 or 2, opacity above zero, in no optional-content group, drawn once
at its origin, on a left-to-right line of a page with no `/Rotate`. Text is
then removed with each box trimmed clear of the kept lines, and the boxes
are drawn and applied to images and graphics unchanged, in a second pass
that leaves text alone. The old text removal is the floor: it is tried on a
one-page copy of the result, and if it would still take any character other
than a kept one, it is applied to the page. A kept glyph can sit under the
half point of border the burn draws around each box, over the top of its
tallest letters; it is still text. A page on which nothing is kept burns
exactly as without the flag, and so does every page when the flag is
absent. An older subprocess ignores the field and burns the old way.

**Result:** `{"pdf_b64": str, "protection_applied": bool}`, plus, **exactly
when the request carried `remove_text`**, `"text_removal": {"removed":
[[page, index], ...], "kept": [[page, index], ...]}`. `index` is the entry's
position among that page's entries. A `kept` word was not found, or could not
be removed without touching other text, and was left as it was. An older
subprocess ignores the unknown field and returns no `text_removal`, which is
how a client tells "removed nothing" from "not supported".

`protection_applied` is `false` when re-encryption was requested but
failed (op falls back to unprotected output rather than blocking).

Malformed match payloads (bad page index, non-numeric rect coords,
inverted bounds) return `error_type: "ValueError"` with a
named-field message.

### `strip_metadata`

Remove document metadata + XMP. No re-encryption.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |

**Result:** `{"pdf_b64": str}`.

### `page_count`

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |

**Result:** `{"count": int}`.

### `page_size`

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `page` | int | `0` |

**Result:** `{"width": float, "height": float}` (in PDF point-space).

### `is_encrypted`

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |

**Result:** `{"encrypted": bool}`.

PyMuPDF auto-authenticates empty-password PDFs silently, so the common
"encrypted-but-empty-pw" case reports `false`. Distinguishes "needs
unlock UI" from "open but flagged as encrypted."

### `get_metadata`

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |

**Result:** `{"metadata": dict, "has_xmp": bool}`.

Nested shape so callers can pass through `metadata` (a string-coerced
dict with empty values dropped) without rendering `has_xmp` as a metadata
field.

### `extract_pages` (v6+)

Extract one contiguous 0-based inclusive page range as a standalone
PDF. Built for splitting an over-sized document into scannable parts:
page content is copied untouched (`insert_pdf`), and the source's
bookmark outline is sliced to the range and re-based to the part's
local 1-based numbering, so citations into the master remain
resolvable through the part. Outline entries whose destination falls
outside the range (or that have no in-document destination) are
dropped; hierarchy levels are clamped so a slice that orphans children
still forms a legal outline. Output is saved `garbage=3, deflate=True`.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `from_page` | int (0-based) | `0` |
| `to_page` | int (0-based, inclusive) | required in practice (`-1` fails validation) |

**Result:** `{"pdf_b64": str, "page_count": int}`.

An out-of-range or inverted range returns `error_type: "ValueError"`.
`extract_pages_h` is the handle variant; it is read-only against the
cached document (never mutates or closes it).

### `decrypt`

Authenticate a password-protected PDF and return cleartext bytes.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `password` | string | `""` |

**Result:** `{"pdf_b64": str, "page_count": int}`.

Wrong password returns `error_type: "WrongPasswordError"` (op-level error;
instance stays alive, caller can retry with a different password).

Unencrypted input is re-saved cleanly with the password ignored
(defensive path).

### `encrypt`

AES-256 encrypt a **cleartext** PDF under a password — the encrypt-on-exit
counterpart to `decrypt`. The Lex Cloak route calls it as the final pipeline
step, after redaction + Spec 13/14 stamping, so those steps always operate on
cleartext.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `password` | string | `""` |

**Result:** `{"pdf_b64": str, "protection_applied": bool}`.

`protection_applied` is `true` iff the encrypted save succeeded. An **empty
password** is a no-op that returns the input bytes unchanged with
`protection_applied: false`. A PyMuPDF save failure degrades to unprotected
output (`protection_applied: false`) rather than raising — a failed encryption
never blocks the download (shares the fallback with `apply_redactions`).
**Already-encrypted input** returns `error_type: "ValueError"` (the op requires
cleartext; authenticate with `decrypt` first). Permissions are locked to
accessibility-only, matching `apply_redactions`' re-encrypt path.

### `reduce_size`

Shrink a (cleartext) PDF locally. Lossless by default (orphan/metadata
scrub + font subsetting); opt-in image downsample when `dpi` is given.

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |
| `dpi` | int or null | `null` (lossless — no image downsample) |
| `quality` | int 1..100 | `75` (JPEG quality when downsampling) |
| `grayscale` | bool | `false` (convert downsampled images to gray) |
| `preserve_metadata` | array of string or null | `null` (v0.6.6 — strip everything, historical behaviour) |

**Result:** `{"pdf_b64": str, "info": {"orig_size": int, "new_size": int, "applied_dpi": int|null}}`.

`applied_dpi` is the DPI that actually shaped the returned bytes: `null`
for the lossless path, for a downsample that failed and fell back to
lossless, or when the no-grow guard returned the original. The op never
returns bytes larger than the input (no-grow guard) and preserves the
OCR/selectable text layer. Encrypted/password-protected input returns
`error_type: "ValueError"` — the op requires cleartext (the redaction
route compresses before any optional encryption).

`preserve_metadata` (v0.6.6) names metadata keys to carry across the
scrub. The lossless scrub strips the whole metadata dict, including any
marking the **caller** applied after redacting — Lex Cloak's Spec-13
notice spans `subject` + `producer` + `keywords` and all three were
erased by every successful compression before this field existed. Send
`["subject", "producer", "keywords"]` to keep them.

Known keys are `title`, `author`, `subject`, `keywords`, `creator`,
`producer`, `creationDate`, `modDate`. Anything else — including a
misspelling like `"Subject"`, and the derived-not-settable `format` /
`encryption` — returns `error_type: "ValueError"`. A bare string instead
of an array is likewise rejected rather than iterated into characters.

That validation is a **known-key check, not a safety allowlist**.
Whether a key is safe to keep is the caller's decision and this op cannot
make it: `producer` holding `"Lex Cloak 1.8.19"` is a deliberate
post-redaction marking, `producer` holding `"HP Scanner 4.2"` is a
fingerprint leak — same key, same type, opposite meanings. What the check
buys is typo rejection, since a misspelled key would otherwise be a
silent no-op and a silently-dropped marking is the failure this field
exists to prevent. Name the narrowest set that carries your marking.

The field is opt-in rather than default-on because this is a general op:
flipping the default would start preserving arbitrary third-party
document metadata that callers rely on it to strip.

An older subprocess that predates v0.6.6 ignores the field (unknown keys
have always been ignored on this wire) and strips the metadata as before
— the caller sees a missing stamp, not a protocol error.

### `exit`

Terminates the subprocess cleanly. No response frame.

## Stateful handle ops (v4+)

Each `_h` op takes a `handle` field (UUID4 string from a prior `open_doc`)
instead of a `pdf_b64` field. All other args + result shapes match the
stateless counterpart exactly. A handle remains valid until `close_doc`,
process exit, or LRU eviction (the cache holds at most 16 docs; oldest
evicted on overflow).

### `open_doc`

| Input field | Type | Default |
|---|---|---|
| `pdf_b64` | base64 string | required |

**Result:** `{"handle": str}` (UUID4 in canonical 8-4-4-4-12 hex form).

### `open_doc_path` (v6+)

| Input field | Type | Default |
|---|---|---|
| `pdf_path` | string (absolute filesystem path) | required |

**Result:** `{"handle": str}` — identical to `open_doc`, including LRU
eviction and `close_doc` semantics. An encrypted document is opened and
handed back exactly as `open_doc` does; ask `is_encrypted_h`.

**Why it exists.** `open_doc` receives the document inline, so N concurrent
readers of one document hold N private copies — and in the closed app each
OCR worker is such a reader, with a second copy in its own subprocess.
PyMuPDF mmaps a file opened by path, so the OS page cache holds one shared
set of pages instead. Measured on a 185 MB document: per-reader USS
285 MB → 100 MB, marginal cost per added reader 318 MB → 135 MB.

**Errors.** `KeyError` (field absent), `ValueError` (not a non-empty
string), `FileNotFoundError` (missing, or not a regular file — a directory
lands here), `PermissionError` (present but unreadable). A file that exists
and is readable but is not a PDF surfaces PyMuPDF's own `FileDataError`.

**The path never appears in an error message.** Callers pass real user
filesystem paths, and a filename can itself be sensitive. Every error from
this op — including PyMuPDF's, which does interpolate the filename — is
scrubbed to `<pdf_path>` before it crosses the wire. The exception *class*
is preserved so callers can still distinguish causes. Do not add the path
back for debuggability; the caller already knows which path it sent.

### `close_doc`

| Input field | Type | Default |
|---|---|---|
| `handle` | string | required |

**Result:** `{"closed": bool}` — `true` if the handle existed, `false` if
already gone (idempotent; never raises).

### Per-op `_h` variants

The following ops accept `handle` (string, required) instead of `pdf_b64`:

| Stateless op | Handle op | Result shape |
|---|---|---|
| `render` | `render_h` | `{"png_b64": str}` |
| `extract_native` | `extract_native_h` | `{"words": [...]}` |
| `extract_ocr` | `extract_ocr_h` | `{...}` or `null` |
| `extract_text_dict` | `extract_text_dict_h` | `{"blocks": [...]}` |
| `extract_text_plain` | `extract_text_plain_h` | `{"text": str}` |
| `trace_text` | `trace_text_h` | `{"rect": [...], "rotation": int, "image_cover": float, "words": [...]}` |
| `residue_report` | — | `{"report": {...}}` |
| `render_removed` | `render_removed_h` | `{"png_b64": str, "lost": [...], "kept": [...], "missed": [...]}` |
| `search_for` | `search_for_h` | `{"rects": [...]}` |
| `apply_redactions` | `apply_redactions_h` | `{"pdf_b64": str, "protection_applied": bool}` |
| `strip_metadata` | `strip_metadata_h` | `{"pdf_b64": str}` |
| `page_count` | `page_count_h` | `{"count": int}` |
| `page_size` | `page_size_h` | `{"width": float, "height": float}` |
| `all_page_sizes` | `all_page_sizes_h` | `{"sizes": [[w, h], ...]}` |
| `is_encrypted` | `is_encrypted_h` | `{"encrypted": bool}` |
| `get_metadata` | `get_metadata_h` | `{"metadata": dict, "has_xmp": bool}` |
| `set_metadata` | `set_metadata_h` | `{"pdf_b64": str}` |
| `insert_cover_page` | `insert_cover_page_h` | `{"pdf_b64": str}` |
| `reduce_size` | `reduce_size_h` | `{"pdf_b64": str, "info": {...}}` |
| `encrypt` | `encrypt_h` | `{"pdf_b64": str, "protection_applied": bool}` |

Calling a handle op with a missing, closed, or non-string `handle` field
returns `error_type: "HandleNotFound"`. Distinct from `ProtocolError` so
clients can distinguish "subprocess crashed" (instance broken — respawn)
from "handle stale" (subprocess fine — reopen via `open_doc`).

`apply_redactions_h`, `strip_metadata_h`, `set_metadata_h`,
`insert_cover_page_h`, and `reduce_size_h` mutate the cached doc in place;
subsequent reads on the same handle reflect the mutation. `encrypt_h` is the
exception: PyMuPDF applies encryption at save time, so the cached doc stays
cleartext and reusable after an `encrypt_h` call. Callers that need to
preserve a mutated original should keep their own `pdf_bytes` reference.

## CLI flags

The binary is normally driven entirely over the stdin/stdout frame protocol,
but one out-of-band flag is recognized before the loop starts:

* `--version` — print the package version (a bare `MAJOR.MINOR.PATCH` semver)
  to **stdout** and exit `0`, without reading a single protocol frame. The
  closed app probes this to confirm the bundled subprocess matches the pinned
  release tag. It reads stdout only — the startup banner (below) is on stderr
  and carries `pymupdf_version`, which must not be mistaken for the package
  version. The version is sourced from the installed distribution metadata,
  falling back to the packaged `__version__` in the frozen binary.

## Observability

Every successful or failed op writes one stderr line:

```
<op> ok=<bool> duration_ms=<int>
```

On startup, exactly one stderr line:

```
lexcloak_pdf_tool starting protocol_version=4 pymupdf_version=<x.y.z>
```

Stdout is reserved for protocol frames -- a downstream pipe-consumer will
never see observability bytes mixed with response bytes.

## Error model

All ops surface failures as a structured response with `ok=false`,
`error`, and `error_type`. There is no bare `except: pass` anywhere in
the dispatch loop -- every exception either re-raises (after logging),
returns a structured error, or writes a diagnostic line.

Frame-level errors (oversized prefix, malformed JSON inside a valid-length
frame) return a structured response and then exit non-zero -- frame
boundary is lost once the parser walks past a bad payload, so the parent
must restart a fresh subprocess.

## Stability

`protocol_version` is the public commitment for `lexcloak-pdf-tool`'s
wire surface. Future bumps (v4+) will be released as a minor or major
version of the package and will document additions and removals in the
CHANGELOG. The supported-set policy is "current version + previous"
during the rolling-upgrade window, then narrow back to one once every
shipping client has caught up.

## Versioning

| package version | protocol_version | notes |
|---|---|---|
| 0.2.0 | 2 | Initial public release. 13 ops. |
| 0.3.0 | 3 | Adds `all_page_sizes` batch op (14 ops). v2 stays supported for backward compat during closed-app rolling upgrade. |
| 0.4.0 | 4 | Adds stateful handle protocol (`open_doc` + `close_doc` + 13 `_h` per-op variants, 29 ops total). Subprocess holds parsed `pymupdf.Document` instances keyed by UUID handle with LRU eviction (cache size 16). v2 + v3 stay supported during the rolling closed-app upgrade. `decrypt` has no `_h` variant by design — decryption is byte-in/byte-out, then callers open a fresh handle on the cleartext. |
| 0.6.0 | 4 | Adds `reduce_size` op (+ `reduce_size_h`) for local PDF compression: lossless scrub + font subset, opt-in DPI image downsample, no-grow guard, cleartext-only. Additive — no protocol bump; v2–v4 unaffected. Doc gap: the 0.5.0–0.5.4 op additions (`set_metadata`, `insert_cover_page`, `blackout_pages`) predate this row and are not yet captured in the per-op contracts above. |
| 0.6.1–0.6.2 | 4 | Patch fixes (redaction sliver + AcroForm widget-flatten guard-widen). No new ops. |
| 0.6.3 | 4 | Adds `encrypt` op (+ `encrypt_h`) — AES-256 encrypt-on-exit, the symmetric counterpart to `decrypt`; and a `--version` CLI flag (bare semver on stdout). Additive — no protocol bump. The encrypted-save block is now shared with `apply_redactions` via `redact._save_encrypted`. The enum + `_h` table above are brought current as of this row (the earlier 0.5.x/0.6.0 op names were backfilled here). |
| 0.6.4 | 4 | **Superseded by 0.6.5 — do not pin.** Same per-match `redact_label` as 0.6.5, but made the numeric token-boundary rule *unconditional*, which silently changed the semantics of every `search_whole_word_in_chars` caller including ones searching for human-typed needles. 0.6.5 puts that rule behind an opt-in flag. The tag remains published (tags are immutable) but nothing should reference it. |
| 0.6.5 | 4 | No new ops, no wire-surface change. (1) `apply_redactions` accepts an optional per-match `redact_label` overriding the document-level label for that box; absent/empty falls back, so a label-free payload is byte-identical to 0.6.3 (verified A/B across three document-label shapes, modulo the random trailer `/ID`). (2) `search_whole_word_in_chars` gains a keyword-only `numeric_token_boundary=False`: when True, a numeric-shaped needle no longer matches inside a longer number through an intra-number separator (`12` in `18-12-107.5`). **The default is the historical behavior**, so no existing caller changes — including the `search_for` op, whose `whole_word=True` path is untouched. Alpha and mixed needles are unaffected either way. The flag exists because whether a numeric fragment is noise depends on the needle's provenance, which only the caller knows: detector-inferred needles want it True, human-typed needles want it False. |
| 0.6.7 | 4 | No new ops, no wire-surface change — frames are byte-identical to 0.6.6. Retires the deprecated `fitz` alias: the package and its tests now `import pymupdf`. `fitz` is a `from pymupdf import *` shim, so every name this package uses resolves to the identical object (verified against PyMuPDF 1.27.2.3 and 1.28.2) — but `import fitz` writes a deprecation warning to **stdout** at import time on 1.28.2+, and stdout is this protocol's frame channel. An import-time write lands ahead of every in-process mitigation, so not importing the alias is the only fix that reaches it. PyMuPDF also states the alias will be removed in a future release, which would make the subprocess unstartable. `PROTOCOL_VERSION` stays 4; the supported set stays {2, 3, 4}. |
| 0.7.0 | **6** | Adds `extract_pages` (+ `extract_pages_h`) — page-range split with re-based bookmarks, backing the closed app's scan-cost preflight "split into scannable parts" offer — **and `open_doc_path`**, a handle opened from a filesystem path so concurrent readers share one mmap instead of each holding a private copy. Both ops landed under v6 before 0.7.0 was released, so no shipped binary ever advertised 6 with only one of them; once 0.7.0 ships, a further op needs v7. Bumps the protocol so a client can capability-gate the split offer from the startup banner instead of discovering an unknown op mid-flow. The supported set widens to {2, 3, 4, 5, 6}; every existing client is unaffected. |
| 0.8.0 | **7** | Adds `trace_text` (+ `trace_text_h`): every word a page carries, with its render mode, fill opacity, optional-content group and state, whether it survives clip-respecting extraction, and whether a later fill or image covers it, all in the rotated page frame. It lets a client compare a page's text content against a render of the page. **And** `apply_redactions` (+ `_h`) gains the optional `remove_text` field: remove named words' text with no fill and no change to images or graphics, verified on a one-page copy before it is applied, reported back as `text_removal`. Additive; the supported set widens to {2, 3, 4, 5, 6, 7}, so every existing client is unaffected. |
| 0.9.0 | **8** | Adds `residue_report`, and widens what `apply_redactions` (+ `_h`) and `reduce_size` (+ `_h`) remove from the delivered file, with **no new request field**. Removed: every action other than go-to, URI and named (an `/OpenAction` survives only as a go-to), all `/AA` dictionaries; `/Info` keys beyond the eight standard ones, page and image `/Metadata`, `/PieceInfo`, catalog and page keys outside the PDF specification's, and the trailer `/ID` (regenerated on save); `/AF` associated files; comment, EXIF and XMP segments inside plain-DCT JPEG images, without re-encoding; and `/ActualText` and `/Alt` on the pages a redaction touched (page content, forms, property lists and structure elements), because a tag that repeats a sentence keeps the words the burn removed from the glyphs. A tag on a page nothing was redacted from is kept. A client that ignores the new op is unaffected; one that needs the check should treat a missing op as "could not check". Additive; the supported set widens to {2, 3, 4, 5, 6, 7, 8}. |
| 0.9.1 | 8 | No new ops, no wire-surface change. `apply_redactions` (+ `_h`) and `reduce_size` no longer fail on a PDF whose xref leaves an object number undefined (its `/Size` exceeds what the xref sections cover, which is common in linearized files carrying an incremental update). PyMuPDF's `Document.scrub` walks every number for `javascript=True` / `xml_metadata=True` and raised `cannot find object in xref` on the first undefined one, so every export of such a file failed, with or without boxes. That walk now runs in the package (`redact.scrub_objects`) and passes over only the numbers `pdf_object_exists` reports undefined, which have no body and are written as free entries on save; a defined object that fails to load still raises. JavaScript and XMP removal are unchanged. `reduce_size_h` was not affected: it saves before it scrubs, and the save rebuilds the xref. |
| 0.9.2 | 8 | No new ops, no wire-surface change. Flattening a tagged form in `apply_redactions` (+ `_h`) now leaves no widget object in the file. `bake(widgets=True)` takes each widget off its page, but a tagged form's structure tree points at every widget (`/K << /Type /OBJR /Obj N 0 R >>`), so the widget objects and the parent field dictionaries behind them survived the save, `/V` values included, in a file with no live field on any page. Every widget object no page lists after the bake is now replaced with `null` (`redact.drop_baked_widget_objects`); the parent field dictionaries are then unreferenced and collected on save. A widget still on a page is left alone. The structure tree stays, its object references resolving to null. The page renders the same. |
| 0.10.0 | **9** | Adds `render_removed` (+ `render_removed_h`): a page rendered as it would look with some text removed, and the boxes of any other character that removal took, so a client can compare it with the page's own render and tell which characters change nothing when they go. `trace_text` (+ `_h`) gains, for words a later fill, shading or image reaches into, each character's box (`chars`), and the page gains the later draws themselves (`covers`). `remove_text` entries on `apply_redactions` (+ `_h`) may name some of a word's characters (`chars`), removed and verified below word scale, for a word a drawn box covers only part of. **And** `remove_text` no longer keeps a word it removed cleanly: its verify step keys every character by character, origin (rounded to 0.01 pt) and drawing properties, and when MuPDF rewrites a text run to drop the target's glyphs, the glyphs it keeps can come back a few millionths of a point from where they were (measured 138.394989 -> 138.395004 on pymupdf 1.28.2), which tipped the rounded key one step, so an untouched glyph read as lost and the word was reported `kept` (14 of 80 random positions of a run's first word on a synthetic 11pt line). Keys now pair when their character and properties match and their origins are within one rounding step (`KEY_STEPS`), exact partners first and one to one. A glyph really lost or a target glyph really left still fails the check. Additive; the supported set widens to {2, 3, 4, 5, 6, 7, 8, 9}. |
| 0.11.0 | 9 | `apply_redactions` (+ `_h`) gains the optional `keep_uncovered_lines` (default `false`, so every existing client gets the historical burn, verified identical by page text and render on a multi-page document, labelled and unlabelled). With it, a box keeps the glyphs of a line it does not reach: MuPDF's filter takes a glyph when a box reaches a tenth of the way into its ascender-to-descender box (measured at 7, 12 and 24 pt on pymupdf 1.28.2), so a box over one line removed letters of the line below whose ink it never touched. Only plainly drawn, horizontal text on an unrotated page is kept; the fill, images and graphics use the box unchanged; and the old text removal, tried on a one-page copy of the result, is the floor. A non-boolean value is a `ValueError`. No new ops; `PROTOCOL_VERSION` stays 9 and an older subprocess ignores the field. |
| 0.6.8 | **5** | Adds `render_clip` and `list_annotations` (v5+). **Bumps the protocol version, departing from the additive-no-bump precedent set at 0.6.0/0.6.3** — deliberately. Those additions were optional enhancements a client could simply not call; these two back a closed-app export-integrity gate that fails CLOSED, so a client built against them has no safe degraded mode. Advertising 5 lets that client detect an too-old subprocess from the startup banner and refuse to start, instead of discovering it as a per-export refusal once a user is mid-document. The supported set widens to {2, 3, 4, 5}, so every existing client — including the closed app, which declares 2 on stateless calls — is unaffected. |
