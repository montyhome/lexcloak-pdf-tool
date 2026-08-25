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

**Result:** `{"pdf_b64": str, "protection_applied": bool}`.

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
| 0.7.0 | **6** | Adds `extract_pages` (+ `extract_pages_h`) — page-range split with re-based bookmarks, backing the closed app's scan-cost preflight "split into scannable parts" offer (Session 991) — **and `open_doc_path`**, a handle opened from a filesystem path so concurrent readers share one mmap instead of each holding a private copy (Session 992). Both ops landed under v6 before 0.7.0 was released, so no shipped binary ever advertised 6 with only one of them; once 0.7.0 ships, a further op needs v7. Bumps the protocol so a client can capability-gate the split offer from the startup banner instead of discovering an unknown op mid-flow. The supported set widens to {2, 3, 4, 5, 6}; every existing client is unaffected. |
| 0.6.8 | **5** | Adds `render_clip` and `list_annotations` (v5+). **Bumps the protocol version, departing from the additive-no-bump precedent set at 0.6.0/0.6.3** — deliberately. Those additions were optional enhancements a client could simply not call; these two back a closed-app export-integrity gate that fails CLOSED, so a client built against them has no safe degraded mode. Advertising 5 lets that client detect an too-old subprocess from the startup banner and refuse to start, instead of discovering it as a per-export refusal once a user is mid-document. The supported set widens to {2, 3, 4, 5}, so every existing client — including the closed app, which declares 2 on stateless calls — is unaffected. |
