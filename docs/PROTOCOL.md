# Wire Protocol

`lexcloak-pdf-tool` is invoked as a subprocess and communicates with its
parent via length-prefixed JSON frames over stdin/stdout. **v0.2.0 ships
protocol version 2.**

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
  "protocol_version": 2,
  "op": "render"
       | "extract_native"
       | "extract_ocr"
       | "extract_text_dict"
       | "extract_text_plain"
       | "search_for"
       | "apply_redactions"
       | "strip_metadata"
       | "page_count"
       | "page_size"
       | "is_encrypted"
       | "get_metadata"
       | "decrypt"
       | "exit",
  ...op-specific fields (see below)
}
```

`protocol_version` values outside the supported set (v0.2.0: `{2}`) are
rejected with `error_type: "ProtocolVersionMismatch"`.

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
  "enabled": bool
}
```

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

### `exit`

Terminates the subprocess cleanly. No response frame.

## Observability

Every successful or failed op writes one stderr line:

```
<op> ok=<bool> duration_ms=<int>
```

On startup, exactly one stderr line:

```
lexcloak_pdf_tool starting protocol_version=2 pymupdf_version=<x.y.z>
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
wire surface. Bumps to v3 will be released as a major version of the
package and will document additions and removals in the CHANGELOG.

## Versioning

| package version | protocol_version | notes |
|---|---|---|
| 0.2.0 | 2 | Initial public release. 13 ops. |
