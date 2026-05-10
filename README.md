# lexcloak-pdf-tool

AGPL-licensed PyMuPDF I/O wrapper used by [Lex Cloak](https://lexcloak.com)
desktop. Communicates with the closed Lex Cloak app via length-prefixed JSON
over stdin/stdout.

## What this is

A subprocess CLI that wraps a narrow set of PyMuPDF operations:

- Render PDF pages to PNG.
- Extract native-text words and character coordinates.
- Run Tesseract OCR on rendered pages and parse the hOCR output.
- Search for text on a page (substring, whole-word, or head/tail split).
- Apply content-stream redactions (black-box matches, optional re-encryption).
- Strip metadata and XMP.
- Page count, single-page size, batch all-page sizes, encryption-state probes.
- Authenticate password-protected PDFs.

Built around the specific call patterns Lex Cloak needs -- **not a
general-purpose PyMuPDF library**.

## Why public and AGPL

PyMuPDF (and the underlying MuPDF) are dual-licensed: GNU AGPL v3 or a paid
Artifex commercial license. Distributing PyMuPDF inside a closed-source
desktop application requires either the commercial license or a clean
subprocess boundary.

Lex Cloak chose the subprocess split. This repository is the AGPL'd
subprocess; the closed Lex Cloak desktop spawns the bundled CLI binary as a
child process and communicates over stdin/stdout. The desktop application
itself does not link or embed PyMuPDF.

See `LICENSE` (AGPL v3 full text) and `NOTICE` (PyMuPDF and MuPDF
attribution).

## What this does NOT contain

This package is the I/O layer, not the product:

- No detection rules, NER models, or signature heuristics.
- No OCR orchestration (deskew, page selection, multi-page workers).
- No coordinate-mapping logic beyond character-position search.
- No license validation, telemetry, billing, or UI code.

The Lex Cloak desktop application has all of those; this package has none.
Anyone wanting a generic PyMuPDF subprocess wrapper is welcome to fork this
repository and extend the CLI's op set under the AGPL v3.

## Install from GitHub

```
pip install git+https://github.com/montyhome/lexcloak-pdf-tool@v0.3.0
```

PyPI publish is deferred until the package stabilizes.

## Run as a CLI

```
python -m lexcloak_pdf_tool
```

Reads length-prefixed JSON commands from stdin and writes length-prefixed
JSON responses to stdout. See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the
wire contract.

## Build a standalone binary

```
pip install pyinstaller
pyinstaller --onefile \
  --name lexcloak-pdf-tool \
  lexcloak_pdf_tool/__main__.py
```

Output: `dist/lexcloak-pdf-tool` (or `.exe` on Windows). Bundle that into
your own application and spawn it as a subprocess.

## OCR

The `extract_ocr` op shells out to a system Tesseract binary. To enable OCR:

1. Install Tesseract (`brew install tesseract` on macOS, the UB-Mannheim
   installer on Windows, `apt install tesseract-ocr` on Debian/Ubuntu).
2. Ensure `tesseract` is on `PATH`, or set `TESSDATA_PREFIX` to the
   `tessdata` directory containing language files.

When Tesseract is unavailable, `extract_ocr` returns `null` rather than
raising -- callers can fall back to native-text extraction.

## Tests

```
pip install -e ".[dev]"
pytest
```

Tesseract-dependent tests skip automatically when no Tesseract binary is
discoverable. CI installs Tesseract explicitly on each platform.

## Contributing

Bug reports and pull requests are welcome via GitHub Issues. Significant
changes require coordination with the Lex Cloak release cadence -- the wire
protocol is versioned, and bumps require coordinated releases across this
repository and the closed Lex Cloak desktop.

## License

AGPL v3 -- full text in `LICENSE`. If you need a closed-source PyMuPDF
integration without AGPL terms, contact Artifex Software, Inc. directly for
a commercial PyMuPDF license; that is generally cheaper than maintaining a
fork of this repository.

Copyright (C) 2026 Monty Home LLC.
