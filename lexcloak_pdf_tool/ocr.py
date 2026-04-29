"""Tesseract subprocess pipeline + hOCR parser.

Pure-byte primitive ``_ocr_png_to_dict`` takes a rendered page (PNG bytes)
plus DPI / page-segmentation-mode and returns a dict with extracted text,
character-level coordinates, and per-line spans. Internally:

* Tesseract is invoked via stdin/stdout with ``hocr`` output mode -- no
  temp files on disk.
* hOCR is parsed with the stdlib ``html.parser`` -- no extra dependencies.
* Word-level bounding boxes are expanded to character bounding boxes by
  proportional split (downstream coordinate search tolerates small slop).

Use ``--psm 3`` (auto page segmentation, the default) -- correctly handles
forms, tables, sidebars, captions, and multi-column layouts. PyMuPDF's
``page.get_textpage_ocr`` internally forces ``--psm 6`` ("single uniform
block") which mis-OCRs non-uniform layouts.

Tesseract availability
----------------------
``find_tesseract_binary()`` resolves the Tesseract executable to an absolute
path (PyInstaller-bundled location, ``shutil.which``, then OS-specific
fallback paths). ``find_tessdata()`` resolves the language-data directory
similarly. When either is unavailable, ``_ocr_png_to_dict`` returns
``None`` rather than raising, so callers can fall back to native-text
extraction.

Test cache
----------
Setting ``OCR_CACHE_DIR`` enables an on-disk cache keyed by SHA-256 of
``(png_bytes || psm || tesseract_version)``. Production leaves this unset.
Used by the test suite to avoid re-running Tesseract on the same fixture
PNGs across hundreds of tests.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


# Higher DPI improves OCR of characters at page margins.
OCR_DPI = 300

# Tesseract subprocess timeout -- generous upper bound.
_TESSERACT_TIMEOUT_S = 120

# Windows console suppression: ``subprocess.run`` with a GUI parent pops a
# console window per invocation by default. Multi-page OCR with worker
# subprocesses produces a visible window per page on Windows -- user-hostile.
# CREATE_NO_WINDOW (0x08000000) is Windows-only; falls back to 0 on POSIX.
_SUBPROCESS_CREATION_FLAGS = (
    getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if sys.platform == "win32" else 0
)


# ── Tesseract / tessdata discovery ───────────────────────────────────────────


def find_tessdata() -> str | None:
    """Find the Tesseract data directory.

    Resolution order:
      1. PyInstaller-bundled ``_MEIPASS/tesseract/tessdata`` -- consumers
         that ship Tesseract inside a frozen binary place language files
         here.
      2. ``TESSDATA_PREFIX`` environment variable.
      3. Common system install locations per OS.
    """
    if hasattr(sys, "_MEIPASS"):
        bundled = os.path.join(sys._MEIPASS, "tesseract", "tessdata")
        if os.path.isdir(bundled):
            return bundled
    if p := os.environ.get("TESSDATA_PREFIX"):
        return p
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tessdata",
            r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        ]
        # Non-admin UB-Mannheim installs land here -- the installer silently
        # picks this when run without elevation.
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            candidates.append(os.path.join(
                local_appdata, "Programs", "Tesseract-OCR", "tessdata"))
    else:
        candidates = [
            "/opt/homebrew/share/tessdata",         # macOS Homebrew (Apple Silicon)
            "/usr/local/share/tessdata",            # macOS Homebrew (Intel) / Linux
            "/usr/share/tesseract-ocr/5/tessdata",  # Ubuntu 24.04+ (Tesseract 5)
            "/usr/share/tesseract-ocr/4.00/tessdata",  # Ubuntu/Debian (Tesseract 4)
            "/usr/share/tessdata",
        ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


TESSDATA_PATH = find_tessdata()


def find_tesseract_binary() -> str | None:
    """Resolve the Tesseract executable to an absolute path.

    ``shutil.which("tesseract")`` suffices for normal shell invocations but
    fails in frozen worker subprocesses on Windows: ``subprocess.run`` with
    a bare name does not reliably resolve user-local installs under
    ``%LOCALAPPDATA%`` even when the binary is on PATH for ``shutil.which``.

    Resolution order:
      1. ``_MEIPASS/tesseract/tesseract.exe`` -- bundled binary on packaged
         builds.
      2. ``shutil.which("tesseract")`` -- primary path for dev runs and
         POSIX systems.
      3. ``tesseract.exe`` sibling of the ``find_tessdata()`` directory.
      4. Explicit candidate paths matching ``find_tessdata()``'s search set.

    Returns an absolute filesystem path or ``None``.
    """
    if hasattr(sys, "_MEIPASS"):
        for name in ("tesseract.exe", "tesseract"):
            bundled_exe = os.path.join(sys._MEIPASS, "tesseract", name)
            if os.path.isfile(bundled_exe):
                return bundled_exe

    which_result = shutil.which("tesseract")
    if which_result:
        return which_result

    if TESSDATA_PATH:
        tessdata_parent = os.path.dirname(TESSDATA_PATH)
        for name in ("tesseract.exe", "tesseract"):
            candidate = os.path.join(tessdata_parent, name)
            if os.path.isfile(candidate):
                return candidate

    candidate_roots = []
    if sys.platform == "win32":
        candidate_roots.extend([
            r"C:\Program Files\Tesseract-OCR",
            r"C:\Program Files (x86)\Tesseract-OCR",
        ])
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            candidate_roots.append(
                os.path.join(local_appdata, "Programs", "Tesseract-OCR"))
    else:
        candidate_roots.extend([
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
        ])

    for root in candidate_roots:
        for name in ("tesseract.exe", "tesseract"):
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                return candidate
    return None


TESSERACT_BINARY = find_tesseract_binary()


# ── OCR result cache (test-only) ─────────────────────────────────────────────
# Setting ``OCR_CACHE_DIR`` enables on-disk caching keyed by content-hash
# before invoking Tesseract. Cache hit -> return cached bytes. Cache miss ->
# run Tesseract, write result, return. Intended for test suites that
# re-OCR the same fixture PNGs many times.
#
# Cache key:  SHA-256(png_bytes + b"|psm=" + psm + b"|tess=" + tesseract_version)
# Cache file: $OCR_CACHE_DIR/{sha256}.hocr (raw tesseract stdout bytes)
#
# Atomic writes via tmp + os.replace make cache writes safe across worker
# subprocesses; concurrent writers may produce identical files but no reader
# observes a partial file.

_TESSERACT_VERSION_CACHE: str | None = None


def _tesseract_version_string() -> str:
    """Return the first line of ``tesseract --version`` for cache-key salting.

    Memoized at module-process scope. Returns ``"unknown"`` on probe failure
    -- a stable fallback is preferable to disabling the cache.
    """
    global _TESSERACT_VERSION_CACHE
    if _TESSERACT_VERSION_CACHE is not None:
        return _TESSERACT_VERSION_CACHE
    binary = TESSERACT_BINARY
    if binary is None:
        _TESSERACT_VERSION_CACHE = "unknown"
        return _TESSERACT_VERSION_CACHE
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=_SUBPROCESS_CREATION_FLAGS,
        )
        # Tesseract 4.x writes version to stderr, 5.x to stdout. Combine.
        combined = (proc.stdout or b"") + b"\n" + (proc.stderr or b"")
        for line in combined.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                _TESSERACT_VERSION_CACHE = line.strip()
                return _TESSERACT_VERSION_CACHE
    except (OSError, subprocess.SubprocessError):
        pass
    _TESSERACT_VERSION_CACHE = "unknown"
    return _TESSERACT_VERSION_CACHE


def _cache_key(png_bytes: bytes, psm: int) -> str:
    h = hashlib.sha256()
    h.update(png_bytes)
    h.update(b"|psm=")
    h.update(str(psm).encode("ascii"))
    h.update(b"|tess=")
    h.update(_tesseract_version_string().encode("utf-8", errors="replace"))
    return h.hexdigest()


def _cache_lookup(cache_dir: str, key: str) -> bytes | None:
    """Return cached hOCR bytes for the key, or ``None`` on miss / error.

    A zero-byte cache file is treated as a miss (indicates a previous crash
    mid-write or external corruption). Caller will re-OCR and overwrite.
    """
    try:
        path = os.path.join(cache_dir, f"{key}.hocr")
        with open(path, "rb") as f:
            data = f.read()
        return data if data else None
    except OSError:
        return None


def _cache_store(cache_dir: str, key: str, data: bytes) -> None:
    """Atomically write ``data`` to ``{cache_dir}/{key}.hocr``. Best-effort."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        final_path = os.path.join(cache_dir, f"{key}.hocr")
        # PID + key-suffixed tmp file so concurrent writers from worker
        # subprocesses don't clobber each other. ``os.replace`` is atomic.
        tmp_path = f"{final_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, final_path)
    except OSError:
        pass


def _cache_record_event(cache_dir: str, hit: bool) -> None:
    """Append a single byte (``H`` or ``M``) to ``{cache_dir}/.stats``.

    Single-byte appends are atomic on POSIX (well under PIPE_BUF) and on
    Windows when opened with ``O_APPEND`` semantics, so worker processes can
    write concurrently without serializing.
    """
    try:
        with open(os.path.join(cache_dir, ".stats"), "ab") as f:
            f.write(b"H" if hit else b"M")
    except OSError:
        pass


# ── hOCR parsing ─────────────────────────────────────────────────────────────
# Tesseract's hOCR output is HTML with class markers (``ocr_page``,
# ``ocr_line``, ``ocrx_word``) and ``title="bbox X0 Y0 X1 Y1; ..."`` attributes
# carrying pixel-space coordinates. Parsed via stdlib html.parser to avoid an
# extra dependency for a one-format reader.


def _parse_bbox(title: str) -> tuple[float, float, float, float] | None:
    """Extract ``bbox X0 Y0 X1 Y1`` from an hOCR title attribute.

    Returns ``(x0, y0, x1, y1)`` in pixel-space, or ``None`` if the bbox
    clause is missing or malformed. Tolerates extra clauses (baseline,
    x_wconf, etc.).
    """
    if not title:
        return None
    for clause in title.split(";"):
        clause = clause.strip()
        if clause.startswith("bbox "):
            try:
                nums = clause[5:].split()[:4]
                if len(nums) == 4:
                    return (float(nums[0]), float(nums[1]),
                            float(nums[2]), float(nums[3]))
            except ValueError:
                return None
    return None


# Tesseract emits text lines under several class names depending on the
# detected region type. Treat them all as line containers -- the word spans
# inside share the same downstream contract regardless of classification.
_HOCR_LINE_CLASSES = frozenset({
    "ocr_line",       # standard body text
    "ocr_header",     # heading text (section titles, form-field labels in bold)
    "ocr_footer",     # page footers
    "ocr_caption",    # figure captions
    "ocr_textfloat",  # text wrapping around figures
})


class _HocrParser(HTMLParser):
    """Streaming parser that builds a list of lines -> words with bboxes.

    Handles the hOCR class markers carrying spatial data: ``ocr_page`` (page
    bbox), any line-level class in ``_HOCR_LINE_CLASSES``, and ``ocrx_word``
    (word bbox + text content). Other classes (``ocr_carea``, ``ocr_par``,
    ``ocr_separator``) are ignored.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._tag_stack: list[tuple[str, str]] = []
        self._current_line: dict | None = None
        self._current_word: dict | None = None
        self._word_text_parts: list[str] = []
        self.page_bbox: tuple[float, float, float, float] | None = None
        self.lines: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        cls = attr_map.get("class") or ""
        title = attr_map.get("title") or ""
        bbox = _parse_bbox(title)

        if cls == "ocr_page":
            self.page_bbox = bbox
        elif cls in _HOCR_LINE_CLASSES:
            self._current_line = {"bbox": bbox, "words": []}
        elif cls == "ocrx_word":
            self._current_word = {"bbox": bbox}
            self._word_text_parts = []

        self._tag_stack.append((tag, cls))

    def handle_data(self, data):
        if self._current_word is not None:
            self._word_text_parts.append(data)

    def handle_endtag(self, tag):
        for i in range(len(self._tag_stack) - 1, -1, -1):
            if self._tag_stack[i][0] == tag:
                cls = self._tag_stack[i][1]
                del self._tag_stack[i:]

                if cls == "ocrx_word" and self._current_word is not None:
                    text = "".join(self._word_text_parts).strip()
                    bbox = self._current_word.get("bbox")
                    if text and bbox and self._current_line is not None:
                        self._current_word["text"] = text
                        self._current_line["words"].append(self._current_word)
                    self._current_word = None
                    self._word_text_parts = []
                elif cls in _HOCR_LINE_CLASSES \
                        and self._current_line is not None:
                    if self._current_line.get("bbox") and \
                            self._current_line["words"]:
                        self.lines.append(self._current_line)
                    self._current_line = None
                return


def _parse_hocr(hocr: str) -> dict:
    """Parse hOCR text into ``{"page_bbox": ..., "lines": [...]}``.

    Each line: ``{"bbox": (x0,y0,x1,y1), "words": [{"text": str, "bbox": ...}]}``.
    Coordinates are in pixel-space -- caller converts to point-space.
    """
    parser = _HocrParser()
    parser.feed(hocr)
    return {"page_bbox": parser.page_bbox, "lines": parser.lines}


def _expand_word_bbox_to_chars(text: str, bbox: tuple[float, float, float, float]
                                ) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Split a word's bounding box across its characters proportionally.

    Tesseract hOCR gives word-level bboxes, not character-level. Equal
    proportional split is acceptable for downstream coordinate search
    (callers typically tolerate ~2pt slop).
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    n = len(text)
    if n == 0 or width <= 0:
        return []
    char_w = width / n
    return [
        (ch, (x0 + i * char_w, y0, x0 + (i + 1) * char_w, y1))
        for i, ch in enumerate(text)
    ]


# ── Subprocess OCR pipeline ──────────────────────────────────────────────────


def _run_tesseract(png_bytes: bytes, psm: int, tessdata_path: str) -> bytes | None:
    """Invoke Tesseract via stdin -> stdout, requesting hOCR output.

    Returns the hOCR bytes on success, ``None`` on any failure (missing
    binary, non-zero exit, timeout). Keeps PNG and OCR output in memory --
    no temp files on disk.

    Uses the absolute ``TESSERACT_BINARY`` path resolved at module load.
    Bare-name subprocess lookups don't reliably traverse user PATH inside
    frozen worker subprocesses on Windows when Tesseract is installed under
    ``%LOCALAPPDATA%`` (user-local UB-Mannheim install).
    """
    cache_dir = os.environ.get("OCR_CACHE_DIR") or None
    cache_key: str | None = None
    if cache_dir:
        cache_key = _cache_key(png_bytes, psm)
        cached = _cache_lookup(cache_dir, cache_key)
        if cached is not None:
            _cache_record_event(cache_dir, hit=True)
            return cached

    binary = TESSERACT_BINARY
    if binary is None:
        logger.warning(
            "tesseract binary not found at module-load time; "
            "check find_tesseract_binary() candidate paths"
        )
        return None
    try:
        argv = [
            binary, "stdin", "stdout",
            "-l", "eng",
            "--psm", str(psm),
            "--tessdata-dir", tessdata_path,
            "hocr",
        ]
        proc = subprocess.run(
            argv,
            input=png_bytes,
            capture_output=True,
            timeout=_TESSERACT_TIMEOUT_S,
            check=False,
            creationflags=_SUBPROCESS_CREATION_FLAGS,
        )
        if proc.returncode != 0:
            return None
        if not proc.stdout:
            return None
        if cache_dir and cache_key is not None:
            _cache_store(cache_dir, cache_key, proc.stdout)
            _cache_record_event(cache_dir, hit=False)
        return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.SubprocessError, OSError) as exc:
        logger.exception("tesseract subprocess failed: %s", exc)
        return None


def _ocr_png_to_dict(png_bytes: bytes, dpi: int = OCR_DPI,
                     psm: int = 3) -> dict | None:
    """Run Tesseract on ``png_bytes`` and parse the result.

    Returns ``{"text": str, "chars": list[(char, bbox|None)],
    "spans": list[{"text", "bbox", "size"}]}`` in PDF point-space (using the
    caller-provided ``dpi`` for pixel->point conversion), or ``None`` on
    failure (Tesseract unavailable, empty page, unparseable hOCR).

    ``psm=3`` (auto page segmentation) is the default -- correctly handles
    forms, tables, sidebars, captions, multi-column layouts.
    """
    if not TESSDATA_PATH:
        return None

    hocr_bytes = _run_tesseract(png_bytes, psm, TESSDATA_PATH)
    if hocr_bytes is None:
        return None

    try:
        hocr = hocr_bytes.decode("utf-8", errors="replace")
        parsed = _parse_hocr(hocr)
    except Exception as exc:
        logger.exception("hocr parse failed: %s", exc)
        return None

    if not parsed["lines"]:
        return None

    # Pixel -> PDF point conversion: 72pt per inch / dpi px per inch.
    scale = 72.0 / dpi

    text_lines: list[str] = []
    chars: list[tuple[str, tuple[float, float, float, float] | None]] = []
    spans: list[dict] = []

    for line in parsed["lines"]:
        words = line["words"]
        if not words:
            continue
        line_text = " ".join(w["text"] for w in words)
        text_lines.append(line_text)

        line_bbox_pt = tuple(v * scale for v in line["bbox"])
        line_size = max(line_bbox_pt[3] - line_bbox_pt[1], 1.0)
        spans.append({
            "text": line_text,
            "bbox": line_bbox_pt,
            "size": line_size,
        })

        for idx, word in enumerate(words):
            if idx > 0:
                chars.append((" ", None))
            word_bbox_pt = tuple(v * scale for v in word["bbox"])
            for ch, bbox in _expand_word_bbox_to_chars(word["text"], word_bbox_pt):
                chars.append((ch, bbox))
        chars.append(("\n", None))

    full_text = "\n".join(text_lines).strip()
    if not full_text:
        return None

    return {"text": full_text, "chars": chars, "spans": spans}
