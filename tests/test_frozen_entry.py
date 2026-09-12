"""The entry script must work the way the frozen binary runs it.

Every consumer builds the standalone binary with ``__main__.py`` as the
PyInstaller entry script (README, "Build a standalone binary"). A frozen entry
script runs as a top-level ``__main__`` with **no parent package**, which is a
different import environment from ``python -m lexcloak_pdf_tool``:

* a relative import (``from .render import render_clip``) raises
  ``ImportError: attempted relative import with no known parent package``;
* a bare import of a sibling module (``import render``) finds nothing, because
  the bundle does not put the package directory on ``sys.path``.

Both work under ``-m``, which is how ``test_cli.py`` spawns the subprocess, so
the rest of this suite cannot see either one. v0.6.8 through v0.7.0 shipped
the first shape in the ``render_clip`` and ``list_annotations`` handlers: the
import is function-local, so the binary started cleanly, answered every other
op, and failed only those two, on every input. Lex Cloak's export checks
depend on both, so every packaged export was refused (fixed in v0.7.1).

Two guards, because each covers a gap in the other. The static check finds the
import wherever it sits, including after argument validation a minimal request
would never get past. The spawn test catches a frozen-only import failure of
any other kind, by running every op with the entry script started as a plain
script, which is exactly the condition the binary runs under.

Synthetic fixtures only.
"""
from __future__ import annotations

import ast
import base64
import json
import pathlib
import struct
import subprocess
import sys

import pymupdf

import lexcloak_pdf_tool
from lexcloak_pdf_tool.__main__ import _OPS, PROTOCOL_VERSION

PACKAGE_DIR = pathlib.Path(lexcloak_pdf_tool.__file__).resolve().parent
ENTRY = PACKAGE_DIR / "__main__.py"
LENGTH = struct.Struct(">I")
SIBLINGS = {p.stem for p in PACKAGE_DIR.glob("*.py")} - {"__init__", "__main__"}


def test_entry_script_has_no_relative_or_bare_sibling_imports():
    tree = ast.parse(ENTRY.read_text(encoding="utf-8"), filename=str(ENTRY))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                offenders.append(f"line {node.lineno}: relative import "
                                 f"from {'.' * node.level}{node.module or ''}")
            elif node.module and node.module.split(".")[0] in SIBLINGS:
                offenders.append(f"line {node.lineno}: bare sibling import "
                                 f"from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in SIBLINGS:
                    offenders.append(f"line {node.lineno}: bare sibling "
                                     f"import {alias.name}")
    assert not offenders, (
        "__main__.py is the frozen entry script and must import the package "
        "absolutely (from lexcloak_pdf_tool.x import y):\n  "
        + "\n  ".join(offenders))


def _one_page_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(pymupdf.Point(72, 100), "Synthetic entry-script probe",
                     fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _frame(obj: dict) -> bytes:
    payload = json.dumps(obj).encode("utf-8")
    return LENGTH.pack(len(payload)) + payload


def _read_frames(buf: bytes) -> list[dict]:
    frames, i = [], 0
    while i + LENGTH.size <= len(buf):
        (n,) = LENGTH.unpack_from(buf, i)
        i += LENGTH.size
        frames.append(json.loads(buf[i:i + n].decode("utf-8")))
        i += n
    return frames


def test_every_op_loads_when_the_entry_runs_as_a_plain_script():
    pdf_b64 = base64.b64encode(_one_page_pdf()).decode("ascii")
    ops = sorted(_OPS)
    # Plausible arguments, not correct ones: an op may still refuse its input
    # (a missing handle, a bad path), and that is fine. Only an import failure
    # is a finding here, and a handler raises that on its first line.
    common = {"protocol_version": PROTOCOL_VERSION, "pdf_b64": pdf_b64,
              "page": 0, "dpi": 72, "clip": [0, 0, 100, 100],
              "handle": "no-such-handle", "path": str(ENTRY.parent / "nope.pdf")}
    stdin = b"".join(_frame({**common, "op": op}) for op in ops)
    stdin += _frame({"op": "exit"})

    proc = subprocess.run([sys.executable, str(ENTRY)], input=stdin,
                          capture_output=True, timeout=180)
    responses = _read_frames(proc.stdout)

    assert len(responses) == len(ops), (
        f"expected one response per op ({len(ops)}), got {len(responses)}; "
        f"the script-mode subprocess died early.\nstderr:\n"
        f"{proc.stderr.decode('utf-8', 'replace')[-2000:]}")
    broken = {
        op: resp.get("error", "")
        for op, resp in zip(ops, responses)
        if not resp.get("ok")
        and resp.get("error_type") in {"ImportError", "ModuleNotFoundError"}
    }
    assert not broken, (
        "these ops cannot load their code when __main__.py runs the way the "
        f"frozen binary runs it: {broken}")
