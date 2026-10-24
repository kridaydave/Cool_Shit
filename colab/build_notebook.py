"""Build a Colab .ipynb from the percent-format .py.

Usage: python3 colab/build_notebook.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "mola_sr_v1.py"
DST = ROOT / "mola_sr_v1.ipynb"

raw = SRC.read_text().splitlines()
blocks = []
cur = None
for line in raw:
    if line.startswith("# %%"):
        if cur is not None:
            blocks.append(cur)
        cur = {"marker": line, "lines": []}
    elif cur is not None:
        cur["lines"].append(line)
if cur is not None:
    blocks.append(cur)


def strip_md(lines):
    out = []
    for line in lines:
        if line == "#":
            out.append("")
        elif line.startswith("# "):
            out.append(line[2:])
        else:
            out.append(line)
    return out


def trim(lines):
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines


cells = []
for b in blocks:
    is_md = "[markdown]" in b["marker"]
    lines = strip_md(b["lines"]) if is_md else list(b["lines"])
    lines = trim(lines)
    text = "\n".join(lines)
    if not text.strip():
        continue
    if is_md:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
    else:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        })

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
DST.write_text(json.dumps(nb, indent=1))
print(f"wrote {DST} with {len(cells)} cells")
