#!/usr/bin/env python3
"""
Extract full text from all .docx in docx/ and save as .txt in txt/ (same base name).
Run from: research strategy/
"""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import re

DOCX_DIR = "docx"
TXT_DIR = "txt"


def docx_to_text(path: Path) -> str:
    with zipfile.ZipFile(path, "r") as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    parts = []
    for t in root.iter(f"{{{ns}}}t"):
        if t.text:
            parts.append(t.text)
        if t.tail:
            parts.append(t.tail)
    text = "".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def main():
    base = Path(__file__).resolve().parent
    docx_dir = base / DOCX_DIR
    txt_dir = base / TXT_DIR
    if not docx_dir.exists():
        print(f"No {DOCX_DIR}/ folder.")
        return
    txt_dir.mkdir(parents=True, exist_ok=True)
    for docx in sorted(docx_dir.glob("*.docx")):
        txt_path = txt_dir / docx.with_suffix(".txt").name
        try:
            text = docx_to_text(docx)
            txt_path.write_text(text, encoding="utf-8")
            print(f"OK: {docx.name} -> {TXT_DIR}/{txt_path.name} ({len(text)} chars)")
        except Exception as e:
            print(f"FAIL: {docx.name} - {e}")

if __name__ == "__main__":
    main()
