#!/usr/bin/env python3
"""
Extract all figures from:
  - PDFs in Reference paper/  -> figures/Reference_paper/{source_stem}/fig-000.png, ...
  - DOCX in research strategy/ -> figures/docx/{source_stem}_fig_00.png, ...
Naming: source file is clear from folder/filename; fig index = order in document.
"""
from pathlib import Path
import re
import subprocess
import zipfile
import shutil

BASE = Path(__file__).resolve().parent
REF = BASE / "Reference paper"
FIG_REF = BASE / "figures" / "Reference_paper"
FIG_DOCX = BASE / "figures" / "docx"


def sanitize(s: str) -> str:
    """Safe folder/filename: no spaces, no problematic chars."""
    s = re.sub(r"[^\w\-\.]", "_", s)
    return s.strip("_") or "unnamed"


def extract_pdf_images():
    """Extract images from each PDF in Reference paper/ using pdfimages."""
    if not REF.exists():
        return []
    pdfs = list(REF.glob("*.pdf"))
    log = []
    for pdf in sorted(pdfs):
        stem = sanitize(pdf.stem)
        out_dir = FIG_REF / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()
        try:
            subprocess.run(
                ["pdfimages", "-png", str(pdf.resolve()), "fig"],
                cwd=str(out_dir),
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            log.append(f"  PDF skip {pdf.name}: {e}")
            continue
        # pdfimages creates fig-000.png, fig-001.png in out_dir (prefix path used as base)
        # Actually pdfimages writes to CWD when prefix is relative! So we ran from out_dir, so we get fig-000.png in out_dir.
        created = sorted(out_dir.glob("fig-*.png"))
        log.append(f"  {pdf.name} -> {out_dir.relative_to(BASE)}/ ({len(created)} images)")
    return log


def extract_docx_images():
    """Extract word/media/* from each DOCX in docx/ and save with source name + index."""
    docx_dir = BASE / "docx"
    docxs = list(docx_dir.glob("*.docx")) if docx_dir.exists() else []
    log = []
    for docx in sorted(docxs):
        stem = sanitize(docx.stem)
        try:
            with zipfile.ZipFile(docx, "r") as z:
                media = [n for n in z.namelist() if n.startswith("word/media/") and not n.endswith("/")]
            if not media:
                log.append(f"  {docx.name} -> (no images)")
                continue
            # Sort by name so image1, image2,... order
            media.sort(key=lambda x: (len(x), x))
            for i, name in enumerate(media):
                ext = Path(name).suffix.lower() or ".png"
                out_name = f"{stem}_fig_{i:02d}{ext}"
                out_path = FIG_DOCX / out_name
                with zipfile.ZipFile(docx, "r") as z:
                    out_path.write_bytes(z.read(name))
            log.append(f"  {docx.name} -> figures/docx/ ({len(media)} images)")
        except Exception as e:
            log.append(f"  DOCX skip {docx.name}: {e}")
    return log


def main():
    FIG_REF.mkdir(parents=True, exist_ok=True)
    FIG_DOCX.mkdir(parents=True, exist_ok=True)
    pdf_log = extract_pdf_images()
    docx_log = extract_docx_images()
    for line in pdf_log:
        print(line)
    for line in docx_log:
        print(line)


if __name__ == "__main__":
    main()
