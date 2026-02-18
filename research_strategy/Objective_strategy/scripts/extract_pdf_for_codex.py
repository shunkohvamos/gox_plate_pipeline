#!/usr/bin/env python3
"""
Extract text and figures/tables from PDFs in Objective_strategy/reference
for Codex-readable format.

For each PDF in reference/:
  - Creates a subfolder by paper title (from PDF metadata or filename).
  - Puts the original PDF, extracted full_text.txt, and figures_tables/ in that folder.
  - figures_tables/ contains: figure_page_N_*.png (extracted images).

Usage:
  uv run --optional ref python research_strategy/Objective_strategy/scripts/extract_pdf_for_codex.py
  # Or from repo root with pymupdf installed:
  python research_strategy/Objective_strategy/scripts/extract_pdf_for_codex.py

Requires: pymupdf (pip install pymupdf / uv add --optional ref).
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("pymupdf is required. Install with: uv add --optional ref", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = REPO_ROOT / "research_strategy" / "Objective_strategy" / "reference"


def sanitize_folder_name(name: str, max_len: int = 120) -> str:
    """Build a filesystem-safe folder name from a title or filename stem."""
    # Replace problematic chars with underscore, collapse multiple
    s = re.sub(r'[<>:"/\\|?*\s]+', "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    # Keep only printable ASCII + common unicode
    s = "".join(c for c in s if ord(c) < 0x10000 and (c.isalnum() or c in "._- "))
    s = s.replace(" ", "_").strip("_")
    return s[:max_len] if s else "unnamed"


def _infer_title_from_first_page(doc: "fitz.Document") -> str:
    """
    Heuristic: infer paper title from the first page text when metadata.title is empty.

    Strategy:
      - Take non-empty text lines on page 1.
      - Prefer the first line that is reasonably long and contains alphabetic characters.
      - Fall back to joining the first 2–3 lines if the very first line looks short.
    """
    try:
        page0 = doc[0]
    except Exception:
        return ""
    text = page0.get_text() or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # First, look for a line that looks like a title.
    for ln in lines:
        if len(ln) >= 10 and any(c.isalpha() for c in ln):
            return ln

    # Fallback: concatenate first few lines.
    head = " ".join(lines[:3])
    return head


def get_paper_folder_name(pdf_path: Path, doc: "fitz.Document") -> str:
    """Prefer PDF metadata title; else infer from first-page text; else filename stem (sanitized)."""
    meta = doc.metadata
    title = (meta.get("title") or "").strip()
    if title:
        return sanitize_folder_name(title)

    inferred = _infer_title_from_first_page(doc).strip()
    if inferred:
        return sanitize_folder_name(inferred)

    stem = pdf_path.stem
    return sanitize_folder_name(stem)


def extract_text(doc: "fitz.Document") -> str:
    """Extract full text with page breaks for readability."""
    lines = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        if text.strip():
            lines.append(f"\n--- Page {i + 1} ---\n")
            lines.append(text)
    return "\n".join(lines).strip() or "(No text extracted)"


def extract_images(doc: "fitz.Document", out_dir: Path) -> list[Path]:
    """Extract all embedded images to out_dir. Returns list of saved paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    seen_xref: set[int] = set()

    for page_num in range(len(doc)):
        for xref in doc.get_page_images(page_num):
            xref_id = xref[0]
            if xref_id in seen_xref:
                continue
            seen_xref.add(xref_id)
            try:
                base = doc.extract_image(xref_id)
                ext = base.get("ext", "png")
                if ext.lower() == "jpeg":
                    ext = "jpg"
                name = f"figure_page{page_num + 1}_xref{xref_id}.{ext}"
                path = out_dir / name
                with open(path, "wb") as f:
                    f.write(base["image"])
                saved.append(path)
            except Exception:
                try:
                    pix = fitz.Pixmap(doc, xref_id)
                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    path = out_dir / f"figure_page{page_num + 1}_xref{xref_id}.png"
                    pix.save(str(path))
                    saved.append(path)
                except Exception:
                    continue
    return saved


def render_pages_as_images(doc: "fitz.Document", out_dir: Path, dpi: int = 150) -> list[Path]:
    """Render each page as PNG (for tables/figures as layout). Returns list of paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        path = out_dir / f"page_{i + 1:03d}.png"
        pix.save(str(path))
        saved.append(path)
    return saved


def process_pdf(pdf_path: Path, reference_dir: Path) -> Path | None:
    """
    Process one PDF: create paper folder, copy PDF, write txt, extract images.
    Returns the created paper folder path or None on failure.
    """
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return None
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Skip (open failed): {pdf_path.name} — {e}")
        return None

    try:
        folder_name = get_paper_folder_name(pdf_path, doc)

        # If this PDF was already processed before, find its existing folder
        # (any subdirectory under reference_dir that already contains this PDF),
        # and rename it to the new folder_name when needed.
        existing_dir: Path | None = None
        for child in reference_dir.iterdir():
            if not child.is_dir():
                continue
            if (child / pdf_path.name).is_file():
                existing_dir = child
                break

        if existing_dir is not None and existing_dir.name != folder_name:
            # Rename existing per-paper folder to match the inferred title.
            new_dir = existing_dir.parent / folder_name
            if new_dir.exists() and new_dir != existing_dir:
                # If a folder with the new name already exists, we fall back to it.
                paper_dir = new_dir
            else:
                existing_dir.rename(new_dir)
                paper_dir = new_dir

            # If there is an old full_text file using the previous prefix, rename it too.
            old_txt_candidates = list(new_dir.glob("*_full_text.txt"))
            if old_txt_candidates:
                old_txt = old_txt_candidates[0]
                new_txt = new_dir / f"{folder_name}_full_text.txt"
                if old_txt.name != new_txt.name:
                    old_txt.rename(new_txt)
        else:
            paper_dir = reference_dir / folder_name

        paper_dir.mkdir(parents=True, exist_ok=True)

        # Copy original PDF into paper folder (keep original name)
        dest_pdf = paper_dir / pdf_path.name
        if dest_pdf.resolve() != pdf_path.resolve():
            shutil.copy2(pdf_path, dest_pdf)

        # Full text
        text = extract_text(doc)
        txt_name = folder_name + "_full_text.txt"
        txt_path = paper_dir / txt_name
        txt_path.write_text(text, encoding="utf-8", errors="replace")

        # Figures and tables: embedded images + full-page renders (for tables/layout)
        figures_dir = paper_dir / "figures_tables"
        saved_images = extract_images(doc, figures_dir)
        saved_pages = render_pages_as_images(doc, figures_dir, dpi=150)

        doc.close()
        print(f"  OK: {folder_name}  (text + {len(saved_images)} figures + {len(saved_pages)} pages)")
        return paper_dir
    except Exception as e:
        print(f"  Error: {pdf_path.name} — {e}")
        doc.close()
        return None


def main() -> None:
    ref = Path(REFERENCE_DIR)
    if not ref.is_dir():
        print(f"Reference dir not found: {ref}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(ref.glob("*.pdf"))
    # Only process PDFs that are directly in reference (not already in a paper subfolder)
    pdfs = [p for p in pdfs if p.is_file()]
    if not pdfs:
        print("No PDFs found in", ref)
        return

    print(f"Processing {len(pdfs)} PDF(s) in {ref}")
    for pdf_path in pdfs:
        process_pdf(pdf_path, ref)
    print("Done.")


if __name__ == "__main__":
    main()
