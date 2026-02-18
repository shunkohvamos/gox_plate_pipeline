# Reference papers (Codex-readable)

This folder contains reference PDFs and their **Codex-readable extractions**: one subfolder per paper.

## Layout (per paper)

Each **paper-title folder** contains:

- **Original PDF** — same filename as in the top-level `reference/` (copied here).
- **`<folder_name>_full_text.txt`** — full text extracted from the PDF, with `--- Page N ---` separators.
- **`figures_tables/`** — extracted figures and page renders:
  - `figure_page<N>_xref<id>.<ext>` — embedded images (figures) from the PDF.
  - `page_001.png`, `page_002.png`, … — full-page renders (150 dpi) so tables and layout are visible.

## For Codex / AI

- Read **`*_full_text.txt`** for the main text.
- Use **`figures_tables/`** for figures (embedded images) and for tables/layout (page PNGs).

## Regenerating extractions

From repo root:

```bash
uv run python research_strategy/Objective_strategy/scripts/extract_pdf_for_codex.py
```

Requires: `pymupdf` (`uv sync --extra ref` or `uv add pymupdf`).

## Note

PDFs are **copied** into each paper folder; the originals remain in the top-level `reference/` for backup. You can delete the top-level PDFs after confirming the subfolders if you want to avoid duplication.
