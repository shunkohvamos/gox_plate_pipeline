#!/usr/bin/env python3
"""
Rename extracted figures to match captions (Figure 1, Table S2, etc.) when possible.
- PDF: read existing .txt, detect caption lines in order, rename fig-000.png -> Figure_01.png etc.
- DOCX: parse document.xml for caption order and image order, then rename.
Run after extract_figures.py. Uses existing figures/ and Reference paper/*.txt.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

BASE = Path(__file__).resolve().parent
REF = BASE / "Reference paper"
FIG_REF = BASE / "figures" / "Reference_paper"
FIG_DOCX = BASE / "figures" / "docx"

# Caption patterns: line must start (after optional whitespace) with Figure N. / Table N. / Fig. N.
CAPTION_LINE = re.compile(
    r"^\s*(?:Figure\s+(S?\d+[a-z]?)|Table\s+(S?\d+)|Fig\.\s+(S?\d+[a-z]?))\s*[\.\:]",
    re.IGNORECASE,
)
# Group 1 = Figure number, Group 2 = Table number, Group 3 = Fig. number


def caption_to_filename(caption_label: str) -> str:
    """Turn 'Figure 1' / 'Table S2' into safe filename stem: Figure_01, Table_S02."""
    caption_label = caption_label.strip()
    # Normalize: "Figure 1" -> "Figure_01", "Table S1" -> "Table_S01"
    s = re.sub(r"\s+", "_", caption_label)
    s = re.sub(r"_+", "_", s)  # collapse multiple underscores
    # Pad number: Figure_1 -> Figure_01, Table_S1 -> Table_S01
    def pad(m):
        pre, num = m.group(1), m.group(2)
        return f"{pre}{int(num):02d}" if num.isdigit() else f"{pre}{num}"

    s = re.sub(r"^(.+?_)(\d+)$", pad, s)
    s = re.sub(r"^(.+?_S)(\d+)$", pad, s)
    s = re.sub(r"[^\w\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Fig"


def extract_captions_from_text(text: str) -> list[str]:
    """Return list of caption labels in order of first appearance (Figure 1, Table S1, ...)."""
    seen = set()
    captions = []
    for line in text.splitlines():
        m = CAPTION_LINE.search(line)
        if not m:
            continue
        if m.lastindex >= 1 and m.group(1):
            label = f"Figure {m.group(1)}"
        elif m.lastindex >= 2 and m.group(2):
            label = f"Table {m.group(2)}"
        elif m.lastindex >= 3 and m.group(3):
            label = f"Figure {m.group(3)}"
        else:
            continue
        if label not in seen:
            seen.add(label)
            captions.append(label)
    return captions


def rename_pdf_figures():
    """For each PDF's extracted folder, get captions from .txt and rename fig-*.png."""
    if not FIG_REF.exists():
        return []
    log = []
    for txt_path in sorted(REF.glob("*.txt")):
        stem = txt_path.stem
        # Sanitize stem to match folder name (same as extract_figures)
        safe = re.sub(r"[^\w\-\.]", "_", stem).strip("_") or "unnamed"
        out_dir = FIG_REF / safe
        if not out_dir.exists():
            continue
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.append(f"  Skip {stem}: read txt failed - {e}")
            continue
        captions = extract_captions_from_text(text)
        files = sorted(out_dir.glob("fig-*.png"))
        if not files:
            continue
        for i, f in enumerate(files):
            if i < len(captions):
                new_stem = caption_to_filename(captions[i])
                new_name = f"{new_stem}.png"
            else:
                new_name = f"Fig_{i:03d}.png"
            new_path = out_dir / new_name
            if new_path == f:
                continue
            if new_path.exists() and new_path != f:
                new_path = out_dir / f"{new_stem}_{i:02d}.png"
            f.rename(new_path)
        log.append(f"  {stem}: {len(files)} images, {len(captions)} captions -> renamed")
    return log


def get_docx_image_order_and_captions(docx_path: Path) -> tuple[list[str], list[str]]:
    """
    Return (ordered_media_filenames, ordered_caption_labels).
    Media order = order of first blip reference in document.xml.
    Captions = paragraphs that start with Figure/Table in document order.
    """
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }
    with zipfile.ZipFile(docx_path, "r") as z:
        try:
            xml = z.read("word/document.xml")
        except KeyError:
            return [], []
        try:
            rels_xml = z.read("word/_rels/document.xml.rels")
        except KeyError:
            return [], []

    # rId -> media filename (e.g. rId5 -> word/media/image1.png)
    rels_root = ET.fromstring(rels_xml)
    rid_to_media = {}
    for rel in rels_root.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
        rid = rel.get("Id")
        target = rel.get("Target", "")
        if "media/" in target:
            rid_to_media[rid] = target.split("/")[-1]  # image1.png

    root = ET.fromstring(xml)
    body = root.find("{%s}body" % ns["w"])
    if body is None:
        body = root
    captions = []
    image_rids = []

    def get_text(elem):
        parts = []
        for t in elem.iter("{%s}t" % ns["w"]):
            if t.text:
                parts.append(t.text)
            if t.tail:
                parts.append(t.tail)
        return "".join(parts)

    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            para_text = get_text(child)
            m = CAPTION_LINE.search(para_text)
            if m:
                if m.group(1):
                    captions.append(f"Figure {m.group(1)}")
                elif m.group(2):
                    captions.append(f"Table {m.group(2)}")
                elif m.group(3):
                    captions.append(f"Figure {m.group(3)}")
        for blip in child.iter():
            if blip.tag.endswith("}blip"):
                embed = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                if embed and embed not in image_rids:
                    image_rids.append(embed)

    media_order = [rid_to_media.get(rid, f"rid_{rid}") for rid in image_rids]
    return media_order, captions


def rename_docx_figures():
    """Rename docx-extracted images by caption when possible (1:1 by current fig_00, fig_01 order)."""
    if not FIG_DOCX.exists():
        return []
    log = []
    docx_dir = BASE / "docx"
    for docx_path in sorted(docx_dir.glob("*.docx")) if docx_dir.exists() else []:
        stem = re.sub(r"[^\w\-\.]", "_", docx_path.stem).strip("_") or "unnamed"
        media_order, captions = get_docx_image_order_and_captions(docx_path)
        files = sorted(FIG_DOCX.glob(f"{stem}_fig_*.*"))
        if not files:
            continue
        try:
            for i, f in enumerate(files):
                if i < len(captions):
                    new_stem = caption_to_filename(captions[i])
                    ext = f.suffix
                    new_name = f"{stem}_{new_stem}{ext}"
                else:
                    new_name = f"{stem}_Fig_{i:03d}{f.suffix}"
                new_path = FIG_DOCX / new_name
                if new_path == f:
                    continue
                if new_path.exists() and new_path != f:
                    new_path = FIG_DOCX / f"{stem}_{new_stem}_{i:02d}{f.suffix}"
                f.rename(new_path)
            log.append(f"  {docx_path.name}: {len(files)} images, {len(captions)} captions -> renamed")
        except Exception as e:
            log.append(f"  Skip {docx_path.name}: {e}")
    return log


def main():
    print("PDF figures (from Reference paper .txt):")
    for line in rename_pdf_figures():
        print(line)
    print("DOCX figures:")
    for line in rename_docx_figures():
        print(line)


if __name__ == "__main__":
    main()
