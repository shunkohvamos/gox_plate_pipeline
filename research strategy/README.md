# Research strategy — フォルダ構成

研究方針・グループディスカッション・参照論文を整理したフォルダ。

---

## フォルダ一覧

| フォルダ | 内容 |
|----------|------|
| **docx/** | 研究方針・グループディスカッションの Word 文書（.docx） |
| **txt/** | 上記 .docx から抽出した全文テキスト（.txt）。参照・検索用。 |
| **Reference paper/** | 参照論文の PDF・その全文 .txt・要約（REFERENCE_PAPERS_SUMMARY.md） |
| **figures/** | PDF / DOCX から抽出した図（Reference_paper/ と docx/ の画像）。一覧は figures/FIGURES_INDEX.md |

---

## スクリプト（直下）

- **extract_docx_to_txt.py** — `docx/*.docx` からテキストを抽出し `txt/*.txt` に保存。
- **extract_figures.py** — `Reference paper/*.pdf` と `docx/*.docx` から図を抽出し `figures/` に保存。
- **rename_figures_by_caption.py** — 抽出した図をキャプション（Figure 1, Table S2 等）に合わせてリネーム。

---

## 参照のしかた

- テキスト一覧・図の一覧: **README_TXT_INDEX.md**
- 研究の全体像・参照論文の図の種類: **RESEARCH_OVERVIEW_AND_REFERENCES.md**
