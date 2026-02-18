# Research strategy — テキスト一覧（全文参照用）

**構成**: `.docx` は **`docx/`**、抽出した **`.txt`** は **`txt/`** に格納。参照論文の PDF/テキストは `Reference paper/` 内。

---

## 研究方針・グループディスカッション（docx/ → txt/）

| テキストファイル | 元 | 内容 |
|------------------|-----|------|
| `txt/250812_研究方針.txt` | docx/250812_研究方針.docx | 研究方針：熱失活メカニズム、BMA/MPC/MTAC、Tamasi との差別化、BO・DLS 等 |
| `txt/250820須藤_グループディスカッション_英語.txt` | 同 .docx | グループディスカッション |
| `txt/250926須藤_グループディスカッション_英語_修正.txt` | 同 .docx | グループディスカッション（修正版） |
| `txt/251024須藤_グループディスカッション_英語_完成.txt` | 同 .docx | グループディスカッション（完成版・英語） |
| `txt/251118須藤_グループディスカッション_英語.txt` | 同 .docx | グループディスカッション |
| `txt/251226須藤_グループディスカッション_英語.txt` | 同 .docx | グループディスカッション |

**追加した .docx がある場合**: `docx/` に .docx を置き、`python3 extract_docx_to_txt.py` を実行すると `txt/` に .txt が生成される。

---

## 参照論文・SI（.pdf → .txt、全文）

いずれも `Reference paper/` フォルダ内。`pdftotext` で PDF 全文を抽出済み。

### 本論文 4 本

| テキストファイル | 元 PDF | 内容 |
|------------------|--------|------|
| `Reference paper/Advanced Materials - 2022 - Tamasi - Machine Learning on a Robotic Platform for the Design of Polymer Protein Hybrids.txt` | 同 .pdf | Tamasi et al. 2022, Adv. Mater. — ML + ロボット、REA、Learn–Design–Build–Test |
| `Reference paper/High throughput screening for the design of protein binding polymers.txt` | 同 .pdf | RSC Chemical Science 2025 — FRET HTS、288 ポリマー、8 酵素、熱安定性 |
| `Reference paper/RSC_Nagao.txt` | RSC_Nagao.pdf | Nagao et al., Nanoscale 2024 — 糖鎖ポリマー、BO、GPR、CTB |
| `Reference paper/Science_RHP.txt` | Science_RHP.pdf | Panganiban et al., Science — RHP、タンパク質表面パターン、有機溶媒中安定化 |

### SI（Supporting Information）

| テキストファイル | 元 PDF | 内容 |
|------------------|--------|------|
| `Reference paper/SI_Advanced.txt` | SI_Advanced.pdf | Tamasi 論文の SI |
| `Reference paper/SI_High throughput screening for the design of protein binding polymers.txt` | 同 .pdf | RSC HTS 論文の SI |
| `Reference paper/SI_Nagao.txt` | SI_Nagao.pdf | Nagao 論文の SI |
| `Reference paper/Science_SI.txt` | Science_SI.pdf | Science RHP 論文の SI |

---

## 図の抽出（PDF / DOCX 内の画像）

図だけ画像ファイルとして保存したものは **`figures/`** に入っている。

- **参照論文・SI**: `figures/Reference_paper/{元PDF名}/fig-000.png`, `fig-001.png`, …（文書内の 1 枚目, 2 枚目, …）
- **研究方針・ディスカッション**: `figures/docx/{元DOCX名}_fig_00.png`, `_fig_01.png`, …

**「どのファイルの Figure いくつか」** の対応は **`figures/FIGURES_INDEX.md`** に一覧した。  
ファイル名はキャプション（Figure 1, Table S2 等）にそろえてあり、例: `Figure_01.png`, `Table_S02.png`。  
再抽出するときは `python3 extract_figures.py` のあと `python3 rename_figures_by_caption.py` を実行する。

---

## 参照のしかた

- **自分 / Codex**: チャットで `@research strategy/txt/250812_研究方針.txt` や `@research strategy/Reference paper/SI_Advanced.txt` のように指定すると、その全文を参照できる。
- **要約のみ**: `Reference paper/REFERENCE_PAPERS_SUMMARY.md` に論文ごとの要約とパイプラインとの対応を記載している。
- **図**: `@research strategy/figures/Reference_paper/SI_Advanced/fig-005.png` のように画像パスを指定すると参照できる。一覧は `figures/FIGURES_INDEX.md`。
