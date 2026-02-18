# 図の抽出一覧 — キャプション（Figure XX / Table X）にそろえたファイル名

PDF および DOCX から画像を抽出したあと、**元のキャプション（Figure 1, Table S2 など）を検出してファイル名に反映**している。  
「どのファイルの Figure いくつか」がそのままファイル名になる。

---

## ルール

- **フォルダ名 / ファイル名のプレフィックス** = 元の PDF または DOCX
- **ファイル名** = キャプションから生成（`Figure_01.png`, `Table_S02.png` など）。キャプションが検出できなかった画像は `Fig_001.png` のように通し番号。
- キャプションは「行の先頭の Figure N. / Table N. / Fig. N.」で検出。本文中の参照「(Figure 1)」は含めない。
- 1 枚目・2 枚目…の順にキャプションを 1:1 で割り当てるため、**図の順序とキャプションの順序が原稿と異なる場合は、目視で確認**すると確実。

---

## 参照論文・SI（PDF → `figures/Reference_paper/`）

各 PDF ごとに **1 フォルダ**。中身はキャプションにそろえた名前（例: `Figure_01.png`, `Figure_02.png`, `Table_S01.png`）と、キャプションがなかった分の `Fig_003.png` など。

| 元 PDF | 保存先フォルダ | 命名の例 |
|--------|----------------|----------|
| Advanced Materials - 2022 - Tamasi - ... | `Reference_paper/Advanced_Materials_-_2022_-_Tamasi_-_Machine_Learning_on_a_Robotic_Platform_for_the_Design_of_Polymer_Protein_Hybrids/` | Figure_01 … Figure_04, Fig_004, Fig_005（キャプション 4 件検出） |
| High throughput screening ... | `Reference_paper/High_throughput_screening_for_the_design_of_protein_binding_polymers/` | Fig_000 …（キャプション形式により 0 件の場合は通し番号のみ） |
| RSC_Nagao.pdf | `Reference_paper/RSC_Nagao/` | 同上 |
| SI_Advanced.pdf | `Reference_paper/SI_Advanced/` | キャプションがあれば Figure_01 等、なければ Fig_xxx |
| SI_High throughput ... | `Reference_paper/SI_High_throughput_screening_for_the_design_of_protein_binding_polymers/` | Figure_01 等（14 件検出時） |
| SI_Nagao.pdf | `Reference_paper/SI_Nagao/` | Figure_01 等（28 件検出時） |
| Science_RHP.pdf | `Reference_paper/Science_RHP/` | Figure_01 等（3 件検出時） |
| Science_SI.pdf | `Reference_paper/Science_SI/` | Figure_01 等（32 件検出時） |

**例**: Tamasi 本論文の **Figure 1** → `.../Advanced_Materials_.../Figure_01.png` を参照。

---

## 研究方針・グループディスカッション（DOCX → `figures/docx/`）

**ファイル名** = `{元DOCXのファイル名（拡張子なし）}_{キャプション}.png`（例: `251024須藤_..._完成_Figure_03.png`）。キャプションが足りない分は `…_Fig_012.png` のように通し番号。

| 元 DOCX（`docx/` 内） | 命名の例 |
|------------------------|----------|
| docx/250812_研究方針.docx | 250812_研究方針_Figure_01.png, _Fig_002.png 等（キャプション 2 件検出） |
| docx/250820須藤_グループディスカッション_英語.docx | 250820須藤_..._英語_Figure_01.png …（6 件検出） |
| docx/251024須藤_グループディスカッション_英語_完成.docx | 251024須藤_..._完成_Figure_01.png …（19 件検出） |
| 他も同様 | {stem}_Figure_XX または {stem}_Fig_xxx |

**例**: 「251024完成版の Figure 3」→ `figures/docx/251024須藤_グループディスカッション_英語_完成_Figure_03.png`。

---

## 再抽出・リネームの手順

1. **図の抽出**（上書きされる）  
   `python3 extract_figures.py`（PDF は `Reference paper/`、DOCX は `docx/` を参照）

2. **キャプションにそろえたリネーム**  
   `python3 rename_figures_by_caption.py`

上記の順で実行すると、ファイル名が「Figure XX / Table X」にそろう。  
キャプションが検出されない PDF（表記が異なるなど）では、従来どおり `Fig_000.png` … のままになる。
