# FoG計算時の警告情報

生成日時: 2026-02-04 18:10:23

## ⚠️ 異常GOx t50値の検出

### 1. Round R1

- **GOx t50 中央値**: 34.022 min
- **異常値の閾値**: [11.227, 102.067] min
- **検出された異常値**:
  - 260203-2/plate3: 0.227 min

**処理**: 異常値はround平均GOx t50の計算に**含まれています**。

---

## 「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」を実行した場合

`--exclude_outlier_gox`フラグを有効にして実行すると、以下の変更が行われます：

### 変更点

1. **異常GOx t50値の除外**:
   - 検出された異常GOx t50値がround平均GOx t50の計算から除外されます
   - これにより、round平均GOx t50がより安定した値になります

2. **FoG値への影響**:
   - 異常GOx t50値を持つplateのポリマー:
     - そのplateにGOxがある場合: `same_plate`のGOxを使うため、**影響なし**
     - そのplateにGOxがない場合: `same_round`のGOxを使うため、**round平均GOx t50の変更の影響を受ける**
   - 他のplateのポリマー:
     - `same_round`のGOxを使う場合: **round平均GOx t50の変更の影響を受ける**
     - `same_plate`のGOxを使う場合: **影響なし**

3. **出力ファイルの変更**:
   - `fog_plate_aware.csv`: 各ポリマーのFoG値が変更される可能性があります
   - `fog_plate_aware_round_averaged.csv`: Round平均FoG値が変更される可能性があります

### 実行方法

VS Codeの「実行とデバッグ」パネルから、以下の設定を選択してください：

- **「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」**

または、コマンドラインから：

```bash
python scripts/build_fog_plate_aware.py \
  --run_round_map meta/bo_run_round_map.tsv \
  --processed_dir data/processed \
  --out_dir data/processed \
  --exclude_outlier_gox
```

### 注意事項

- 異常値を除外すると、round平均GOx t50が変わるため、FoG値も変わります
- 除外するかどうかは、データの品質と研究の目的に応じて判断してください
- 異常値が多数ある場合、除外によりround平均GOx t50が大きく変わる可能性があります