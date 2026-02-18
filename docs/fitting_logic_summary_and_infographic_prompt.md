# フィッティングロジック要約（研究報告書用）とインフォグラフィック用プロンプト

**対象範囲**: ここでいう**フィッティング**は、**生データ（時間–シグナル）から傾き（初速）を推定するところまで**のみを指す。REA の計算・t50 の算出・ランキング図などはフィッティングの外の後段処理とする。

---

## 1. 研究報告書に書く「フィッティングの方法」要約（傾きフィットのみ）

### 1.1 初速（initial rate）の定義と目的

- **定義**: 反応の**初期レジーム**における、時間–シグナル曲線の**局所線形近似の傾き**を初速（絶対活性）とする。基質枯渇・生成物阻害・熱失活の影響がまだ小さい区間で、傾きが「反応速度」として解釈できる範囲を対象とする。
- **目的**: GOx の熱耐性評価のため、加熱時間ごとに「その時点での酵素活性」を再現可能に推定する。できるだけ早い妥当な線形窓を採用し、すでに曲がり始めた区間を直線でフィットしない。

### 1.2 データの前処理と窓の候補生成

- **入力**: ウェル単位の時系列（`time_s`, `signal`）。欠損・非数値は除外し、`time_s` でソート。
- **ステップジャンプ検出**: シグナルの急な段差（閾値 25%）を検出し、ジャンプ以降の点は除外。ジャンプ前の区間のみをフィット定義域とする。
- **開始点の検出（ラグの扱い）**:
  - **本当のラグ**（混合・平衡・デッドタイムなどによる持続的な平坦/ノイジーな立ち上がり）がある場合に限り、先頭数点をスキップして開始インデックスを遅らせる。スキップ上限（点数）を設ける。
  - **浅いディップ**（1点程度の下がりや小さな揺れ）はラグとみなさず、開始点は 0 のままにし、先頭の点も候補に含める。
- **窓候補**: 開始インデックス以降で、**連続した区間**のみを対象に、最小点数〜最大点数（例: 6〜12 点）、最小時間スパンを満たすすべての (開始, 終了) の組み合わせを列挙する。

### 1.3 各窓のフィットと診断指標

- **フィット**: 各窓内の (time_s, signal) に線形回帰（デフォルト OLS、オプションで Theil–Sen）を適用し、傾き・切片・R² を算出。必要に応じて 1 点/2 点トリムした再フィットも計算する。
- **反応らしさの指標**: 窓内の単調性（mono_frac, down_steps）、正のステップ数（pos_steps）、シグナル変化量（dy）、SNR（|dy|/RMSE）、曲率の目安（前半・後半の傾き差 slope_half_drop_frac）などを算出し、窓選択の判定に用いる。

### 1.4 窓の選択（どの窓を初速として採用するか）

- **ハード条件**: 傾き ≥ 0、R² ≥ 閾値（例 0.98）、t_end が上限以内、dy ≥ min_delta_y、mono_frac ≥ 閾値、down_steps ≤ 許容数、pos_steps ≥ 最小数、SNR ≥ 閾値。これらを満たす窓のみを候補とする。
- **選択方針（method: initial_positive）**:
  1. 上記ハード条件を満たす窓のうち、傾きが「最大傾きの (1 − slope_drop_frac) 以上」のものに絞る（曲がり区間を避ける）。
  2. **できるだけ早い t_end** の窓を優先する（初期レジームを採用）。
  3. 同程度なら、R² が高く、点数が多く、t_start が早い順で tie-break。
  4. 早期の窓の傾きが後期より明らかに大きい（例: 1.3 倍以上）場合は、早期で傾きが最大に近い窓を優先する（true initial rate の保護）。
- **長窓の優先（force_whole）**: 同じ開始点からなる窓のうち、点数が十分で R² が高く、傾きの落ち込みが小さい「ほぼ全区間線形」な窓があれば、それを採用するオプションがある。

### 1.5 救済・後処理（局所的な修正のみ）

- **段階的緩和**: 厳しい閾値で候補が残らない場合、R²・mono_frac・min_pos_steps などを段階的に緩和して再選択を試す。
- **短窓救済**: それでも選べない場合、点数を 5→4→3 と減らし、R² も緩和した候補で再試行。救済時も start_idx=0 を優先して初速を維持する。
- **外れ値対応**: 1 点飛ばしの full-range フィット、窓内の内部外れ値の除去、 curvature 検出による窓の短縮などを行うが、いずれも「初速を過小評価しない」ガード（_protect_initial_rate）をかける。
- **最終安全チェック**: 選択された窓が、傾き・R²・t_end・dy・mono・SNR の最終閾値を満たすことを強制する。
- **last resort**: 上記すべてで選べない場合、データの**前半のみ**を対象に短い窓（4〜8 点）を走査し、R²・SNR の緩い閾値で「最初に許容できる窓」を返す（初期レジーム優先）。

### 1.6 フィッティングの出力

- **出力**: 選択された窓の**傾き**をそのウェルの初速（絶対活性）として出力する（行平均はとらない）。この傾きが後段の REA 計算・t50 算出・ランキングなどに使われる。

### 1.7 追跡可能性

- 先頭点をスキップした場合は、スキップした点数または開始時刻を CSV/ログに記録する。
- 選択に用いたメソッド（initial_positive, force_whole, last_resort, col1_rescue など）を `select_method_used` として出力する。
- すべての派生データは run_id と manifest により生データに追跡可能とする。

---

## 2. フィッティング（傾きフィット）が直接関係する図

傾きフィッティングの結果を**そのまま可視化している**図のみ。

| 図の種類 | 内容の一言 |
|----------|------------|
| ウェル別 PNG (`{well}.png`) | 1 ウェルの時系列＋選択された線形窓＋フィット線 |
| plate_grid__{run_id}__plate{N}.png | プレート上のウェル配置でフィット結果（傾き等）を一覧 |
| fit_qc_report.md / 関連 CSV | フィット QC サマリ |

※ REA・t50・ランキング・FoG などの図は、フィッティングで得た**傾き**を入力とする後段処理の出力である。

---

## 3. 目的関数の説明（研究報告書用）

熱耐性評価と最適化のため、**「耐熱性（FoG）を上げつつ、初期活性（U0）を維持・向上させる」**という方針を一つのスカラーにまとめた目的関数を定義している。研究報告書では以下のように説明できる。

### 3.1 用語と前提

- **FoG（Fold over GOx）**: 各ポリマーの t50 を、同一ラン（同一プレート）の裸 GOx の t50 で割った比。FoG > 1 は GOx より耐熱性が高いことを意味する。
- **U0（絶対活性 at 0 min）**: 加熱前（heat_min = 0）の初速フィットから得た傾き。溶媒・条件を揃えた基準（GOx または溶媒別コントロール）で正規化した値を U0* とする。
- **θ（シータ）**: 初期活性の** feasibility（実行可能性）閾値**。ネイティブ相対活性（GOx 基準）が θ 未満のポリマーは「活性を損なっている」とみなし、主目的では不採用または別扱いとする。デフォルトは **θ = 0.70**（感度解析では 0.50〜0.90 を検討）。

### 3.2 主目的関数（log-linear）

**方針**: 耐熱性（FoG*）と初期活性（U0*）の両方を正に評価し、対数空間で線形に結合する。

- **定義**（溶媒マッチ正規化 FoG*, U0* を用いる場合）:
  - **S = log(FoG\*) + λ·log(U0\*)**
  - デフォルトは **λ = 1** のため、S = log(FoG* · U0*) となり、指数を取ると **exp(S) = FoG* × U0*** である。
- **解釈**: FoG* と U0* の両方が大きいほど S は大きくなる。耐熱性だけを追うと U0 が下がりがちなトレードオフを、λ で「活性の重み」を明示してバランスさせる。
- **実行可能性**: 主解析では θ を設け、U0（ネイティブ相対）≥ θ のポリマーだけをランキング・意思決定の対象とする場合がある（mainA の縦線 θ など）。θ 未満は「非実行」とみなし、FoG のみで順位づけしない。

### 3.3 溶媒バランス付き目的（FoG–activity）

溶媒別コントロールで正規化した FoG*・U0* を使う場合、**活性によるペナルティ・ボーナス**をかけた形もある。

- **S_solv = FoG* × activity_factor(U0\*)**
- **activity_factor**:
  - U0* ≤ 1 のとき: **down_penalty** = clip(U0*, 0, 1)^指数（指数は例: 2）。U0* が小さいほど減点。
  - U0* > 1+deadband のとき: **up_bonus** = 1 + 係数×[U0* − (1+deadband)] の上限クリップ。基準より高い活性にボーナス。
- 報告書では「溶媒を揃えた上で、耐熱性（FoG*）に初期活性（U0*）のペナルティ・ボーナスを乗じた複合スコア」と述べればよい。

### 3.4 補足指標（t_θ）

- **t_θ**: 加熱時間 t に対して正規化活性 U(t) が**初めて θ を下回る**時間。U(0) < θ の場合は「already_below」、一度も下回らない場合は「never_cross」（右打ち切り）として記録する。
- 報告書では「活性が閾値 θ を下回るまでの時間」として、耐熱性と活性維持の両方を一つの曲線で議論する際に用いると書ける。

### 3.5 図との対応（簡潔に）

- **mainA**: 横軸 U0（ネイティブ相対または log U0*）、縦軸 FoG（または log FoG*）。θ の縦線で「実行可能域」を示す。等スコア線は S = const の直線（log 空間）。
- **mainB**: U0–FoG のトレードオフとパレートフロント。
- **mainC**: U0–FoG 平面上の目的関数の等高線（contour）。
- **mainD**: 目的関数を 3D で表示（hill）。

報告書では「目的関数 S を定義し、mainA〜mainD で可視化してランキング・意思決定に用いた」とまとめられる。

---

## 4. インフォグラフィック用プロンプト（傾きフィット用・Google 画像生成 Nano banana pro 向け）

**対象**: フィッティング＝**生データから傾きのフィットまで**のみ。1 枚絵はこの部分に絞る。

以下を **英語で**、1 枚絵のインフォグラフィックを生成するための**詳細な描画プロンプト**としてそのまま使うか、必要に応じて短くして使用してください。

---

**Prompt (English, for a single-panel infographic):**

Create a clear, professional **single-panel scientific infographic** that explains **only the slope-fitting step**: from raw time–signal data to a chosen linear window and its slope (initial rate). Do not show REA, t50, or ranking; the output of this step is the slope per well. Style: flat design, high contrast, sans-serif labels, suitable for a slide or poster. No 3D or photorealistic rendering.

**Layout (top to bottom or left-to-right flow):**

1. **Title area**  
   Text: “Initial rate fitting — from raw signal to slope”.  
   Subtitle: “Earliest valid linear window → slope = initial rate (per well).”

2. **Data flow (3–4 horizontal blocks with short arrows):**

   - **Block A — Raw & prep**  
     - Sketch: a **time–signal curve** (time x-axis, signal y-axis) with a short “lag” at the start, then rise, then curvature.  
     - Labels: “Raw well trace”, “Step-jump truncation”, “Start index (skip only true lag)”.

   - **Block B — Windows**  
     - Several **overlapping horizontal segments** on the same time axis: “early short”, “early long”, “late”.  
     - Highlight the **earliest acceptable segment** in a different color.  
     - Labels: “Candidate windows (min–max points)”, “Earliest linear window selected”.

   - **Block C — Selection rules**  
     - Checklist/badges: “Slope ≥ 0”, “R² ≥ threshold”, “Early t_end preferred”, “Curvature guard (slope drop)”, “Reaction-like (mono, SNR)”.  
     - One line: “Rescue: relax R² / points → last resort: first-half short window”.

   - **Block D — Output**  
     - One box: “Slope = initial rate (per well)”. Optional one short line: “(Used later for REA, t50, ranking.)”

3. **Single main sketch:**  
   One **time vs signal** plot with the **selected linear segment** and the **fitted straight line** on the early part. Label: “Well fit: chosen window + slope”.

4. **Principles (short text callouts):**  
   - “Earliest valid linear window = initial rate.”  
   - “Do not fit into curved region.”  
   - “Skip leading points only for true lag; shallow dip → keep start at 0.”  
   - “Provenance: run_id, manifest, lineage.”

Use a **consistent color scheme**: e.g. blue for “raw/data”, green for “selected window / slope”, orange for “rescue”. All text in **English**, labels readable at small size. No file paths or code in the image.

---

**短縮版（キャプション用）:**  
“Infographic: slope fitting only — raw time–signal → candidate windows → earliest linear window selected (R², slope, curvature guard) → output = slope per well (initial rate). One panel, flat design, English labels.”

---

以上で、研究報告書用の「傾きフィッティングの方法」（§1）、「目的関数の説明」（§3）、および傾きフィット部分に限定した 1 枚絵プロンプト（§4）を揃えています。
