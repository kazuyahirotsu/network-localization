# 今後のタスクリスト (TODO)

## 優先度：高 (Chapter 5のための追加実験・解析)

1.  [ ] **ドローン測位の感度解析 (Sensitivity Analysis)**
    *   既存のドローン飛行データ（Paper 2の実験データまたはシミュレーションデータ）を用意する。
    *   可能であればMatlab環境上でドローン軌跡を模擬し、環境の不整合（Matlab vs Omnet）を低減する。
    *   以下の2パターンのノイズをビーコン座標に乗せて、ドローンの測位誤差（Huber Loss使用）がどう変化するかグラフ化する。
        *   **Bias Noise:** 全ビーコンを同じ方向（例: X+50m, Y+50m）にずらす。
        *   **Random Noise:** 各ビーコンをガウス分布 $\mathcal{N}(0, \sigma^2)$ でずらす（$\sigma = 10, 50, 100, 200$m）。
    *   目的：ランダムノイズへの耐性と、バイアスノイズの脆弱性（そのまま誤差になること）を確認し、Chapter 5の議論の土台にする。

2.  [ ] **GCNの精度飽和（Saturation）の検証実験**
    *   Free-space環境において、ビーコン密度をさらに高くした（例：32倍、64倍など可能な範囲で）データセットを生成する。
    *   GCNを学習させ、誤差がどこで下げ止まるか（Saturation Point）を確認する。
    *   目的：「50mの壁」が環境要因（地形）だけではなく、手法やRSSI自体の限界であることを示唆するデータを得る。

## 優先度：中 (論文執筆・統合)

3.  [ ] **表記の統一 (Notation Standardization)**
    *   Paper 2 (Drone) の数式表記を、Paper 1 (GCN) の物理モデルベースの記法に書き換える。
    *   $TP \to P_t$, $L2 \to \mathcal{L}_{L2}$ などの置換作業。

4.  [ ] **Chapter 3 (Drone) のロジック修正**
    *   高度制約（Altitude Constraint）の導入理由を、「インフラ側の不確実性（Z軸推定の難しさ）からのDecoupling（分離）」という文脈で説明し直す文章を作成する。

5.  [ ] **Chapter 4 (GCN) のロジック修正**
    *   "Physically-Informed" の定義を、「物理法則（対数距離則）を事前知識として構造に組み込んだもの」と明確に記述する。
    *   **Site-Specific Learning:** 本モデルは対象エリアの地形データを用いた過学習（Site-Specific Optimization）を前提としており、汎化性能ではなく特定環境での最適化を目指すものであると記述する。

6.  [ ] **Chapter 5 (Integrated Analysis) のDiscussion執筆**
    *   **Computation Offloading:** 計算はベースステーションに集約するモデルであることを記述。
    *   **Wake-up Radio:** バッテリー/法規制対策として、Wake-up on Demand技術の利用を将来展望として記述。
    *   **Scalability:** ビーコン密度向上に伴う通信衝突（Collision）のリスクと、容量限界について言及する。

## 優先度：低 (体裁・その他)

7.  [ ] **タイトルの最終決定**
    *   案：*Resilient UAV Localization in GNSS-Denied Environments via LPWA: Coupling Graph-Based Infrastructure Estimation with Robust Navigation*
    *   指導教官と相談して確定する。

8.  [ ] **参考文献の統合**
    *   2つの論文のBibTeXをマージし、重複を削除して整理する。
