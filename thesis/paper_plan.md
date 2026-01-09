# Thesis Plan: Resilient UAV Navigation in GNSS-Denied Environments via Physics-Aware Graph Learning and Robust Multilateration
# (物理モデルを組み込んだグラフ学習とロバスト三辺測量によるGNSS拒否環境下での自律UAV航法)

## Thesis Abstract (要旨)
*   **Overview (To be expanded to ~4 pages in thesis):**
    本論文は、GNSS（衛星測位システム）が利用できない、または信頼できない過酷な環境（災害地、山岳部、妨害下）において、LPWA（Low-Power Wide-Area）通信網を用いてUAVの自律航行を支援する「即応的ナビゲーションインフラ」の構築手法と実現可能性を論じるものである。本研究は、インフラの自己組織化とUAVのロバスト航行を統合したシステム全体を設計・評価する。

*   **1. Background & Motivation:**
    *   UAVの産業利用（物流、監視、災害対応）において、GNSSへの依存は重大な脆弱性（Single Point of Failure）となっている。特に山間部や都市の峡谷、あるいは意図的な妨害（Jamming/Spoofing）が存在する環境では、代替手段が必要となる。
    *   Visual SLAMやLiDARといった既存の自律航法技術は、計算コストが高く、特徴点の少ない環境（砂漠、海上）や視界不良下では機能しない課題がある。
    *   本研究では、低消費電力・長距離通信・低コストを特徴とするLPWA規格「LoRa」に着目し、RSSI（受信信号強度）のみを用いた軽量な測位システムの実現を目指す。

*   **2. Related Work:**

*   **3. System Architecture:**
    *   **Concept:** インフラ（ビーコン）とクライアント（UAV）を機能的に分離した非対称システム。
    *   **Infrastructure (Beacons):** 地上セグメント。投下直後（位置未知）は相互通信によりデータを収集し、中央ハブでのGCN推論により位置推定を受ける（Self-Organization Process）。位置確定後、推定座標をグローバルにブロードキャストする（Broadcast Mode）。
    *   **Client (UAV):** 飛翔体セグメント。ビーコンからの信号を一方的に受信（Passive Receive）し、オンボードで自身の位置を計算する。これにより、UAV台数が増加しても通信帯域を圧迫しないスケーラビリティを確保する。

*   **4. Robust UAV Navigation (Chapter 4):**
    *   **Objective:** インフラ位置が既知の場合のクライアント側測位精度の向上。
    *   **Methodology:** RSSIの大きな変動に対処するため、「高度（Z軸）の分離（Decoupling）」と「Huber Lossによる外れ値抑制」を導入。
    *   **Results:** 安価なLPWA信号でも安定した水平位置推定を実現（Sim: 4.5m median error）。

*   **5. Infrastructure Self-Organization (Chapter 5):**
    *   **Objective:** インフラ位置が未知（GNSS拒否下の災害地等）の場合の自己位置推定。
    *   **Methodology:** 航空機から投下されたビーコンが、相互通信と地形情報（DEM）を用いた「Physics-Aware GCN」により、自律的に自身の位置を推定する。
    *   **Results:** 従来手法（Multilateration）と比較して90%以上の精度向上（Sim: ~200m mean error）。

*   **6. Integrated Feasibility and Deployment Strategy (Chapter 6):**
    *   **System Integration:** Chapter 5のインフラ誤差（~150m）環境下で、Chapter 4のUAV測位が実用に耐えうるかを統合的に評価。
    *   **Geometric Stability:** UAVの測位誤差が大きくても、目的地までの距離が十分長ければ方位角誤差は10度未満に収まり、安定した巡航が可能。
    *   **Operational Feasibility:** ピンポイント着陸は困難だが、広域捜索（Area Sweep）や回廊航行においては、飛行パスのオーバーラップ率調整により任務遂行が可能。
    *   **Turning Point Problem:** ウェイポイント接近時の不安定性に対し、到達判定閾値緩和（~200m）による解決策を提示。

*   **7. Conclusion:**
    *   提案システムは、従来の精密測位（GPS）の代替ではないが、GNSS喪失下における「生存可能なナビゲーション（Survivable Navigation）」を提供する実用的なソリューションである。

## Chapter 1: Introduction (序論)
*   **1.1 Background**
    *   UAVの産業利用の拡大（物流、監視、災害対応）とGNSS依存。
    *   GNSS脆弱性の3カテゴリ:
        1.  環境的遮蔽（都市峡谷、森林、山岳、屋内）
        2.  意図的妨害（ジャミング、スプーフィング）
        3.  地球外環境（月、火星）
    *   既存代替手段の課題: Visual SLAM（計算負荷、特徴点依存）、LiDAR（コスト、重量）、INS（ドリフト蓄積）。
    *   なぜLPWA (LoRa) か: 長距離通信（2-15km）、低消費電力、低コスト、既存ハードウェア利用可能。
*   **1.2 Problem Statement**
    *   **Challenge 1（クライアント側）:** RSSIの高分散（マルチパス、シャドーイング）により測位精度低下。TDoA/AoAは高精度だがGNSS同期や専用アンテナが必要。
    *   **Challenge 2（インフラ側）:** GNSS拒否環境ではビーコン自体の位置決めが困難。Iterative Multilaterationは疎なネットワークで誤差伝播。
*   **1.3 Contributions**
    1.  **Robust Client-Side Localization:** 高度制約 + Huber損失による多辺測量。Static 4.5m、Dynamic 62.9m（従来比25%改善）。
    2.  **Physics-Aware GCN:** 対数距離パスロスモデルをGCNに組み込み。Mean 206m（Multilateration 1974m比90%改善）。
    3.  **Integrated Feasibility Analysis:** 150-200m誤差でも幾何学的安定性により10度未満の方位誤差。Turning Point Problem対策。
*   **1.4 Thesis Outline**
    *   Chapter 2: Related Work（無線測位技術、UAV測位、ネットワーク自己位置推定）
    *   Chapter 3: System Architecture（全体設計、インフラ・クライアント・ハブの役割）
    *   Chapter 4: Robust UAV Localization（高度制約 + Huber損失、シミュレーション、実験）
    *   Chapter 5: Infrastructure Self-Localization（Physics-Aware GCN、Longley-Rice評価）
    *   Chapter 6: Integrated System Analysis（感度解析、航行実現性、デプロイメント戦略）
    *   Chapter 7: Conclusion（貢献まとめ、今後の課題）

## Chapter 2: Related Work (関連研究)
*   **2.1 Wireless Positioning Fundamentals**
    *   **Signal Metrics Comparison:** RSSI (強度), ToF/TDoA (時間), AoA (角度) の比較。
    *   **Why LoRa/RSSI?:** ToF/TDoAは高精度だが厳密な時刻同期（通常はGPS由来）が必要。AoAはアレイアンテナが必要。本研究は、最も安価で既存のIoTハードウェア（単一アンテナ・非同期）で実現可能なRSSI方式に焦点を当てる。

*   **2.2 Client-Side: UAV Localization Approaches**
    *   **Non-RF Methods (SLAM/LiDAR):** 高精度だが計算負荷が高く、特徴点のない環境（海、砂漠）や視界不良時（煙、霧）に機能しない。
    *   **RF-based UAV Localization:**
        *   *TDoA methods (e.g., Fargas et al.):* 高精度だが、インフラ側の時刻同期にGPSを必要とするケースが多く、GNSS拒否環境下のバックアップとしては矛盾が生じる。
        *   *Hybrid RSSI/AoA (e.g., Baik et al.):* RSSIと到来角を併用して精度（~3m）を出す研究もあるが、専用のデュアルアンテナや位相変調器が必要となる。
    *   **Our Positioning:** 特別なハードウェアを必要とせず、一般的なLoRaモジュールのみを用い、アルゴリズム（高度制約 + Huber Loss）のみで実用的なロバスト性を確保する点に独自性がある。

*   **2.3 Infrastructure-Side: Network Self-Localization**
    *   **Classical Geometric Solvers:**
        *   *Iterative Multilateration:* 位置が判明したノードを順次アンカーとして利用する手法。疎なネットワークやNLOS環境では、初期の推定誤差が連鎖的に拡大（Error Propagation）し、システム全体が破綻する課題がある (Hada et al.)。
    *   **Data-Driven Approaches:**
        *   *Fingerprinting (e.g., Purohit et al.):* 高精度だが、事前の実地調査（サイトサーベイ）によるデータベース構築が必須。即応性が求められる災害地や未踏地（Unknown Environment）には適用不可能。
    *   **Graph Neural Networks (GNN):**
        *   センサーネットワークや屋内WiFi測位へのGNN応用が進んでいる (Vishwakarma et al.)。しかし、多くは屋内などの高密度環境を想定しているか、物理特性を考慮しないブラックボックス的なアプローチである。
    *   **Our Positioning:** 本研究は、物理モデル（対数距離パスロス式）をGNNの構造に明示的に組み込む「Physics-Aware」なアプローチを提案する。また、地形効果（Longley-Riceモデル）が支配的な、疎な屋外ネットワークでの自己位置推定に特化している点で新規性がある。

## Chapter 3: System Architecture (システムアーキテクチャ)
*   *New Content: Defining the Asymmetric System*
*   **3.1 Overall Concept**
    *   **Goal:** GNSS拒否環境下での即応的なナビゲーションインフラの構築。
    *   **Roles:** インフラ側（ビーコン）は「位置情報の提供者」、クライアント側（UAV）は「利用者」として明確に分離する。
*   **3.2 Infrastructure Segment (Beacons)**
    *   **Function:** 自律的な位置推定と、グローバルな位置情報のブロードキャスト。
    *   **Mode A (Self-Organization):** 近隣ビーコンとのRSSI計測を行い、中央ハブへデータを送信。
    *   **Mode B (Service Operation):** 推定された座標とIDを周期的に送信（Beaconing）。
    *   **Energy Management (Wake-up on Demand):** バッテリー消費を抑えるため、デフォルトではスリープ（Deep Sleep）状態。UAVからの「Wake-up信号」を受信した際のみ、一定時間（例: 数分間）ブロードキャストを行う。
*   **3.3 Client Segment (UAV)**
    *   **Function:** パッシブな受信による自己位置推定。
    *   **Privacy & Scalability:** UAVは信号を発しないため、無限の台数が同時に利用可能であり、かつ自身の位置を外部に漏らさない。
    *   **Active Request:** 必要な信号が受信できない場合のみ、Wake-up信号をブロードキャストする。単一の信号でエリア内の全ビーコンを起動でき、他のUAVも便乗できるため、多数機運用時でも通信帯域を圧迫しない。
*   **3.4 Hub Segment (Base Station / Mother UAV)**
    *   **Function:** 計算リソースの集約。GCNのトレーニングと推論を行い、ビーコンへ位置情報を配布する。

## Chapter 4: Robust UAV Localization with Altitude Constraints
*   *Based on: LoRa-based Localization for UAVs (IEEE Access, 2024)*

*   **4.1 Problem Formulation**
    *   **Scenario:** ビーコン位置が既知の環境（都市部、既設インフラ利用時）におけるUAV測位。
    *   **Challenge:** RSSIベース測位は、信号減衰・マルチパス伝搬・干渉による大きな分散を持ち、精度が低下しやすい。
    *   **Objective:** 追加のハードウェア（時刻同期、アンテナアレイ）を必要とせず、アルゴリズムのみで精度とロバスト性を向上させる。

*   **4.2 Proposed Method**
    *   **4.2.1 System Overview**
        *   複数のビーコンがLoRa信号を周期的にブロードキャスト（ID・位置情報を含む）。
        *   UAVは受信専用（Passive Receive）で、双方向通信不要 → UAV数増加時もスケーラブル。
    *   **4.2.2 RSSI-to-Distance Conversion**
        *   対数モデル: $d_i = A \log_{10}(\text{RSSI}_i) + B$
        *   係数 $A, B$ は事前の実験データによるカーブフィッティングで決定。
    *   **4.2.3 Multilateration with Huber Loss**
        *   従来手法: 二乗誤差 $L_2 = (d_i - \hat{d}_i)^2$ を最小化。
        *   提案手法: Huber損失関数を使用。
            *   小さな誤差に対しては二乗誤差（精度重視）、大きな誤差に対しては線形誤差（外れ値抑制）。
            *   RSSIの大きな分散に対してロバスト。
    *   **4.2.4 Altitude Constraints (Decoupling Strategy)**
        *   気圧計/レーザー高度計から得られる高度データを制約条件として導入。
        *   3D最適化問題を2D（水平方向）に削減し、垂直方向の誤差が水平方向に伝播することを防止。

*   **4.3 Simulation Study**
    *   **4.3.1 Setup**
        *   シミュレータ: OMNeT++ + INET + FLoRa + OSM Buildings。
        *   環境: ベルリン市街地（0.55 km²）、10ビーコンをビル屋上に配置、UAV高度100m。
        *   パラメータ: SF=7, TP=14dBm, BW=500kHz, Nakagami Fading。
    *   **4.3.2 Evaluation Scenarios**
        *   **Static:** UAVホバリング、500秒間の平均RSSIを使用。4手法の純粋な精度比較。
        *   **Dynamic:** UAV円周飛行（半径200m, 速度10m/s）、5秒間のサンプリング、1Hz更新。
    *   **4.3.3 Compared Methods**
        *   Method 1: 二乗誤差（従来手法）
        *   Method 2: 二乗誤差 + 高度制約
        *   Method 3: Huber損失
        *   Method 4: Huber損失 + 高度制約（提案手法）
    *   **4.3.4 Results**
        *   **Static:** Method 4 は Median Error **4.5m**（2D/3D）、Method 1 は 29.7m/44.3m。
        *   **Dynamic:** Method 4 は Median Error **62.98m**、Std **43.05m**。Method 1 は 84.16m、Std 105.16m。
        *   **Scalability:** Huber損失使用手法はビーコン数増加に伴い精度向上、二乗誤差手法は悪化。
        *   **Error Monitoring:** 最小化誤差と推定誤差の間に強い相関（R²=0.99）があり、大きな誤差を検出可能。

*   **4.4 Experimental Study**
    *   **4.4.1 Hardware**
        *   LoRaモジュール: RFM95W、アンテナ: VERT900、コンピュータ: Raspberry Pi 3B。
        *   UAV: Holybro PX4 Development Kit - X500 V2、フライトコントローラ: Pixhawk 6C。
    *   **4.4.2 Setup**
        *   場所: イタリア・ペルジネ市の駐車場（610 m²）。
        *   4ビーコンを矩形配置、信号送信間隔は平均0.5秒の正規分布。
        *   実験: 手動制御（5-10m高度）と自律制御（15m高度）で楕円軌道・8の字軌道を飛行。
    *   **4.4.3 Results**
        *   手動制御（低高度）: Median Error 7-9m（2D）、提案手法が従来手法をわずかに上回る。
        *   自律制御（高高度）: Median Error 14-26m（2D）、シミュレーションほどの改善は見られず。
        *   改善幅が小さい原因: ビーコン数の不足（4台 vs シミュレーション10台）。

*   **4.5 Discussion**
    *   **4.5.1 Antenna Directivity Issue**
        *   高高度でのパフォーマンス低下は、モノポールアンテナの放射パターンに起因。
        *   アンテナ直上では信号が弱くなる「Cone of Silence」効果。
        *   解決策: 高度パラメータを含むRSSI-距離モデルの導入。
    *   **4.5.2 Sampling Time Optimization**
        *   サンプリング時間と精度にはトレードオフが存在。
        *   静的環境では長時間サンプリングがノイズ低減に有効、動的環境では短時間が必要。
    *   **4.5.3 Scalability**
        *   LoRaの長距離通信（最大15km）により、少数のビーコンで広域をカバー可能。
        *   双方向通信不要の設計により、UAV数増加時も通信帯域を圧迫しない。

## Chapter 5: Infrastructure Self-Localization via Graph Learning
*   *Based on: GCN-Based Localization for Low-Density LPWA Networks in GNSS-Denied Environments*

*   **5.1 Problem Formulation**
    *   **Scenario:** GNSS拒否環境下でのビーコンネットワーク自己位置推定。
        *   アンカーノード（位置既知）: 境界配置、航空投下時のGPS記録、または短時間のGNSS利用で取得。
        *   未知ノード: ピアツーピアRSSI計測のみから位置を推定。
    *   **Graph Formulation:**
        *   有向グラフ $G = (\mathcal{V}, \mathcal{E})$、ノード数64、アンカー:未知 = 1:3（16:48）。
        *   各エッジ $(i, j)$ に対し、$K=10$ 個のRSSIサンプル時系列 $\{r_{ij}^{(k)}\}_{k=1}^K$ を観測。
    *   **Path-Loss Model:**
        *   対数距離モデル: $r_{ij} = P_t - 10n \log_{10}(d_{ij}) + o + \varepsilon$
        *   距離推定: $\hat{d}_{ij} = 10^{(P_t + o - \bar{r}_{ij}) / (10n)}$
        *   $P_t$ は固定（13dBm）、$n$（パスロス指数）と $o$（オフセット）は学習対象。

*   **5.2 Proposed Method**
    *   **5.2.1 Physically-Informed Edge-Conditioned GCN**
        *   Edge-Conditioned Convolution (NNConv): エッジ属性に基づいて動的に重み行列を生成。
        *   $\mathbf{x}_i^{(\ell+1)} = \sigma\left( \text{mean}_{j \in \mathcal{N}(i)} \left( \mathbf{W}_{ij}^{(\ell)} \mathbf{x}_j^{(\ell)} \right) \right)$
        *   $\mathbf{W}_{ij}^{(\ell)} = \text{EdgeNet}(\mathbf{e}_{ij})$: 小規模MLPがエッジ特徴量から重みを生成。
    *   **5.2.2 Edge Features**
        *   ノード特徴量: $\mathbf{x}_i^{(0)} = [x_i, y_i, \text{is\_anchor}]$（未知ノードはアンカー重心で初期化）。
        *   エッジ特徴量: $\mathbf{e}_{ij} = [r_{ij}^{(1)}, \dots, r_{ij}^{(K)}, \hat{d}_{ij}]$（RSSI時系列 + 学習された距離推定）。
        *   物理的解釈可能な距離推定をエッジ特徴量に含めることで、ネットワークにスケール感覚を与える。
    *   **5.2.3 Trainable Path-Loss Module**
        *   グローバルパスロスパラメータ $(n, o)$ をスカラーとして学習。
        *   数値安定性のための制約: 分母クランプ、指数クランプ（[-2, 5]）、距離下限 $\epsilon > 0$。
    *   **5.2.4 Anchor Handling**
        *   アンカー座標は訓練・推論を通じて固定（Ground Truth）。
        *   メッセージパッシングには全ノードが参加するが、アンカーの予測座標は上書きされ、損失計算から除外。
    *   **5.2.5 Training Objective**
        *   損失関数: Smooth L1 (Huber) Loss on unknown nodes.
        *   $\mathcal{L} = \sum_{i \in \mathcal{U}} \rho(\hat{\mathbf{p}}_i - \mathbf{p}_i)$

*   **5.3 Evaluation Setup**
    *   **5.3.1 Dataset**
        *   対象地域: トルコ・チョルム県の4km × 4km 山岳地域（GPS干渉が報告されている地域）。
        *   MATLABベースのデータ生成器で1000グラフインスタンスを生成（800訓練、200テスト）。
        *   ノード配置: 16アンカー（グリッド制約配置）、48未知ノード（一様ランダム）、高さ1.0m。
    *   **5.3.2 Propagation Regimes**
        *   **Terrain-Aware:** Longley-Riceモデルによる地形効果（回折・遮蔽）を考慮。主要な評価設定。
        *   **Free-Space:** 障害物・地形効果なしの理想環境。手法間の性能ランキングへの影響を検証。
    *   **5.3.3 Baselines**
        *   **Iterative Multilateration:** 古典的な幾何学的解法。アンカー間リンクでパスロスモデルを推定 → 距離変換 → 最小二乗法で位置推定 → 新たに推定されたノードをアンカーに昇格（10イテレーション）。
        *   **Plain GCN Ablation:** 提案手法と同一構造だが、エッジ特徴量から学習された距離推定 $\hat{d}_{ij}$ を除外（RSSI時系列のみ）。
    *   **5.3.4 Training Details**
        *   フレームワーク: PyTorch Geometric。
        *   ハイパーパラメータ: Adam optimizer, lr=1e-4, weight decay=1e-5, epochs=50, hidden dim=64, 2×NNConv layers, gradient clipping=1.0。

*   **5.4 Results**
    *   **5.4.1 Terrain-Aware Results**
        *   **Proposed:** Mean **206.74m**, Median 182.42m, P90 381.01m, P95 455.37m。
        *   **Plain GCN:** Mean **750.87m**, Median 662.12m, P90 1330.32m。
        *   **Iterative Multilateration:** Mean **1974.40m**, Median 1747.99m, P90 3581.93m。
        *   提案手法はPlain GCNに対し**72%以上**、Multilaterationに対し**90%近く**の誤差削減。
        *   誤差分布の裾（P90/P95）も大幅に改善 → ロバスト性の向上。
    *   **5.4.2 Free-Space Results**
        *   **Proposed:** Mean 51.15m。
        *   **Plain GCN:** Mean 257.80m。
        *   **Iterative Multilateration:** Mean **6.28m**。
        *   理想環境では幾何学的手法が最高精度 → 地形効果が手法間の性能ランキングを逆転させることを示唆。
        *   Free-Space評価のみでは実世界性能を誤評価するリスク。

*   **5.5 Discussion**
    *   **5.5.1 Performance Drivers**
        *   地形効果によりRSSI-距離関係がノイジーかつ多峰的 → 単一RSSI値が複数距離に対応。
        *   Iterative Multilaterationは「ハード」な距離推定に依存 → 幾何学的に矛盾するシステムを生成し、誤差が連鎖的に拡大。
        *   提案手法は学習された距離を「ソフト」な事前分布として使用 → GCNがエッジごとに距離推定の信頼度を学習。
        *   メッセージパッシングによる局所一貫性の強制 → グラフ全体での同時推論により、孤立した大誤差を抑制。
    *   **5.5.2 Limitations**
        *   絶対精度: GNSS精度には及ばない。ビーコン密度と精度のトレードオフを要検討。
        *   アンカー配置: グリッド制約配置を使用。実世界ではアクセス制約により最適配置が困難な場合あり。
        *   Sim-to-Real Gap: Longley-Riceモデルは地形を考慮するが、デバイス差異・アンテナ配置・時間変動などは未考慮。
        *   グローバルパスロス: 領域全体で共通のパラメータを学習。局所的な変動への対応が今後の課題。
    *   **5.5.3 Proposed Deployment Workflow**
        1.  **Pre-Deployment Simulation:** 対象地域のDEMを用いてLongley-Riceシミュレータで合成データセットを生成し、モデルを事前学習。
        2.  **Post-Installation Calibration:** ハードウェア設置後、アンカー間RSSIデータを収集し、GCN重みを固定したままパスロスパラメータのみをファインチューニング（ラベル不要）。
        3.  **Inference:** アンカー固定のまま、未知ノードの位置を推論。

## Chapter 6: Integrated System Analysis & Navigation Feasibility (統合システム解析と航行実現性)
*   *New Content: Sensitivity Analysis & Closed-Loop Navigation*
*   **6.1 Sensitivity Analysis: Impact of Beacon Error**
    *   **目的:** Chapter 5のインフラ推定誤差（$\sigma \approx 150$m）が、Chapter 4のUAV測位にどう伝播するか。
    *   **Results (MATLAB Simulation):**
        *   ビーコン位置にガウスノイズ（$\sigma=150$m）を付加しても、UAVの測位誤差は指数関数的には増大しない（Median Error: 116m $\rightarrow$ 138m）。
        *   インフラ側の誤差とクライアント側の誤差は線形に近い関係であり、システムは破綻しない（Graceful Degradation）。
*   **6.2 Navigation Feasibility Analysis (Closed-Loop Control)**
    *   **Scenario:** 推定位置のみに頼ってウェイポイント航行を行うフィードバック制御シミュレーション。
    *   **Key Insight 1: Geometric Stability (幾何学的安定性)**
        *   ターゲットまでの距離（>1km）に対して、測位誤差（~150m）が十分小さければ、方向（Heading）の角度誤差は**10度未満**に収まる。
        *   これにより、測位精度が悪くても、長距離移動においては安定した経路追従が可能。
    *   **Key Insight 2: The Turning Point Problem**
        *   ウェイポイント接近時（<200m）は、距離に対する誤差の比率が大きくなり、方向推定が不安定になる。
        *   **Solution:** 到達判定閾値（Waypoint Threshold）を測位誤差と同程度（**200m**）に設定することで、ターゲット付近での"Wobble"（ふらつき）を許容しつつ、ミッションを継続可能とする。
    *   **Key Insight 3: Area Sweep Feasibility (広域捜索の実現可能性)**
        *   広域カバレッジミッションにおいて、飛行パスを**50%オーバーラップ**で計画する（例：200mセンサースワスに対し100m間隔）ことで、効果的な地表カバレッジを保証。
        *   この戦略により、飛行効率を犠牲にして信頼性を確保し、最大50mの横方向位置誤差があってもミッション成功を保証する。
*   **6.3 Deployment Strategy**
    *   **Coarse-to-Fine Approach:** LoRa測位は「広域巡回（Area Coverage）」や「初期位置推定」に適しており、精密着陸等は別センサーで補完する階層的アプローチを提案。
    *   **Computation Offloading:** GCNによるインフラ推定（Centralized）とMultilaterationによるUAV測位（Distributed/On-board）の役割分担。

## Chapter 7: Conclusion (結論)
*   **7.1 Summary of Contributions**
    *   厳しい環境下（低密度・山岳・インフラ位置未知）でも、システム全体として「破綻しない」ロバストなUAV航行システムを実証。
    *   GCNによるインフラ自己推定と、Multilaterationによる航行を統合した際のフィージビリティ（~200m精度での運用可能性）を定量的に示した。
*   **7.2 Future Work**
    *   Local Parameter Learningによる精度向上。
    *   Antenna Orientationの補正。
    *   実機での統合実験。
    *   カルマンフィルタ
