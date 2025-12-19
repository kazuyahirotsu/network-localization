# Thesis Plan: Resilient UAV Navigation in GNSS-Denied Environments via Physics-Aware Graph Learning and Robust Multilateration
# (物理モデルを組み込んだグラフ学習とロバスト三辺測量によるGNSS拒否環境下での自律ドローン航法)

## Thesis Abstract (要旨)
*   **Overview (To be expanded to ~4 pages in thesis):**
    本論文は、GNSS（衛星測位システム）が利用できない、または信頼できない過酷な環境（災害地、山岳部、妨害下）において、LPWA（Low-Power Wide-Area）通信網を用いてドローンの自律航行を支援する「即応的ナビゲーションインフラ」の構築手法と実現可能性を論じるものである。本研究は、インフラの自己組織化とドローンのロバスト航行を統合したシステム全体を設計・評価する。

*   **1. Background & Motivation:**
    *   ドローンの産業利用（物流、監視、災害対応）において、GNSSへの依存は重大な脆弱性（Single Point of Failure）となっている。特に山間部や都市の峡谷、あるいは意図的な妨害（Jamming/Spoofing）が存在する環境では、代替手段が必要となる。
    *   Visual SLAMやLiDARといった既存の自律航法技術は、計算コストが高く、特徴点の少ない環境（砂漠、海上）や視界不良下では機能しない課題がある。
    *   本研究では、低消費電力・長距離通信・低コストを特徴とするLPWA規格「LoRa」に着目し、RSSI（受信信号強度）のみを用いた軽量な測位システムの実現を目指す。

*   **2. Related Work:**

*   **3. System Architecture:**
    *   **Concept:** インフラ（ビーコン）とクライアント（ドローン）を機能的に分離した非対称システム。
    *   **Infrastructure (Beacons):** 地上セグメント。投下直後（位置未知）は相互通信によりデータを収集し、中央ハブでのGCN推論により位置推定を受ける（Self-Organization Phase）。位置確定後、推定座標をグローバルにブロードキャストする（Broadcast Mode）。
    *   **Client (Drone):** 飛翔体セグメント。ビーコンからの信号を一方的に受信（Passive Receive）し、オンボードで自身の位置を計算する。これにより、ドローン台数が増加しても通信帯域を圧迫しないスケーラビリティを確保する。

*   **4. Phase 1: Robust Drone Navigation (Chapter 4):**
    *   **Objective:** インフラ位置が既知の場合のクライアント側測位精度の向上。
    *   **Methodology:** RSSIの大きな変動に対処するため、「高度（Z軸）の分離（Decoupling）」と「Huber Lossによる外れ値抑制」を導入。
    *   **Results:** 安価なLPWA信号でも安定した水平位置推定を実現（Sim: 4.5m median error）。

*   **5. Phase 2: Infrastructure Self-Organization (Chapter 5):**
    *   **Objective:** インフラ位置が未知（GNSS拒否下の災害地等）の場合の自己位置推定。
    *   **Methodology:** 航空機から投下されたビーコンが、相互通信と地形情報（DEM）を用いた「Physics-Aware GCN」により、自律的に自身の位置を推定する。
    *   **Results:** 従来手法（Multilateration）と比較して90%以上の精度向上（Sim: ~200m mean error）。

*   **6. Integrated Feasibility and Deployment Strategy (Chapter 6):**
    *   **System Integration:** Phase 2のインフラ誤差（~150m）環境下で、Phase 1のドローン測位が実用に耐えうるかを統合的に評価。
    *   **Geometric Stability:** ドローンの測位誤差が大きくても、目的地までの距離が十分長ければ方位角誤差は数度（<5°）に収まり、安定した巡航が可能。
    *   **Operational Feasibility:** ピンポイント着陸は困難だが、広域捜索（Area Sweep）や回廊航行においては、飛行パスのオーバーラップ率調整により任務遂行が可能。
    *   **Turning Point Problem:** ウェイポイント接近時の不安定性に対し、到達判定閾値緩和（~200m）による解決策を提示。

*   **7. Conclusion:**
    *   提案システムは、従来の精密測位（GPS）の代替ではないが、GNSS喪失下における「生存可能なナビゲーション（Survivable Navigation）」を提供する実用的なソリューションである。

## Chapter 1: Introduction (序論)
*   **1.1 Background**
    *   ドローンの産業利用の拡大（物流、監視、災害対応）。
    *   GNSS依存のリスク（ジャミング、スプーフィング、都市部や山間部での信号途絶）。
    *   既存の代替手段（Visual SLAM, LiDAR）の課題（コスト、計算資源、環境依存性）。
*   **1.2 Motivation**
    *   なぜLPWA (LoRa) なのか？：低消費電力、長距離通信、低コスト、既存ハードウェアの利用可能性。
    *   なぜRSSIなのか？：追加のハードウェア（高精度時計など）が不要で、最も安価に実装可能。
*   **1.3 Problem Statement**
    *   **課題1（クライアント側）：** RSSIの物理的な不安定さと変動（マルチパス、フェージング）。ドローン側での測位精度の確保。
    *   **課題2（インフラ側）：** GNSS拒否環境下（砂漠、災害地）では、基準局となるビーコン自体の位置決めが困難。広域展開時の測量コスト。
*   **1.4 Thesis Outline**
    *   本論文の構成：
        *   Chapter 3: システムアーキテクチャ（全体設計）。
        *   Chapter 4: インフラ位置が既知の場合のドローン測位（ロバスト性の向上）。
        *   Chapter 5: インフラ位置が未知の場合の自己組織化（GCNによる推定）。
        *   Chapter 6: 統合システム解析とデプロイメント戦略。

## Chapter 2: Related Work (関連研究)
*   **2.1 Wireless Positioning Fundamentals**
    *   **Signal Metrics Comparison:** RSSI (強度), ToF/TDoA (時間), AoA (角度) の比較。
    *   **Why LoRa/RSSI?:** ToF/TDoAは高精度だが厳密な時刻同期（通常はGPS由来）が必要。AoAはアレイアンテナが必要。本研究は、最も安価で既存のIoTハードウェア（単一アンテナ・非同期）で実現可能なRSSI方式に焦点を当てる。

*   **2.2 Client-Side: Drone Localization Approaches (Phase 1 Context)**
    *   **Non-RF Methods (SLAM/LiDAR):** 高精度だが計算負荷が高く、特徴点のない環境（海、砂漠）や視界不良時（煙、霧）に機能しない。
    *   **RF-based Drone Localization:**
        *   *TDoA methods (e.g., Fargas et al.):* 高精度だが、インフラ側の時刻同期にGPSを必要とするケースが多く、GNSS拒否環境下のバックアップとしては矛盾が生じる。
        *   *Hybrid RSSI/AoA (e.g., Baik et al.):* RSSIと到来角を併用して精度（~3m）を出す研究もあるが、専用のデュアルアンテナや位相変調器が必要となる。
    *   **Positioning of Phase 1:** 特別なハードウェアを必要とせず、一般的なLoRaモジュールのみを用い、アルゴリズム（高度制約 + Huber Loss）のみで実用的なロバスト性を確保する点に独自性がある。

*   **2.3 Infrastructure-Side: Network Self-Localization (Phase 2 Context)**
    *   **Classical Geometric Solvers:**
        *   *Iterative Multilateration:* 位置が判明したノードを順次アンカーとして利用する手法。疎なネットワークやNLOS環境では、初期の推定誤差が連鎖的に拡大（Error Propagation）し、システム全体が破綻する課題がある (Hada et al.)。
    *   **Data-Driven Approaches:**
        *   *Fingerprinting (e.g., Purohit et al.):* 高精度だが、事前の実地調査（サイトサーベイ）によるデータベース構築が必須。即応性が求められる災害地や未踏地（Unknown Environment）には適用不可能。
    *   **Graph Neural Networks (GNN):**
        *   センサーネットワークや屋内WiFi測位へのGNN応用が進んでいる (Vishwakarma et al.)。しかし、多くは屋内などの高密度環境を想定しているか、物理特性を考慮しないブラックボックス的なアプローチである。
    *   **Positioning of Phase 2:** 本研究は、物理モデル（対数距離パスロス式）をGNNの構造に明示的に組み込む「Physics-Aware」なアプローチを提案する。また、地形効果（Longley-Riceモデル）が支配的な、疎な屋外ネットワークでの自己位置推定に特化している点で新規性がある。

## Chapter 3: System Architecture (システムアーキテクチャ)
*   *New Content: Defining the Asymmetric System*
*   **3.1 Overall Concept**
    *   **Goal:** GNSS拒否環境下での即応的なナビゲーションインフラの構築。
    *   **Roles:** インフラ側（ビーコン）は「位置情報の提供者」、クライアント側（ドローン）は「利用者」として明確に分離する。
*   **3.2 Infrastructure Segment (Beacons)**
    *   **Function:** 自律的な位置推定と、グローバルな位置情報のブロードキャスト。
    *   **Mode A (Self-Organization):** 近隣ビーコンとのRSSI計測を行い、中央ハブへデータを送信。
    *   **Mode B (Service Operation):** 推定された座標とIDを周期的に送信（Beaconing）。
    *   **Energy Management (Wake-up on Demand):** バッテリー消費を抑えるため、デフォルトではスリープ（Deep Sleep）状態。ドローンからの「Wake-up信号」を受信した際のみ、一定時間（例: 数分間）ブロードキャストを行う。
*   **3.3 Client Segment (Drone)**
    *   **Function:** パッシブな受信による自己位置推定。
    *   **Privacy & Scalability:** ドローンは信号を発しないため、無限の台数が同時に利用可能であり、かつ自身の位置を外部に漏らさない。
    *   **Active Request:** 必要な信号が受信できない場合のみ、Wake-up信号をブロードキャストする。単一の信号でエリア内の全ビーコンを起動でき、他のドローンも便乗できるため、多数機運用時でも通信帯域を圧迫しない。
*   **3.4 Hub Segment (Base Station / Mother Drone)**
    *   **Function:** 計算リソースの集約。GCNのトレーニングと推論を行い、ビーコンへ位置情報を配布する。

## Chapter 4: Robust UAV Localization with Altitude Constraints (Phase 1)
*   *Based on: LoRa-based Localization for Drones*
*   **4.1 Scenario Definition**
    *   都市部や既設インフラ利用時など、ビーコン位置が正確にわかっているケース。
*   **4.2 Methodology: Decoupling Strategy**
    *   **Decoupling Z-axis:** 信頼性の高い気圧/レーザー高度計でZ軸を固定し、不確実性の高いRSSI測位を水平方向（X/Y）に限定する。これにより、インフラ側のZ軸推定誤差や通信環境の影響をクライアント側で遮断（Isolate）する。
    *   **Huber Loss Integration:** RSSIの大きな外れ値（Outliers）の影響を抑制するための損失関数の変更（L2 Loss vs Huber Loss）。
*   **4.3 Evaluation**
    *   **Simulation:** Omnet++を用いた動的/静的環境での評価。
    *   **Field Experiment:** イタリア・駐車場での実機飛行実験。
    *   **Results:** 高度制約とHuber Lossによる精度向上と安定性の実証。アンテナ指向性による課題の発見。

## Chapter 5: Infrastructure Self-Localization via Graph Computing (Phase 2)
*   *Based on: GCN-Based Localization for Low-Density LPWA Networks*
*   **5.1 Problem Transition**
    *   災害地や未踏地など、ビーコンの正確な配置が不可能なシナリオへの拡張。
    *   **Scenario Definition:** アンカーは境界配置（Boundary Expansion）またはドローン投下時記録（Aerial Deployment）により座標が既知と仮定する。
*   **5.2 Methodology**
    *   **Site-Specific Learning:** 対象エリアの地形データ（DEM）を用いて事前学習を行う。
    *   **Physics-Aware GCN:** 物理法則（対数距離則）を明示的なPriorとしてネットワーク構造に組み込む。
    *   **Edge-Conditioned Convolution:** RSSI時系列データをエッジ特徴量として扱い、距離の「確からしさ」を学習。
*   **5.3 Evaluation & Characterization**
    *   **Terrain-Aware Simulation:** Longley-Riceモデルを用いた地形効果（回折・遮蔽）を含む現実的なデータセットでの評価。
    *   **Results:** 反復三辺測量（Iterative Multilateration）との比較で、疎なネットワークかつ地形影響下での圧倒的な優位性。
    *   **Empirical Saturation Point:** ビーコン密度やモデルを改善しても、RSSIの物理的性質により推定精度は一定レベル（~50-150m）で飽和する。この「残留誤差」を許容できるかが次の課題となる。

## Chapter 6: Integrated System Analysis & Navigation Feasibility (統合システム解析と航行実現性)
*   *New Content: Sensitivity Analysis & Closed-Loop Navigation*
*   **6.1 Sensitivity Analysis: Impact of Beacon Error**
    *   **目的:** Chapter 4のインフラ推定誤差（$\sigma \approx 150$m）が、Chapter 3のドローン測位にどう伝播するか。
    *   **Results (MATLAB Simulation):**
        *   ビーコン位置にガウスノイズ（$\sigma=150$m）を付加しても、ドローンの測位誤差は指数関数的には増大しない（Median Error: 116m $\rightarrow$ 138m）。
        *   インフラ側の誤差とクライアント側の誤差は線形に近い関係であり、システムは破綻しない（Graceful Degradation）。
*   **6.2 Navigation Feasibility Analysis (Closed-Loop Control)**
    *   **Scenario:** 推定位置のみに頼ってウェイポイント航行を行うフィードバック制御シミュレーション。
    *   **Key Insight: Geometric Stability (幾何学的安定性)**
        *   ターゲットまでの距離（>1km）に対して、測位誤差（~150m）が十分小さければ、方向（Heading）の角度誤差は数度（<5°）に収まる。
        *   これにより、測位精度が悪くても、長距離移動（Cruise Phase）においては安定した経路追従が可能。
    *   **Constraint: The Turning Point Problem**
        *   ウェイポイント接近時（<200m）は、距離に対する誤差の比率が大きくなり、方向推定が不安定になる。
        *   **Solution:** 到達判定閾値（Waypoint Threshold）を測位誤差と同程度（~200m）に設定することで、ターゲット付近での"Wobble"（ふらつき）を許容しつつ、ミッションを継続可能とする。
    *   **Update Rate Analysis:**
        *   LoRaの通信速度（0.1-0.5Hz）であっても、幾何学的安定性により長距離航行には十分であることを確認。
*   **6.3 Deployment Strategy**
    *   **Coarse-to-Fine Approach:** LoRa測位は「広域巡回（Area Coverage）」や「初期位置推定」に適しており、精密着陸等は別センサーで補完する階層的アプローチを提案。
    *   **Computation Offloading:** GCNによるインフラ推定（Centralized）とMultilaterationによるドローン測位（Distributed/On-board）の役割分担。

## Chapter 7: Conclusion (結論)
*   **7.1 Summary of Contributions**
    *   厳しい環境下（低密度・山岳・インフラ位置未知）でも、システム全体として「破綻しない」ロバストなドローン航行システムを実証。
    *   GCNによるインフラ自己推定と、Multilaterationによる航行を統合した際のフィージビリティ（~200m精度での運用可能性）を定量的に示した。
*   **7.2 Future Work**
    *   Local Parameter Learningによる精度向上。
    *   Antenna Orientationの補正。
    *   実機での統合実験。
