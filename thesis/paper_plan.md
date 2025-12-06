# Thesis Plan: Resilient UAV Localization in GNSS-Denied Environments via LPWA Networks: Coupling Graph-Based Infrastructure Estimation with Robust Navigation

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
        *   Chapter 3: インフラ位置が既知の場合のドローン測位（ロバスト性の向上）。
        *   Chapter 4: インフラ位置が未知の場合の自己組織化（GCNによる推定）。
        *   Chapter 5: 両者を統合したシステムとしての評価と運用戦略。

## Chapter 2: Related Work (関連研究)
*   **2.1 LPWA and LoRa Technology**
    *   LoRaの物理層特性、通信範囲、消費電力。
*   **2.2 RSSI-based Localization Techniques**
    *   Fingerprinting（事前調査が必要、コスト高）vs Multilateration（モデルベース、柔軟）。
    *   対数距離パスロスモデルとその限界。
*   **2.3 Drone Localization Methods**
    *   GNSSフリー環境での手法（Optical Flow, SLAM, UWB等）との比較。
*   **2.4 Network Self-Localization**
    *   センサーネットワークの自己位置推定。
    *   Graph Neural Networks (GNN) の測位への応用。

## Chapter 3: Robust UAV Localization with Altitude Constraints (高度制約を用いたロバストなドローン測位)
*   *Based on: LoRa-based Localization for Drones*
*   **3.1 Scenario Definition**
    *   都市部や既設インフラ利用時など、ビーコン位置が正確にわかっているケース。
*   **3.2 Methodology: Decoupling Strategy**
    *   **Decoupling Z-axis:** 信頼性の高い気圧/レーザー高度計でZ軸を固定し、不確実性の高いRSSI測位を水平方向（X/Y）に限定する。これにより、インフラ側のZ軸推定誤差や通信環境の影響をクライアント側で遮断（Isolate）する。
    *   **Huber Loss Integration:** RSSIの大きな外れ値（Outliers）の影響を抑制するための損失関数の変更（L2 Loss vs Huber Loss）。
*   **3.3 Evaluation**
    *   **Simulation:** Omnet++を用いた動的/静的環境での評価。
    *   **Field Experiment:** イタリア・駐車場での実機飛行実験。
    *   **Results:** 高度制約とHuber Lossによる精度向上と安定性の実証。アンテナ指向性による課題の発見。

## Chapter 4: Infrastructure Self-Localization via Graph Computing (グラフ計算によるインフラ自己位置推定)
*   *Based on: GCN-Based Localization for Low-Density LPWA Networks*
*   **4.1 Problem Transition**
    *   災害地や未踏地など、ビーコンの正確な配置が不可能なシナリオへの拡張。
    *   **Scenario Definition:** アンカーは境界配置（Boundary Expansion）またはドローン投下時記録（Aerial Deployment）により座標が既知と仮定する。
*   **4.2 Methodology**
    *   **Site-Specific Learning:** 対象エリアの地形データ（DEM）を用いて事前学習を行う、Fingerprintingの高度化アプローチとしての位置付け。
    *   **Physically-Informed GCN:** 物理法則（対数距離則）を明示的なPriorとしてネットワーク構造に組み込む。
    *   **Edge-Conditioned Convolution:** RSSI時系列データをエッジ特徴量として扱い、距離の「確からしさ」を学習。
*   **4.3 Evaluation & Characterization**
    *   **Terrain-Aware Simulation:** Longley-Riceモデルを用いた地形効果（回折・遮蔽）を含む現実的なデータセットでの評価。
    *   **Results:** 反復三辺測量（Iterative Multilateration）との比較で、疎なネットワークかつ地形影響下での圧倒的な優位性。
    *   **Empirical Saturation Point (RSSI Limitations):** ビーコン密度を上げても、Free-space環境であっても、単純なGCNモデルでは精度が一定レベル（~50m）で飽和する現象の報告。グローバルパラメータ学習の限界と、局所的なフェージングや個体差の影響についての考察。

## Chapter 5: Integrated System Analysis (統合システム解析と考察)
*   *New Content: Sensitivity Analysis & Deployment Strategy*
*   **5.1 The Impact of Beacon Error on Drone Navigation (感度解析)**
    *   **目的:** Chapter 4のインフラ推定誤差（50m~200m）が、Chapter 3のドローン測位に与える影響を定量化する。
    *   **Assumption:** 位置誤差によるチャネル変動（RSSIの変化）は無視した一次近似解析とする。
    *   **手法:**
        *   **Bias Noise Analysis:** 全ビーコンに一定のズレ（Offset）があるWorst Caseシナリオでのドローン挙動分析。
        *   **Random Noise Analysis:** ビーコンごとにランダムな誤差がある場合のHuber Lossによるフィルタリング効果の検証。
    *   **結果:** ドローン測位が許容できるインフラ誤差のマージン（Break Point）の特定。
*   **5.2 Deployment Strategy**
    *   **Computation Offloading:** GCNによるインフラ位置推定は、親ドローンやベースステーションなどの集中処理リソースで行い、計算負荷の軽いビーコンとドローンを実現する。
    *   **Energy Efficiency:** ビーコンは常時発信ではなく、ドローン接近時のみ起動するWake-up on Demand方式を想定し、バッテリーと法規制（Duty Cycle）の問題を回避する。
    *   **Coarse-to-Fine Approach:** 「絶対位置の初期化」としての役割定義。
    *   **Deployment Workflow:** Pre-deployment Simulation -> Rapid Deployment -> Centralized Self-Localization -> Drone Operation。

## Chapter 6: Conclusion (結論)
*   **6.1 Summary of Contributions**
    *   Decouplingによる信頼性の確保と、Physically-Informed GCNによる自己組織化。
    *   物理的限界（Saturation Point）の実験的特定。
*   **6.2 Future Work**
    *   Local Parameter Learningによる精度向上。
    *   Antenna Orientationの補正（偏波ダイバーシティ等）。
