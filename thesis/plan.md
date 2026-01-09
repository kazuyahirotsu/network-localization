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

do not add citation by yourself since you could hallucinate. just state that it needs citation and i can find it for you

abstract:
keep it as extended abstract
match it to chapter 6

chapter 1
keep extraterrestrial environments but as short remark
Explicitly state that end-to-end integration is evaluated in simulation, and field experiments validate only the client-side method under GNSS-available conditions for ground truth.

chapter 2
apply the fixes you suggesteed

chapter 3
do these
State assumptions explicitly for the scalability formula and cite a LoRa/LoRaWAN capacity reference.
Reframe wake-on-demand as “proposed” and separate what is implemented vs conceptual.

chapter 4
huber parameter is correct, it is not from the notebook for chapter 6
fix the rest based on the original paper

chapter 5

Continuing with the deep read: I’m going to finish Chapters 4–7 in `paper.tex` (in chunks), then I’ll cross-check Chapter 6’s stated numbers against `thesis/chapter6_results.json` + the notebook, and then I’ll produce the blunt section-by-section review with specific examples and fixes.

[4 tools called]

I’m now finishing Chapter 6 (from the sensitivity figures through navigation + discussion) and then Chapter 7. After that, I’ll cross-check the Chapter 6 numbers against `thesis/chapter6_results.json` and only then I’ll write the section-by-section critique (with concrete line-level examples).

[18 tools called]

### Thesis-wide issues (before chapter-by-chapter feedback)

- **Strengths (actually present)**
  - **Clear two-layer framing**: The thesis consistently separates **client-side UAV localization** (Ch.4) from **infrastructure self-localization** (Ch.5), then integrates them (Ch.6). This is a coherent story arc.
  - **Quantitative reporting exists**: Chapters 4–6 include concrete numbers, tables, and comparisons, not just qualitative claims.

- **Critical issues (must fix)**
  - **Internal inconsistency between the extended abstract and Chapter 6**: The extended abstract states “Adding Gaussian noise (\(\sigma=150\) m) increases the UAV’s localization error from 116 m to 138 m.” Chapter 6 uses **mean 2D error** for beacon noise and reports baseline mean **122 m**, and at **150 m** beacon error the mean is **125 m**. This is a direct contradiction and will immediately reduce credibility.
    - **Actionable fix**: Rewrite the extended abstract’s sensitivity paragraph so it uses the **same noise definition** as Chapter 6 (mean 2D beacon error) and the same reported numbers.
  - **One missing bibliography entry**: `\cite{huber1964robust}` is cited but not present in `thesis/paper/references.bib`.
    - **Actionable fix**: Add the Huber (1964) reference to `references.bib` and ensure the citation compiles without “?”.
  - **Your own published paper is not cited**: `hirotsu2024lora` exists in the bib but is never cited in the thesis. This is academically sloppy given Chapter 4 is adapted from that paper.
    - **Actionable fix**: Add explicit citations at the start of Chapter 4 and Chapter 5 (and optionally a short disclosure sentence in Chapter 1 or a preface) indicating those chapters are adapted from your published works.
  - **Mathematical correctness problem in Chapter 4 RSSI-to-distance model**: You define \(d_i = A \log_{10}(\text{RSSI}_i) + B\). RSSI in dBm is typically negative, so \(\log_{10}(\text{RSSI}_i)\) is undefined. Even if you implicitly mean \(\log_{10}(|\text{RSSI}_i|)\), that is not stated and it is not the usual derivation.
    - **Actionable fix**: Replace this with a physically consistent model and derivation (for example log-distance path-loss, then invert to distance), and align the text with what your code actually uses.
  - **Parameter mismatch between thesis text and code**: In Chapter 4 you state Huber \(\delta = 100\) m. In the Chapter 6 notebook, the multilateration function’s default Huber `delta` is `10.0`. That is not a minor discrepancy, it changes robustness behavior.
    - **Actionable fix**: Decide a single \(\delta\) definition and value, and make thesis + code agree. If they differ by chapter, justify explicitly and rename parameters to avoid confusion (e.g., \(\delta_{\text{mlat}}\) vs \(\delta_{\text{robust}}\)).
  - **Strict “present tense” rule is violated in multiple places**: Example: “We conducted field experiments…”, “Experiments were conducted…”. Your own rules explicitly forbid past tense.
    - **Actionable fix**: Do a global tense sweep. If your professor truly insists on present tense, you must rewrite those passages, even though past tense is conventional for experiments.
  - **Terminology inconsistency (“drone” appears in thesis body)**: You still write “commercial drones” once in Chapter 1 background. This violates your own terminology rule.
    - **Actionable fix**: Replace with “commercial UAVs”.

---

### Extended Abstract (`eabstract`)

- **Strengths**
  - **Full-system narrative exists**: It covers both layers and the integration, with numerical results. That is stronger than a generic abstract.

- **Major weaknesses**
  - **It is not an abstract, it is a mini-thesis**: It has “1. Introduction … 7. Conclusion” sections and includes many details. Unless your program explicitly requires an extended abstract, this is structurally non-standard and increases the chance of internal inconsistencies (which already happened).
  - **Outdated / inconsistent Chapter 6 numbers**: The sensitivity numbers and noise parameterization do not match Chapter 6.
  - **Misleading parameterization**: It uses “\(\sigma=150\) m” but Chapter 6 uses mean 2D error. This is exactly the confusion you already identified earlier.

- **Actionable fixes**
  - **Either**: convert it into a true abstract (single block, 200 to 400 words, only key results).
  - **Or**: if it must remain extended, treat it like a compressed executive summary and enforce **strict consistency** by regenerating it from the final chapter numbers and definitions.

---

### Chapter 1: Introduction

- **Strengths**
  - **Problem motivation is clear**: The GNSS vulnerability framing is understandable and relevant.
  - **Two challenges are explicit**: Client-side robustness and infrastructure self-localization are properly separated.
  - **Contribution list is concrete**: It includes quantitative performance claims and an integrated feasibility claim.

- **Weaknesses**
  - **Scope creep harms focus**: The “extraterrestrial environments” paragraph is a distraction unless you later connect it to your system design constraints. It reads as padding.
  - **Terminology slip**: “commercial drones” should be “commercial UAVs”.
  - **Overclaim risk**: Some statements read like you validate a complete system in real-world GNSS-denied environments, but the integrated analysis (Ch.6) is simulation-based. Your wording should reflect that.

- **Actionable fixes**
  - **Tighten the motivation**: Either cut extraterrestrial discussion or demote it to a short remark.
  - **Add a “scope and assumptions” paragraph**: Explicitly state that end-to-end integration is evaluated in simulation, and field experiments validate only the client-side method under GNSS-available conditions for ground truth.

---

### Chapter 2: Related Work

- **Strengths**
  - **Good taxonomy**: RSSI vs time-based vs AoA vs CSI, plus non-RF approaches, is a sensible structure.
  - **The summary table is useful**: It forces clarity about what your system does and does not assume.

- **Weaknesses**
  - **Path-loss equation presentation is sloppy**: You write \( \text{RSSI} = P_t - 10n\log_{10}(d) + X_\sigma\). This is not the standard form and the intercept term is unclear. This matters because later chapters depend on path-loss inversion and “physics-aware” modeling.
  - **Insufficient citations for broad claims**: Example: “Commodity LoRa modules cost under \$10” is not cited; “LoRa achieves 2–15 km” is cited, but the conditions (SF, bandwidth, environment) are not described.
  - **LPWA / LPWAN naming inconsistency**: You use both. That is minor but it is an avoidable professionalism hit.

- **Actionable fixes**
  - **Rewrite the RSSI-distance model subsection**: Use a standard reference model with clear variable definitions and a citation that actually derives it. Then keep the same notation in Chapters 4 and 5.
  - **Add a few missing cornerstone references**: Right now the thesis cites surprisingly few papers overall for a master thesis. This chapter should carry more of that weight.

---

### Chapter 3: System Architecture

- **Strengths**
  - **Design goals are clear and relevant**: Rapid deployment, GNSS independence, scalability, privacy, cost all make sense.
  - **Pre-deployment training stage is described**: This matches the practical constraint that the GCN model is trained offline and deployed for inference.
  - **Operational timeline is useful**: It clarifies stages from planning to service mode.

- **Weaknesses**
  - **The “unlimited clients” scalability argument is oversimplified**: Channel utilization \(B\tau/T\) ignores collisions, duty-cycle regulations, multi-SF behavior, and interference. As written, it looks naive.
  - **Wake-up mechanism is asserted without evidence**: A wake-up receiver is non-trivial in LoRa deployments; you need either citations or a clear statement that it is a proposed design element, not evaluated.
  - **Placeholder figures pollute the List of Figures**: If you keep placeholders with full captions, the LoF looks like a finished document while figures are not finished.

- **Actionable fixes**
  - **State assumptions explicitly for the scalability formula** and cite a LoRa/LoRaWAN capacity reference.
  - **Reframe wake-on-demand as “proposed”** and separate what is implemented vs conceptual.
  - **Consider removing placeholders from the LoF** until real figures exist (e.g., use `\caption*{}` for placeholders).

---

### Chapter 4: Robust UAV Localization with Altitude Constraints

- **Strengths**
  - **Problem formulation is clear and relevant**: Variance, NLOS, antenna effects, 3D geometry are correctly identified.
  - **Comparison design is strong**: The 2x2 factorial comparison (loss function x altitude constraint) is a solid experimental structure.
  - **You include both simulation and field experiments**: That is a real strength compared to many theses that are simulation-only.

- **Major weaknesses (serious)**
  - **RSSI-to-distance equation is mathematically invalid as written**: \(d_i = A \log_{10}(\text{RSSI}_i) + B\) is not acceptable in a thesis without clarification, because RSSI is negative in dBm.
  - **Huber parameter inconsistency**: The thesis states \(\delta = 100\) m; the notebook uses `delta=10.0` by default. This undermines reproducibility.
  - **Method implementation details are under-specified**: You say “iterative nonlinear least squares” with “Huber weights” but do not specify the solver, the update rule, convergence criteria, or how the Huber loss is implemented (for example IRLS). A thesis should be explicit.
  - **Experimental results table is misleading**: Caption says “median error / standard deviation”, but the entries look like “2D/3D medians” and you omit standard deviations entirely for Methods 2 and 4. This is objectively incorrect presentation.
  - **Beacon density claim looks wrong**: You write that ~18 beacons per square kilometer gives ~11.6 m median accuracy while ~7.2 beacons per square kilometer gives ~4.5 m. That implies fewer beacons yields better accuracy, contradicting both intuition and your own scalability discussion.

- **Actionable fixes**
  - **Fix the RSSI-to-distance section first**: Put a correct model, show inversion, state what you actually fit, and make that consistent with the rest of the thesis and the code.
  - **Make robustness implementation reproducible**: Specify the exact loss, \(\delta\), solver, initialization, stopping criteria.
  - **Fix the experimental results table**: Either present median and standard deviation consistently for all methods, or change caption and text so it accurately describes what is shown.
  - **Re-audit the beacon density numbers**: If they come from a specific experiment, show the underlying curve or table so it is not a random claim.

---

### Chapter 5: Infrastructure Self-Localization via Graph Learning

- **Strengths**
  - **Architecture explanation is substantially better than typical student theses**: NNConv and EdgeNet are explained clearly.
  - **Baselines are described concretely**: Iterative multilateration steps are listed, and the plain GCN ablation is well motivated.
  - **Evaluation realism point is strong**: The terrain-aware vs free-space ranking inversion is a valuable contribution and is correctly emphasized.

- **Weaknesses**
  - **Baseline implementation details are still incomplete**: Iterative multilateration depends on thresholds, bounds, and stopping conditions. These are not specified, which weakens the fairness of comparison.
  - **Hyperparameter choices lack justification**: Batch size 1, 2 layers, 64 hidden units, K=10, anchor ratio 25 percent appear arbitrary without justification or ablation.
  - **Standardization details are unclear**: You say “all node features are standardized” but node features include a binary anchor flag. Standardizing that is questionable unless explained.

- **Actionable fixes**
  - **Add a reproducibility table** for Chapter 5: exact thresholds, optimizer settings, seeds, training schedule, and baseline parameters.
  - **Add minimal justification** for key hyperparameters (or cite the original paper choices explicitly and state you follow them).
  - **Clarify how the anchor indicator is handled** during scaling.

---

### Chapter 6: Integrated System Analysis and Navigation Feasibility

- **Strengths**
  - **Internal consistency with your stored results is good**: The sensitivity table and navigation table match the saved results exactly.
  - **The diagonal 200 m path figure is used effectively**: It concretely illustrates the “stable when far, unstable when close” narrative, which is one of the chapter’s strongest arguments.

- **Weaknesses**
  - **Incorrect overlap example**: “50% overlap (e.g., 300 m spacing for a 200 m swath)” is wrong arithmetic and needs to be corrected.
  - **Geometric stability figure caption contradicts the text**: You state “at 1 km, 200 m error gives 11 degrees”, but the figure caption says “beyond 1 km, up to 200 m stays below 10 degrees”. That is false unless you define “beyond” as significantly greater than 1 km.
  - **Missing controller and trial details**: You do not state control gain, exact update rate assumptions beyond 1 Hz, number of navigation trials, or the “stuck recovery” logic that exists in code. These choices influence success rates and should be documented.
  - **Metric mismatch**: You interpret localization error as “lateral deviation” in the coverage paragraph without computing cross-track error.

- **Actionable fixes**
  - **Correct the overlap example** and ensure it is dimensionally consistent with your stated sensor swath and expected cross-track error.
  - **Fix the stability caption** so it matches the equation and text.
  - **Add a short “simulation parameters” table** listing controller parameters, trial counts, noise application (fixed per run), boundary handling, and stuck recovery.

---

### Chapter 7: Conclusion

- **Strengths**
  - **Clear structure**: Contributions, limitations, future work, concluding remarks is a sensible organization.
  - **Limitations are mostly honest**: You acknowledge simulation dependence and accuracy limits.

- **Weaknesses**
  - **Repetition**: It largely restates Chapter 1 contributions and Chapter summaries rather than synthesizing new insight.
  - **Overclaiming**: Phrases that imply validated operational deployment are too strong without end-to-end field validation.
  - **Future work lacks prioritization**: It lists many items without a clear “top 3 next steps”.

- **Actionable fixes**
  - **Cut repetition** and add a short “what the thesis proves, what it does not prove” paragraph.
  - **Prioritize future work**: Rank items and justify the ranking (field validation and sim-to-real calibration should be first).

---

### Overall Assessment

- **Main strengths**
  - **Coherent system narrative across Chapters 4–6**: The two-layer architecture and integration logic are the thesis’s strongest structural asset.
  - **Non-trivial technical content**: Chapter 5 in particular reads like serious research work, not a superficial student write-up.
  - **Chapter 6 provides a plausible operational argument**: The geometric stability framing is genuinely persuasive when supported correctly.

- **Key areas needing major improvement (highest priority)**
  - **Fix correctness and consistency issues that undermine credibility**: Chapter 4 distance model, Huber \(\delta\) mismatch, abstract vs Chapter 6 contradiction, and the overlap arithmetic error.
  - **Improve reproducibility**: Add explicit parameter listings and baseline details so the work can be replicated.
  - **Align with strict writing rules**: Present tense sweep, UAV terminology sweep, and explicit attribution to the published papers.

- **Direct recommended plan**
  - **Pass 1 (must-do)**: Repair the extended abstract consistency, correct the RSSI-to-distance math, add `huber1964robust`, fix the experimental table caption/content mismatch, fix the overlap example, remove “drone(s)” in body text.
  - **Pass 2 (quality)**: Add reproducibility tables for Chapters 4–6, add missing baseline parameters in Chapter 5, and correct the geometric stability caption wording.
  - **Pass 3 (polish)**: Reduce repetition in Chapter 7, strengthen citation density in Chapter 2, and clean tense throughout to comply with your program’s rule.