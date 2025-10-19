# Customer Segmentation with Hypothesis Testing & K-Means

---

## 0) Executive Summary

This project explores **mall customer behavior** using a small but well-known retail dataset (200 customers; features: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1–100)`). After validating basic data quality, we:

- **Test associations** among demographics, income, and spending with χ² tests, Pearson correlation, Levene tests, and one-way ANOVA.  
  **Key findings:** Gender is not associated with spending or income; income and spending are **positively correlated** (p ≈ 0.0099); spending and income **differ across age groups**. (See Sections 2.2–2.4.)
- **Segment customers** via **K-Means** on three features—`Annual_Income`, `Score`, and **binned** `Age`—and select **k = 5** using the elbow method. (Sections 3–4.)
- **Interpret clusters** to propose actionable personas (e.g., **High-income / low-spend** cohort to target with activation offers; **Low-income / high-spend** cohort for loyalty programs). (Section 5.)

> **So what?** Even a compact dataset can yield **clear, testable insights** and **behavior-oriented segments** that guide targeting, promotions, and retention design.

---

## 1) Project Overview

### 1.1 Business Context
Customer segmentation divides a population into **behaviorally or demographically coherent groups** so we can tailor product, messaging, and offers. The notebook begins with a brief introduction to common B2C dimensions (age, gender, life stage, income, etc.), motivating the segmentation use-case for marketing efficiency. (p. 1)
<p align="center">
  <img src="https://github.com/user-attachments/assets/2de0af9e-6e85-4d48-86ae-6da9092009b6"
       alt="Customer segmentation figure"
       width="85%" />
</p>



### 1.2 Data at a Glance
- **Dataset:** `Mall_Customers.csv`  
- **Rows / columns:** 200 × 5 (all non-null, per `df.info`) (p. 2)  
- **Renames:** `Annual Income (k$)` → `Annual_Income`, `Spending Score (1-100)` → `Score` (p. 2)  
- **Descriptives:**  
  - `Age` is **right-skewed**; `Annual_Income` and `Score` ≈ **roughly normal** (inference following `df.describe()` and univariate plots). (pp. 3–4)

---

## 2) Problem Statement & Methodology

### 2.1 Goal
**Discover natural groupings** of customers that differ in **ability and propensity to spend**, and validate basic assumptions about how **demographics relate to income and spending** before modeling. (p. 1)

### 2.2 Exploratory Data Analysis (EDA)
- Checked shape, dtypes, and missingness (all five columns have 200 non-null entries). (p. 2)  
- Visualized histograms, KDEs, and boxplots for `Age`, `Annual_Income`, `Score`. (pp. 3–4)

**Takeaways:**  
`Annual_Income` and `Score` are approximately symmetric, while `Age` skews right. No data gaps are evident. (pp. 2–4)

### 2.3 Hypothesis Testing
We formalize questions with statistical tests (threshold α = 0.05). All tests and visual corroboration are implemented in the notebook.

- **Is Spending independent of Gender?**  
  **Test:** χ² on crosstab(`Gender`, `Score`)  
  **Result:** p ≈ **0.3412** → **Fail to reject** independence → **Spending is independent of Gender**. (pp. 5–6)  
  **Visuals:** Bar/box/strip plots support the null. (p. 6)

- **Is Annual Income independent of Gender?**  
  **Test:** χ² on crosstab(`Gender`, `Annual_Income`)  
  **Result:** p ≈ **0.3495** → **Fail to reject** independence → **Income is independent of Gender**. (p. 7)  
  **Visuals:** Same conclusion. (p. 7)

- **Are Annual Income and Spending correlated?**  
  **Test:** Pearson correlation `Annual_Income` vs `Score`  
  **Result:** p ≈ **0.0099** → **Reject H₀** → **Small positive correlation**. (p. 8)  
  **Visuals:** Scatterplot suggests a mild upward trend. (p. 8)

- **Do Spending scores differ across Age groups?**  
  **Binning:** `Age` → [0–20), [20–40), [40–60), [60–80) labeled 0–3. (pp. 8–9)  
  **Variance check:** Levene p ≈ **0.0004** (variances unequal). (p. 9)  
  **ANOVA:** F ≈ **20.40**, p ≈ **1.51e−11** → **At least one age group mean differs** (on Spending). (pp. 9–10)  
  **Visuals:** Differences clear across groups. (p. 10)

- **Do Incomes differ across Age groups?**  
  **Variance check:** Levene p ≈ **0.0029** (variances unequal). (p. 10)  
  **ANOVA:** The narrative/inference states **age groups differ on income**; figures show distinct distributions by age. *(Note: the printed F-test cell repeats “Score” in code, but the interpretation and plots discuss **Annual Income**.)* (p. 11)

> **Interpretation:** Gender adds little signal. Age captures **systematic differences** in both spending and income; income and spending are **linked**, motivating their joint use in clustering. (pp. 5–11)

---

## 3) Feature Set for Clustering
- **Included:** `Annual_Income`, `Score`, `binned_Age`  
- **Excluded:** `Gender` (lowest relevance per hypothesis tests/visuals)  
- **Rationale:** These three variables jointly capture **ability** (`Annual_Income`), **propensity** (`Score`), and **life stage proxy** (`binned_Age`). (p. 12)  
- **Note on scaling:** The notebook feeds raw values to K-Means. Because K-Means uses Euclidean distance, **standardization** is usually recommended in production; here, binned age (0–3) naturally down-weights age relative to income/spend. (p. 12)

---

## 4) Modeling — K-Means

### 4.1 Choosing **k** (Elbow Method)
We fit models for k = 1..14, plot WCSS (inertia), and observe the **elbow at k ≈ 5**—a good balance between fit and parsimony. (p. 13)

### 4.2 Final Model
- **Algorithm:** `KMeans(init='k-means++', max_iter=500, n_clusters=5)`  
- **Input matrix:** `X = [Annual_Income, Score, binned_Age]`  
- **Visualization:** 3D scatter (`Income` × `Score` × `binned_Age`) and three 2D pairwise plots. (pp. 13–14)

---

## 5) Results — Segment Narratives
The **2D plot of Annual Income vs. Spending Score** offers the **clearest separation** (center figure, p. 15). The notebook labels cluster colors in plots as **red, blue, orange, green, violet**; the business interpretations below summarize the qualitative patterns observed. (p. 15)

- **Green — “Value Maximizers” (Low-income, High-spend)**  
  Customers with relatively **lower income** yet **high spending** rates—indicating strong **share-of-wallet** capture.  
  **Action:** Maintain loyalty; protect experience; consider cash-flow-friendly offers. (p. 15)

- **Blue — “Untapped Affluents” (High-income, Low-spend)**  
  **Higher income**, but **limited spend**.  
  **Action:** Prime target for **activation** and **cross-sell** campaigns; test differentiated value propositions. (p. 15)

- **Red / Violet / Orange — “Proportional Spenders”**  
  Spend **roughly proportional** to income, following the expected linear trend.  
  **Action:** Standard lifecycle marketing; nudge to move up the value curve. (p. 15)

> The **Income × Score** plane (p. 15) is the most diagnostic view; age adds nuance but less separation after binning.

---

## 6) Reproducibility — How to Run
The original analysis is a notebook exported to PDF. Below are minimal steps to reproduce it as a Python notebook or script—**no additional code provided here**.

### 6.1 Environment
- Python ≥ 3.9  
- Packages: NumPy, Pandas, Matplotlib, Seaborn, SciPy, scikit-learn

### 6.2 Data
- Download `Mall_Customers.csv` (same schema as used in the notebook) and place it in your working directory.  
- The original path in the notebook points to a local Windows directory; adjust accordingly when running locally. (p. 1–2)

### 6.3 Quickstart (Notebook-style)
- Follow the sequence of cells in the original notebook/report: data load → renames → EDA → hypothesis tests → age binning → elbow selection → K-Means fit → plots.  
- *Tip:* Consider standardizing `Annual_Income` and `Score` with `StandardScaler` for production-grade K-Means.

---

## 7) Validation, Assumptions & Sensitivity
- **Cluster number (k):** Selected via **elbow** at k ≈ 5; one could confirm with **Silhouette** scores for robustness. (p. 13)  
- **Scaling:** Not applied in the notebook; recommend trying **StandardScaler** (especially if extending feature set). (p. 12)  
- **Age binning:** Loses within-bin variation; future work could test **continuous age** (after scaling) or **ordinal encodings**. (pp. 8–12)  
- **Statistical tests:**  
  - χ² used with crosstabs (Gender vs Score/Income).  
  - Pearson detects **small positive** Income–Score correlation.  
  - Levene + ANOVA show **age group** differences on spending and (per interpretation) income; the **ANOVA code cell for income appears to reuse “Score”**—plots and narrative nonetheless discuss income differences. (pp. 9–11)

---

## 8) What to Do with These Segments
- **Untapped Affluents (High-income, Low-spend | “Blue”)**  
  **Activation:** curated bundles, concierge onboarding, targeted trials.  
  **Messaging:** premium value, time savings, exclusivity. (p. 15)

- **Value Maximizers (Low-income, High-spend | “Green”)**  
  **Retention:** loyalty perks, predict churn risk early, protect experience. (p. 15)

- **Proportional Spenders (Red/Violet/Orange)**  
  **Upsell:** progressive laddering (good, better, best); **CLV** improvements through gentle cross-sell. (p. 15)

---

## 9) Limitations & Next Steps
- **Small sample (n=200):** Use as a **teaching / prototyping** dataset; validate on larger first-party data. (p. 2)  
- **Feature scope:** Only 3 features inform K-Means; consider **behavioral variables** (visits, basket size, recency/frequency) if available. (p. 12)  
- **Scaling & distance metric:** Standardize and evaluate **k-means vs. k-medoids** or **Gaussian Mixtures**.  
- **Model selection:** Complement elbow with **Silhouette**, **Calinski–Harabasz**, and **Davies–Bouldin**.  
- **Statistical rigor:** For unequal variances, use **Welch’s ANOVA** post-hoc; add **multiple-comparison controls** (Tukey/Tamhane). (pp. 9–11)


---

## 10) Tech Stack
**Python**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, **SciPy** (χ², Pearson, Levene, ANOVA), **scikit-learn** (`KMeans`), **mpl_toolkits** (3D plotting).

---

## Appendix A — K-Means Objective (for completeness)
Given data \(X=\{x_i\}_{i=1}^n\) and centroids \(\{\mu_k\}_{k=1}^K\), K-Means minimizes:
\[
\min_{\{\mu_k\},\{z_{ik}\}} \sum_{i=1}^n \sum_{k=1}^K z_{ik}\,\lVert x_i - \mu_k\rVert^2
\quad \text{s.t. } z_{ik}\in\{0,1\},\ \sum_{k=1}^K z_{ik}=1
\]
The **elbow method** inspects inertia vs. \(K\) and selects the knee (here, **K = 5**).
