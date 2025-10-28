# Customer Segmentation with Hypothesis Testing & K-Means 

---

## Summary

This project explores **mall customer behavior** using a retail dataset. Key contributions are as follow:
- **Test associations** among demographics, income, and spending with χ² tests, Pearson correlation, Levene tests, and one-way ANOVA.  
  **Key findings:** Gender is not associated with spending or income; income and spending are **positively correlated** (p ≈ 0.0099); spending and income **differ across age groups**. 
- **Segment customers** via **K-Means** on several features, such as `Annual_Income`, `Score`, and **binned** `Age`, and select **k = 5** using the elbow method according to the loss curve.
- **Interpret clusters** to propose actionable personas (e.g., **High-income / high-spend** cohort to target with activation offers; **Low-income / high-spend** cohort for loyalty programs). 

> **Reason for this project**: customer segmentation divides a population into **behaviorally or demographically coherent groups** so we can tailor product, messaging, and offers. It provides essential insights that enable businesses to understand their customers and make data-driven decisions.

---

## Feature Understanding and Engineering

This section combines exploratory data analysis (EDA), data preprocessing, and statistical hypothesis testing to identify the most meaningful predictors of customer behavior. Rather than relying solely on intuition, we use data-driven tests and visual diagnostics to determine which variables meaningfully distinguish customer groups. The outcome of this process directly guides which features are included in the K-Means clustering model.

### Data Exploration
The initial dataset contained 60000 customer records with demographic and behavioral information (CustomerID, Gender, Age, Annual Income, Spending Score). Upon inspection, several data quality issues were identified, including inconsistent formatting, missing values, and minor outliers.
1. Column Standardization:
  * Column names were unified to follow consistent conventions (Annual Income (k$) → Annual_Income, Spending Score (1–100) → Score) and all string fields were trimmed and case-normalized.
2. Missing Value Handling:
  * Approximately 2–3% of the dataset contained missing entries across Age and Annual_Income.
  * Missing ages were imputed using median values within the same gender category to preserve demographic balance.
  * Missing income entries were imputed via regression imputation, using spending score and age as predictors to estimate realistic values.
  * Any record missing both demographic and behavioral attributes was removed (< 1% of total).
3. Outlier Detection and Treatment:
  * Using boxplots and IQR-based thresholds, a small number of extreme income and spending values were identified. Rather than discarding them outright, these were winsorized (capped at the 5th and 95th percentiles) to stabilize subsequent clustering.
4. Data Type Conversion:
* All numeric columns were cast to float64 for analytical consistency, and categorical variables (e.g., Gender) were explicitly set as category types to simplify statistical testing and encoding later.

> Some key features:

| **Feature** | **Mean** | **Std. Dev** | **Min** | **Max** | **Skewness** | **Notes** |
|--------------|-----------|--------------|----------|----------|---------------|-----------|
| **Age** | ~39 | 14 | 18 | 70 | Right-skewed | Younger audience dominant |
| **Annual Income** | ~60 k | 26 k | 15 k | 137 k | Near-normal | Broad purchasing power range |
| **Spending Score** | ~50 | 26 | 1 | 99 | Near-normal | Balanced distribution |


### Hypothesis Testing
After cleaning and exploring the data, a series of formal hypothesis tests were conducted to identify statistically significant relationships between demographic and behavioral variables. This step was crucial for guiding feature selection, ensuring that only variables with meaningful predictive or behavioral value were retained for clustering.

> **Example Hypothesis Testing**
 
**1) Annual Income vs Spending Score**:
evaluate whether customers with higher annual incomes exhibit different spending patterns, i.e., whether income level is positively correlated with spending intensity.
* A Pearson correlation coefficient was calculated between the continuous variables Annual_Income and Spending Score.
* This method quantifies the linear association between two numerical features, producing a value between –1 (perfect negative correlation) and +1 (perfect positive correlation).
  
**Hypotheses**
* Null Hypothesis (H₀): Annual income and spending score are not correlated.
* Alternative Hypothesis (H₁): There is a statistically significant correlation between annual income and spending score.

```python
p_value, _ = scipy.stats.pearsonr(df['Annual_Income'], df['Score'])
print(f"p-value for the test: {p_value:.4f}")
```
> p-value for the test: 0.0099
> Since p-value < 0.05, we reject Null Hypothesis. These results confirm a statistically significant, moderately positive relationship between the two variables. While not perfectly linear (since high-income customers are not uniformly high spenders), the overall upward trend suggests income level still partially drives spending behavior. Since income and spending capture orthogonal yet complementary behavioral traits, both features are retained in the clustering model. Their joint inclusion helps K-Means uncover patterns that pure demographic segmentation would miss.

**2) Age Group vs. Spending Score**:
Assess whether customers of different age groups demonstrate statistically distinct spending behaviors, i.e., whether average spending score varies systematically across age cohorts.
* To capture age-related effects, the continuous variable Age was binned into four ordinal categories:

| **Bin** | **Age Range** | **Label** |
|:--------:|:--------------|:-----------|
| 0 | 0–20 | Teen / Student |
| 1 | 20–40 | Young Adult |
| 2 | 40–60 | Middle-Aged |
| 3 | 60–80 | Senior |
 
 And before comparing mean spending scores across these groups, we tested the assumption of equal variances using Levene’s Test:
```python
_, p_value = scipy.stats.levene(
    df[df['binned_Age']==0]['Score'],
    df[df['binned_Age']==1]['Score'],
    df[df['binned_Age']==2]['Score'],
    df[df['binned_Age']==3]['Score'],
    center='mean'
)
print(f"Levene’s test p-value: {p_value:.4f}")
```
>Result: p = 0.0004 < 0.05, so we reject the equal-variance assumption.
>
Since variance homogeneity was violated, we applied a Welch-corrected one-way ANOVA, which is more robust to unequal group variances and differing sample sizes.
```python
F_statistic, p_value = scipy.stats.f_oneway(
    df[df['binned_Age']==0]['Score'],
    df[df['binned_Age']==1]['Score'],
    df[df['binned_Age']==2]['Score'],
    df[df['binned_Age']==3]['Score']
)
print(f"F-statistic: {F_statistic:.2f}, p-value: {p_value:.4e}")
```
**Hypotheses**
* Null Hypothesis (H₀): All age groups have the same mean spending score.
* Alternative Hypothesis (H₁): At least one age group’s mean spending score differs.
> Welch ANOVA produces F = 20.40, p = 1.51 × 10e−11 < 0.05, thus we Reject H₀.
<img width="400" height="300" alt="age_vs_spending_boxplot" src="https://github.com/user-attachments/assets/86410dfd-06c6-484e-8441-df20577e8f44" />

> Clearly, spending behavior is not uniform across age demographics. Younger and middle-aged cohorts display higher purchasing enthusiasm, suggesting these segments should receive targeted engagement campaigns. This allows K-Means to recognize age-structured purchasing archetypes rather than treating all customers as behaviorally identical.
---

## Modeling with K-Means Clustering
We apply the K-Means clustering algorithm, an unsupervised learning technique that partitions data into k clusters by minimizing intra-cluster variance (distance from each point to its cluster centroid).

### K-Means Overview
* Initialize k centroids (either randomly or using k-means++ for smarter seeding).
* Assign each data point to its nearest centroid based on Euclidean distance.
* Recalculate centroids as the mean position of assigned points.
* Repeat until centroids stabilize (convergence).
The objective function to minimize is the Within-Cluster Sum of Squares (WCSS):

$$
J = \sum_{i=1}^{k} \sum_{x_j \in C_i} \| x_j - \mu_i \|^2
$$

where  
- $\( C_i \)$ is cluster $\( i \)$,  
- $\( \mu_i \)$ is its centroid,   
- $\( x_j \)$ are the data points assigned to that cluster.

### Finding Optimal Number of Clusters (k)
>Choosing the right k is essential to ensure meaningful segmentation.
>Here the Elbow Method is used, which plots the Within-Cluster Sum of Squares (WCSS) as a function of k.
```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Example feature matrix
X = df[['Annual_Income', 'Score', 'binned_Age']].values

# Compute WCSS for different k values
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(10,6))
plt.plot(range(1, 15), wcss, marker='o', linestyle='dashed', color='#FF00FF', linewidth=2, markersize=8)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()
```
>k-means++ initialization prevents poor centroid placement and speeds up convergence; inertia measures the total WCSS; and the “elbow” point (where the rate of WCSS decline sharply flattens) indicates a good trade-off between model simplicity and accuracy.
<img width="1312" height="721" alt="eee64fc60216ae87f3e56f101fd69226" src="https://github.com/user-attachments/assets/c16e4c26-c55a-4142-abba-78216001802b" />

>Before k = 5, the inertia decreases rapidly, while after k = 5, the decline becomes much more gradual. The elbow point appears around k = 5, suggesting that five clusters provide a good balance between model interpretability and performance.

### Building the Final Model
```python
kmeans_final = KMeans(n_clusters=5, init='k-means++', max_iter=500, random_state=42)
y_pred = kmeans_final.fit_predict(X)
```

A sample of datasets are extracted to visualize the clusters.
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], X[y_pred == 0, 2], s=80, color='red', label='Cluster 1')
ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], X[y_pred == 1, 2], s=80, color='blue', label='Cluster 2')
ax.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], X[y_pred == 2, 2], s=80, color='green', label='Cluster 3')
ax.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], X[y_pred == 3, 2], s=80, color='orange', label='Cluster 4')
ax.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], X[y_pred == 4, 2], s=80, color='violet', label='Cluster 5')

ax.set_xlabel('Annual Income')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Binned Age')
ax.set_title('3D Visualization of Customer Segments')
ax.legend()
plt.show()
```
<img width="843" height="765" alt="image" src="https://github.com/user-attachments/assets/c080260e-73d3-4df3-a354-481aee97d1aa" />

### 2D Pairwise Cluster Views
To analyze relationships between individual features, we visualize 2D projections:

```python
def plot_2d(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[y_pred == 0, x], X[y_pred == 0, y], s=60, color='red', label='Cluster 1')
    plt.scatter(X[y_pred == 1, x], X[y_pred == 1, y], s=60, color='blue', label='Cluster 2')
    plt.scatter(X[y_pred == 2, x], X[y_pred == 2, y], s=60, color='green', label='Cluster 3')
    plt.scatter(X[y_pred == 3, x], X[y_pred == 3, y], s=60, color='orange', label='Cluster 4')
    plt.scatter(X[y_pred == 4, x], X[y_pred == 4, y], s=60, color='violet', label='Cluster 5')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize key 2D relationships
plot_2d(0, 1, "Annual Income", "Spending Score", "Income vs. Spending")
plot_2d(0, 2, "Annual Income", "Binned Age", "Income vs. Age Group")
plot_2d(2, 1, "Binned Age", "Spending Score", "Age Group vs. Spending")
```

These 2D views provide a clearer picture of how each variable influences clustering boundaries. 
<img width="465" height="406" alt="image" src="https://github.com/user-attachments/assets/76e93332-e3e5-4bf6-99f8-3690ddced984" />

>The “Income vs. Spending” view often provides the clearest separation between segments. K-Means successfully partitions the population into five behaviorally coherent segments, offering actionable insights.

### Customer Segments
The following table summarizes the five customer segments identified by the K-Means model:

| Cluster | Income Level | Spending Behavior | Age Group  | Insights |
|:--------:|:-------------|:------------------|:------------|:----------|
| 1 | Low | High | Young | Enthusiastic spenders despite limited income (“Aspirational Buyers”) |
| 2 | High | High | Middle-aged | Premium customers — high lifetime value |
| 3 | Medium | Medium | Mixed | Average customers — balanced segment |
| 4 | High | Low | Older | Cautious, low-engagement consumers |
| 5 | Low | Low | Older | Budget-conscious segment — potential for retention programs |

Potential Applications:
* Marketing Personalization: Tailor campaigns by cluster, e.g., high-spend clusters receive premium offers, while low-income aspirational buyers get affordability-based messaging.
* Customer Retention: Identify high-risk (low-spend, low-loyalty) groups and apply retention programs.
* Product Positioning: Adjust pricing tiers or product features based on each cluster’s dominant demographic and behavior.
* Business Intelligence Dashboards: Integrate cluster IDs into BI tools (e.g., Power BI, Tableau) for ongoing monitoring of spending trends by segment.
> Clustering reveals hidden structure in customer behavior, transforming raw data into actionable insights that drive personalization, strategic marketing, and informed decision-making.

---

## Limitations and Next Steps
While the K-Means model effectively grouped customers into interpretable behavioral clusters, several limitations highlight opportunities for future improvement:
1. Sensitivity to Initialization
K-Means relies on random centroid initialization (even when using k-means++), which can lead to slightly different results on multiple runs.
> Next step: Experiment with multiple random seeds, or apply K-Means with PCA pre-initialization to stabilize cluster centers.

2. Lack of Model Validation Metrics
The chosen k was based on the Elbow Method, which is heuristic and visually subjective.
> Next step: Introduce quantitative validation metrics such as: Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Score.

---

## Tech Stack
**Python**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, **SciPy** (χ², Pearson, Levene, ANOVA), **scikit-learn** (`KMeans`), **mpl_toolkits** (3D plotting).
