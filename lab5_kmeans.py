import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Generate Customer Dataset
# ==========================================
print("--- 1. GENERATING CUSTOMER DATASET ---")
# Creating 5 distinct customer groups
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.5, random_state=42)

# Mapping the random blobs to realistic 'Annual Income' and 'Spending Score'
income = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), (15, 130))  # Income in $1000s
spending = np.interp(X[:, 1], (X[:, 1].min(), X[:, 1].max()), (1, 100))  # Score 1-100

df = pd.DataFrame({'Annual_Income_k$': income.round(1), 'Spending_Score': spending.round(1)})
print("First 5 rows of the dataset:")
print(df.head())

# ==========================================
# 2. The Elbow Method
# ==========================================
print("\n--- 2. DETERMINING OPTIMAL CLUSTERS (ELBOW METHOD) ---")
wcss = []  # Within-Cluster Sum of Squares

# Test k values from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()  # NOTE: Close this plot window to continue the script!

# ==========================================
# 3. Implement K-Means Clustering
# ==========================================
print("\n--- 3. IMPLEMENTING K-MEANS ---")
# Based on the Elbow method (which will show a sharp bend at K=5), we choose 5 clusters.
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)

# Assign each data point to its respective cluster
cluster_labels = kmeans_final.fit_predict(df)
df['Cluster'] = cluster_labels

print(f"\nCluster Labels successfully assigned! First 10 labels:\n{df['Cluster'].head(10).tolist()}")

# ==========================================
# 4. Visualization & Centroids
# ==========================================
print("\n--- 4. VISUALIZATION & ANALYSIS ---")
centroids = kmeans_final.cluster_centers_

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual_Income_k$', y='Spending_Score', hue='Cluster', data=df,
                palette='tab10', s=70, alpha=0.8, legend='full')

# Plotting the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='red', marker='X',
            edgecolor='black', label='Centroids')

plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (in $1000s)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 5. Cluster Interpretation
# ==========================================
print("\n--- CLUSTER CHARACTERISTICS ---")
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    mean_income = cluster_data['Annual_Income_k$'].mean()
    mean_spending = cluster_data['Spending_Score'].mean()
    size = len(cluster_data)

    print(f"Cluster {i} (Size: {size} customers):")
    print(f"  - Avg Income:   ${mean_income:.2f}k")
    print(f"  - Avg Spending: {mean_spending:.2f}")

    # Simple logic to generate an interpretation
    income_level = "High" if mean_income > 75 else "Low" if mean_income < 45 else "Medium"
    spending_level = "High" if mean_spending > 65 else "Low" if mean_spending < 35 else "Medium"
    print(f"  - Profile: {income_level} Income, {spending_level} Spending\n")