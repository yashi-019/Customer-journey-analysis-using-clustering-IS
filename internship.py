import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the dataset (Simulated Customer Journey Data)
data = {
    'CustomerID': range(1, 101),
    'Website_Visits': np.random.randint(1, 20, 100),
    'Product_Views': np.random.randint(1, 50, 100),
    'Cart_Additions': np.random.randint(0, 10, 100),
    'Purchases': np.random.randint(0, 5, 100),
    'Time_Spent_Minutes': np.random.randint(5, 500, 100)
}
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
features = df.drop(columns=['CustomerID'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Find optimal number of clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 4: Apply K-Means Clustering
optimal_k = 3  # Based on Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 5: Visualizing Clusters using PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue=df['Cluster'], palette='viridis', data=df)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Segments using PCA')
plt.legend(title='Cluster')
plt.show()

# Step 6: Analyzing Cluster Characteristics
cluster_summary = df.groupby('Cluster').mean()
print("Cluster Summary:")
print(cluster_summary)
