import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

train = pd.read_csv("data/train.csv")

# One-hot encode Soil Type and Crop Type
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(train[['Soil Type', 'Crop Type']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Soil Type', 'Crop Type']))

# Concatenate with Fertilizer Name
df_encoded = pd.concat([train[['Fertilizer Name']], encoded_df], axis=1)

# Group by Fertilizer Name and take mean of encoded features
grouped = df_encoded.groupby('Fertilizer Name').mean().reset_index()

# KMeans clustering
X = grouped.drop('Fertilizer Name', axis=1)
kmeans = KMeans(n_clusters=3, random_state=42)
grouped['Cluster'] = kmeans.fit_predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
grouped['PCA1'] = X_pca[:, 0]
grouped['PCA2'] = X_pca[:, 1]

# Plot clusters in PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(data=grouped, x='PCA1', y='PCA2', hue='Cluster', style='Fertilizer Name', palette='tab10', s=120)
plt.title('KMeans Clusters of Fertilizer Name (One-hot Encoded Soil & Crop Types)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
