import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Cargar el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df = pd.DataFrame(X_scaled, columns=feature_names)
df['true_labels'] = y

# Visualizar el conjunto de datos
sns.pairplot(df, hue='true_labels', palette='viridis')
plt.show()

# Entrenar el modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
df['kmeans'] = kmeans.predict(X_scaled)
sns.pairplot(df, hue='kmeans', palette='viridis')
plt.show()

# Entrenar el modelo K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(X_scaled)
df['kmedoids'] = kmedoids.predict(X_scaled)
sns.pairplot(df, hue='kmedoids', palette='viridis')
plt.show()
