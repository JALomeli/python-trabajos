import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import seaborn as sns
import matplotlib.pyplot as plt

# Crear el dataset inicial
df1 = pd.DataFrame({
    'x': np.random.normal(5, 3, 100),
    'y': np.random.normal(-2, 2, 100)
})

df2 = pd.DataFrame({
    'x': np.random.normal(15, 2, 100),
    'y': np.random.normal(22, 2, 100)
})

df3 = pd.DataFrame({
    'x': np.random.normal(-5, 3, 100),
    'y': np.random.normal(8, 2, 100)
})

# Concatenar los DataFrames
df = pd.concat([df1, df2, df3], ignore_index=True)
outliers = pd.DataFrame({'x': [200, -200], 'y': [200, -200]})
df = pd.concat([df, outliers], ignore_index=True)
sns.relplot(data=df, x='x', y='y')
plt.title('Datos Originales')
plt.show()

# Entrenar el modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
sns.scatterplot(data=df, x='x', y='y', hue=kmeans.predict(df), palette='viridis')
plt.title('Resultado de K-Means')
plt.show()

# Entrenar el modelo K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(df)
sns.scatterplot(data=df, x='x', y='y', hue=kmedoids.predict(df), palette='viridis')
plt.title('Resultado de K-Medoids')
plt.show()
