'''''
Este codigo realiza el análisis de clustering en el conjunto de datos Iris utilizando los algoritmos K-Means y K-Medoids.

Se siguen los siguientes pasos:
1. Carga del conjunto de datos Iris.
2. Normalización de los datos.
3. Creación de un DataFrame para facilitar la visualización.
4. Visualización de los datos originales 
5. Aplicación del algoritmo K-Means para clustering.
6. Visualización de los resultados de K-Means.
7. Aplicación del algoritmo K-Medoids para clustering.
8. Visualización de los resultados de K-Medoids.

Librerías requeridas:
- numpy
- pandas
- scikit-learn
- sklearn_extra
- matplotlib
- seaborn

Instrucciones:
1. Asegúrate de tener instaladas las librerías requeridas.
2. Ejecuta el script para ver las visualizaciones del análisis de clustering.

'''''
#Librerias requeridas
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
