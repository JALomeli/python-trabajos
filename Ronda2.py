'''''
Este script realiza el análisis de clustering en un conjunto de datos utilizando los algoritmos K-Means y K-Medoids.

Se siguen los siguientes pasos:
1. Creación de datos sintéticos con tres grupos de puntos y un outlier.
2. Concatenación de los datos y el outlier.
3. Visualización de los datos originales.
4. Aplicación del algoritmo K-Means para clustering.
5. Visualización de los resultados del clustering con K-Means.
6. Aplicación del algoritmo K-Medoids para clustering.
7. Visualización de los resultados del clustering con K-Medoids.

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
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns

# Crear datos de ejemplo
c1 = np.random.normal(5, 3, (100, 2))
c2 = np.random.normal(15, 5, (100, 2))
c3 = np.random.normal(-5, 2, (100, 2))

# Crear un outlier
outlier = np.array([[100, 100]])

# Concatenar los datos y el outlier
d = np.concatenate((c1, c2, c3, outlier), axis=0)

# Verificar la forma del conjunto de datos
print(d.shape)
sns.scatterplot(x=d[:, 0], y=d[:, 1])
plt.title('Datos Originales')
plt.show()

# Entrenar el modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=0).fit(d)
sns.scatterplot(x=d[:, 0], y=d[:, 1], hue=kmeans.predict(d), palette='viridis')
plt.title('Resultado de K-Means')
plt.show()

# Entrenar el modelo K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(d)
sns.scatterplot(x=d[:, 0], y=d[:, 1], hue=kmedoids.predict(d), palette='viridis')
plt.title('Resultado de K-Medoids')
plt.show()
