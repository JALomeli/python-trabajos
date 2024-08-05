'''''
Este script realiza el análisis de clustering en un conjunto de datos que incluye un outlier, utilizando los algoritmos K-Means y K-Medoids.

Se siguen los siguientes pasos:
1. Creación de tres conjuntos de datos con distribuciones normales diferentes.
2. Adición de un outlier al conjunto de datos.
3. Concatenación de los DataFrames para formar un único conjunto de datos.
4. Visualización de los datos originales con el outlier.
5. Aplicación del algoritmo K-Means para clustering.
6. Visualización de los resultados del clustering con K-Means.
7. Aplicación del algoritmo K-Medoids para clustering.
8. Visualización de los resultados del clustering con K-Medoids.

Librerías requeridas:
- numpy
- pandas
- scikit-learn
- sklearn_extra
- seaborn
- matplotlib

Instrucciones:
1. Asegúrate de tener instaladas las librerías requeridas.
2. Ejecuta el script para ver las visualizaciones del análisis de clustering, incluyendo el efecto del outlier.

'''''
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import seaborn as sns
import matplotlib.pyplot as plt

# Crear el dataset con outlier
df1 = pd.DataFrame({'x': np.random.normal(5, 3, 100),
                    'y': np.random.normal(-2, 2, 100)})

df2 = pd.DataFrame({'x': np.random.normal(15, 2, 100),
                    'y': np.random.normal(22, 2, 100)})

df3 = pd.DataFrame({'x': np.random.normal(-5, 3, 100),
                    'y': np.random.normal(8, 2, 100)})

# Concatenar los DataFrames
df = pd.concat([df1, df2, df3], ignore_index=True)
outlier = pd.DataFrame({'x': [100], 'y': [100]})
df = pd.concat([df, outlier], ignore_index=True)
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
