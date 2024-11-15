import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar el dataset
df = pd.read_csv('data.csv')

# Visualizar las primeras filas del dataset
print(df.head())

# Exploración inicial del dataset
print(df.info())
print(df.describe())

# Seleccionamos solo las columnas numéricas para realizar análisis
numeric_df = df.select_dtypes(include=[np.number])

# Correlación entre las variables numéricas
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Graficamos la distribución de las variables numéricas
numeric_df.hist(bins=20, figsize=(10, 8))
plt.show()