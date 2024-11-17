import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tabulate import tabulate

# Hipotesis 1:
#El genero de accion es el genero mas votado y el que mas le gusta a la gente.



# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Cargar el dataset
df = pd.read_csv('data.csv')

# --------------------------------------------------------------------------------------------------------------------------------------
# Calcular la correlacion entre variables
# Seleccionamos solo las columnas numéricas para realizar análisis
numeric_df = df.select_dtypes(include=[np.number])

# Correlación entre las variables numéricas
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------
# Graficamos la distribución de las variables numéricas

numeric_df.hist(bins=20, figsize=(10, 8))
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------
# Obtener una lista de todos los géneros únicos, eliminando los valores NaN
genres_list = df['genres'].dropna().str.split(',').explode().str.strip().unique()

# Crear columnas binarias para cada género
for genre in genres_list:
    df[genre] = df['genres'].apply(lambda x: 1 if isinstance(x, str) and genre in x else 0)

# Ver el dataframe resultante con las nuevas columnas
print(df.head(), end="\n\n")

print(df.info(), end="\n\n")
print('Cantidad de Filas y columnas:',df.shape)
print('Nombre columnas:',df.columns, end="\n\n")

print(df.describe())
# Descripción estadística de los datos numéricos
# Pandas filtra las features numéricas y calcula datos estadísticos que pueden ser útiles: 
#        cantidad, media, desvío estándar, valores máximo y mínimo.


# --------------------------------------------------------------------------------------------------------------------------------------
# Calcular el total de votos de todas las películas
total_votes = df['imdbNumVotes'].sum()

# Calcular la cantidad de votos por género (en porcentaje)
votes_per_genre = {genre: (df[genre] * df['imdbNumVotes']).sum() for genre in genres_list}

# Crear un DataFrame con la suma de votos por género
votes_df = pd.DataFrame(list(votes_per_genre.items()), columns=['Genre', 'TotalVotes'])

# Filtrar los géneros que tienen votos (evitar géneros con 0 votos)
votes_df = votes_df[votes_df['TotalVotes'] > 0]

# Calcular el porcentaje de votos por género
votes_df['Percentage'] = (votes_df['TotalVotes'] / total_votes) * 100

# Calcular la media de la columna 'imdbNumVotes'
mean_votes = df['imdbNumVotes'].mean()

# Graficar el gráfico de barras en porcentaje
plt.figure(figsize=(10, 6))
sns.barplot(x='Percentage', y='Genre', data=votes_df, palette='viridis')

# Añadir la línea de la media (en porcentaje)
mean_percentage = (mean_votes / total_votes) * 100
plt.axvline(x=mean_percentage, color='r', linestyle='--', label=f'Media: {mean_percentage:.2f}%')

# Añadir etiquetas y título
plt.title('Porcentaje de votos por género con línea de la media')
plt.xlabel('Porcentaje de votos')
plt.ylabel('Género')
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------
# Grafico de distribucion de votos

# Filtrar las películas con 'imdbNumVotes' entre 0 y 100
df_filtered = df[(df['imdbNumVotes'] > 0) & (df['imdbNumVotes'] <= 100)]

# Crear el gráfico de distribución proporcional
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered['imdbNumVotes'], bins=20, kde=False, color='skyblue', edgecolor='black', stat='density')

# Añadir etiquetas y título
plt.title('Distribución proporcional de IMDb Num Votes (0-100)')
plt.xlabel('Número de votos (imdbNumVotes)')
plt.ylabel('Densidad (proporcional)')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------
# Progresion del genero de accion a traves de los años

# Filtrar las películas del género "Action" (donde la columna "Action" es 1)
action_movies = df[df['Action'] == 1]

# Agrupar por el año de lanzamiento y contar el número de películas
action_movies_per_year = action_movies.groupby('releaseYear').size()

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(action_movies_per_year.index, action_movies_per_year.values, marker='o', color='b')
plt.title("Progresión de películas de acción a través de los años", fontsize=14)
plt.xlabel("Año de lanzamiento", fontsize=12)
plt.ylabel("Cantidad de películas", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------
# Obtener las 10 películas con mayor 'imdbAverageRating'
top_10_rating = df[['title', 'imdbAverageRating', 'genres']].sort_values(by='imdbAverageRating', ascending=False).head(10)

# Obtener las 10 películas con mayor 'imdbNumVotes'
top_10_votes = df[['title', 'imdbNumVotes', 'genres']].sort_values(by='imdbNumVotes', ascending=False).head(10)

# Convertir a listas para usar con `tabulate`
rating_table = top_10_rating.values.tolist()
votes_table = top_10_votes.values.tolist()

# Imprimir tablas con `tabulate`
print("Top 10 películas con mayor IMDB Average Rating:")
print(tabulate(rating_table, headers=['Title', 'IMDB Rating', 'Genres'], tablefmt='grid'), end="\n\n")

print("Top 10 películas con mayor número de IMDB Votes:")
print(tabulate(votes_table, headers=['Title', 'IMDB Votes', 'Genres'], tablefmt='grid'), end="\n\n")

#--------------------------------------------------------------------------------------------------------------------------------------
# Obteniendo el genero con mejor imdbAverageRating promedio

# Filtrar las filas con datos válidos en `imdbAverageRating` y `releaseYear`
df = df[df['imdbAverageRating'].notnull() & df['releaseYear'].notnull()]

# Convertir `releaseYear` a entero
df['releaseYear'] = df['releaseYear'].astype(int)

# Lista de columnas de géneros, excluyendo "Kids"
genre_columns = [
    'Action', 'Adventure', 'Sci-Fi', 'Drama', 'Western', 'Romance', 'Crime', 'Sport',
    'Music', 'Comedy', 'Mystery', 'Family', 'Thriller', 'Biography', 'War', 'Horror',
    'Fantasy', 'Animation', 'History', 'Musical', 'Documentary', 'Short',
    'Talk-Show', 'TV Movie', 'Science Fiction', 'Reality-TV', 'Game-Show',
    'Reality', 'Talk', 'Sci-Fi & Fantasy', 'Action & Adventure', 'Adult'
]  # "Kids" está excluido

# Calcular el promedio de IMDb para cada género
genre_ratings = {genre: df[df[genre] == 1]['imdbAverageRating'].mean() for genre in genre_columns}

# Encontrar el género con el mayor promedio
best_genre = max(genre_ratings, key=genre_ratings.get)
best_rating = genre_ratings[best_genre]

# Mostrar el resultado
print(f"El género con el mejor promedio de IMDb es: {best_genre} con un rating promedio de {best_rating:.2f}")
print("Se excluyen generos que no tengan valores para al menos 3 años ")

# Calcular la cantidad total de votos para cada género
genre_votes = {genre: df[df[genre] == 1]['imdbNumVotes'].sum() for genre in genre_columns}

# Encontrar el género con la mayor cantidad de votos
best_genre_votes = max(genre_votes, key=genre_votes.get)
best_votes = genre_votes[best_genre_votes]

# Mostrar los resultados
print(f"El género con la mayor cantidad de votos es: {best_genre_votes} con {best_votes:,} votos")

#--------------------------------------------------------------------------------------------------------------------------------------
# Progresion del valoraciones del genro de accion y war a traves de los años

# Filtrar las filas con datos válidos en `imdbAverageRating` y `releaseYear`
df = df[df['imdbAverageRating'].notnull() & df['releaseYear'].notnull()]

# Convertir `releaseYear` a entero para mayor claridad
df['releaseYear'] = df['releaseYear'].astype(int)

# Función para calcular la progresión de un género
def genre_progression(data, genre):
    genre_data = data[data[genre] == 1]  # Filtrar por género
    return genre_data.groupby('releaseYear')['imdbAverageRating'].mean()

# Calcular la progresión para los géneros "Action" y "Kids"
action_progression = genre_progression(df, 'Action')
war_progression = genre_progression(df, 'War')

# Graficar las progresiones
plt.figure(figsize=(12, 6))
plt.plot(action_progression.index, action_progression.values, label="Action", marker='o', color='b')
plt.plot(war_progression.index, war_progression.values, label="War", marker='o', color='g')

# Personalizar el gráfico
plt.title("Progresión del promedio de IMDb a través de los años", fontsize=14)
plt.xlabel("Año de lanzamiento", fontsize=12)
plt.ylabel("Promedio de IMDb", fontsize=12)
plt.legend(title="Género", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
