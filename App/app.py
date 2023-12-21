import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import requests

st.set_page_config(layout='wide')
st.title('Movie Recommender System')

def fetch_api(movie_id):
    r = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=750553a5d65f40cd8cf97f9dc22952fe&language=en-US'.format(movie_id))
    data = r.json()
    return 'https://image.tmdb.org/t/p/w500/' + data['poster_path']

# Chargement des données
movies = []
with open("data.item", "r") as f:
    for line in f:
        items = line.split('|')
        items[2] = 'https://image.tmdb.org/t/p/w500/' + items[2].replace('\n', '')
        movies.append(items)

movies_df = pd.DataFrame(movies)
movies_df.columns = ["Id", "title", "path"]
movies_df.drop(["path"], axis=1, inplace=True)

# Création de la matrice de similarité
vectorizer = CountVectorizer()
title_matrix = vectorizer.fit_transform(movies_df['title'])
similarity_matrix = cosine_similarity(title_matrix)

# Fonction de recommandation
def recommend(movie_name):
    top5_movies = []
    recommended_movie_posters = []
    movie_index = movies_df[movies_df['title'] == movie_name].index[0]
    distances = similarity_matrix[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[0:5]
    for i in movie_list:
        top5_movies.append(movies_df.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_api(movies_df.iloc[i[0]].Id))
    return top5_movies, recommended_movie_posters

# Streamlit UI
selected_movie_name = st.selectbox('Enter Movie Name', movies_df['title'])

if st.button('Recommend'):
    recommendations, posters = recommend(selected_movie_name)
    st.write('Movies')
    
    # Utiliser st.beta_columns pour afficher les films sur la même ligne
    col1, col2, col3, col4, col5 = st.columns(5)
    for i in range(len(recommendations)):
        with [col1, col2, col3, col4, col5][i % 5]:
            st.image(posters[i])
            st.header(recommendations[i])
