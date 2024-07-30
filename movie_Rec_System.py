import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# Merging datasets
df = pd.merge(ratings, movies, on='movieId')
df['genres'] = df['genres'].astype(str).str.split('|')
df = df.explode('genres')

# Extract unique genres and movie titles
unique_genres = df['genres'].unique()
unique_movie_titles = df['title'].unique()

# Pre-compute popularity-based recommendations
@st.cache_data
def precompute_popularity():
    genre_recommendations = {}
    for genre in unique_genres:
        genre_data = df[df['genres'] == genre]
        reviews_count = genre_data['title'].value_counts()
        reviews_count = reviews_count.rename('Num Reviews')
        avg_rating = genre_data.groupby('title')['rating'].mean()
        avg_rating = avg_rating.rename('Average Movie Rating')
        popular_movies = pd.merge(reviews_count, avg_rating, on='title')
        popular_movies = popular_movies.sort_values(by='Average Movie Rating', ascending=False)
        popular_movies = popular_movies.reset_index().rename(columns={'title': 'Movie Title'})
        genre_recommendations[genre] = popular_movies[['Movie Title', 'Num Reviews', 'Average Movie Rating']]
    return genre_recommendations

genre_recommendations = precompute_popularity()

def popularity_based(genre, min_reviews, num_recommendations):
    if genre in genre_recommendations:
        popular_movies = genre_recommendations[genre]
        popular_movies = popular_movies[popular_movies['Num Reviews'] >= min_reviews]
        return popular_movies.head(num_recommendations)['Movie Title']
    return []

# Pre-compute content-based similarity matrix
@st.cache_data
def precompute_content_similarity():
    df_cb = movies.copy()
    df_cb['genres'] = df_cb['genres'].str.split('|')
    df_cb = df_cb.explode('genres')
    df_cb = df_cb.drop_duplicates('movieId').reset_index(drop=True)

    count_vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    genre_vectors = count_vectorizer.fit_transform(df_cb['genres'])

    similarity = cosine_similarity(genre_vectors)
    return df_cb, similarity

df_cb, similarity = precompute_content_similarity()

def content_based(movie, num_rec):
    if movie not in df_cb['title'].values:
        return []
    movie_index = df_cb[df_cb['title'] == movie].index[0]
    similar_movies = similarity[movie_index]
    movie_list = sorted(list(enumerate(similar_movies)), reverse=True, key=lambda x: x[1])[1:num_rec + 1]
    recommended_movies = [df_cb.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Pre-compute collaborative-based user similarity matrix
@st.cache_data
def precompute_collaborative_similarity():
    df_cbr = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    user_similarity = cosine_similarity(df_cbr)
    user_similarity_df = pd.DataFrame(user_similarity, index=df_cbr.index, columns=df_cbr.index)
    return df_cbr, user_similarity_df

df_cbr, user_similarity_df = precompute_collaborative_similarity()

def collaborative_based(userid, num_rec, similar_users):
    if userid not in df_cbr.index:
        return []
    similar_users_list = user_similarity_df[userid].sort_values(ascending=False)[1:similar_users + 1].index
    similar_users_ratings = df_cbr.loc[similar_users_list].mean(axis=0)
    user_rated_movies = df_cbr.loc[userid]
    already_rated = user_rated_movies[user_rated_movies > 0].index.tolist()
    recommendations = similar_users_ratings.drop(already_rated).sort_values(ascending=False)
    top_recommendations = recommendations.head(num_rec)
    return top_recommendations.index.tolist()

# Streamlit UI
st.title("Movie Recommendation System")
st.write("This system recommends movies based on popularity, content, and collaborative filtering.")

# User inputs
option = st.selectbox(
    'Choose a recommendation method',
    ('Popularity-based', 'Content-based', 'Collaborative-based')
)

if option == 'Popularity-based':
    genre = st.selectbox('Select a genre:', unique_genres)
    min_reviews = st.number_input('Enter minimum number of reviews:', min_value=1)
    num_recommendations = st.number_input('Enter number of recommendations:', min_value=1)
    if genre and min_reviews and num_recommendations:
        recommendations = popularity_based(genre, min_reviews, num_recommendations)
        st.write("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)

elif option == 'Content-based':
    selected_movie = st.selectbox('Select a movie:', unique_movie_titles)
    num_recommendations = st.number_input('Enter number of recommendations:', min_value=1)
    if selected_movie and num_recommendations:
        recommendations = content_based(selected_movie, num_recommendations)
        if recommendations:
            st.write("Recommended Movies:")
            for movie in recommendations:
                st.write(movie)
        else:
            st.write("No recommendations found. Please check the movie name.")

elif option == 'Collaborative-based':
    user_id = st.number_input('Enter your user ID:', min_value=1)
    num_recommendations = st.number_input('Enter number of recommendations:', min_value=1)
    similar_users = st.number_input('Enter number of similar users to consider:', min_value=1)
    if user_id and num_recommendations and similar_users:
        recommendations = collaborative_based(user_id, num_recommendations, similar_users)
        if recommendations:
            st.write("Recommended Movies:")
            for movie in recommendations:
                st.write(movie)
        else:
            st.write("No recommendations found. Please check the user ID.")
