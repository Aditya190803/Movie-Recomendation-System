import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import requests
import streamlit as st
import os
# Load the data
df = pd.read_csv("https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv")
features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')
x = features['Movie_Genre'] + ' ' + features['Movie_Keywords'] + ' ' + features['Movie_Tagline'] + ' ' + features['Movie_Cast'] + ' ' + features['Movie_Director']

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(x)
Similarity_Score = cosine_similarity(x)

# Directly setting the API key for testing
API_KEY = "4be41a6b"

def fetch_movie_details(movie_title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={movie_title}"
    response = requests.get(url)
    data = response.json()
    poster_url = data.get('Poster', 'https://via.placeholder.com/150x225?text=Poster+Not+Found')
    plot = data.get('Plot', 'Plot not available')
    year = data.get('Year', 'Year not available')
    imdb_rating = data.get('imdbRating', 'Rating not available')
    return poster_url, plot, year, imdb_rating

# Streamlit app layout
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎥")
st.title("🎥 Movie Recommendation System 🎬")

with st.form(key='movie_form'):
    movie_name = st.text_input("Enter your favorite movie name:")
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if movie_name:
        list_of_all_titles = df['Movie_Title'].tolist()
        close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if not close_match:
            st.error("No close match found. Please try again with a different movie name.")
        else:
            match = close_match[0]
            index_of_movie = df[df.Movie_Title == match]['Movie_ID'].values[0]
            recommendation_score = list(enumerate(Similarity_Score[index_of_movie]))
            sorted_similar_movies = sorted(recommendation_score, key=lambda x: x[1], reverse=True)
            
            st.header(f"Top 5 similar movies to '{movie_name}':")
            
            for i, movie in enumerate(sorted_similar_movies[:5]):
                index = movie[0]
                title_from_index = df[df.Movie_ID == index]['Movie_Title'].values[0]
                poster_url, plot, year, imdb_rating = fetch_movie_details(title_from_index)
                
                col1, col2 = st.columns([1, 2])  # Adjusted column sizes for better layout
                with col1:
                    st.image(poster_url, use_column_width=True, width=120)  # Adjusted width for posters
                with col2:
                    st.subheader(title_from_index)
                    st.write(f"Year: {year}")
                    st.write(f"IMDB Rating: {imdb_rating}")
                    st.write(f"Plot: {plot}")
    else:
        st.error("Please enter a movie name.")
