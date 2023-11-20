import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample Movie Data
movies_data = {
    'Title': ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'],
    'Director': ['Director1', 'Director2', 'Director1', 'Director3', 'Director2'],
    'Description': ['Action-packed movie with thrilling scenes',
                    'A hilarious comedy that will make you laugh',
                    'Another action movie with intense sequences',
                    'Drama film with emotional storyline',
                    'Funny comedy with great performances']
}

movies_df = pd.DataFrame(movies_data)

# User's Preferences
user_preferences = {'Genre': 'Action', 'Director': 'Director1'}

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Description'])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies_df.index[movies_df['Title'] == title].tolist()[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return movies_df['Title'].iloc[movie_indices]

# Get movie recommendations based on user preferences
recommended_movies = get_recommendations('Movie1')
print(recommended_movies)
