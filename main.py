import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Movie Data
movies = pd.read_csv("movies_metadata.csv",
                     usecols=["id", "overview", "title", "vote_average", "vote_count", "release_date"])
movies = movies.dropna().drop_duplicates().reset_index(drop=True)
movies = movies.rename(columns={"id": "movieId"})
movies["movieId"] = movies["movieId"].astype("int64")

# Load Ratings Data
ratings = pd.read_csv("ratings_small.csv")
ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings = ratings.drop("timestamp", axis=1)

# Merge Movie and Ratings Data
movie_ratings = pd.merge(ratings, movies, on="movieId")

# Create a pivot table with user ratings
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0 (assuming missing values mean the user hasn't rated that movie)
user_movie_ratings = user_movie_ratings.fillna(0)

# Feature extraction using TF-IDF on movie overviews
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=5)
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])

# Compute similarity scores
try:
    # Compute similarity scores
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
except Exception as e:
    print(f"Error computing similarity scores: {e}")


# Function to get movie recommendations
def get_content_based_recommendations(title, cosine_sim=cosine_sim, user_ratings=user_movie_ratings, movies=movies):
    if title not in movies['title'].values:
        raise ValueError("Movie title not found in the dataset.")
    # Get the index of the movie that matches the title
    idx = movies.index[movies['title'].str.lower() == title.lower()].tolist()[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Filter movies that the user hasn't rated yet
    unrated_movies = [movie for movie in movies['title'].iloc[movie_indices] if movie not in user_ratings.columns]

    # Return the top 5 most similar movies that the user hasn't rated yet
    return unrated_movies[:5]


# Get content-based movie recommendations based on user preferences
recommended_movies = get_content_based_recommendations('The God father')
print(recommended_movies)
