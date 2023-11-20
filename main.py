import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_and_preprocess_data():
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

    return movie_ratings, movies


def create_user_movie_ratings_pivot(movie_ratings):
    # Create a pivot table with user ratings
    user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

    # Fill missing values with 0 (assuming missing values mean the user hasn't rated that movie)
    user_movie_ratings = user_movie_ratings.fillna(0)

    return user_movie_ratings


def calculate_cosine_similarity(movies):
    # Feature extraction using TF-IDF on movie overviews
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=5)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])

    # Compute similarity scores
    try:
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    except Exception as e:
        raise RuntimeError(f"Error computing similarity scores: {e}")

    return cosine_sim


def get_content_based_recommendations(title, cosine_sim, user_ratings, movies):
    # Input validation
    matching_indices = movies.index[movies['title'].str.lower() == title.lower()].tolist()
    if not matching_indices:
        raise ValueError("Movie title not found in the dataset.")
    elif len(matching_indices) > 1:
        raise ValueError("Multiple movies with the same title found in the dataset. Provide a more specific title.")
    idx = matching_indices[0]

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


def main():
    movie_ratings, movies = load_and_preprocess_data()
    user_movie_ratings = create_user_movie_ratings_pivot(movie_ratings)
    cosine_sim = calculate_cosine_similarity(movies)

    # Take user input for the movie title
    user_input_title = input("Enter the title of a movie you've watched: ")

    try:
        # Get content-based movie recommendations based on user preferences
        recommended_movies = get_content_based_recommendations(user_input_title, cosine_sim, user_movie_ratings, movies)
        print("Recommended movies:")
        print(recommended_movies)
    except ValueError as ve:
        print(f"Error: {ve}")


if __name__ == "__main__":
    main()
