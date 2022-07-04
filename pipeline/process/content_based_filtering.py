import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

PATH = './data/'

movies = pd.read_csv(PATH + 'movies.csv')
ratings = pd.read_csv(PATH + 'ratings.csv')

# drop 0 > rating > 5 data
ratings = ratings.drop(ratings[ratings.rating > 5].index)
ratings = ratings.drop(ratings[ratings.rating < 0].index)

# remove empty UserID rated
ratings['userId'] = ratings['userId'].fillna('')
ratings = ratings.drop(ratings[ratings.userId == ''].index)

# Transforms text to feature vectors that can be used as input
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
movies['genres'] = movies['genres'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data filtered by genres
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Replace NaN with an empty string
movies['title'] = movies['title'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data filtered by title
tfidf_matrix_2 = tfidf.fit_transform(movies['title'])

# Compute the cosine similarity matrix
cosine_sim_2 = linear_kernel(tfidf_matrix_2, tfidf_matrix_2)

indices = pd.Series(movies.index, index=movies['movieId']).drop_duplicates()


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations_idx(movie_id, threshold=0.8, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[movie_id]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 3 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices and similarity score over 0.7
    movie_indices = [i[0] for i in sim_scores if i[1] >= threshold]

    # Return the top 10 most similar movies
    return movie_indices


def run_content_based_recommendation(user_id):
    # get history indices that this user rated
    history_indices_list = ratings[ratings.userId == user_id].index.tolist()

    # get percentile 60 of score and drop it
    score_list = ratings["rating"].iloc[history_indices_list[:]].tolist()
    percent_60 = np.percentile(score_list, q=50)
    drop_list = ratings[ratings.rating <= percent_60].index.tolist()
    history_indices_list = list(set(history_indices_list) - set(drop_list))

    # get movieId list
    movies_id_list = ratings["movieId"].iloc[history_indices_list[:]].tolist()
    recommendation_list = []
    for movie_id in movies_id_list:
        # get genres based recommendation indices & union list which will exclude duplication movie id
        genres_based_recommendation_indices = get_recommendations_idx(movie_id, 0.9, cosine_sim)
        recommendation_list = list(set().union(recommendation_list, genres_based_recommendation_indices))

        # get title based recommendation indices & union list
        title_based_recommendation_indices = get_recommendations_idx(movie_id, 0.9, cosine_sim_2)
        recommendation_list = list(set().union(recommendation_list, title_based_recommendation_indices))

    # remove already watched movie
    recommendation_list = set(recommendation_list) - set(movies_id_list)

    return list(recommendation_list)
