from .call_model import collaborative_filtering_predict
from .content_based_filtering import run_content_based_recommendation
import numpy as np
import pandas as pd

PATH = './data/'


def get_top_3(recommended_movie_id, predict_output):
    sorted_prediction = sorted(list(predict_output))
    top_3_movie_id = [recommended_movie_id[list(predict_output).index(v)] for v in sorted_prediction[-3:]]
    return top_3_movie_id


def movie_metadata(movie_id):
    movies = pd.read_csv(PATH + 'movies.csv')
    temp_ds = movies.loc[movies['movieId'] == movie_id]
    genres = temp_ds["genres"].tolist()[0].split("|")
    return {"id": movie_id, "title": temp_ds["title"].tolist()[0], "genres": genres}


def run_process(user_id, returnMetadata=False):
    # recommended_movie_id is movie_ids based on input user content based filtering first
    recommended_movie_id = run_content_based_recommendation(user_id)

    # predict results from collaborative based filtering
    predict_output = collaborative_filtering_predict(user_id, recommended_movie_id)

    # get top 3 predicted result
    top_3_movie_id = get_top_3(recommended_movie_id, predict_output)

    if returnMetadata:
        output = {"item": [movie_metadata(movie_id) for movie_id in top_3_movie_id]}
    else:
        output = {"item": [{"id": movie_id} for movie_id in top_3_movie_id]}
    return output
