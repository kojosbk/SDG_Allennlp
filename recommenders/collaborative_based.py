"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp
import random
# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)
#movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'])]

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD_Team3.pkl', 'rb'))
df_merged = pd.read_csv('resources/data/df_merged.csv')


def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    #Create pivot table
    matrix = df_merged.pivot_table(index=['title'],columns=['userId'],values='rating')
    #Normalise each row
    matrix_norm = matrix.apply(lambda i: (i-np.mean(i))/(np.max(i)-np.min(i)), axis = 1)
    
    # Fill missing values with zeros
    matrix_norm.fillna(0, inplace=True)

    # Transpose the matrix
    matrix_norm = matrix_norm.T


    matrix_norm = matrix_norm.loc[:,(matrix_norm !=0).any(axis=0)]

    #Create a sparse matrix
    matrix_sparse = sp.sparse.csr_matrix(matrix_norm.values)

    # Calculate the cosine similarity matrix
    similarity_score = cosine_similarity(matrix_sparse.T)

    df_sim_movies = pd.DataFrame(similarity_score,index=matrix_norm.columns,columns=matrix_norm.columns)


    if movie_list[0] not in df_sim_movies.columns:
        first_op = pd.DataFrame()
    else:
        first_op = pd.DataFrame(df_sim_movies[movie_list[0]])
        first_op = first_op.reset_index()
        first_op['similarity'] = first_op[movie_list[0]]
        first_op = pd.DataFrame(first_op,columns=['title','similarity'])

    if movie_list[1] not in df_sim_movies.columns:
        second_op = pd.DataFrame()
    else:
        second_op = pd.DataFrame(df_sim_movies[movie_list[1]])
        second_op = second_op.reset_index()
        second_op['similarity'] = second_op[movie_list[1]]
        second_op = pd.DataFrame(second_op,columns=['title','similarity'])

    if movie_list[2] not in df_sim_movies.columns:
        third_op = pd.DataFrame()
    else:
        third_op = pd.DataFrame(df_sim_movies[movie_list[2]])
        third_op = third_op.reset_index()
        third_op['similarity'] = third_op[movie_list[2]]
        third_op = pd.DataFrame(third_op,columns=['title','similarity'])

    options_movies = pd.concat([first_op,second_op,third_op])
    if options_movies.empty:
        reco = df_merged.groupby('title').mean().sort_values(by='rating', ascending=False).index[:top_n].to_list()
        recommended_movies=random.sample(reco,top_n)
    else:
        recommended_movies = options_movies.sort_values('similarity',ascending=False)
        recommended_movies = recommended_movies[~(recommended_movies['title'].isin(movie_list))]
        recommended_movies=list(recommended_movies[0:top_n]['title'])
    return recommended_movies
