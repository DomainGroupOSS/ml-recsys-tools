import os

import io
import pandas as pd
import requests
import zipfile

MOVIE_LENS_1M_URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
ratings_csv_name = 'readable_ratings.csv'
users_csv_name = 'users.csv'
movies_csv_name = 'movies.csv'


def get_and_prep_data(root_dir=None):
    """
    Downloads raw data files and makes readable self explanatory csv files
    :param root_dir: choice of root data directory
    :return: paths to ratings, users, and movies csv files
    """

    if not root_dir:
        root_dir = os.path.join(os.path.abspath(os.curdir), 'out')
        os.makedirs(root_dir, exist_ok=True)

    ml_1m_dir = os.path.join(root_dir, 'ml-1m')
    rating_csv_path = os.path.join(ml_1m_dir, ratings_csv_name)
    users_csv_path = os.path.join(ml_1m_dir, users_csv_name)
    movies_csv_path = os.path.join(ml_1m_dir, movies_csv_name)

    if not os.path.exists(rating_csv_path):

        # download raw data if it's not there
        if not os.path.exists(ml_1m_dir):
            download_raw_data(root_dir)

        # make readable csv files
        prep_readable_csvs(ml_1m_dir)

    return rating_csv_path, users_csv_path, movies_csv_path


def download_raw_data(dir_path):
    r = requests.get(MOVIE_LENS_1M_URL)
    if r.ok:
        print('downloading and unzipping raw data..')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dir_path)
    else:
        raise ConnectionError('Failed downloading dataset from %s' % MOVIE_LENS_1M_URL)


def prep_readable_csvs(out_dir):
    # read the movies data
    movies_df = pd.read_csv(os.path.join(out_dir, 'movies.dat'),
                            delimiter='::', header=None,
                            names=['item_ind', 'title', 'genres']). \
        rename({'title': 'itemid'}, axis=1)

    # read the users data
    users_df = pd.read_csv(os.path.join(out_dir, 'users.dat'),
                           delimiter='::', header=None,
                           names=['user_ind', 'gender', 'age', 'occupation', 'zipcode'])

    # change occupations index to occupation names
    occupations_df = get_occupation_names_df()
    users_df = pd.merge(users_df, occupations_df, left_on='occupation', right_on='index')

    # make a userid string out of all the user features: gender-age-occupation_name-zipcode
    users_df['userid'] = users_df[
        ['gender', 'age', 'occupation_name', 'zipcode']]. \
        apply(lambda x: '-'.join([str(el) for el in x]), axis=1)

    # read the ratings data
    ratings_df = pd.read_csv(os.path.join(out_dir, 'ratings.dat'),
                             delimiter='::', header=None,
                             names=['user_ind', 'item_ind', 'rating', 'timestamp'])

    # join with movie titles (itemids)
    ratings_df = pd.merge(ratings_df, movies_df[['item_ind', 'itemid']],
                          on='item_ind').drop('item_ind', axis=1)
    # join with userid strings
    ratings_df = pd.merge(ratings_df, users_df[['user_ind', 'userid']],
                          on='user_ind').drop('user_ind', axis=1)

    # save everything
    movies_df.to_csv(os.path.join(out_dir, movies_csv_name), index=None)
    users_df.to_csv(os.path.join(out_dir, users_csv_name), index=None)
    ratings_df.to_csv(os.path.join(out_dir, ratings_csv_name), index=None)


def get_occupation_names_df():
    # form the dataset README file
    occupations = \
        """
            *  0:  "other"
            *  1:  "academic/educator"
            *  2:  "artist"
            *  3:  "clerical/admin"
            *  4:  "college/grad student"
            *  5:  "customer service"
            *  6:  "doctor/health care"
            *  7:  "executive/managerial"
            *  8:  "farmer"
            *  9:  "homemaker"
            * 10:  "K-12 student"
            * 11:  "lawyer"
            * 12:  "programmer"
            * 13:  "retired"
            * 14:  "sales/marketing"
            * 15:  "scientist"
            * 16:  "self-employed"
            * 17:  "technician/engineer"
            * 18:  "tradesman/craftsman"
            * 19:  "unemployed"
            * 20:  "writer"
        """
    occupations_df = pd.read_csv(io.BytesIO(occupations.encode()),
                                 delim_whitespace=True, header=None,
                                 names=['star', 'ind', 'occupation_name']). \
        drop(['star', 'ind'], axis=1).reset_index()
    return occupations_df
