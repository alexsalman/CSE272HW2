import pandas as pd
import numpy as np
import prediction
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
# import csv
from collections import OrderedDict


def main():
    # data = list(csv.reader(open('ratings_Movies_and_TV.csv')))
    # print(data)
#################################################################################
    # data selection and processing
    # I chose Amazon Movies and TV shows for this HW2
    data_frame = pd.read_json('reviews_Movies_and_TV_5.json', lines=True)
    # grouping by reviewerID to split users' ratings 80 training / 20 testing
    data_frame_group_by_id = data_frame.groupby(data_frame.reviewerID)
    # print(data_frame_group_by_id.get_group("AQP1VPK16SVWM"))
    # split to %80 training
    training_data = data_frame_group_by_id.sample(frac=0.8, random_state=1)
    training_data = training_data.drop(columns=['reviewerName', 'helpful', 'reviewText', 'summary', 'unixReviewTime', 'reviewTime'])
    # print(training_data.info())
    # print(training_data.head())
    # print(training_data.groupby(training_data.reviewerID).get_group("AQP1VPK16SVWM"))
    # drop the training set from the main set to get the 20% testing
    testing_data = data_frame.drop(training_data.index)
#################################################################################
    # print(testing_data.groupby(testing_data.reviewerID).get_group("AQP1VPK16SVWM"))
    # rating prediction class using user based collaborative filtering
    # ubcf = prediction.UserBasedCF(training_data.reviewerID, training_data.asin, training_data.overall)
    # user_id_list, item_id_list, rating_list, uir, user_set, item_set = ubcf.prep()
    # user_id_list = list(OrderedDict.fromkeys(user_id_list))
    # nn = ubcf.nearest_neighbors(user_id_list, item_id_list, rating_list, uir, user_set, item_set)
    # # prediction = ubcf.predict(user_id_list, nn)
    # print(nn)
    missing_rating = training_data.pivot_table(values='overall', index='reviewerID', columns='asin')
    # print(missing_rating.head())
    rate = {}
    rows_indexes = {}
    for i, row in missing_rating.iterrows():
        rows = [x for x in range(0, len(missing_rating.columns))]
        combine = list(zip(row.index, row.values, rows))
        rated = [(x, z) for x, y, z in combine if str(y) != 'nan']
        index = [i[1] for i in rated]
        row_names = [i[0] for i in rated]
        rows_indexes[i] = index
        rate[i] = row_names
    # print(rate)
    pivot_table = training_data.pivot_table(values='overall', index='reviewerID', columns='asin').fillna(0)
    pivot_table = pivot_table.apply(np.sign)
    print(pivot_table.head())

    notrated = {}
    notrated_indexes = {}
    for i, row in pivot_table.iterrows():
        rows = [x for x in range(0, len(missing_rating.columns))]
        combine = list(zip(row.index, row.values, row))
        idx_row = [(idx, col) for idx, val, col in combine if not val > 0]
        indices = [i[1] for i in idx_row]
        row_names = [i[0] for i in idx_row]
        notrated_indexes[i] = indices
        notrated[i] = row_names
    print(notrated)

    # Nearest Neighbor Recommender
    n = 5
    cosine_nn = NearestNeighbors(n_neighbors=n, algorithm='brute', metric='cosine')
    item_cosine_nn_fit = cosine_nn.fit(pivot_table.T.values)
    item_distances, item_indices = item_cosine_nn_fit.kneighbors(pivot_table.T.values)

    # Item Based Recommender



if __name__ == "__main__":
    main()
