import collections
import pandas as pd
import numpy as np
import prediction
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    data_frame = pd.read_json('reviews_Video_Games_5.json', lines=True)
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
#################################################################################
    # preping testing data for evaluation
    testing_data = data_frame.drop(training_data.index)
    testing_data = testing_data.drop(columns=['reviewerName', 'helpful', 'reviewText', 'summary', 'unixReviewTime', 'reviewTime'])
    testing_data = testing_data.pivot_table(values='overall', index='reviewerID', columns='asin')
    testing_data_user_items = {}
    for i, row in testing_data.iterrows():
        rows = [x for x in range(0, len(testing_data.columns))]
        combine = list(zip(row.index, row.values, rows))
        rated = [(x, z) for x, y, z in combine if str(y) != 'nan']
        row_names = [i[0] for i in rated]
        testing_data_user_items[i] = row_names
    print(testing_data_user_items)
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
    pivot_table = training_data.pivot_table(values='overall', index='reviewerID', columns='asin').fillna(0)
    pivot_table = pivot_table.apply(np.sign)
    # print(pivot_table.head())

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
    # print(notrated)

    # Nearest Neighbor Recommender
    n = 10
    cosine_nn = NearestNeighbors(n_neighbors=n, algorithm='brute', metric='cosine')
    item_cosine_nn_fit = cosine_nn.fit(pivot_table.T.values)
    item_distances, item_indices = item_cosine_nn_fit.kneighbors(pivot_table.T.values)

    # Item Based Recommender
    items_dic = {}
    for i in range(len(pivot_table.T.index)):
        item_idx = item_indices[i]
        col_names = pivot_table.T.index[item_idx].tolist()
        items_dic[pivot_table.T.index[i]] = col_names
    # print(items_dic)

    topRecs = {}
    for k,v in rows_indexes.items():
        item_idx = [j for i in item_indices[v] for j in i]
        item_dist = [j for i in item_distances[v] for j in i]
        combine = list(zip(item_dist, item_idx))
        diction = {i: d for d, i in combine if i not in v}
        zipped = list(zip(diction.keys(), diction.values()))
        sort = sorted(zipped, key=lambda x: x[1])
        recommendations = [(pivot_table.columns[i], d) for i, d in sort]
        topRecs[k] = recommendations

    # predictions
    item_distances = 1 - item_distances
    predictions = item_distances.T.dot(pivot_table.T.values) / np.array([np.abs(item_distances.T).sum(axis=1)]).T
    ground_truth = pivot_table.T.values[item_distances.argsort()[0]]

    # Eval predictions
    def mae(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return mean_absolute_error(prediction, ground_truth)

    def rmse(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    # error_rate = rmse(predictions, ground_truth)
    # print("Accuracy: {:.3f}".format(100 - error_rate))
    print("MAE: {:.5f}".format(mae(predictions, ground_truth)))
    print("RMSE: {:.5f}".format(rmse(predictions, ground_truth)))

    # making a 10-item list of recommendations to all users
    def get_recommendations(user):
        to_file = '\n\n'+user+' - '+str(rate[user])
        users_recommendations.write(to_file)
        count = 0
        for a, b in topRecs.items():
            if user == a:
                for j in b[:10]:
                    to_file = '\n{} with similarity: {:.4f}'.format(j[0], 1 - j[1])
                    users_recommendations.write(to_file)
                    if j[0] in testing_data_user_items.get(user):
                        count += 1
        items_count = len(testing_data_user_items.get(user))
        return count/10, count/items_count
    users = np.unique(training_data.reviewerID.to_numpy())
    users_recommendations = open('users_recommendations.txt', 'a')
    users_counter = 0
    sum_precision = 0
    sum_recall = 0
    conversion_rate = 0
    for user in users:
        users_counter += 1
        count_precision, count_recall = get_recommendations(user)
        sum_precision += count_precision
        sum_recall += count_recall
        if count_precision > 0:
            conversion_rate += 1
    users_recommendations.close()

    # Eval recommendation
    # Precision
    # if recomm in test then add to a list then take the len of both and take the percentage
    precision = (sum_precision / users_counter) * 100
    recall = (sum_recall / users_counter) * 100
    f_measure = 2 * precision * recall / (precision + recall)
    conversion_rate = (conversion_rate / users_counter) * 100
    print('Precision {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    print('F-measure: {:.4f}'.format(f_measure))
    print('Conversion rate: {:.4f}'.format(conversion_rate))


if __name__ == "__main__":
    main()
