# Item Item Based Collaborative Filtering Recommender
# Alex Salman - aalsalma@ucsc.edu
# May 26 Fall 2021
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors


def main():
    print('Item Item Based Collaborative Filtering Recommender: \n')
    # Data selection and processing
    # Any http://jmcauley.ucsd.edu/data/amazon/ (5-core) works on this algorithm
    # Reading JSON file and convert it to a pandas dataframe
    dataframe = pd.read_json('reviews_Video_Games_5.json', lines=True)
    # Drop all the columns that I am not using in this algorithm
    dataframe = dataframe.drop(columns=['reviewerName', 'helpful', 'reviewText', 'summary', 'unixReviewTime', 'reviewTime'])
    # Grouping by reviewerID to split users' ratings 80 training / 20 testing
    dataframe_group_by_id = dataframe.groupby(dataframe.reviewerID)
    # Split to %80 training - 80% of each user ratings
    training_data = dataframe_group_by_id.sample(frac=0.8, random_state=1)
    # Drop the training set from the main set to get the 20% testing data
    # Processing testing data to be used for evaluation
    testing_data = dataframe.drop(training_data.index)
    # Create a dictionary to include all users in the testing set with their rated items as values for their ids
    testing_data = testing_data.pivot_table(values='overall', index='reviewerID', columns='asin')
    testing_data_user_items = {}
    for i, row in testing_data.iterrows():
        rows = [x for x in range(0, len(testing_data.columns))]
        combine = list(zip(row.index, row.values, rows))
        rated = [(x, z) for x, y, z in combine if str(y) != 'nan']
        row_names = [i[0] for i in rated]
        testing_data_user_items[i] = row_names
    # Create a dictionary to include all users in the training set with their rated items as values for their ids
    training_pivot_rating = training_data.pivot_table(values='overall', index='reviewerID', columns='asin')
    # A dictionary with users as keys and lists of items rated as list of values
    user_items_rated = {}
    # To store indices
    rows_indexes = {}
    for i, row in training_pivot_rating.iterrows():
        rows = [x for x in range(0, len(training_pivot_rating.columns))]
        combine = list(zip(row.index, row.values, rows))
        rated = [(x, z) for x, y, z in combine if str(y) != 'nan']
        index = [i[1] for i in rated]
        row_names = [i[0] for i in rated]
        rows_indexes[i] = index
        user_items_rated[i] = row_names
    # User item table
    pivot_table = training_data.pivot_table(values='overall', index='reviewerID', columns='asin').fillna(0)
    pivot_table = pivot_table.apply(np.sign)
    # Nearest Neighbor Recommender
    # Closest nearest neighbors to an item, I used the algorithm (brute) because it is faster
    # I used the metric (cosine) because it measures similarity between vectors. Cosine measures
    # the distance between two vectors. The closest the vectors are, the more correlated they are
    # the farther the are, the more un correlated they are
    n = 5
    cosine_nn = NearestNeighbors(n_neighbors=n, algorithm='brute', metric='cosine')
    item_cosine_nn_fit = cosine_nn.fit(pivot_table.T.values)
    # item indices give the index of the actual item
    item_distances, item_indices = item_cosine_nn_fit.kneighbors(pivot_table.T.values)
    # The Predictions
    item_distances = 1 - item_distances
    predictions = item_distances.T.dot(pivot_table.T.values) / np.array([np.abs(item_distances.T).sum(axis=1)]).T
    ground_truth = pivot_table.T.values[item_distances.argsort()[0]]

    # Eval for predictions
    # Mean Absolute Error (MAE)
    def mae(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return mean_absolute_error(prediction, ground_truth)

    # Root Mean Square Error
    def rmse(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    print("MAE: {:.2f}".format(mae(predictions, ground_truth)))
    print("RMSE: {:.2f}".format(rmse(predictions, ground_truth)))

    # Item Based Recommender
    items = {}
    for i in range(len(pivot_table.T.index)):
        item_index = item_indices[i]
        # get the name of the index and turn it to a list
        col_names = pivot_table.T.index[item_index].tolist()
        # turn the list into a dictionary with keys of items and values of items that are associated with it
        items[pivot_table.T.index[i]] = col_names
    # Top recommendations
    top_recommendations = {}
    for userID, idx in rows_indexes.items():
        # indices for all the items rated
        item_index = [j for i in item_indices[idx] for j in i]
        # distances for all the items rated
        item_dist = [j for i in item_distances[idx] for j in i]
        combine = list(zip(item_dist, item_index))
        # filtering out the items that have been rated
        dictionary = {i: d for d, i in combine if i not in idx}
        zipped = list(zip(dictionary.keys(), dictionary.values()))
        # list the similarity scores from the most similar to the least similar
        # based of the items the user rated and items the user have not rated
        sort = sorted(zipped, key=lambda x: x[1])
        recommendations = [(pivot_table.columns[i], d) for i, d in sort]
        top_recommendations[userID] = recommendations

    # Making a 10-item list of recommendations to all users
    def recommendations(user):
        to_file = '\n\n'+user+' - '+str(user_items_rated[user])
        users_recommendations.write(to_file)
        count = 0
        for a, b in top_recommendations.items():
            if user == a:
                for j in b[:10]:
                    to_file = '\n{} with similarity: {:.4f}'.format(j[0], 1 - j[1])
                    users_recommendations.write(to_file)
                    # Check if training data item in list of testing list items
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
        count_precision, count_recall = recommendations(user)
        sum_precision += count_precision
        sum_recall += count_recall
        if count_precision > 0:
            conversion_rate += 1
    users_recommendations.close()

    # Eval for recommendations
    # Precision
    precision = (sum_precision / users_counter) * 100
    # Recall
    recall = (sum_recall / users_counter) * 100
    # F measure
    f_measure = 2 * precision * recall / (precision + recall)
    # Conversion rate
    conversion_rate = (conversion_rate / users_counter) * 100
    print('Precision: {:.2f}%'.format(precision))
    print('Recall: {:.2f}%'.format(recall))
    print('F-measure: {:.2f}'.format(f_measure))
    print('Conversion rate: {:.2f}%'.format(conversion_rate))


if __name__ == "__main__":
    main()
