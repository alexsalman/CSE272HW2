import pandas as pd
import prediction
# import csv


def main():
    # data = list(csv.reader(open('ratings_Movies_and_TV.csv')))
    # print(data)
    # data selection and processing
    # I chose Amazon Movies and TV shows for this HW2
    data_frame = pd.read_json('reviews_Movies_and_TV_5.json', lines=True)
    # data_frame.info()
    # grouping by reviewerID to split users' ratings 80 training / 20 testing
    data_frame_group_by_id = data_frame.groupby(data_frame.reviewerID)
    # print(data_frame_group_by_id.get_group("AQP1VPK16SVWM"))
    # split to %80 training
    training_data = data_frame_group_by_id.sample(frac=0.8, random_state=1)
    # print(training_data.groupby(training_data.reviewerID).get_group("AQP1VPK16SVWM"))
    # drop the training set from the main set to get the 20% testing
    testing_data = data_frame.drop(training_data.index)
    # print(testing_data.groupby(testing_data.reviewerID).get_group("AQP1VPK16SVWM"))
    # rating prediction class using user based collaborative filtering
    ubcf = prediction.UserBasedCF(training_data.reviewerID, training_data.asin, training_data.overall)
    user_id_list, item_id_list, rating_list, uir, user_set, item_set = ubcf.prep()
    nn = prediction.nearest_neighbors(user_id_list, item_id_list, rating_list, uir, user_set, item_set)


if __name__ == "__main__":
    main()
