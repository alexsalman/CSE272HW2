# rating prediction class using user based collaborative filtering
import math


class UserBasedCF:
    def __init__(self, user_id, item_id, rating):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.uir = None
        self.user_set = None
        self.item_set = None

    def prep(self):
        user_id_list = self.user_id.values.tolist()
        item_id_list = list(self.item_id)
        rating_list = self.rating.values.tolist()
        # dataset user, item, rating
        uir = []
        # build a two dimensional array with user, item, and rate
        for count in range(0, len(user_id_list)):
            uir.append([user_id_list[count], item_id_list[count], rating_list[count]])
        # build a user set and an item set
        user_set = {}
        item_set = {}
        for row in uir:
            # user set
            user_set.setdefault(row[0], {})
            user_set[row[0]].setdefault(row[1], float(row[2]))
            # item set
            item_set.setdefault(row[1], {})
            item_set[row[1]].setdefault(row[0], float(row[2]))
        return user_id_list, item_id_list, rating_list, uir, user_set, item_set

    def nearest_neighbors(self, user_id_list, item_id_list, rating_list, uir, user_set, item_set):
        neighbors = []
        last = []
        for user in user_id_list:
            for (uid, data) in user_set.items():
                if uid != user:
                    user_pc = self.pearson_correlation(user, uid, user_set)
                    neighbors.append((uid, user_pc))
            # sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]), reverse=True)
            sorted_neighbors = sorted(neighbors, reverse=True)
        for i in range(5):
            if i >= len(sorted_neighbors):
                break
            last.append(sorted_neighbors[i])
        return last

    def pearson_correlation(self, user, uid, user_set):
        user_data = user_set[user]
        uid_data = user_set[uid]
        # find user average rating
        rx_avg = self.user_average_rating(user_data)
        ry_avg = self.user_average_rating(uid_data)
        # find common items
        sxy = self.common_items(user_data, uid_data)
        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user_data[item]
            rys = uid_data[item]
            top_result += (rxs - rx_avg) * (rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)
        if top_result != 0 and bottom_left_result * bottom_right_result != 0:
            correlation_coefficient = top_result / (bottom_left_result * bottom_right_result)
        else:
            correlation_coefficient = 0.0
        return correlation_coefficient

    def user_average_rating(self, user_data):
        avg_rating = 0.0
        vol = len(user_data)
        for (movie, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= vol * 1.0
        return avg_rating

    def common_items(self, user_data, uid_data):
        items = []
        ht = {}
        for (movie, rating) in user_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (movie, rating) in uid_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (k, v) in ht.items():
            if v == 2:
                items.append(k)
        return items

    # def predict(self, user_id_list, nn):
    #     valid_neighbors = self.check_neighbors_validattion(item, nn)
    #     if not len(valid_neighbors):
    #         return 0.0
    #     top_result = 0.0
    #     bottom_result = 0.0
    #     for neighbor in valid_neighbors:
    #         neighbor_id = neighbor[0]
    #         neighbor_similarity = neighbor[1]   # Wi1
    #         rating = self.uu_dataset[neighbor_id][item] # rating i,item
    #         top_result += neighbor_similarity * rating
    #         bottom_result += neighbor_similarity
    #     result = top_result/bottom_result
    #     return result
    #
    # def check_neighbors_validattion(self, item, k_nearest_neighbors):
    #     result = []
    #     for neighbor in k_nearest_neighbors:
    #         neighbor_id = neighbor[0]
    #         # print item
    #         if item in self.uu_dataset[neighbor_id].keys():
    #             result.append(neighbor)
    #     return result

