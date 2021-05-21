# rating prediction class using user based collaborative filtering

def common_items(user_data, uid_data):
    items = []
    ht = []
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

def user_average_rating(user_data):
    avg_rating = 0.0
    vol = len(user_data)
    for (movie, rating) in user_data.items():
        avg_rating += float(rating)
    avg_rating /= vol * 1.0
    return avg_rating

def pearson_correlation(user, uid, user_set):
    correlation_coefficient = 0.0
    user_data = user_set[user]
    uid_data = user_set[uid]
    # find user average rating
    rx_avg = user_average_rating(user_data)
    ry_avg = user_average_rating(uid_data)
    sxy = common_items(user_data, uid_data)


def nearest_neighbors(user_id_list, item_id_list, rating_list, uir, user_set, item_set):
    print(user_id_list)
    neighbor = []
    last = []
    for user in user_id_list:
        for (uid, data) in user_set.items():
            if uid != user:
                user_pc = pearson_correlation(user, uid, user_set)


    return 0


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

