# rating prediction class using user based collaborative filtering
def user_based_CF(user_id, item_id, rating):
    user_id_list = user_id.values.tolist()
    item_id_list = list(item_id)
    rating_list = rating.values.tolist()
    uir = []
    for count in range(0, len(user_id_list)):
        uir.append([user_id_list[count], item_id_list[count], rating_list[count]])
    print(uir)
    for i in range(len(uir)):
        print(uir[i][0])
