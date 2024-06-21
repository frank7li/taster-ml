import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similariy(user_preferences, restaurants):
    user_vector = np.array([user_preferences["price"], user_preferences["healthiness"]])

    matched_similarities = []
    unmatched_similarities = []
    for restaurant in restaurants:
        restaurant_vector = np.array([restaurant["price"], restaurant["healthiness"]])
        similarity = cosine_similarity([user_vector], [restaurant_vector])[0][0]
        matched = True if restaurant["type"] in user_preferences["type"] else False
        if matched:
            matched_similarities.append([restaurant["name"], similarity, matched])
        else:
            unmatched_similarities.append([restaurant["name"], similarity, matched])

    return sorted(matched_similarities, key=lambda x: x[1], reverse=True) + sorted(unmatched_similarities, key=lambda x: x[1], reverse=True)
