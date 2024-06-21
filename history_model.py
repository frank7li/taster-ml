import numpy as np
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime



def preprocess_data(restaurant_mapping, eating_history):
    
    unique_types = list({attrs['type'] for attrs in restaurant_mapping.values()})

    type_encoder = OneHotEncoder()
    type_encoded = type_encoder.fit_transform(np.array(unique_types).reshape(-1, 1)).toarray()
    type_mapping = {unique_types[i]: type_encoded[i] for i in range(len(unique_types))}

    processed_data = []

    for entry in eating_history:
        restaurant = entry["restaurant"]
        time = entry["time"]
        swiped_right = entry["swiped_right"]
        time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S").timestamp()
        restaurant_attributes = restaurant_mapping[restaurant]
        type_vector = type_mapping[restaurant_attributes["type"]]
        # features = [restaurant_attributes["price"], restaurant_attributes["healthiness"]] + list(type_vector) + [time]
        features = [restaurant_attributes["price"], restaurant_attributes["healthiness"]] + list(type_vector)
        processed_data.append((features, swiped_right))

    X = np.array([item[0] for item in processed_data])
    y = np.array([item[1] for item in processed_data]).astype(int)
    
    return X, y, type_mapping


def preprocess_input(time, restaurant, restaurant_mapping, type_mapping):
    # time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S").timestamp()
    # attrs = restaurant_mapping[restaurant]
    # type_vector = type_mapping[attrs["type"]]
    # features = [attrs["price"], attrs["healthiness"]] + list(type_vector) + [time]

    attrs = restaurant_mapping[restaurant]
    type_vector = type_mapping[attrs["type"]]
    features = [attrs["price"], attrs["healthiness"]] + list(type_vector)


    return np.array(features)


def prepare_sequence(X, sequence_length):
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
    return np.array(sequences)
