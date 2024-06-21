import json
import preference_model
import history_model
import numpy as np

with open('sample_preference.json') as f:
    data = json.load(f)

restaurants = data["restaurants"]
users = data["users"]


for user in users:
    results = preference_model.compute_cosine_similariy(user, restaurants)
    # print(results)


with open('sample_restaurant.json') as f:
    restaurant_info = json.load(f)

with open('sample_history.json') as f:
    eating_history = json.load(f)

restaurant_mapping = restaurant_info["restaurants"]

for user_name in eating_history:
    user_history = eating_history[user_name]
    X, y, type_mapping = history_model.preprocess_data(restaurant_mapping, user_history)

print()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

sequence_length = 5
sequences = []
for i in range(len(X) - sequence_length):
    sequences.append(X[i:i + sequence_length])

# Convert to numpy array
sequences = np.array(sequences)
y = y[sequence_length:]

# Split data into training and testing sets
split_index = int(len(sequences) * 0.8)
X_train, X_test = sequences[:split_index], sequences[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

new_time = "2024-06-23T16:50:12"

input_data = []
for restaurant in restaurant_mapping.keys():
    features = history_model.preprocess_input(new_time, restaurant, restaurant_mapping, type_mapping)
    input_data.append(features)

X_new = np.array(input_data)
sequence_length = 5
X_new = history_model.prepare_sequence(X_new, sequence_length)

probabilities = model.predict(X_new)

for i, restaurant in enumerate(restaurant_mapping.keys()):
    if i >= len(probabilities):
        break
    print(f"Restaurant: {restaurant}, Probability of swiping right: {probabilities[i][0]:.4f}")