import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample data (replace with your dataset)
import numpy as np
data = np.random.rand(1000, 10)  # 1000 samples, 10 features
labels = np.random.randint(2, size=(1000, 1))  # 0 or 1 for each sample


# Sequential Model: This type of model is a linear stack of layers.
# Dense Layer: Fully connected layer where each input node is connected to each output node.
# Activation Functions: relu for hidden layers to introduce non-linearity, and sigmoid for the output layer to produce a probability.
# Keras is a powerful tool that simplifies the process of building, training, and deploying neural networks. If you have any more questions or need further examples, feel free to ask!

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(data.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary crossentropy loss for binary classification
              metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)

print("Model trained successfully!")

# Evaluate the model on the test data
test_data = np.random.rand(200, 10)  # Replace with your test dataset
test_labels = np.random.randint(2, size=(200, 1))  # Replace with your test labels

results = model.evaluate(test_data, test_labels)
print('Test loss, Test accuracy:', results)

# New data for predictions
new_data = np.random.rand(5, 10)  # Replace with your new data samples

# Predict whether the users will buy something (output close to 1 indicates 'yes', close to 0 indicates 'no')
predictions = model.predict(new_data)
print(predictions)

# Print the predictions
for i, prediction in enumerate(predictions):
    print(f"Data {i + 1}: {prediction[0]}")
    if prediction[0] > 0.5:
        print("Will buy")
    else:
        print("Won't buy")


