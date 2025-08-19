# Example: training a tiny model and saving as image_classifier.h5
# Replace with your real training notebook steps.
import tensorflow as tf
from tensorflow.keras import layers, models

# Dummy tiny CNN for 224x224x3 inputs (replace with your own model)
model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ... train your model here ...

model.save("image_classifier.h5")
print("Saved image_classifier.h5")
