import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define a deep convolutional neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)  # 3 output channels for RGB color
])

# Compile the model
model.compile(optimizer='adam',
              loss='mse',  # Mean Squared Error loss for image generation
              metrics=['accuracy'])

# Generate a dataset of random 3D sculptures
sculptures = np.random.rand(100, 64, 64, 3)
sculptures = sculptures * 255  # Scale to 0-255 range

# Train the model to generate sculptures
model.fit(sculptures, sculptures, epochs=10, batch_size=16)

# Generate a new sculpture
new_sculpture = model.predict(np.random.rand(1, 64, 64, 3) * 255)

# Display the generated sculpture
plt.imshow(new_sculpture[0].astype(np.uint8))
plt.show()
