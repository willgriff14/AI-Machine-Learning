# loading the data

# Importing required module to fetch the MNIST dataset
from tensorflow.keras.datasets import mnist
# Loading MNIST data into training and testing sets
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

# Data preprocessing

# Convert the pixel values from integers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the pixel values to be between 0 and 1 (0-255 maps to 0-1)
x_train /= 255 # this is shorthand for x = x / 255
x_test /= 255

# Convert integer labels to one-hot encoded vectors
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(labels_train, 10) # 10 classes for 10 digits (0-9)
y_test = to_categorical(labels_test, 10)

# Reshaping the data to fit the model's expected input shape
# This is necessary for Conv2D layer as it expects the input in the format (batch_size, height, width, channels)
# Here, channels = 1 because MNIST images are grayscale
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Defining the neural network model

# Importing necessary layers and Model API from Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input

# Start defining the neural network
inputs = Input(shape=x_train.shape[1:]) # Defining the input shape

# First Convolution layer with 32 filters of size 5x5 and 'relu' activation
x = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(inputs)
# Pooling layer to reduce dimensions
x = MaxPool2D(pool_size=(2, 2))(x)
# Second Convolution layer with 32 filters of size 3x3 and 'relu' activation
x = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
# Second pooling layer
x = MaxPool2D(pool_size=(2, 2))(x)

# Flatten the 2D data to 1D
x = Flatten()(x)
# Fully connected layer with 256 neurons and 'relu' activation
x = Dense(256, activation='relu')(x)
# Dropout layer to prevent overfitting. It drops 50% of its inputs randomly.
x = Dropout(rate=0.5)(x)
# Output layer with 10 neurons (for 10 classes) and 'softmax' activation for multi-class classification
outputs = Dense(10, activation='softmax')(x)

# Finalizing the model definition
net = Model(inputs=inputs, outputs=outputs)
# Print the model summary to show its architecture
net.summary()

# Model Compilation and Training

# Compile the model by setting the loss, optimizer and metrics
net.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model using the training data, validate using the testing data
history = net.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=20,batch_size=256)

# Visualizing the training process

# Importing the required module for plotting
import matplotlib.pyplot as plt

# Plotting the training and validation loss over epochs
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Save the trained model to disk
net.save("network_for_mnist.h5")
