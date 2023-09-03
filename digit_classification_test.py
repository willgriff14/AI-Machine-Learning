# Use the trained network (CNN) to predict the labels of the test data
outputs = net.predict(x_test)

# Convert the softmax outputs (probabilities for each class) to class labels.
# The predicted class label is the index of the maximum value in the output array.
labels_predicted = np.argmax(outputs, axis=1)

# Calculate the number of misclassified samples by comparing the predicted labels with the true test labels
misclassified = sum(labels_predicted != labels_test)

# Print the percentage of misclassified samples
print('Percentage misclassified = ', 100 * misclassified / labels_test.size)

# Plotting the test samples and their corresponding predictions

# Set the size of the figure
plt.figure(figsize=(8, 2))

# Plot the first 8 test samples
for i in range(0,8):
    ax = plt.subplot(2, 8, i + 1) # Create a subplot for each image
    plt.imshow(x_test[i,:].reshape(28, 28), cmap=plt.get_cmap('gray_r')) # Display the image
    plt.title(labels_test[i]) # Set the title of the subplot to the true label of the test sample
    ax.get_xaxis().set_visible(False) # Hide the x-axis
    ax.get_yaxis().set_visible(False) # Hide the y-axis

# Plot the network's predictions (in the form of a bar graph) for the first 8 test samples
for i in range(0,8):
    # Predict the label for the test sample. Note: CNN expects the input shape to be (batch_size, height, width, channels).
    # Here, we are predicting one sample at a time, so batch_size = 1.
    output = net.predict(x_test[i,:].reshape(1, 28, 28, 1))
    
    output = output[0, 0:] # Extract the output array from the batched result
    plt.subplot(2, 8, 8 + i + 1) # Create a subplot for the bar graph
    plt.bar(np.arange(10.), output) # Create a bar graph of the predicted probabilities for each class
    plt.title(np.argmax(output)) # Set the title of the subplot to the predicted label
