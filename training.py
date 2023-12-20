import cv2
import numpy as np
# from sklearn.svm import SVC

# # Define the HOG parameters
winSize = (20, 20)
blockSize = (10, 10)
blockStride = (5, 5)
cellSize = (5, 5)
nbins = 9

# # Create the HOG descriptor
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# # Prepare the dataset
# data = []
# labels = []
# for i in range(0, 10):
#     # Load the image
#     image_path = f'trainingSet/{i}.png'
#     image = cv2.imread(image_path, 0) #gray 0-255

#     # Resize the image to a fixed size
#     image = cv2.resize(image, winSize)

#     # Compute the HOG features
#     hog_features = hog.compute(image)
#     hog_features = hog_features.flatten().tolist()

#     # Append the features and label to the dataset
#     data.append(hog_features)
#     labels.append(i)

# print(data[0])
# print(labels[0])

# # Train an SVM classifier
# svm = cv2.ml.SVM_create()
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.train(data, cv2.ml.ROW_SAMPLE, labels)

# # Make predictions on the training set
# _, y_pred = svm.predict(data)
# print(y_pred)
# # Calculate the accuracy of the classifier
# # accuracy = accuracy_score(labels, y_pred)
# # print("Training Accuracy:", accuracy)

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define HOG parameters
winSize = (20, 20)
blockSize = (10, 10)
blockStride = (5, 5)
cellSize = (5, 5)
nbins = 9

# Create the HOG descriptor
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# Prepare the dataset
data = []
labels = []

# Assuming you have images in 'trainingSet' folder named 'number_0.png', 'number_1.png', ..., 'number_9.png'
for i in range(10):
        # Load the image
    # image_path = f'trainingSet/number_{i}_{j}.png'
    image_path = f'trainingSet/{i}.png'

    image = cv2.imread(image_path, 0)  # gray

        # Resize the image to a fixed size
    image = cv2.resize(image, winSize)

        # Compute the HOG features
    hog_features = hog.compute(image)
    hog_features = hog_features.flatten().tolist()

        # Append the features and label to the dataset
    data.append(hog_features)
    labels.append(i)

# Convert data and labels to NumPy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Split the dataset into training and testing sets
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(data_train, cv2.ml.ROW_SAMPLE, labels_train)

# Save the trained SVM model
svm.save('svm_model.xml')

# Load the trained SVM model
svm = cv2.ml.SVM_load('svm_model.xml')

# Function to predict a digit using HOG features
def predict_digit(image_path):
    # Read the image
    image = cv2.imread(image_path, 0)  # gray

    # Resize the image to the specified window size
    image = cv2.resize(image, winSize)

    # Compute HOG features
    hog_features = hog.compute(image)
    hog_features = hog_features.flatten().tolist()

    # Reshape the features into a 2D array for prediction
    hog_features = np.array(hog_features).reshape(1, -1)

    # Predict using the trained SVM model
    _, digit_pred = svm.predict(hog_features)

    return int(digit_pred[0, 0])

# Test the function for each digit
for i in range(10):
    image_path = f'testSet/number_{i}_test.png'  # Assuming you have test images named 'number_0_test.png', ..., 'number_9_test.png'
    predicted_digit = predict_digit(image_path)
    print(f"Predicted digit for number_{i}_test.png: {predicted_digit}")
