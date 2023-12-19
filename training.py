import cv2
import numpy as np
# from sklearn.svm import SVC

# Define the HOG parameters
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
for i in range(0, 10):
    # Load the image
    image_path = f'trainingSet/{i}.png'
    image = cv2.imread(image_path, 0) #gray 0-255

    # Resize the image to a fixed size
    image = cv2.resize(image, winSize)

    # Compute the HOG features
    hog_features = hog.compute(image)
    hog_features = hog_features.flatten().tolist()

    # Append the features and label to the dataset
    data.append(hog_features)
    labels.append(i)

print(data[0])
print(labels[0])

# Train an SVM classifier
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(data, cv2.ml.ROW_SAMPLE, labels)

# Make predictions on the training set
_, y_pred = svm.predict(data)
print(y_pred)
# Calculate the accuracy of the classifier
# accuracy = accuracy_score(labels, y_pred)
# print("Training Accuracy:", accuracy)