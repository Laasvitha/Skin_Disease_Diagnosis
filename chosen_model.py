import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump


# Function to load and preprocess images
def load_images_and_labels(data_dir):
    images = []
    labels = []

    for label in os.listdir(data_dir):
        label_folder = os.path.join(data_dir, label)

        if os.path.isdir(label_folder):
            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)
                try:
                    # Read and resize image
                    image = cv2.imread(file_path)
                    image = cv2.resize(image, (64, 64))
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Load data
train_data_dir = r'C:\Users\elist\OneDrive\Desktop\PY_INFO\Task_2\Task_2\Skin Disease Trained Data Set\skin-disease-datasaet\train_set'
test_data_dir = r'C:\Users\elist\OneDrive\Desktop\PY_INFO\Task_2\Task_2\Skin Disease Trained Data Set\skin-disease-datasaet\test_set'

# Load training and test data
X_train_images, y_train = load_images_and_labels(train_data_dir)
X_test_images, y_test = load_images_and_labels(test_data_dir)

# Normalize the pixel values (values are already between 0 and 255)
X_train = X_train_images.reshape(X_train_images.shape[0], -1) / 255.0
X_test = X_test_images.reshape(X_test_images.shape[0], -1) / 255.0

# Apply Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
print("Training Logistic Regression...")
model = LogisticRegression(max_iter=2000, random_state=42)  # Increased iterations
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}\n")

# Save the model and label encoder
dump(model, 'logistic_regression_model.joblib')
print("Model saved as logistic_regression_model.joblib")
