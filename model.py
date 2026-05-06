import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configuration
DATASET_PATH = "dataset"
CATEGORIES = ["bad", "good"] # 'bad' is 0, 'good' is 1
IMG_SIZE = 100 # Resize all images to 100x100 pixels

def load_data():
    data = []
    labels = []
    
    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        class_num = CATEGORIES.index(category) # 0 or 1
        
        if not os.path.exists(path):
            print(f"Warning: Folder '{path}' not found.")
            continue

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                # Read image in grayscale to simplify the data
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_array is not None:
                    # Resize to ensure all images are exactly the same size
                    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    
                    # Flatten the 2D image array into a 1D list of pixels
                    data.append(resized_array.flatten())
                    labels.append(class_num)
            except Exception as e:
                pass
                
    return np.array(data), np.array(labels)

print("Loading and processing images...")
X, y = load_data()

if len(X) == 0:
    print("No images found! Please add images to dataset/good and dataset/bad.")
    exit()

# Split the data: 80% for training, 20% for testing the model's accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the SVM model (this might take a moment)...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Test the model to see how well it learned
print("Evaluating model...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=CATEGORIES))

# Save the trained model to a .pkl file
MODEL_NAME = "gear_model.pkl"
joblib.dump(model, MODEL_NAME)
print(f"Model successfully saved as {MODEL_NAME}!")