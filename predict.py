import cv2
import joblib
import sys

# Configuration
MODEL_NAME = "gear_model.pkl"
IMG_SIZE = 100
CATEGORIES = ["Bad Gear", "Good Gear"]

# Load the saved model
try:
    model = joblib.load(MODEL_NAME)
    print(f"Loaded {MODEL_NAME} successfully.")
except FileNotFoundError:
    print(f"Error: {MODEL_NAME} not found. Please run train.py first.")
    sys.exit()

def predict_gear(image_path):
    # Load and process the new image exactly how we did during training
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_array is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    flattened_array = resized_array.flatten().reshape(1, -1) # Reshape for a single prediction
    
    # Make prediction
    prediction = model.predict(flattened_array)
    confidence = model.predict_proba(flattened_array)
    
    result_class = CATEGORIES[prediction[0]]
    confidence_score = max(confidence[0]) * 100
    
    print(f"Result: {result_class}")
    print(f"Confidence: {confidence_score:.2f}%\n")

# Example usage:
if __name__ == "__main__":
    # You can change this to any image path you want to test
    test_image_path = "test_images/sample_gear.jpg" 
    
    print(f"Analyzing {test_image_path}...")
    predict_gear(test_image_path)
