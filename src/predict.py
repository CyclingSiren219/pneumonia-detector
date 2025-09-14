# code to run predictions on new images using the trained model
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os


# Define paths
MODEL_PATH = "models/pneumoniaDetectionModel.keras"
IMG_SIZE = (224, 224)

# Load the trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Function to preprocess a single image
def preprocess_image(image_path):
    """
    Load an image, resize it, scale pixel values, and expand dimensions for batch input.
    """
    img = load_img(image_path, target_size=IMG_SIZE)       # load and resize image
    img_array = img_to_array(img) / 255.0                  # convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)          # add batch dimension
    return img_array

# Function to make predictions
def predict_image(image_path):
    """
    Returns 0 for NORMAL and 1 for PNEUMONIA
    """
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
    return label, confidence


if __name__ == "__main__":
    # Replace with your test image path
    test_image_path = "sample-images/sample-normal-1.jpeg"
    
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print(f"Image not found: {test_image_path}")
