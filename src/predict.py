import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from explainability import generate_gradcam_heatmap, save_and_display_gradcam

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumoniaDetectionModel.keras")
IMG_SIZE = (224, 224)

# --- LOAD MODEL ---
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_prediction_model()

# --- PREDICTION ---
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    if model is None: return

    try:
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        if prediction > 0.5:
            label = "PNEUMONIA"
            confidence = prediction
            
            print("Generating diagnostic heatmap...")
            heatmap = generate_gradcam_heatmap(model, img_array)
            if heatmap is not None:
                save_and_display_gradcam(image_path, heatmap)
        else:
            label = "NORMAL"
            confidence = 1 - prediction
        
        print(f"\n--- Result ---")
        print(f"File: {os.path.basename(image_path)}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_image_path = os.path.join(BASE_DIR, "sample-images", "sample-pneumonia-1.jpeg")
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print(f"Image not found at {test_image_path}")