import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- CONFIGURATION ---

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumoniaDetectionModel.keras")
IMG_SIZE = (224, 224)

# --- MODEL HANDLING ---

def build_model_architecture():
    """Reconstructs the model architecture manually for the fallback loader."""
    model = Sequential()
    
    # Input Layer
    model.add(Input(shape=(224, 224, 3)))
    
    # Frozen VGG16 Base
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model.add(base_model)
    
    # Pooling and Dense Layers (Matching trained model structure)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def load_prediction_model():
    """Loads model, falling back to manual build if Keras versions mismatch."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None

    try:
        # Try standard load first
        return load_model(MODEL_PATH)
    except Exception:
        print(f"Standard load failed. Using reconstruction fix...")
        try:
            # Fallback: Build shell and load weights only
            model = build_model_architecture()
            model.load_weights(MODEL_PATH)
            print("Model weights loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading weights: {e}")
            return None

# Load model on startup
model = load_prediction_model()

# --- PREDICTION FUNCTIONS ---

def preprocess_image(image_path):
    """Loads and normalizes image for VGG16 (scale 0-1, batch dimension)."""
    img = load_img(image_path, target_size=IMG_SIZE)       
    img_array = img_to_array(img) / 255.0                  
    img_array = np.expand_dims(img_array, axis=0)          
    return img_array

def predict_image(image_path):
    """Predicts NORMAL or PNEUMONIA with confidence score."""
    if model is None:
        print("Cannot predict: Model not loaded.")
        return

    try:
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Determine label (Threshold 0.5)
        if prediction > 0.5:
            label = "PNEUMONIA"
            confidence = prediction
        else:
            label = "NORMAL"
            confidence = 1 - prediction
        
        print(f"\n--- Result ---")
        print(f"File: {os.path.basename(image_path)}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence*100:.2f}%")
        return label, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")

# --- MAIN ---

if __name__ == "__main__":

    # Change the path below to test with different images
    test_image_path = os.path.join(BASE_DIR, "sample-images", "sample-normal-1.jpeg")
    
    print(f"Running prediction on: {test_image_path}")
    
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print(f"Image not found at {test_image_path}")