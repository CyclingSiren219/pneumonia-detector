import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name="block5_conv3"):
    """
    Generates a Grad-CAM heatmap using a manual forward pass.
    """
    try:
        # 1. Split model into Feature Extractor (VGG) and Classifier
        vgg_layer = model.get_layer("vgg16")
        target_conv_layer = vgg_layer.get_layer(last_conv_layer_name)
        
        vgg_submodel = Model(
            inputs=vgg_layer.input, 
            outputs=[target_conv_layer.output, vgg_layer.output]
        )
        
        # Classifier layers are everything after VGG16 (GAP -> Dense -> Dropout -> Output)
        classifier_layers = model.layers[2:] 

        # 2. Compute Gradients
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            conv_outputs, vgg_output = vgg_submodel(img_tensor)
            
            # Manual forward pass through classifier
            x = vgg_output
            for layer in classifier_layers:
                x = layer(x)
            
            class_channel = x[:, 0]

        # 3. Generate Heatmap
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to keep only positive influence
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()

    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        return None

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap onto the original image and saves it.
    """
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # Thresholding: Only show heatmap where activation is > 20%
        threshold = 0.2 
        heatmap[heatmap < threshold] = 0
        
        # Convert heatmap to RGB (Jet Colormap)
        heatmap_uint8 = np.uint8(255 * heatmap)
        jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Resize heatmap to match image
        jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))

        # Create transparency mask
        mask = heatmap > threshold
        mask = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))
        mask = mask[..., np.newaxis] 

        # Superimpose
        superimposed_img = (jet_heatmap * alpha * mask) + (img * (1 - (alpha * mask)))
        superimposed_img = np.uint8(superimposed_img)
        
        output_path = img_path.replace(".jpeg", "_gradcam.jpg")
        cv2.imwrite(output_path, superimposed_img)
        print(f"Heatmap saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving heatmap: {e}")