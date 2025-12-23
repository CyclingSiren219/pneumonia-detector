import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
import os

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name="block5_conv3"):
    """
    Generates a Grad-CAM heatmap (numpy array) for a given image and model.
    """
    try:
        # Connect the last conv layer to the classifier output
        vgg_layer = model.get_layer("vgg16")
        target_conv_layer = vgg_layer.get_layer(last_conv_layer_name)
        
        vgg_submodel = Model(
            inputs=vgg_layer.input, 
            outputs=[target_conv_layer.output, vgg_layer.output]
        )
        
        classifier_layers = model.layers[2:] 

        # Compute gradients
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            conv_outputs, vgg_output = vgg_submodel(img_tensor)
            x = vgg_output
            for layer in classifier_layers:
                x = layer(x)
            class_channel = x[:, 0]

        # Generate heatmap via average pooling
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize to [0,1]
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        return None

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image and saves it to disk.
    Auto-corrects colors so Red indicates high importance (infection).
    """
    try:
        # Load and resize original image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # Thresholding: Ignore low confidence areas (< 20%)
        threshold = 0.2 
        heatmap[heatmap < threshold] = 0
        
        # Colorize heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Color Correction: Swap BGR to RGB so Red = Infection, Blue = Background
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
        jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))

        # Create overlay
        mask = heatmap > threshold
        mask = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))[..., np.newaxis]
        superimposed_img = (jet_heatmap * alpha * mask) + (img * (1 - (alpha * mask)))
        superimposed_img = np.uint8(superimposed_img)
        
        # Smart Filename: Avoid "_gradcam_gradcam.jpg" duplicates
        root, ext = os.path.splitext(img_path)
        if root.endswith("_gradcam"):
            output_path = img_path 
        else:
            output_path = root + "_gradcam" + ext 

        cv2.imwrite(output_path, superimposed_img)
        print(f"Heatmap saved to: {output_path}")
        
        return output_path

    except Exception as e:
        print(f"Error saving heatmap: {e}")
        return None