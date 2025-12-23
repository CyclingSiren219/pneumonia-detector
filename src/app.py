import streamlit as st
import os
import uuid
import cv2
from predict import load_prediction_model, preprocess_image
from explainability import generate_gradcam_heatmap, save_and_display_gradcam

# Page Config
st.set_page_config(page_title="Pneumonia AI Detector", layout="wide")
st.title("🫁 Pneumonia Detection AI")
st.markdown("Upload a Chest X-Ray to detect pneumonia and visualize the affected regions.")

# Load Model (Cached)
@st.cache_resource
def get_model():
    return load_prediction_model()

model = get_model()

# File Uploader
uploaded_file = st.file_uploader("Choose an X-Ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use unique ID to prevent multi-user collisions
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}.jpg"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original X-Ray")
            st.image(temp_path, width="stretch")

        if st.button("Analyze Image"):
            with st.spinner("Analyzing lungs..."):
                img_array = preprocess_image(temp_path)
                prediction = model.predict(img_array, verbose=0)[0][0]

                if prediction > 0.5:
                    label = "PNEUMONIA"
                    confidence = prediction
                    color_hex = "#ff4b4b" # Red
                    
                    # Generate Explanation
                    heatmap = generate_gradcam_heatmap(model, img_array)
                    if heatmap is not None:
                        # Save visualization to disk
                        output_path = save_and_display_gradcam(temp_path, heatmap)
                        
                        if output_path and os.path.exists(output_path):
                            with col2:
                                st.subheader("AI Explainability (Grad-CAM)")
                                
                                # Display Fix: Read as BGR -> Convert to RGB for Streamlit
                                display_img = cv2.imread(output_path)
                                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                                st.image(display_img, caption="Red areas indicate infection", width="stretch")
                                
                                # Clean up visualization file
                                os.remove(output_path)
                        else:
                            st.error("Error: Image file could not be saved.")
                    else:
                        st.error("Could not generate heatmap.")

                else:
                    label = "NORMAL"
                    confidence = 1 - prediction
                    color_hex = "#09ab3b" # Green
                    with col2:
                        st.subheader("Diagnosis")
                        st.success("No signs of Pneumonia detected.")

                # Results Banner
                st.markdown(f"""
                <div style="background-color:{color_hex};padding:20px;border-radius:10px;text-align:center;">
                    <h2 style="color:white;margin:0;">{label}</h2>
                    <p style="color:white;margin:0;">Confidence: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    finally:
        # Cleanup input file
        if os.path.exists(temp_path):
            os.remove(temp_path)