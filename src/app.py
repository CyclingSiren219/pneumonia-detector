import streamlit as st
import os
import uuid
import cv2
from predict import load_prediction_model, preprocess_image
from explainability import generate_gradcam_heatmap, save_and_display_gradcam

# 1. PAGE CONFIGURATION & SIDEBAR
st.set_page_config(page_title="Pneumonia AI Detector", page_icon="🫁", layout="wide")

with st.sidebar:
    st.title("About this App")
    st.markdown("""
    This AI system uses a **VGG16 Convolutional Neural Network** trained on chest X-rays to detect Pneumonia. It also provides **Grad-CAM** visual explanations to highlight areas of concern in the lungs.
    
    **Features:**
    - ⚡ Instant Diagnosis
    - 🔍 **Grad-CAM** Explainability
    - 🔒 Privacy-Focused (Images processed locally)
    """)
    st.warning("**Disclaimer:** This tool is for educational purposes only and should not be used for medical diagnosis.")

# 2. MAIN HEADER
st.title("🫁 Pneumonia Detection AI")
st.markdown("### Upload a Chest X-Ray to detect signs of pneumonia.")

# 3. LOAD MODEL (Cached for Performance)
@st.cache_resource
def get_model():
    return load_prediction_model()

model = get_model()

# 4. FILE UPLOADER
uploaded_file = st.file_uploader("Choose an X-Ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- UNIQUE FILENAME (SCALING FIX) ---
    # Use a random ID so multiple users don't overwrite each other
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}.jpg"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original X-Ray")
            st.image(temp_path, width="stretch")

        # 5. RUN PREDICTION
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing lungs..."):
                img_array = preprocess_image(temp_path)
                prediction = model.predict(img_array, verbose=0)[0][0]

                if prediction > 0.5:
                    label = "PNEUMONIA"
                    confidence = float(prediction)
                    
                    # --- GRAD-CAM GENERATION ---
                    heatmap = generate_gradcam_heatmap(model, img_array)
                    
                    if heatmap is not None:
                        # Save result to disk (logic inside explainability.py handles colors)
                        output_path = save_and_display_gradcam(temp_path, heatmap)
                        
                        if output_path and os.path.exists(output_path):
                            with col2:
                                st.subheader("AI Explainability (Grad-CAM)")
                                
                                # DISPLAY FIX: OpenCV reads as BGR, so we convert to RGB for Streamlit
                                display_img = cv2.imread(output_path)
                                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                                st.image(display_img, caption="Red areas indicate infection", width="stretch")
                                
                                # Clean up the Grad-CAM file after showing it
                                os.remove(output_path)
                        else:
                            st.error("Error: Image file could not be saved.")
                    else:
                        st.error("Could not generate heatmap.")

                    # --- METRICS & EXPLANATION ---
                    st.divider()
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Diagnosis", label, delta="Review Needed", delta_color="inverse")
                    with res_col2:
                        st.progress(confidence, text=f"Confidence Score: {confidence*100:.2f}%")
                    
                    with st.expander("ℹ️ How to read the Heatmap"):
                        st.info("""
                        - **Red/Orange Zones:** High importance. The AI believes these areas show signs of infection (opacity/consolidation).
                        - **Blue/Transparent:** Low importance. Healthy tissue or background.
                        - **Goal:** The red zones should overlap with the cloudy patches in the lungs.
                        """)

                else:
                    label = "NORMAL"
                    confidence = float(1 - prediction)
                    
                    with col2:
                        st.subheader("Diagnosis")
                        st.success("No signs of Pneumonia detected.")
                    
                    st.divider()
                    st.metric("Diagnosis", label, delta="Healthy", delta_color="normal")
                    st.progress(confidence, text=f"Confidence Score: {confidence*100:.2f}%")

    finally:
        # --- CLEANUP ---
        # Delete the temp input file to keep the server clean
        if os.path.exists(temp_path):
            os.remove(temp_path)