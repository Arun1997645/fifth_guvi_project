# src/streamlit_app.py
"""
SolarGuard: Streamlit Web App
A user-friendly interface to upload solar panel images and get AI-powered defect predictions.
"""

# Import Streamlit for building interactive web apps
import streamlit as st
# Import numpy for numerical operations
import numpy as np
# Import PIL's Image class for opening and manipulating image files
from PIL import Image
# Import TensorFlow for loading and running the trained model
import tensorflow as tf
# Import pickle for loading Python objects (used for class indices)
import pickle
# Import Path from pathlib for handling file paths
from pathlib import Path

# Set page configuration for the Streamlit app (title, icon, layout, sidebar)
st.set_page_config(
    page_title="SolarGuard",
    page_icon="🌞",
    layout="centered",
    initial_sidebar_state="auto"
)

# Define paths relative to project root for model and class indices
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_cnn_model.h5"           # Path to trained model
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.pkl"   # Path to class indices file

@st.cache_resource
def load_model_and_classes():
    """
    Load the trained model and class names.
    Decorated with @st.cache_resource to avoid reloading on every interaction.
    """
    try:
        # Load the trained model from disk
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Load class indices (mapping from class names to integer labels)
        if not CLASS_INDICES_PATH.exists():
            raise FileNotFoundError(f"Class indices file not found: {CLASS_INDICES_PATH}")
        with open(CLASS_INDICES_PATH, 'rb') as f:
            class_indices = pickle.load(f)
        
        # Sort and return class names in the correct order
        class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
        return model, class_names

    except Exception as e:
        # Show error message in the app and stop execution if loading fails
        st.error(f"❌ Error loading model or class indices: {e}")
        st.stop()
        return None, None

def preprocess_image(image):
    """
    Preprocess the uploaded image to match model input requirements.
    - Resize to 224x224
    - Convert to numpy array
    - Normalize pixel values (0–1)
    - Add batch dimension
    """
    image = image.resize((224, 224))              # Resize image to model's expected input size
    image = np.array(image) / 255.0               # Normalize pixel values to range [0, 1]
    image = np.expand_dims(image, axis=0)         # Add batch dimension (shape: 1, 224, 224, 3)
    return image

def get_recommendation(pred_class):
    """
    Return maintenance recommendation based on predicted class.
    """
    # Dictionary mapping each class to a maintenance recommendation
    recommendations = {
        "Clean": "✅ Panel is clean. No action required.",
        "Dusty": "🧹 Dust detected. Schedule cleaning within 1 week to maintain efficiency.",
        "Bird-Drop": "🐦 Bird droppings found. Clean soon to avoid corrosion and hotspots.",
        "Electrical-Damage": "⚡ Electrical damage detected. Inspect wiring and connections immediately.",
        "Physical-Damage": "🔧 Physical crack or break detected. Panel may need replacement.",
        "Snow-Covered": "❄️ Snow coverage detected. Wait for natural melt or use heating system."
    }
    # Return recommendation for the predicted class, or a default message
    return recommendations.get(pred_class, "🔍 Inspection recommended for optimal performance.")

def main():
    # Set the app title and description
    st.title("🌞 SolarGuard: AI-Based Solar Panel Defect Detection")
    st.markdown("""
    Upload a solar panel image to detect its condition using deep learning.  
    SolarGuard classifies defects and provides actionable maintenance recommendations.
    """)
    
    # File uploader widget for user to upload an image
    uploaded_file = st.file_uploader("📤 Choose a solar panel image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='📷 Uploaded Image', use_column_width=True)

        # Show a spinner while the model is analyzing the image
        with st.spinner("🔍 Analyzing image... Please wait."):
            # Load the trained model and class names
            model, class_names = load_model_and_classes()
            
            # If loading failed, show error and exit
            if model is None or class_names is None:
                st.error("Could not load model. Please check the logs.")
                return

            # Preprocess the image for prediction
            processed_img = preprocess_image(image)
            # Make prediction using the model
            pred = model.predict(processed_img)[0]
            # Get the predicted class name
            pred_class = class_names[np.argmax(pred)]
            # Get the confidence score for the prediction
            confidence = np.max(pred)

        # Show the predicted condition and confidence score
        st.success(f"**✅ Predicted Condition:** {pred_class}")
        st.info(f"**📊 Confidence:** {confidence:.2%}")

        # Show maintenance recommendation based on prediction
        st.markdown("---")
        st.subheader("🔧 Maintenance Recommendation")
        st.write(get_recommendation(pred_class))

        # Show prediction probabilities for all classes as a bar chart
        st.markdown("---")
        st.subheader("📈 Prediction Probabilities")
        prob_df = {cls: float(prob) for cls, prob in zip(class_names, pred)}
        st.bar_chart(prob_df)

        # Optional: Show model information
        st.markdown("---")
        st.caption("Model: MobileNetV2 (Transfer Learning) | Accuracy: ~75%")

# If this script is run directly (not imported as a module), execute the main function
if __name__ == "__main__":
    main()