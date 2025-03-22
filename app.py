import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image

# Load the trained model and preprocessing tools
svm_model = joblib.load("svm_model.pkl")  # Load SVM model
pca = joblib.load("pca.pkl")  # Load PCA
label_encoder = joblib.load("label_encoder.pkl")  # Load Label Encoder

# Function to predict Batik class
def predict_image(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, (90, 90))  # Resize to match training data

    # Extract HOG features
    feature = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    # Apply PCA transformation
    feature_pca = pca.transform([feature])
    
    # Predict using SVM
    label = svm_model.predict(feature_pca)[0]
    
    # Convert label to class name
    class_name = label_encoder.inverse_transform([label])[0]
    
    return class_name

# Streamlit UI
st.title("Batik Banyumasan Classifier")
st.write("Upload an image to classify its Batik variety.")

uploaded_file = st.file_uploader("Choose a Batik Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    prediction = predict_image(image)
    st.write(f"**Predicted Class:** {prediction}")
