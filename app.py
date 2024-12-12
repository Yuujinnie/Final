import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Cache the model loading function
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model_checkpoint.keras')
    return model

model = load_model()

# Title of the app
st.write("""
# Weather Classification
""")

# File uploader for image input
file = st.file_uploader("Choose a weather photo from your computer (Cloudy, Sunrise, Shine)", type=["jpg", "png"])

# Function to preprocess the image and predict
def import_and_predict(image_data, model):
    size = (64, 64)  # Resize the image to 64x64 pixels
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)  # Fit image to target size
    img = np.asarray(image)  # Convert to numpy array
    img_reshape = img[np.newaxis, ...]  # Reshape for model input
    prediction = model.predict(img_reshape)  # Make prediction
    return prediction

# Handle file upload and display results
if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)  # Open the uploaded image
    st.image(image, use_column_width=True)  # Display the image in the app
    prediction = import_and_predict(image, model)  # Get prediction
    class_names = ['Cloudy', 'Shine', 'Sunrise']  # Define class labels
    result = "OUTPUT: " + class_names[np.argmax(prediction)]  # Get the class with the highest probability
    st.success(result)  # Display the prediction result
