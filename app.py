import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_my_model():
    return load_model('mnist_digit_recognizer.h5')

model = load_my_model()

# App layout
st.title("🎨 Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below and see the AI prediction!")

# Sidebar controls
st.sidebar.title("Controls")
stroke_width = st.sidebar.slider("Brush Size", 1, 25, 15)
clear_button = st.sidebar.button("Clear Canvas")

# Drawing canvas
st.write("### Draw Here")
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=stroke_width,
    stroke_color="rgba(255, 255, 255, 1)",
    background_color="rgba(0, 0, 0, 1)",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
    display_toolbar=True
)

# Prediction
st.write("### Prediction")
predict_button = st.button("Predict Digit")

if predict_button and canvas_result.image_data is not None:
    # Convert canvas to grayscale and resize
    img_array = np.array(canvas_result.image_data)
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    resized_img = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)
    normalized_img = resized_img.astype('float32') / 255.0
    input_img = normalized_img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(input_img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display results
    st.write(f"## Predicted Digit: **{predicted_digit}**")
    st.write(f"Confidence: **{confidence * 100:.1f}%**")

    # Show probabilities
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0], color="skyblue")
    ax.set_title("Prediction Probabilities")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

if clear_button:
    st.experimental_rerun()

# Instructions
st.markdown("---")
st.write("### How to Use:")
st.write("1. **Draw** a digit on the black canvas (click and drag)")
st.write("2. Click **Predict Digit** to see the AI's guess")
st.write("3. Use **Clear Canvas** to start over")