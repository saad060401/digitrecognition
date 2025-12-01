import numpy as np
import streamlit as st
from tensorflow import keras
from PIL import Image
import cv2

# ============================
# 1. Load trained model
# ============================

@st.cache_resource
def load_model():
    model = keras.models.load_model("mnist_cnn.h5")
    return model

model = load_model()


# ============================
# 2. Preprocessing logic
# ============================

def preprocess_digit_image(pil_img, pad=20):
    # Convert to grayscale
    img = pil_img.convert("L")
    arr = np.array(img)

    # OTSU threshold to separate background and stroke
    _, thresh = cv2.threshold(
        arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If digit is white on black → invert to black-on-white
    if np.mean(thresh) < 128:
        thresh = 255 - thresh

    # Find all dark pixels (digit)
    ys, xs = np.where(thresh < 128)

    if len(xs) == 0 or len(ys) == 0:
        resized = img.resize((28, 28))
        arr = np.array(resized).astype("float32") / 255.0
        arr = 1.0 - arr
        return arr.reshape(28, 28, 1)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Add padding
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(thresh.shape[1], x_max + pad)
    y_max = min(thresh.shape[0], y_max + pad)

    digit_crop = thresh[y_min:y_max, x_min:x_max]

    # Resize to MNIST shape
    crop_img = Image.fromarray(digit_crop)
    crop_img = crop_img.resize((28, 28))

    arr = np.array(crop_img).astype("float32") / 255.0
    arr = 1.0 - arr  # invert so digit = bright and background = dark

    return arr.reshape(28, 28, 1)


def predict_digit(pil_img):
    img_array = preprocess_digit_image(pil_img)
    img_array = np.expand_dims(img_array, axis=0)    # (1,28,28,1)

    preds = model.predict(img_array)
    digit = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))

    return digit, conf


# ============================
# 3. STREAMLIT USER INTERFACE
# ============================

st.title("🧠 Handwritten Digit Recognizer")

st.write("Upload an image containing **one handwritten digit (0–9)**.")

uploaded_file = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    st.image(pil_img, caption="Uploaded Image", width=200)

    with st.spinner("Predicting..."):
        digit, confidence = predict_digit(pil_img)

    st.success(f"Predicted Digit: **{digit}**")
    st.write(f"Confidence: `{confidence:.4f}`")
