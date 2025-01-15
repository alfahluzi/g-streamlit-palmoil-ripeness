import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

st.title("AI Untuk Mendeteksi Kematangan Kelapa Sawit")
st.write(
    "Projek ini bertujuan untuk mengembangkan alat untuk mendeteksi kematangan kelapa sawit. Sistem ini memanfaatkan data foto buah kelapa sawit untuk melihat tingkat kematangannya."
)

st.header("Project Overview")
st.write("Kelapa sawit merupakan komoditas yang sangat besar di Indonesia. Penentuan kematangan kelapa sawit diperlukan untuk menghasilkan produk olahan dengan kualitas yang lebih baik.")
st.write("Sistem pendeteksi kematangan kelapa sawit otomatis dapat bermanfaat untuk mengefisiensikan proses penyortiran buah.")

st.header("Model Overview")
st.image("assets/img/model.png")
st.write('Hyper parameter yang digunakan adalah: batch size = 25, input size = (224,224), rescale = 1/255, dan epoch = 10')
st.write("Hasil training menghasilkan: accuracy: 0.8821 - loss: 0.2886 - val_accuracy: 0.8889 - val_loss: 0.3107")
st.image("assets/img/trainvalgraph.png")
st.write("Hasil ujicoba menghasilkan: Test Loss: 0.3291 & Test Accuracy: 0.8502")
st.image("assets/img/confussionmatrix.png")

st.header("Showcase")
img_file_buffer = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or, Choose a file")

loaded_model = tf.saved_model.load('assets/model')
inference_function = loaded_model.signatures["serving_default"]

input_shape = (224,224)
label = ["Belum matang", "Matang", "Terlalu Matang"]

if img_file_buffer is not None or uploaded_file is not None:
    if uploaded_file is None:
        input_file = img_file_buffer
    else: input_file = uploaded_file
    # To read image file buffer as a PIL Image:
    img = Image.open(input_file)
    img_array = np.array(img)

    image = Image.open(input_file)
    image = image.resize(input_shape)
    image_array = np.array(image)
    image_array = image_array[:, :, :3]  # Keep only the first 3 channels (RGB)
    image_array = image_array / 255.0
    input_image = np.expand_dims(image_array, axis=0)

    # Predict using the model
    input_tensor = tf.constant(input_image, dtype=tf.float32)
    predictions = inference_function(inputs=input_tensor)["output_0"].numpy()
    predicted_label = np.argmax(predictions, axis=-1)[0]  # Extract label index

    st.write("Result:")
    st.write(f"{100 * predictions[0][predicted_label]:.2f}%", label[predicted_label])

    st.write(pd.DataFrame({
        'Label': label,
        'Result': predictions[0],
    }))