import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import os
import tensorflow as tf

from model_utils import load_saved, predict_top2

# --------------------------------------------
#  Streamlit Page Setup
# --------------------------------------------
st.set_page_config(
    page_title="Satellite Land Use Classification",
    layout="centered"
)

st.title(" Satellite Image Classification (Top-2 Predictions)")
st.write("Upload a satellite image and the model will identify the top-2 land-use categories.")

# --------------------------------------------
# Model Paths
# --------------------------------------------
MODEL_PATH = "./saved_models/eurosat_model.keras"
CLASS_PATH = "./saved_models/class_names.npy"

# --------------------------------------------
# Load Model Only Once
# --------------------------------------------
@st.cache_resource
def load_model_cached():
    return load_saved(MODEL_PATH, CLASS_PATH)

model, class_names = load_model_cached()


# --------------------------------------------
# File Upload UI
# --------------------------------------------
uploaded_file = st.file_uploader("📁 Upload Satellite Image", type=["jpg", "jpeg", "png"])


# ==================================================================
#  UPLOADED IMAGE WORKFLOW
# ==================================================================
if uploaded_file:

    # ===============================
    #  Show Uploaded Original Image
    # ===============================
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ===============================
    #  Feature 5: Preprocessing Preview (64×64)
    # ===============================
    st.subheader("🖼 Model Input Preview (64×64)")
    resized_img = img.resize((64, 64))
    st.image(resized_img, width=150)

    # Prepare array for model
    arr = np.array(resized_img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # ===============================
    #  Top-2 Predictions
    # ===============================
    st.subheader(" Top-2 Predictions")
    top2 = predict_top2(model, class_names, img)

    for cls, prob in top2:
        st.write(f"**{cls}** — {prob:.4f}")

    # ===============================
    #  Feature 6: Confidence Gauge (Top-1)
    # ===============================
    st.subheader("📈 Confidence Gauge (Top-1 Prediction)")
    top1_class, top1_prob = top2[0]

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=top1_prob * 100,
        title={'text': f"{top1_class} Confidence"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig_gauge)

    # ===============================
    #  Feature 1: Probability Bar Chart (Top-5)
    # ===============================
    st.subheader("📊 Prediction Probabilities (Top-5)")

    probs = model.predict(arr)[0]
    top5_idx = probs.argsort()[-5:][::-1]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([class_names[i] for i in top5_idx], [probs[i] for i in top5_idx])
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # ===============================
    #  Feature 3: CNN Feature Maps (Fixed)
    # ===============================
    st.subheader("🔍 CNN Feature Maps (First Conv Layer)")

    # Get MobileNetV2 base model inside Sequential
    base_model = model.layers[0]

    # Find first convolution layer
    first_conv_layer = None
    for layer in base_model.layers:
        if "Conv" in layer.name and len(layer.output.shape) == 4:
            first_conv_layer = layer.name
            break

    if first_conv_layer is None:
        st.error("Could not locate convolution layer.")
    else:
        st.info(f"Showing maps from: **{first_conv_layer}**")

        feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(first_conv_layer).output
        )

        feature_maps = feature_extractor.predict(arr)[0]  # (H, W, C)

        num_maps = 6
        fig2, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()

        for i in range(num_maps):
            axes[i].imshow(feature_maps[:, :, i], cmap="viridis")
            axes[i].set_title(f"Map {i+1}")
            axes[i].axis("off")

        st.pyplot(fig2)


# ==================================================================
#  Feature 7: RANDOM TEST IMAGE BUTTON
# ==================================================================
st.subheader("🎲 Random Test Image Prediction")

if st.button("Pick a Random Test Image"):
    test_df = pd.read_csv("test_fixed.csv")
    sample = test_df.sample(1).iloc[0]

    img_path = os.path.join("./sf/EuroSAT", sample["image_path"])
    img_random = Image.open(img_path).convert("RGB")

    st.image(img_random, caption="Random Test Image", use_column_width=True)

    top2_random = predict_top2(model, class_names, img_random)

    st.write(" **Top-2 Predictions for Random Image:**")
    for cls, prob in top2_random:
        st.write(f"**{cls}** — {prob:.4f}")
