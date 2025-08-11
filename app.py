# Importing Necessary Libraries
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Adding Page Configuration
st.set_page_config(page_title="Coral Reef Health Detector", layout="centered")
st.title("🌊 Coral Reef Health Detector")
st.write("Upload a coral reef image to check its health status.")

# Styling UI of page
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #e6f7ff 0%, #ffffff 100%); }
    </style>
    """,
    unsafe_allow_html=True,
)

# CONFIGURATION OF DEVICE
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (3 Classes)
class_names = ['Bleached', 'Dead', 'Healthy']

# TRANSFORMING IMAGES 
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# LOADING THE MODEL
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    # Loading trained weights
    ckpt_path = "coral_mobilenet_ckpt.pth" 
    state_dict = torch.load(ckpt_path, map_location=device)['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# PREDICTION FUNCTION
def predict_image(img: Image.Image):
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs

# FILE UPLOADING
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# MAKING PREDICTIONS AND PLOTTING GRAPH
if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        probs = predict_image(img)
        top_idx = int(np.argmax(probs))

        # Showing all class probabilities
        st.subheader("Prediction Results:")
        for cls, p in zip(class_names, probs):
            st.write(f"{cls:<10}: {p*100:.2f}%")

        # Showing final classification
        st.markdown(
        f"""
        <h2 style='text-align: center; color: #333333;'>
            Classified as (CNN): <span style='color: #2A9D8F;'>{class_names[top_idx]}</span>
        </h2>
        """,
    unsafe_allow_html=True
)

        # Adding pastel colors for bars
        pastel_colors = ["#A8D5BA", "#FBC4AB", "#A0C4FF"]  # Healthy, Bleached, Dead

        # Creating a bar chart
        fig, ax = plt.subplots()
        bars = ax.bar(class_names, probs * 100, color=pastel_colors)
        ax.set_ylabel("Probability (%)")
        ax.set_title("Prediction Confidence")
        ax.set_ylim(0, 100)

        # Adding percentage labels above bars
        for bar, p in zip(bars, probs * 100):
            ax.text(bar.get_x() + bar.get_width()/2, p + 1, f"{p:.2f}%",
                    ha='center', fontsize=10)

        st.pyplot(fig)

# END MESSAGE
st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** This model achieved **93% accuracy** during testing, "
    "but it can still make mistakes. Use predictions as guidance, not absolute truth."
)


