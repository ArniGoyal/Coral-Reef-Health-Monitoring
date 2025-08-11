# 🌊 Coral Reef Health Detector

This project utilizes a MobileNetV2 model to classify coral reef images into three categories: **Healthy**, **Bleached**, and **Dead**. The model was trained using the [BHD Corals Dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals), which contains images of coral reefs categorized into these three classes.

## 📊 Dataset

The dataset comprises images of coral reefs labeled as:

- **Healthy**: Corals that are in a healthy state.
- **Bleached**: Corals that have undergone bleaching.
- **Dead**: Corals that are no longer alive.

You can access and download the dataset from Kaggle here:  
🔗 [BHD Corals Dataset](https://www.kaggle.com/datasets/sonainjamil/bhd-corals)

> **Note**: Due to file size constraints, the dataset is not included in this repository. Please download it from the provided link.

## 🧠 Model

The model used is MobileNetV2, a lightweight deep learning model suitable for mobile and edge devices. The model was fine-tuned on the BHD Corals dataset to classify coral reef images into the three categories mentioned above.

The trained model weights (`coral_mobilenet_ckpt.pth`) are included in this repository and are used for inference.

## 🚀 Deployment

This application is deployed using [Streamlit](https://streamlit.io/), allowing users to upload coral reef images and receive predictions on their health status.


