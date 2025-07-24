# Synthetic-Intelligence-for-Brain-MRI-Augmentation-using-GAN
This project aims to solve the challenge of early and accurate brain tumor detection using MRI scans. By integrating Deep Convolutional Generative Adversarial Networks (DCGANs) for data augmentation, a Convolutional Neural Network (CNN) for classification, and deploying the solution via Gradio and an Android mobile application, this project demonstrates a full pipeline from research to real-world usability.

## 💡 Project Overview
Detecting brain tumors from MRI images is a critical task in medical diagnostics. However, most publicly available datasets are limited in size and often suffer from class imbalance. This reduces the reliability of traditional deep learning models. To overcome this, this project introduces a deep learning solution that:

- Generates synthetic MRI images using a DCGAN to augment the training dataset.

- Trains a CNN classifier to distinguish between tumor and non-tumor images with high accuracy.

- Deploys the trained model using Gradio, creating an interactive web interface.

- Embeds the Gradio interface inside an Android application, enabling mobile-based real-time predictions.

## 🎯 Objectives of the Project
- To tackle data scarcity and imbalance by generating realistic synthetic MRI images using DCGAN.

- To build an accurate and robust CNN model capable of classifying MRI scans into tumor or no tumor.

- To create an accessible platform using Gradio for real-time inference.

- To develop an Android mobile application integrating the web app for easy usage in real-life scenarios.

## 🧪 Technologies and Tools Used
- Deep Learning Frameworks: TensorFlow, Keras

- Data Augmentation: Deep Convolutional GAN (DCGAN)

- Model Deployment: Gradio for web interface

- Mobile Application: Android Studio (Java and XML)

- Development Environment: Google Colab, Visual Studio Code, GitHub

## 🧠 Methodology
1. Data Augmentation using DCGAN
   - A Deep Convolutional GAN was trained on MRI brain images to generate synthetic images resembling real scans. This helped to:

    - Expand the dataset for training.

    - Improve the generalization ability of the CNN.

    - Mitigate overfitting due to limited original samples.

3. CNN Model Training
The synthetic and original MRI images were used to train a CNN model. The CNN architecture includes convolutional layers, ReLU activations, max-pooling, dropout layers, and a fully connected classifier. The model achieved 95.47% test accuracy.

4. Model Deployment via Gradio
The trained .h5 model was integrated into a Gradio interface to provide a user-friendly platform for MRI image classification. Users can upload an MRI image and receive instant feedback about whether a brain tumor is detected.

5. Android Application Integration
The Gradio interface was embedded into a native Android application using a WebView component. This allows the app to connect to the model’s backend in real-time, making the solution mobile and accessible.

## 📊 Results and Discussion
- CNN Classification Accuracy: 95.47%

- Generator Loss (DCGAN): Stabilized between 3–6 during training.

- Discriminator Loss (DCGAN): Stabilized between 0.2–0.6 indicating healthy adversarial training.

## 📌 Confusion Matrix
The confusion matrix demonstrates:

  - High true positive and true negative rates.

  - Minimal false positives and false negatives.

  - Balanced classification performance across classes.

## 📉 Training Graphs
Loss curves for both the generator and discriminator showed stable adversarial training behavior. The CNN model’s accuracy and loss graphs confirmed proper learning and generalization.

## 📱 Android App Features
- Lightweight app that uses WebView to connect to Gradio.

- Allows users to upload MRI scans directly from their device.

- Provides instant predictions without requiring local model inference.

- Minimal and responsive UI with Internet permission handling.

## 🚀 How to Run the Project
✅ Step 1: Train DCGAN
- Open FINAL_MRI_PROJECT.ipynb in Google Colab.

- Train the generator and discriminator to produce synthetic MRI images.

✅ Step 2: Train CNN and Deploy with Gradio
- Open GAN_Gradio Final.ipynb in Colab.

- Load the augmented dataset.

- Train the CNN and save it as .h5.

- Deploy using gr.Interface(...).launch(share=True)

✅ Step 3: Android App Integration
- Open the app/ folder in Android Studio.

- Replace the Gradio URL in MainActivity.java.

- Enable Internet permissions in AndroidManifest.xml.

- Build and run the app on an emulator or Android device.
