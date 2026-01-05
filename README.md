# SATELLITE-IMAGE-CLASSIFICATION-USING-CNN
The **Satellite Image Classification System** is a deep-learning web application that classifies satellite images into land-use categories using a Convolutional Neural Network (CNN).
Instead of manually analyzing satellite scenes, the model automatically predicts what type of area an image represents — such as residential, forest, farmland, roads, or industrial sites.

This project helps students, researchers, geographers, and urban planners quickly interpret satellite images in an interactive and visual way.
## Overview

The system is trained on the EuroSAT dataset, which contains thousands of labeled satellite photos.
It uses a saved TensorFlow model (.keras) and loads class names from a NumPy file.

The app allows you to:
- Upload an image
- Resize and preprocess it for the model
- Generate predictions
- Display the Top-2 most likely classes
- Show confidence visually through charts and feature maps

The application is built as a Streamlit web app, making it simple, fast, and user-friendly.

## Tools and Technologies Used

- **Python:** Core programming language for development.
- **TensorFlow / Keras:** Builds and loads the trained CNN model.
- **Streamlit:** Creates the interactive web interface.
- **NumPy:** Handles image arrays and normalization.
- **Pandas:** Loads CSV files for random test images.
- **Matplotlib:** Displays probability bar charts.
- **Plotly:** Shows interactive confidence gauge meters.
- **Pillow (PIL):** Loads and processes uploaded images.
- **EuroSAT Dataset:** Source of labeled satellite images.

## Why These Tools Were Selected

CNNs are highly effective for image recognition tasks.

TensorFlow provides fast model loading and prediction.

Streamlit makes deployment easy without complex backend code.

Plotly + Matplotlib give clear visual interpretation of predictions.

NumPy + Pandas simplify preprocessing and data handling.

Together, these tools create a smooth, educational, and interactive AI application.

## Features

Upload satellite image and classify instantly

Shows Top-2 most probable predictions

Probability bar graph for Top-5 classes

Confidence gauge for Top-1 prediction

Visual preview of resized model input (64×64)

Random sample prediction from test dataset

CNN feature-map visualization (first convolution layer)

Clean and intuitive UI

## How It Works

 User uploads an image (JPG / PNG).

Image is resized to 64×64 and normalized.

CNN model generates predictions.

System extracts:

Top-2 predictions

Top-5 probabilities

Feature maps from the first CNN layer are visualized to show how the model “sees” patterns.

Confidence gauge displays how strongly the model believes the top result.
## Advantages

Automates satellite land-use identification

Supports visual insights through feature maps

Easy-to-use web interface

No retraining required — model loads instantly

Helps learning and research in remote sensing and AI

Works well even on basic systems
## Limitations

- Predictions depend on training dataset quality

Model may confuse visually similar classes

Requires GPU/CPU for smooth performance with larger inputs

Only supports EuroSAT-like satellite images

Not suitable for critical decision-making (scientific validation needed)
## Real-Time Applications

Urban planning: Monitor land-use growth

Agriculture: Identify farmland and crop areas

Environmental research: Track deforestation and land change

Disaster monitoring: Analyze flood or fire-affected regions

Education: Teach CNNs and satellite vision concepts
## Future Enhancements

- Support multiple datasets

Add Grad-CAM heatmap visualization

Enable batch image uploads

Improve model accuracy with fine-tuning

Allow image downloads with prediction overlay

Deploy on cloud platforms

Add training pipeline for retraining with new data

## Conclusion

The Satellite Image Classification System demonstrates the real power of CNNs in remote sensing.
By combining deep learning with an interactive web interface, it makes satellite interpretation simple, visual, and practical.
With continued improvements, this project can evolve into a powerful land-use monitoring and research tool.

## OUTPUT:
<img width="818" height="492" alt="Image" src="https://github.com/user-attachments/assets/0f24374c-d9eb-4e7a-b46a-adc4869dc0b0" />
<img width="522" height="760" alt="Image" src="https://github.com/user-attachments/assets/412e0660-9391-4155-8993-650913c85bad" />
<img width="693" height="681" alt="Image" src="https://github.com/user-attachments/assets/507aa9bb-9a2c-4399-ab5f-b3fda3d3d2d8" />
<img width="693" height="812" alt="Image" src="https://github.com/user-attachments/assets/14d7e678-62e5-4b23-9b98-730f2955c8b6" />
<img width="773" height="785" alt="Image" src="https://github.com/user-attachments/assets/32902829-414e-4f07-93de-c5667e5e0fee" />
<img width="672" height="790" alt="Image" src="https://github.com/user-attachments/assets/4d827fa6-9398-4af8-bb42-f56a6124b549" />

