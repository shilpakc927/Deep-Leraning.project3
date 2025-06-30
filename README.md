# 🎯  Object Detection via Webcam

This project is a **complete end-to-end  object detection system** built with:

- **TensorFlow / Keras**: Deep learning model to recognize objects from live video.
- **Flask Web Application**: Serves the model predictions via a web interface.
- **HTML Frontend** (`object.html`): Fully styled single-page interface to start and stop webcam detection, and show results dynamically.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model-training)
- [Flask Application](#flask-application)
- [HTML Template](#html-template)
- [How To Run](#how-to-run)
- [Workflow Summary](#workflow-summary)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Files Included](#files-included)
- [Project Screenshots](#project-screenshots)
- [Contact](#contact)

---

## 🎯 Project Overview

**Object Detection** uses your webcam to classify objects live and display labels on each frame.

**Workflow:**

1. **Load the trained deep learning model** (`Web Cam_model.keras`).
2. **Capture frames** from the webcam in real time.
3. **Preprocess each frame** to the model’s expected input size.
4. **Predict object class** for each frame.
5. **Overlay the predicted label** on the video stream.
6. **Display the stream in your browser**, with options to start and stop detection.

---

## 🗂️ Project Structure
```
object_detection/
├── app.py # Flask backend to stream and predict
├── Web Cam_model.keras # Trained model
├── class_labels.json # Class label mapping
├── templates/
│ └── object.html # Fully styled frontend page
├── web Cam.ipynb # Optional Jupyter Notebook (model training or exploration)
└── README.md # Project documentation

```
---


## 📊 Dataset

You can train the model on **any object classification dataset**, for example:
- CIFAR-10 (basic objects)
- Custom dataset (captured images of objects relevant to your project)

**Note:**
This README assumes you have already trained the model and saved it as `Web Cam_model.keras`, along with a `class_labels.json` containing:
```json
{
  "0": "Bench",
  "1": "Bicycle",
  "2": "Branch",
  ...
}


