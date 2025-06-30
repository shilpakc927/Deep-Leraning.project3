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
}
```
---

## 🧠 Model Training (`Web Cam.ipynb`)

The model architecture can be any Keras-compatible classifier.
Example architecture:
```python
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```
**Training Workflow**
- Load dataset.
- Preprocess images (resize, normalize).
- One-hot encode labels.

- Compile model:
  ```python
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  ```

- Train:
   ```python
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)
   ```

- Save:
  ```python
  model.save("Web Cam_model.keras")
  with open("class_labels.json", "w") as f:
    json.dump(class_indices, f)
  ```
---

## 🌐 Flask Application (`app.py`)

The Flask app manages routing and predictions.
- Starting/stopping the webcam
- Streaming annotated video
- Routing and rendering the HTML template

**Main Routes:**
- `/`  
  Renders object.html (interface with Start and Stop buttons).
- `/vedio feed`  
  Streams frames as a multipart HTTP response (MJPEG).
- `/stop`  
  Stops streaming and shows thank you message.

**Example prediction flow:**
```python
image = cv2.resize(frame, (96,96))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
label = class_labels[np.argmax(prediction)]
```

**Video streaming logic:**
- Capture frames continuously with OpenCV.
- Annotate each frame with the predicted label.
- Encode as JPEG and yield to browser.
---

## 🖥️ HTML Templates

**`object.html`** is a single, fully styled page:
```
✅ Webcam video stream embedded
✅ Start Detection button
✅ Stop Detection button
✅ Live object name shown
✅ Thank you message after stopping detection
```

**Behavior:**
- When Start clicked: video feed begins.
- When Stop clicked: stream stops, blank screen with thank you message.

✅ You can customize styles with CSS.


---
## ⚙️ How to Run
1. **Clone the repository:**
    ```python
    git clone https://github.com/your-username/object-detection.git
    cd object-detection

     ```

3. **Create a virtual environment:**
    ```python
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
     ```
    
5. **Install dependencies:**
     ```python 
    pip install -r requirements.txt
      ```
(Example requirements.txt below)

7. **Train model (if needed):**
   - Open the Jupyter Notebook (Web Cam .ipynb).
   - Run all cells to generate Web Cam_model.h5 and class_labels.json.

8. **Run Flask app:**
     ```python
    python app.py
     ```
     
10. **Open your browser:**
    ```
    http://127.0.0.1:5000/
    ```
---
## 🧪 Workflow Summary
```
✅ Step 1: Load main page (/) – shows Start Detection button.
✅ Step 2: Click Start – webcam feed starts, labels update in real time.
✅ Step 3: Click Stop Detection – shows thank you message.

```
---
## 🛠️ Technologies Used
- Python 3
- TensorFlow/Keras
- Flask
- OpenCV
- HTML/CSS

 ---

## 📄 License
   MIT License
 
---

## 📁 Files Included
- `Web Cam.ipynb`    – Jupyter Notebook for training
- `app.py`           – Flask backend
- `object.html`      – frontend UI
- `Web Cam_model.h5` - trained model
- `class_labels.json`– class label mapping
- `README.md`        – project documentation


---
## 📸 Project Screenshots

**Main Interface**
- [screenshot 1](Screenshot%2033.png)
- [screenshot 2](Screenshot%2021.png)

**Live Detection**
- [screenshot 1](Screenshot%2027.png)
- [screenshot 2](Screenshot%2025.png)
- [screenshot 3](Screenshot%2026.png)
- [screenshot 4](Screenshot%2023.png)

**Ending message**
- [Thank You Message](Screenshot%2028.png)

---
## 📝 Example requirements.txt
```python
flask
tensorflow
opencv-python
numpy
```

## 📩 Contact

**Shilpa K C**  
[LinkedIn](https://www.linkedin.com/in/shilpa-kc) | [Email](shilpakcc@gmail.com)

For questions or suggestions, feel free to reach out.

✅ **How to use this:**
- Copy everything **inside the fences above** (including the triple backticks at the start and end).
- Save it as:
  README.md
- Place it in your project folder.
- Commit and push to GitHub.

✅ This is **one single README file** describing:
- Notebook
- Flask
- one HTML page
- Complete workflow

