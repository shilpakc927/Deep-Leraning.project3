from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json

app = Flask(__name__)

# Global variables to manage camera state
camera = None
streaming = False

# Load trained model
model = load_model("Web Cam_model.keras")

# Load class labels
with open("class_labels.json") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

IMG_SIZE = (96, 96)


def generate_frames():
    """
    Generator that captures frames from the webcam,
    performs prediction, and yields JPEG-encoded frames.
    """
    global camera, streaming

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    while streaming:
        success, frame = camera.read()
        if not success:
            break

        # Center crop region of interest
        h, w, _ = frame.shape
        box_size = 224
        x1, y1 = w // 2 - box_size // 2, h // 2 - box_size // 2
        x2, y2 = x1 + box_size, y1 + box_size

        roi = frame[y1:y2, x1:x2]

        # Preprocess for prediction
        roi_resized = cv2.resize(roi, IMG_SIZE)
        roi_array = img_to_array(roi_resized) / 255.0
        roi_array = np.expand_dims(roi_array, axis=0)

        # Predict
        predictions = model.predict(roi_array, verbose=0)
        class_index = np.argmax(predictions)
        label = class_labels[class_index]

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


@app.route("/")
def index():
    """
    Render the main page.
    """
    return render_template("object.html")


@app.route("/video")
def video():
    """
    Start streaming video.
    """
    global streaming
    streaming = True
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop", methods=["GET"])
def stop():
    """
    Stop streaming and release webcam.
    """
    global streaming, camera
    streaming = False
    if camera is not None:
        camera.release()
        camera = None
    return '', 204  # JavaScript updates the UI


@app.route("/shutdown")
def shutdown():
    """
    Manually release webcam.
    """
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "✅ Webcam released."


if __name__ == "__main__":
    app.run(debug=True)
