from tkinter import Image
import streamlit as st  
from streamlit_option_menu import option_menu  
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from time import time
from PIL import Image



with st.sidebar:
    selected = option_menu("Main Menu", 
                           ["Home", "Upload Image", "Live Camera"], 
                           icons=["house", "upload", "camera"], 
                           menu_icon="cast", 
                           default_index=0)

if selected == "Home":
    st.title("Andromeda")
    st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
    st.write("Hello ges howareyu")

# Menu Upload image   
elif selected == "Upload Image":

    st.title("Klasifikasi Gambar Makanan")
    st.write("Unggah gambar makanan untuk diklasifikasikan.")

    IMAGE_SIZE = (224, 224)
    class_names = ['karbohidrat', 'protein', 'buah', 'sayur', 'minuman']

# Load model
    model = tf.keras.models.load_model("./model/best_model.h5")

# Fungsi prediksi
    def predict_image(model, img_array, class_names):
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        predicted_prob = np.max(predictions)
        return predicted_class, predicted_prob



    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocess image
        img_array = np.array(image.resize(IMAGE_SIZE))
        img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
        predicted_class, predicted_prob = predict_image(model, img_array, class_names)

    # Display result
        st.write(f"Prediksi: {predicted_class} dengan probabilitas: {predicted_prob:.2f}")

# Menu live camera 
elif selected == "Live Camera":
    st.title("Live Camera")

# Load model
    model = load_model('./model/best_model.h5')

# Define class names
    class_names = ['karbohidrat', 'protein', 'sayur', 'buah', 'minuman']

    def preprocess_image(image):
        image = cv2.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return np.expand_dims(image, axis=0)

    def get_region_proposals(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = [
            (x, y, w, h) for x, y, w, h in [cv2.boundingRect(c) for c in contours]
            if w > 50 and h > 50
        ]
        return regions

    def detect_objects_in_region(image, box, min_confidence=0.5):
        x, y, w, h = box
        region = image[y:y+h, x:x+w]
        if region.shape[0] < 32 or region.shape[1] < 32:
            return None
        processed_region = preprocess_image(region)
        predictions = model.predict(processed_region, verbose=0)
        confidence = float(np.max(predictions[0]))
        if confidence >= min_confidence:
            class_idx = np.argmax(predictions[0])
            return {'box': box, 'confidence': confidence, 'class': class_names[class_idx]}
        return None

    def apply_nms(detections, nms_threshold=0.3):
        if not detections:
            return []
        boxes = [d['box'] for d in detections]
        scores = [d['confidence'] for d in detections]
        boxes_nms = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores, 0.5, nms_threshold)
        return [detections[i] for i in indices.flatten()]

    def draw_detection(image, detection):
        colors = {
            'karbohidrat': (255, 0, 0), 'protein': (0, 255, 0),
            'sayur': (0, 0, 255), 'buah': (255, 255, 0), 'minuman': (255, 0, 255)
        }
        x, y, w, h = detection['box']
        class_name = detection['class']
        confidence = detection['confidence']
        color = colors.get(class_name, (0, 255, 0))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f'{class_name}: {confidence:.2f}'
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - label_h - 5), (x + label_w, y), color, -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def process_frame(frame, max_regions=10):
        regions = get_region_proposals(frame)
        regions = regions[:max_regions]
        detections = [detect_objects_in_region(frame, box) for box in regions]
        detections = [d for d in detections if d is not None]
        detections = apply_nms(detections)
        for det in detections:
            draw_detection(frame, det)
        return frame

    def realtime_detection():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

    # Set frame resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_time = time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            processed_frame = process_frame(frame)

            frame_count += 1
            if frame_count >= 30:
                fps = frame_count / (time() - fps_time)
                cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame_count = 0
                fps_time = time()

            cv2.imshow('Real-time Food Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        realtime_detection()
