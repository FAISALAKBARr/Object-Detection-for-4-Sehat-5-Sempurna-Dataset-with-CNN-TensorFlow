import os
import tensorflow as tf
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from PIL import Image
import gdown
from tqdm import tqdm
import pandas as pd
import io
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Set page configuration
st.set_page_config(
    page_title="Sistem Deteksi Makanan dan Minuman",
    page_icon="🍽️",
    layout="wide"
)

MODEL_ID = '1FMXOk9ifEoZDl4c7NzpANiP2o_Ednt7P' 
MODEL_PATH = 'FINAL_MODEL.h5'

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background-color: #0245d6;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header aplikasi
st.title("🍽️ Teknologi Cerdas untuk Konsumsi Bijak dan Berkelanjutan")
st.write("Sistem ini dapat mengklasifikasikan makanan dan minuman melalui upload gambar atau secara real-time")

# Define class names and descriptions
class_names = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']
class_descriptions = {
    'buah': """
        🍎 Kategori Buah
        - Sumber vitamin dan mineral alami
        - Mengandung serat tinggi
        - Baik untuk sistem pencernaan
    """,
    'karbohidrat': """
        🍚 Kategori Karbohidrat
        - Sumber energi utama tubuh
        - Termasuk nasi, roti, dan umbi-umbian
        - Penting untuk aktivitas sehari-hari
    """,
    'minuman': """
        🥤 Kategori Minuman
        - Membantu hidrasi tubuh
        - Beragam jenis minuman sehat
        - Penting untuk metabolisme
    """,
    'protein': """
        🍖 Kategori Protein
        - Penting untuk pertumbuhan
        - Sumber protein hewani dan nabati
        - Membantu pembentukan otot
    """,
    'sayur': """
        🥬 Kategori Sayuran
        - Kaya akan vitamin dan mineral
        - Sumber serat yang baik
        - Mendukung sistem imun
    """
}

# Load model
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading model from Google Drive...")
            url = f'https://drive.google.com/uc?id={MODEL_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
        
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate areas
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
        
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def get_color_for_class(class_name):
    """
    Return consistent color for each class
    """
    color_map = {
        'buah': (0, 255, 0),     # Green
        'karbohidrat': (255, 0, 0),  # Blue
        'minuman': (0, 0, 255),   # Red
        'protein': (255, 255, 0), # Cyan
        'sayur': (0, 255, 255)    # Yellow
    }
    return color_map.get(class_name, (128, 128, 128))

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    """
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB if image is from OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
        image = Image.fromarray(image)
    else:
        original = np.array(image)
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array, original

def predict_image(image):
    """
    Make prediction with multiple object detection using optimized sliding window
    """
    try:
        processed_image, original_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        
        height, width = original_image.shape[:2]
        output_image = original_image.copy()
        
        # Increase confidence threshold
        CONFIDENCE_THRESHOLD = 0.3  # Dinaikkan dari 0.1
        detected_objects = []

        # Optimize window sizes and stride
        window_sizes = [(height//2, width//2)]  # Kurangi jumlah window size
        stride = height//4  # Tingkatkan stride untuk mengurangi overlap

        for win_h, win_w in window_sizes:
            for y in range(0, height - win_h, stride):
                for x in range(0, width - win_w, stride):
                    # Extract window
                    window = original_image[y:y+win_h, x:x+win_w]
                    
                    # Skip if window is too small
                    if window.shape[0] < 64 or window.shape[1] < 64:  # Tingkatkan minimum size
                        continue
                    
                    # Preprocess window
                    window_processed = Image.fromarray(window)
                    window_processed = window_processed.resize((224, 224))
                    window_array = tf.keras.preprocessing.image.img_to_array(window_processed)
                    window_array = window_array / 255.0
                    window_array = tf.expand_dims(window_array, 0)
                    
                    # Predict
                    window_pred = model.predict(window_array, verbose=0)
                    
                    # Check for detections
                    max_class_idx = np.argmax(window_pred[0])
                    confidence = window_pred[0][max_class_idx]
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        # Store only the highest confidence detection for this window
                        detected_objects.append({
                            'class': class_names[max_class_idx],
                            'confidence': confidence * 100,
                            'bbox': (x, y, x + win_w, y + win_h)
                        })

        # Apply stricter Non-Maximum Suppression
        final_objects = []
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        while detected_objects:
            current = detected_objects.pop(0)
            final_objects.append(current)
            
            # Increase IoU threshold for stricter filtering
            detected_objects = [
                obj for obj in detected_objects
                if calculate_iou(current['bbox'], obj['bbox']) < 0.3  # Turunkan threshold IoU
            ]
        
        # Draw only top N detections
        MAX_DETECTIONS = 3  # Batasi jumlah deteksi yang ditampilkan
        for obj in final_objects[:MAX_DETECTIONS]:
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']
            class_name = obj['class']
            
            # Generate consistent color based on class
            color = get_color_for_class(class_name)
            
            # Draw thicker bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
            
            # Improve label visualization
            label = f"{class_name}: {confidence:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Position label above bounding box
            label_y = max(y1 - 10, label_size[1])
            label_x = x1
            
            # Draw label background
            cv2.rectangle(
                output_image,
                (label_x, label_y - label_size[1] - 10),
                (label_x + label_size[0], label_y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output_image,
                label,
                (label_x, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return {
            'class': class_names[np.argmax(predictions[0])],
            'confidence': float(np.max(predictions[0])) * 100,
            'output_image': output_image,
            'detected_objects': final_objects[:MAX_DETECTIONS]
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5
        self.current_prediction = None
        self.detection_history = []
        self.frame_count = 0
        self.skip_frames = 2

    def transform(self, frame):
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            if self.frame_count % self.skip_frames != 0:
                return img
            
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                result = predict_image(img)
                
                if result:
                    self.current_prediction = result
                    self.last_prediction_time = current_time
                    
                    if 'detected_objects' in result:
                        self.detection_history = result['detected_objects']

            # Draw detections if available
            if self.current_prediction and 'detected_objects' in self.current_prediction:
                for obj in self.current_prediction['detected_objects']:
                    x1, y1, x2, y2 = obj['bbox']
                    class_name = obj['class']
                    confidence = obj['confidence']
                    
                    color = get_color_for_class(class_name)
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.1f}%"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    label_y = max(y1 - 10, label_size[1])
                    cv2.rectangle(img, 
                                (x1, label_y - label_size[1] - 10),
                                (x1 + label_size[0], label_y),
                                color, -1)
                    cv2.putText(img, label,
                              (x1, label_y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return img
            
        except Exception as e:
            print(f"Error in transform: {str(e)}")
            return img

# Modify the WebRTC configuration
def setup_webrtc():
    rtc_configuration = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]}
    )
    
    return webrtc_streamer(
        key="food-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_transformer_factory=VideoTransformer,
        async_transform=True,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        }
    )

def main():
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Home","Upload Gambar", "Real-time Detection"])

    with tab1:
        st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
        st.write("Selamat datang di Sistem Andromeda! Sistem ini membantu anda menganalisis komposisi makanan sesuai panduan gizi Indonesia '4 Sehat 5 Sempurna'.")
        st.write("Andromeda telah mengembangkan aplikasi ini sebagai hasil penugasan Final project pada track Artificial Intelligence, Startup Campus.")

        # Add your existing Home tab content here

    with tab2:
        st.header("Upload Gambar")
        uploaded_file = st.file_uploader("Pilih gambar makanan...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Add your existing image upload handling code here
            pass

    with tab3:
        st.header("Real-time Detection")
        webrtc_ctx = setup_webrtc()

if __name__ == "__main__":
    main()