import os
import tensorflow as tf
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from PIL import Image
import gdown
from tqdm import tqdm
import io
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Set page configuration
st.set_page_config(
    page_title="Sistem Deteksi Makanan dan Minuman",
    page_icon="🍽️",
    layout="wide"
)

MODEL_ID = '1catZKB9HcjHt4FC40bRyeVvE-P24Ev_Q' 
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
    Make prediction with improved multiple object detection and draw bounding boxes
    """
    try:
        processed_image, original_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        
        height, width = original_image.shape[:2]
        output_image = original_image.copy()
        
        # Meningkatkan threshold untuk mengurangi false positives
        CONFIDENCE_THRESHOLD = 0.45
        
        # Sliding window parameters yang lebih optimal
        WINDOW_SIZES = [(320, 320), (224, 224)]  # Ukuran window yang lebih besar
        STRIDE_RATIO = 0.5  # Stride sebagai rasio dari ukuran window
        
        detected_objects = []
        
        # Improved sliding window detection
        for window_size in WINDOW_SIZES:
            stride = int(min(window_size) * STRIDE_RATIO)
            
            for y in range(0, height - window_size[1] + stride, stride):
                for x in range(0, width - window_size[0] + stride, stride):
                    # Pastikan window tidak melebihi batas gambar
                    end_y = min(y + window_size[1], height)
                    end_x = min(x + window_size[0], width)
                    
                    # Extract and adjust window
                    window = original_image[y:end_y, x:end_x]
                    if window.shape[0] < 10 or window.shape[1] < 10:  # Skip window yang terlalu kecil
                        continue
                        
                    # Process window
                    processed_window, _ = preprocess_image(window)
                    window_predictions = model.predict(processed_window, verbose=0)[0]
                    
                    max_confidence = np.max(window_predictions)
                    if max_confidence > CONFIDENCE_THRESHOLD:
                        class_idx = np.argmax(window_predictions)
                        
                        # Calculate relative area of the detection
                        area_ratio = (end_x - x) * (end_y - y) / (width * height)
                        
                        # Filter out detections that are too small or too large
                        if 0.1 <= area_ratio <= 0.8:
                            detected_objects.append({
                                'class': class_names[class_idx],
                                'confidence': float(max_confidence * 100),
                                'bbox': (x, y, end_x, end_y),
                                'area': area_ratio
                            })
        
        # Improved non-max suppression
        filtered_objects = non_max_suppression(detected_objects, iou_threshold=0.4)
        
        # Post-processing: merge nearby detections of the same class
        merged_objects = merge_nearby_detections(filtered_objects)
        
        # Draw final detections
        for obj in merged_objects:
            x1, y1, x2, y2 = obj['bbox']
            
            # Assign consistent colors for each class
            color = get_class_color(obj['class'])
            
            # Draw bounding box with thicker lines
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
            
            # Improve label visualization
            label = f"{obj['class']}: {obj['confidence']:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Ensure label is visible
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            
            # Draw label with better visibility
            cv2.rectangle(output_image, 
                        (x1, label_y - label_size[1] - 10),
                        (x1 + label_size[0], label_y),
                        color, -1)
            cv2.putText(output_image, label,
                       (x1, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate overall probabilities based on merged detections
        class_counts = {}
        for obj in merged_objects:
            class_name = obj['class']
            if class_name not in class_counts:
                class_counts[class_name] = 1
            else:
                class_counts[class_name] += 1
        
        all_probabilities = {
            class_name: (count / len(merged_objects) * 100 if merged_objects else 0)
            for class_name, count in class_counts.items()
        }
        
        # Fill in missing classes with zero probability
        for class_name in class_names:
            if class_name not in all_probabilities:
                all_probabilities[class_name] = 0.0
        
        return {
            'class': merged_objects[0]['class'] if merged_objects else None,
            'confidence': merged_objects[0]['confidence'] if merged_objects else 0,
            'all_probabilities': all_probabilities,
            'output_image': output_image,
            'detected_objects': merged_objects
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def merge_nearby_detections(detections, distance_threshold=50):
    """
    Merge nearby detections of the same class
    """
    if not detections:
        return []
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
            
        current_group = [det1]
        used.add(i)
        
        for j, det2 in enumerate(detections):
            if j in used or i == j:
                continue
                
            if det1['class'] == det2['class']:
                x1, y1, x2, y2 = det1['bbox']
                x3, y3, x4, y4 = det2['bbox']
                
                # Calculate center points
                center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
                
                # Calculate distance between centers
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < distance_threshold:
                    current_group.append(det2)
                    used.add(j)
        
        # Merge the group
        if current_group:
            merged_box = merge_boxes(current_group)
            merged.append(merged_box)
    
    return merged

def merge_boxes(boxes):
    """
    Merge a group of boxes into one
    """
    if not boxes:
        return None
    
    # Calculate average coordinates
    x1 = min(box['bbox'][0] for box in boxes)
    y1 = min(box['bbox'][1] for box in boxes)
    x2 = max(box['bbox'][2] for box in boxes)
    y2 = max(box['bbox'][3] for box in boxes)
    
    # Calculate average confidence
    avg_confidence = sum(box['confidence'] for box in boxes) / len(boxes)
    
    return {
        'class': boxes[0]['class'],
        'confidence': avg_confidence,
        'bbox': (x1, y1, x2, y2)
    }

def get_class_color(class_name):
    """
    Return consistent color for each class
    """
    color_map = {
        'buah': (0, 255, 0),      # Green
        'karbohidrat': (255, 0, 0),  # Red
        'minuman': (0, 0, 255),    # Blue
        'protein': (255, 165, 0),  # Orange
        'sayur': (128, 0, 128)    # Purple
    }
    return color_map.get(class_name, (200, 200, 200))

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.last_prediction_time = time.time()
        self.prediction_interval = 1.0  # Predict every 1 second
        self.current_prediction = None
        self.detection_history = []  # Untuk tracking deteksi sebelumnya

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        current_time = time.time()
        if current_time - self.last_prediction_time >= self.prediction_interval:
            result = predict_image(img)
            self.current_prediction = result
            self.last_prediction_time = current_time
            
            if result and 'detected_objects' in result:
                self.detection_history = result['detected_objects'][-5:]  # Simpan 5 deteksi terakhir

        if self.current_prediction:
            img = self.current_prediction['output_image']
            
            # Tambahkan overlay informasi
            height, width = img.shape[:2]
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Tampilkan informasi deteksi
            y_offset = 30
            for det in self.detection_history:
                text = f"{det['class']}: {det['confidence']:.1f}%"
                cv2.putText(img, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
        return img

def main():
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Home","Upload Gambar", "Real-time Detection"])

    with tab1:
        st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
        st.write("Selamat datang di Sistem Andromeda! Sistem ini membantu anda menganalisis komposisi makanan sesuai panduan gizi Indonesia '4 Sehat 5 Sempurna'.")
        st.write("Andromeda telah mengembangkan aplikasi ini sebagai hasil penugasan Final project pada track Artificial Intelligence, Startup Campus.")

        # Informasi tentang aplikasi
        st.markdown(""" 
        ### Tentang Aplikasi
        Aplikasi ini bertujuan untuk mengembangkan **Automated Nutritional Analysis**, yaitu deteksi objek untuk evaluasi makanan sehat berdasarkan prinsip *4 Sehat 5 Sempurna*. 
        Dengan memanfaatkan teknologi berbasis **Convolutional Neural Networks (CNN)**, aplikasi ini mampu:
        - Menganalisis komposisi makanan.
        - Mengevaluasi keseimbangan gizi secara otomatis.
        - Memberikan visualisasi interaktif melalui anotasi objek pada gambar makanan.      

        ### Fitur Utama
        - Deteksi dan klasifikasi makanan menggunakan **CNN**.
        - Evaluasi otomatis keseimbangan nutrisi.
        - Tampilan interaktif dengan anotasi visual.
        - Mendukung edukasi masyarakat tentang gizi berbasis teknologi.

        ### Teknologi yang Digunakan
        1. **TensorFlow/Keras** untuk model CNN.
        2. **OpenCV** untuk pemrosesan gambar.
        3. Dataset untuk makanan dan minuman yang dihubungkan dengan Drive.
                    
        ### Prinsip 4 Sehat 5 Sempurna
        - 🍚 **Carbohydrates (Karbohidrat)**
        - 🥩 **Proteins (Protein)**
        - 🥕 **Vegetables (Sayur)**
        - 🍎 **Fruits (Buah)**
        - 🥛 **Beverages (Minuman)**
        """)

        # Tombol untuk akses dataset di Kaggle
        if st.button("Kepo sama dataset lengkap nya??"):
            kaggle_url = "https://www.kaggle.com/datasets/andromedagroup05/data-4-sehat-5-sempurna/data"
            st.warning("Kamu akan diarahkan ke halaman dataset di Kaggle.")
            st.markdown(
                f'<a href="{kaggle_url}" target="_blank" style="text-decoration:none;">'
                '<button style="background-color:#51baff; color:white; padding:10px 20px; border:none; cursor:pointer;">'
                '**Klik di sini yah!**</button></a>',
                unsafe_allow_html=True
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Yuk Analisis Hidanganmu!")
            uploaded_file = st.file_uploader("Pilih file gambar...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Gambar yang diunggah', use_column_width=True)
                
                if st.button('GO!'):
                    with st.spinner('Sedang menganalisis gambar...'):
                        result = predict_image(image)
                        
                        if result:
                            with col2:
                                st.write("### Hasil Analisis")
                                st.image(result['output_image'], 
                                       caption='Hasil Deteksi', 
                                       use_column_width=True)
                                
                                # Tampilkan informasi objek yang terdeteksi
                                st.write("#### Objek Terdeteksi:")
                                for idx, obj in enumerate(result['detected_objects'], 1):
                                    with st.expander(f"Objek {idx}: {obj['class'].upper()} - {obj['confidence']:.2f}%"):
                                        st.markdown(class_descriptions[obj['class']])
                                
                                st.write("#### Distribusi Probabilitas:")
                                for class_name, prob in result['all_probabilities'].items():
                                    st.write(f"{class_name.title()}: {prob:.2f}%")
                                    st.progress(prob/100)
                                    
    with tab3:
        st.write("### Real-time Detection")
        st.write("Gunakan kamera untuk deteksi makanan dan minuman secara real-time")
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="food-detection-streamRealtime",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_transformer_factory=VideoTransformer,
            async_transform=True
        )

    # Sidebar information
    st.sidebar.title("ℹ️ Informasi Sistem")
    st.sidebar.write("""
    Sistem ini menggunakan model Deep Learning (CNN) untuk mengklasifikasikan
    makanan dan minuman ke dalam 5 kategori utama.
    
    **Kategori yang dapat dideteksi:**
    - 🍎 Buah-buahan
    - 🍚 Karbohidrat
    - 🥤 Minuman
    - 🍖 Protein
    - 🥬 Sayuran
    
    **Cara Penggunaan:**
    1. Upload gambar atau gunakan kamera
    2. Sistem akan otomatis mendeteksi kategori
    3. Lihat hasil klasifikasi
    """)

    # Footer
    st.write("<p style='text-align: center;'>© 2023 Andromeda. All rights reserved.</p>", unsafe_allow_html=True)

    # Add a link to the GitHub repository
    st.markdown(
        """
        <p style="text-align: center;">
            <a href="https://github.com/FAISALAKBARr/Object-Detection-for-4-Sehat-5-Sempurna-Dataset-with-CNN-TensorFlow.git" target="_blank" rel="noopener noreferrer">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
            </a>
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()