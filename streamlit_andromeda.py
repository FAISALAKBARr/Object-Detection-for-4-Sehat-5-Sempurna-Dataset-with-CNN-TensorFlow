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
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="SmartPlate",
    page_icon="logo/logo_app(2).png",
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
    Make prediction with multiple object detection and draw bounding boxes
    """
    try:
        processed_image, original_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        
        # Get image dimensions
        height, width = original_image.shape[:2]
        output_image = original_image.copy()
        
        # Threshold untuk confidence score
        CONFIDENCE_THRESHOLD = 0.2
        
        detected_objects = []
        
        # Analyze all predictions above threshold
        for class_idx, confidence in enumerate(predictions[0]):
            if confidence > CONFIDENCE_THRESHOLD:
                # Calculate dynamic bounding box size based on confidence
                box_size = int(min(width, height) * (confidence * 0.5))
                
                # Calculate center point
                center_x = width // 2
                center_y = height // 2
                
                # Calculate box coordinates
                x1 = max(0, center_x - box_size // 2)
                y1 = max(0, center_y - box_size // 2)
                x2 = min(width, center_x + box_size // 2)
                y2 = min(height, center_y + box_size // 2)
                
                # Generate random color for this class
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
                
                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label with class name and confidence
                label = f"{class_names[class_idx]}: {confidence * 100:.2f}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Ensure label background doesn't go outside image
                label_y = max(y1, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(output_image, 
                            (x1, label_y - label_size[1] - 10),
                            (x1 + label_size[0], label_y),
                            color, -1)
                
                # Draw label text
                cv2.putText(output_image, label,
                           (x1, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detected_objects.append({
                    'class': class_names[class_idx],
                    'confidence': confidence * 100,
                    'bbox': (x1, y1, x2, y2)
                })
        
        # Sort detected objects by confidence
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate overall probabilities
        all_probabilities = {
            class_names[i]: float(predictions[0][i]) * 100 
            for i in range(len(class_names))
        }
        
        # Get primary class (highest confidence)
        primary_class = detected_objects[0]['class'] if detected_objects else class_names[np.argmax(predictions[0])]
        primary_confidence = detected_objects[0]['confidence'] if detected_objects else float(np.max(predictions[0])) * 100
        
        return {
            'class': primary_class,
            'confidence': primary_confidence,
            'all_probabilities': all_probabilities,
            'output_image': output_image,
            'detected_objects': detected_objects
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Modify VideoTransformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5  # Reduce prediction frequency
        self.current_prediction = None
        self.detection_history = []
        self.frame_count = 0
        self.skip_frames = 2  # Process every nth frame

    def transform(self, frame):
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Skip frames to reduce processing load
            if self.frame_count % self.skip_frames != 0:
                return img
            
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                # Resize image to reduce processing load
                small_img = cv2.resize(img, (224, 224))
                result = predict_image(small_img)
                
                if result:
                    self.current_prediction = result
                    self.last_prediction_time = current_time
                    
                    if 'detected_objects' in result:
                        self.detection_history = result['detected_objects'][-3:]  # Keep only last 3 detections

            # Draw detections if available
            if self.current_prediction:
                # Draw bounding boxes and labels
                for det in self.detection_history:
                    label = f"{det['class']}: {det['confidence']:.1f}%"
                    cv2.putText(img, label, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                st.image(image, caption='Gambar yang diunggah', use_container_width=True)
                
                if st.button('GO!'):
                    with st.spinner('Sedang menganalisis gambar...'):
                        result = predict_image(image)
                        
                        if result:
                            with col2:
                                st.write("### Hasil Analisis")
                                st.image(result['output_image'], 
                                       caption='Hasil Deteksi', 
                                       use_container_width=True)
                                
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
        
        # Create columns for stream and info
        stream_col, info_col = st.columns([2, 1])
        
        with stream_col:
            try:
                # Setup and start WebRTC stream
                webrtc_ctx = setup_webrtc()
                
                if webrtc_ctx.state.playing:
                    st.success("✅ Stream aktif! Arahkan kamera ke makanan/minuman.")
                    
                    # Add stream statistics
                    stats_placeholder = st.empty()
                    while webrtc_ctx.state.playing:
                        stats = {
                            "Status": "Active",
                            "Resolution": "640x480",
                            "Frame Rate": "~15 fps"
                        }
                        stats_df = pd.DataFrame([stats])
                        stats_placeholder.table(stats_df)
                        time.sleep(1)
                else:
                    st.warning("⚠️ Stream tidak aktif. Klik 'START' untuk memulai.")
                    
            except Exception as e:
                st.error(f"❌ Error saat menginisialisasi WebRTC: {str(e)}")
                st.info("Tips troubleshooting:")
                st.markdown("""
                    - Pastikan browser mengizinkan akses kamera
                    - Coba refresh halaman
                    - Pastikan koneksi internet stabil
                    - Coba gunakan browser berbeda (Chrome/Firefox)
                """)
        
        with info_col:
            st.markdown("""
            ### Panduan Penggunaan
            1. Klik tombol 'START' untuk memulai stream
            2. Izinkan akses kamera jika diminta
            3. Arahkan kamera ke objek makanan/minuman
            4. Sistem akan mendeteksi dan mengklasifikasikan secara otomatis
            
            ### Kategori yang Dapat Dideteksi:
            - 🍎 Buah-buahan
            - 🍚 Karbohidrat
            - 🥤 Minuman
            - 🍖 Protein
            - 🥬 Sayuran
            
            ### Tips Penggunaan:
            - Pastikan pencahayaan cukup
            - Jaga kamera tetap stabil
            - Posisikan objek di tengah frame
            - Hindari gerakan terlalu cepat
            """)
            
            # Add debug information in expander
            with st.expander("Debug Information"):
                if webrtc_ctx is not None:
                    st.write("WebRTC State:", webrtc_ctx.state)
                    st.write("Video Transform:", "Active" if webrtc_ctx.video_transformer else "Inactive")
                    
                    if hasattr(webrtc_ctx.video_transformer, 'frame_count'):
                        st.write("Processed Frames:", webrtc_ctx.video_transformer.frame_count)
                    
                    # Add session state information
                    if 'last_error' in st.session_state:
                        st.error(f"Last Error: {st.session_state.last_error}")
            
            # Add performance metrics
            with st.expander("Performance Metrics"):
                if webrtc_ctx and webrtc_ctx.state.playing:
                    metrics = {
                        "CPU Usage": "Monitoring...",
                        "Memory Usage": "Monitoring...",
                        "FPS": "Calculating...",
                        "Latency": "Measuring..."
                    }
                    st.table(pd.DataFrame([metrics]))
            
            # Add stop button
            if st.button("STOP STREAM", key="stop_stream"):
                if webrtc_ctx is not None:
                    webrtc_ctx.video_transformer = None
                    st.experimental_rerun()
        
        # Add status indicator
        status_placeholder = st.empty()
        if webrtc_ctx and webrtc_ctx.state.playing:
            status_placeholder.success("🎥 Stream berjalan normal")
        else:
            status_placeholder.warning("📵 Stream tidak aktif")
        
        # Add error logging
        if 'errors' not in st.session_state:
            st.session_state.errors = []
        
        # Show recent errors if any
        if st.session_state.errors:
            with st.expander("Error Log"):
                for error in st.session_state.errors[-5:]:  # Show last 5 errors
                    st.error(error)

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
    st.write("<p style='text-align: center;'>© 2024 Andromeda. All rights reserved.</p>", unsafe_allow_html=True)

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