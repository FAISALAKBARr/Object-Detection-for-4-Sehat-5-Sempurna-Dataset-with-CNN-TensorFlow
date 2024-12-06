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
    Make prediction with multiple object detection using sliding window approach
    """
    try:
        processed_image, original_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        
        height, width = original_image.shape[:2]
        output_image = original_image.copy()
        
        CONFIDENCE_THRESHOLD = 0.3
        detected_objects = []

        # Sliding window parameters
        window_sizes = [(height//2, width//2)]
        stride = height//4  # Tingkatkan stride untuk mengurangi overlap

        for win_h, win_w in window_sizes:
            for y in range(0, height - win_h, stride):
                for x in range(0, width - win_w, stride):
                    # Extract window
                    window = original_image[y:y+win_h, x:x+win_w]
                    
                    # Skip if window is too small
                    if window.shape[0] < 64 or window.shape[1] < 64: # Tingkatkan minimum size
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
                        

        # Apply Non-Maximum Suppression to remove overlapping boxes
        final_objects = []
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        while detected_objects:
            current = detected_objects.pop(0)
            final_objects.append(current)
            
            detected_objects = [
                obj for obj in detected_objects
                if calculate_iou(current['bbox'], obj['bbox']) < 0.3
            ]
        MAX_DETECTIONS = 3
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

def get_color_for_class(class_name):
    """
    Return consistent color for each class
    """
    color_map = {
        'buah': (0, 255, 0),     # Hijau
        'karbohidrat': (255, 0, 0),  # Biru
        'minuman': (0, 0, 255),   # Merah
        'protein': (255, 255, 0), # Cyan
        'sayur': (0, 255, 255)    # Kuning
    }
    return color_map.get(class_name, (128, 128, 128))

def calculate_iou(box1, box2):
    """
    Calculate intersection over union between two boxes
    box format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Modify VideoTransformer for better real-time performance
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.2  # Increase prediction frequency
        self.current_prediction = None
        self.frame_buffer = []
        self.max_buffer_size = 3

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()
            
            # Process frame if enough time has passed
            if current_time - self.last_prediction_time >= self.prediction_interval:
                result = predict_image(img)
                
                if result and 'output_image' in result:
                    self.current_prediction = result
                    self.last_prediction_time = current_time
                    
                    # Update frame buffer
                    self.frame_buffer.append(result['output_image'])
                    if len(self.frame_buffer) > self.max_buffer_size:
                        self.frame_buffer.pop(0)
                    
                    return result['output_image']
            
            # Return last processed frame if available
            if self.frame_buffer:
                return self.frame_buffer[-1]
            
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