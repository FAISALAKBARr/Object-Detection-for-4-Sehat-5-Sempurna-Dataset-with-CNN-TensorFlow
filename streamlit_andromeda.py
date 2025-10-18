import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import cv2
from PIL import Image
import gdown
import pandas as pd
import time
import gc
import threading

# Lazy import tensorflow
@st.cache_resource
def get_tensorflow():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    return tf

# Set page configuration
st.set_page_config(
    page_title="SmartPlate",
    page_icon="🍽️",
    layout="wide"
)

# Model configuration - Support both .keras and .h5
# RECOMMENDED: Use best_model (saved by ModelCheckpoint with highest val_acc)
MODEL_ID = '1DgvF7-UyRx_Htjo8urj9Qx-XhkQgLWwl'  # Update this with your Google Drive file ID

# Priority order for model files:
MODEL_PATH_BEST_KERAS = 'best_model.keras'      # ⭐ PRIORITY 1: Best validation accuracy
MODEL_PATH_BEST_H5 = 'best_model.h5'            # ⭐ PRIORITY 2: Best model (legacy format)
MODEL_PATH_FINAL_KERAS = 'FINAL_MODEL_IMPROVED.keras'   # PRIORITY 3: Last epoch (backup)
MODEL_PATH_FINAL_H5 = 'FINAL_MODEL_IMPROVED.h5'          # PRIORITY 4: Last epoch (legacy backup)

# Custom CSS
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

st.title("🍽️ Teknologi Cerdas untuk Konsumsi Bijak dan Berkelanjutan")
st.write("Sistem ini dapat mengklasifikasikan makanan dan minuman melalui upload gambar atau secara real-time")

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

@st.cache_resource
def load_model_safe():
    """
    Load model dengan support untuk format .keras (Keras 3.x) dan .h5 (legacy)
    """
    try:
        tf = get_tensorflow()
        
        # Check which format is available
        model_path = None
        
        # Priority 1: Check for .keras format
        if os.path.exists(MODEL_PATH_BEST_KERAS):
            model_path = MODEL_PATH_BEST_KERAS
            st.info(f"✅ Found model: {MODEL_PATH_BEST_KERAS}")
        # Priority 2: Check for .h5 format
        elif os.path.exists(MODEL_PATH_BEST_H5):
            model_path = MODEL_PATH_BEST_H5
            st.info(f"✅ Found model: {MODEL_PATH_BEST_H5}")
        else:
            # Download model from Google Drive
            with st.spinner('📥 Downloading model from Google Drive...'):
                url = f'https://drive.google.com/uc?id={MODEL_ID}'
                
                # Try downloading as .h5 first (since your current model is .h5)
                try:
                    gdown.download(url, MODEL_PATH_BEST_H5, quiet=False)
                    model_path = MODEL_PATH_BEST_H5
                    st.success(f"✅ Downloaded: {MODEL_PATH_BEST_H5}")
                except Exception as e:
                    st.error(f"❌ Failed to download model: {str(e)}")
                    return None
        
        if model_path is None:
            st.error("❌ No model file found!")
            return None
        
        # Load the model
        with st.spinner(f'🔄 Loading model from {model_path}...'):
            try:
                # Try loading with compile=False first (safer)
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Recompile the model
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                st.success(f"✅ Model loaded successfully from {model_path}")
                
            except Exception as load_error:
                st.error(f"❌ Error loading model: {str(load_error)}")
                
                # Try alternative loading method for .h5 files
                if model_path.endswith('.h5'):
                    try:
                        st.warning("⚠️ Trying alternative loading method for .h5 file...")
                        model = tf.keras.models.load_model(
                            model_path,
                            compile=False
                        )
                        model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        st.success("✅ Model loaded with alternative method")
                    except Exception as e2:
                        st.error(f"❌ Alternative method also failed: {str(e2)}")
                        return None
                else:
                    return None
        
        # Cleanup
        gc.collect()
        return model
        
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        1. Make sure you uploaded the correct model file format (.keras or .h5)
        2. Check if the file is corrupted
        3. Try re-training and re-uploading the model
        4. Verify the MODEL_ID in Google Drive is correct
        """)
        return None

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    """
    tf = get_tensorflow()
    
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
        image = Image.fromarray(image)
    else:
        original = np.array(image)
    
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    return img_array, original

def predict_image(image, model):
    """
    Make prediction with multiple object detection and draw bounding boxes
    """
    try:
        processed_image, original_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        
        height, width = original_image.shape[:2]
        output_image = original_image.copy()
        
        CONFIDENCE_THRESHOLD = 0.2
        detected_objects = []
        
        for class_idx, confidence in enumerate(predictions[0]):
            if confidence > CONFIDENCE_THRESHOLD:
                box_size = int(min(width, height) * (confidence * 0.5))
                center_x = width // 2
                center_y = height // 2
                
                x1 = max(0, center_x - box_size // 2)
                y1 = max(0, center_y - box_size // 2)
                x2 = min(width, center_x + box_size // 2)
                y2 = min(height, center_y + box_size // 2)
                
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
                
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_names[class_idx]}: {confidence * 100:.2f}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_y = max(y1, label_size[1] + 10)
                
                cv2.rectangle(output_image,
                            (x1, label_y - label_size[1] - 10),
                            (x1 + label_size[0], label_y),
                            color, -1)
                
                cv2.putText(output_image, label,
                           (x1, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detected_objects.append({
                    'class': class_names[class_idx],
                    'confidence': confidence * 100,
                    'bbox': (x1, y1, x2, y2)
                })
        
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        all_probabilities = {
            class_names[i]: float(predictions[0][i]) * 100
            for i in range(len(class_names))
        }
        
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

def main():
    try:
        model = load_model_safe()
        
        if model is None:
            st.error("Failed to load model. Please refresh the page.")
            return
        
        tab1, tab2, tab3 = st.tabs(["Home", "Upload Gambar", "Real-time Detection"])
        
        with tab1:
            st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
            st.write("Selamat datang di Sistem Andromeda! Sistem ini membantu anda menganalisis komposisi makanan sesuai panduan gizi Indonesia '4 Sehat 5 Sempurna'.")
            st.write("Andromeda telah mengembangkan aplikasi ini sebagai hasil penugasan Final project pada track Artificial Intelligence, Startup Campus.")
            
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
                            result = predict_image(image, model)
                            
                            if result:
                                with col2:
                                    st.write("### Hasil Analisis")
                                    st.image(result['output_image'],
                                           caption='Hasil Deteksi',
                                           use_container_width=True)
                                    
                                    st.write("#### Objek Terdeteksi:")
                                    for idx, obj in enumerate(result['detected_objects'], 1):
                                        with st.expander(f"Objek {idx}: {obj['class'].upper()} - {obj['confidence']:.2f}%"):
                                            st.markdown(class_descriptions[obj['class']])
                                
                                st.write("#### Distribusi Probabilitas:")
                                for class_name, prob in result['all_probabilities'].items():
                                    st.write(f"{class_name.title()}: {prob:.2f}%")
                                    st.progress(prob/100)
        
        with tab3:
            st.write("### 📸 Real-time Detection")
            st.info("💡 **Fitur ini menggunakan kamera perangkat Anda untuk deteksi real-time**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Camera input dari Streamlit
                camera_image = st.camera_input("Ambil foto untuk deteksi makanan")
                
                if camera_image is not None:
                    # Convert uploaded image to PIL
                    image = Image.open(camera_image)
                    
                    # Placeholder untuk hasil
                    result_placeholder = st.empty()
                    
                    with st.spinner('🔍 Menganalisis gambar...'):
                        result = predict_image(image, model)
                        
                        if result:
                            # Tampilkan hasil deteksi
                            result_placeholder.image(
                                result['output_image'],
                                caption='Hasil Deteksi Real-time',
                                use_container_width=True
                            )
                            
                            # Tampilkan informasi deteksi
                            if result['detected_objects']:
                                st.success(f"✅ Terdeteksi: **{result['class'].upper()}** ({result['confidence']:.2f}%)")
                                
                                with st.expander("📊 Detail Deteksi"):
                                    for idx, obj in enumerate(result['detected_objects'], 1):
                                        st.write(f"**{idx}. {obj['class'].upper()}** - {obj['confidence']:.2f}%")
                                        st.markdown(class_descriptions[obj['class']])
                            else:
                                st.warning("⚠️ Tidak ada objek terdeteksi dengan confidence tinggi")
                            
                            # Tampilkan probabilitas
                            with st.expander("📈 Distribusi Probabilitas"):
                                for class_name, prob in result['all_probabilities'].items():
                                    st.write(f"{class_name.title()}: {prob:.2f}%")
                                    st.progress(prob/100)
            
            with col2:
                st.markdown("""
                ### 📋 Panduan Penggunaan
                
                1. **Klik tombol kamera** di sebelah kiri
                2. **Izinkan akses kamera** jika diminta browser
                3. **Arahkan kamera** ke objek makanan/minuman
                4. **Klik "Take Photo"** untuk mengambil gambar
                5. **Tunggu hasil** deteksi muncul
                
                ### 🎯 Kategori Deteksi:
                - 🍎 Buah-buahan
                - 🍚 Karbohidrat
                - 🥤 Minuman
                - 🍖 Protein
                - 🥬 Sayuran
                
                ### 💡 Tips:
                - ✅ Pastikan pencahayaan cukup
                - ✅ Posisikan objek di tengah
                - ✅ Hindari blur/goyang
                - ✅ Jarak ideal: 20-50 cm
                - ✅ Satu objek per foto
                
                ### ⚙️ Troubleshooting:
                - Jika kamera tidak muncul, cek permission browser
                - Refresh halaman jika ada error
                - Gunakan browser Chrome/Firefox untuk hasil terbaik
                """)
                
                # Status info
                st.divider()
                st.markdown("### 📊 Status Sistem")
                status_data = {
                    "Model": "✅ Loaded",
                    "Camera": "🟢 Ready",
                    "Detection": "🟢 Active"
                }
                for key, value in status_data.items():
                    st.write(f"**{key}:** {value}")
        
        # Sidebar
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
        3. Lihat hasil klasifikasi dan probabilitas
        
        **Teknologi:**
        - TensorFlow/Keras
        - OpenCV
        - Streamlit
        """)
        
        st.sidebar.divider()
        st.sidebar.markdown("### 📈 Model Info")
        
        # Detect model format being used
        if os.path.exists(MODEL_PATH_BEST_KERAS):
            model_format = "Keras 3.x (.keras)"
        elif os.path.exists(MODEL_PATH_BEST_H5):
            model_format = "Legacy (.h5)"
        else:
            model_format = "Unknown"
            
        st.sidebar.info(f"""
        **Arsitektur:** CNN\n
        **Input Size:** 224x224\n
        **Classes:** 5\n
        **Framework:** TensorFlow\n
        **Format:** {model_format}
        """)
        
        # Footer
        st.divider()
        st.write("<p style='text-align: center;'>© 2024 Andromeda. All rights reserved.</p>", unsafe_allow_html=True)
        
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
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page or contact support.")
    finally:
        gc.collect()

if __name__ == '__main__':
    main()