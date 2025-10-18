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

# ✅ FIXED: Model configuration - sesuaikan dengan IMG_SIZE di training (384)
MODEL_ID = '1nFMeE-QzqvvOUWJzSSpMCzb83sZlSWnQ'
IMG_SIZE = 384  # ✅ CRITICAL: Harus sama dengan IMG_SIZE saat training!

MODEL_PATH_BEST_KERAS = 'FINAL_MODEL_NUTRITION.keras'
MODEL_PATH_BEST_H5 = 'best_model.h5'
MODEL_PATH_FINAL_KERAS = 'FINAL_MODEL_NUTRITION.keras'
MODEL_PATH_FINAL_H5 = 'FINAL_MODEL_NUTRITION.h5'

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
        
        model_path = None
        
        if os.path.exists(MODEL_PATH_BEST_KERAS):
            model_path = MODEL_PATH_BEST_KERAS
            st.info(f"✅ Found model: {MODEL_PATH_BEST_KERAS}")
        else:
            with st.spinner('📥 Downloading model from Google Drive...'):
                url = f'https://drive.google.com/uc?id={MODEL_ID}'
                
                try:
                    gdown.download(url, MODEL_PATH_BEST_KERAS, quiet=False)
                    model_path = MODEL_PATH_BEST_KERAS
                    st.success(f"✅ Downloaded: {MODEL_PATH_BEST_KERAS}")
                except Exception as e:
                    st.error(f"❌ Failed to download model: {str(e)}")
                    return None
        
        if model_path is None:
            st.error("❌ No model file found!")
            return None
        
        with st.spinner(f'🔄 Loading model from {model_path}...'):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # ✅ VALIDATION: Cek input shape model
                input_shape = model.input_shape
                st.success(f"✅ Model loaded successfully!")
                st.info(f"📐 Model Input Shape: {input_shape}")
                
            except Exception as load_error:
                st.error(f"❌ Error loading model: {str(load_error)}")
                return None
        
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

def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):  # ✅ FIXED: Gunakan IMG_SIZE dari config
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
                    st.image(image, caption='Gambar yang diunggah', width='stretch')  # ✅ FIXED
                    
                    if st.button('GO!'):
                        with st.spinner('Sedang menganalisis gambar...'):
                            result = predict_image(image, model)
                            
                            if result:
                                with col2:
                                    st.write("### Hasil Analisis")
                                    st.image(result['output_image'],
                                           caption='Hasil Deteksi',
                                           width='stretch')  # ✅ FIXED
                                    
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
                camera_image = st.camera_input("Ambil foto untuk deteksi makanan")
                
                if camera_image is not None:
                    image = Image.open(camera_image)
                    result_placeholder = st.empty()
                    
                    with st.spinner('🔍 Menganalisis gambar...'):
                        result = predict_image(image, model)
                        
                        if result:
                            result_placeholder.image(
                                result['output_image'],
                                caption='Hasil Deteksi Real-time',
                                width='stretch'  # ✅ FIXED
                            )
                            
                            if result['detected_objects']:
                                st.success(f"✅ Terdeteksi: **{result['class'].upper()}** ({result['confidence']:.2f}%)")
                                
                                with st.expander("📊 Detail Deteksi"):
                                    for idx, obj in enumerate(result['detected_objects'], 1):
                                        st.write(f"**{idx}. {obj['class'].upper()}** - {obj['confidence']:.2f}%")
                                        st.markdown(class_descriptions[obj['class']])
                            else:
                                st.warning("⚠️ Tidak ada objek terdeteksi dengan confidence tinggi")
                            
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
                
                st.divider()
                st.markdown("### 📊 Status Sistem")
                status_data = {
                    "Model": "✅ Loaded",
                    "Input Size": f"📐 {IMG_SIZE}x{IMG_SIZE}",
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
        
        if os.path.exists(MODEL_PATH_BEST_KERAS):
            model_format = "Keras 3.x (.keras)"
        elif os.path.exists(MODEL_PATH_BEST_H5):
            model_format = "Legacy (.h5)"
        else:
            model_format = "Unknown"
            
        st.sidebar.info(f"""
        **Arsitektur:** Custom SPP CNN\n
        **Input Size:** {IMG_SIZE}x{IMG_SIZE}\n
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

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import gdown
# import json
# import gc

# # Lazy import tensorflow
# @st.cache_resource
# def get_tensorflow():
#     import tensorflow as tf
#     tf.config.set_visible_devices([], 'GPU')
#     return tf

# st.set_page_config(
#     page_title="SmartPlate - Nutrition Balance Detector",
#     page_icon="🍽️",
#     layout="wide"
# )

# # Model configuration
# MODEL_ID = '1DgvF7-UyRx_Htjo8urj9Qx-XhkQgLWwl'  # ⭐ Update this after training!
# MODEL_PATH = 'best_model.keras'
# CONFIG_PATH = 'model_config.json'
# IMG_SIZE = 300  # Must match training

# # CSS
# st.markdown("""
#     <style>
#     .main {padding: 20px;}
#     .stButton>button {
#         width: 100%;
#         background-color: #0245d6;
#         color: white;
#         border-radius: 10px;
#         padding: 10px;
#         font-weight: bold;
#     }
#     .balance-card {
#         padding: 20px;
#         border-radius: 10px;
#         border: 2px solid #4CAF50;
#         margin: 10px 0;
#         background-color: #f0f9f0;
#     }
#     .unbalanced-card {
#         padding: 20px;
#         border-radius: 10px;
#         border: 2px solid #ff5252;
#         margin: 10px 0;
#         background-color: #fff0f0;
#     }
#     .food-badge {
#         display: inline-block;
#         padding: 5px 15px;
#         border-radius: 20px;
#         margin: 5px;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Load configuration
# @st.cache_data
# def load_config():
#     default_config = {
#         'class_names': ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur'],
#         'num_classes': 5,
#         'img_size': 300,
#         'threshold': 0.5,
#         'nutrition_balance_formula': {
#             'balanced_threshold': 5,
#             'classes': {
#                 'buah': 'Vitamin & Fiber',
#                 'karbohidrat': 'Energy Source',
#                 'minuman': 'Hydration',
#                 'protein': 'Body Building',
#                 'sayur': 'Minerals & Fiber'
#             }
#         }
#     }
    
#     if os.path.exists(CONFIG_PATH):
#         try:
#             with open(CONFIG_PATH, 'r') as f:
#                 return json.load(f)
#         except:
#             return default_config
#     return default_config

# config = load_config()
# CLASS_NAMES = config['class_names']
# THRESHOLD = config['threshold']

# # Class icons
# CLASS_ICONS = {
#     'buah': '🍎',
#     'karbohidrat': '🍚',
#     'minuman': '🥤',
#     'protein': '🍖',
#     'sayur': '🥬'
# }

# # Load model
# @st.cache_resource
# def load_model_safe():
#     try:
#         tf = get_tensorflow()
        
#         if not os.path.exists(MODEL_PATH):
#             with st.spinner('📥 Downloading model from Google Drive...'):
#                 url = f'https://drive.google.com/uc?id={MODEL_ID}'
#                 gdown.download(url, MODEL_PATH, quiet=False)
#                 st.success("✅ Model downloaded!")
        
#         with st.spinner('🔄 Loading model...'):
#             model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#             model.compile(
#                 optimizer='adam',
#                 loss='binary_crossentropy',
#                 metrics=['binary_accuracy']
#             )
        
#         st.success("✅ Model loaded successfully!")
#         gc.collect()
#         return model
        
#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         st.info("""
#         **Troubleshooting:**
#         1. Pastikan MODEL_ID sudah diupdate
#         2. Model file harus format .keras (Keras 3.x)
#         3. Check Google Drive sharing permissions
#         """)
#         return None

# # Grad-CAM function
# def make_gradcam_heatmap(img_array, model, class_index):
#     """Generate Grad-CAM heatmap"""
#     try:
#         tf = get_tensorflow()
        
#         # Find last conv layer
#         last_conv_layer_name = None
#         for layer in reversed(model.layers):
#             if len(layer.output_shape) == 4:  # Conv layer
#                 last_conv_layer_name = layer.name
#                 break
        
#         if last_conv_layer_name is None:
#             return None
        
#         # Create grad model
#         grad_model = tf.keras.models.Model(
#             [model.inputs],
#             [model.get_layer(last_conv_layer_name).output, model.output]
#         )
        
#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(img_array)
#             class_channel = predictions[:, class_index]
        
#         grads = tape.gradient(class_channel, conv_outputs)
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
#         conv_outputs = conv_outputs[0]
#         heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#         heatmap = tf.squeeze(heatmap)
#         heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
#         return heatmap.numpy()
#     except Exception as e:
#         st.warning(f"Grad-CAM generation failed: {str(e)}")
#         return None

# # Preprocess image
# def preprocess_image(image):
#     if isinstance(image, np.ndarray):
#         if len(image.shape) == 3 and image.shape[2] == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         original = image.copy()
#         image = Image.fromarray(image)
#     else:
#         original = np.array(image)
    
#     image = image.resize((IMG_SIZE, IMG_SIZE))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, 0)
    
#     return img_array, original

# # Predict multi-label
# def predict_multilabel(image, model):
#     try:
#         processed_image, original_image = preprocess_image(image)
        
#         # Multi-label predictions (sigmoid output)
#         predictions = model.predict(processed_image, verbose=0)[0]
        
#         # Detect classes above threshold
#         detected_classes = []
#         for i, prob in enumerate(predictions):
#             if prob >= THRESHOLD:
#                 detected_classes.append({
#                     'class': CLASS_NAMES[i],
#                     'confidence': float(prob) * 100,
#                     'index': i,
#                     'icon': CLASS_ICONS[CLASS_NAMES[i]]
#                 })
        
#         # Sort by confidence
#         detected_classes.sort(key=lambda x: x['confidence'], reverse=True)
        
#         # Calculate nutrition balance
#         num_detected = len(detected_classes)
#         is_balanced = num_detected >= 4  # At least 4 out of 5
#         balance_percentage = (num_detected / len(CLASS_NAMES)) * 100
        
#         # Generate Grad-CAM for each detected class
#         gradcam_visualizations = []
#         for det in detected_classes:
#             heatmap = make_gradcam_heatmap(processed_image, model, det['index'])
#             if heatmap is not None:
#                 # Resize heatmap to original image size
#                 h, w = original_image.shape[:2]
#                 heatmap_resized = cv2.resize(heatmap, (w, h))
                
#                 # Create heatmap overlay
#                 heatmap_colored = cv2.applyColorMap(
#                     np.uint8(255 * heatmap_resized), 
#                     cv2.COLORMAP_JET
#                 )
                
#                 # Superimpose
#                 superimposed = cv2.addWeighted(
#                     original_image, 0.6,
#                     heatmap_colored, 0.4,
#                     0
#                 )
                
#                 gradcam_visualizations.append({
#                     'class': det['class'],
#                     'image': superimposed,
#                     'confidence': det['confidence'],
#                     'icon': det['icon']
#                 })
        
#         return {
#             'predictions': predictions,
#             'detected_classes': detected_classes,
#             'is_balanced': is_balanced,
#             'balance_percentage': balance_percentage,
#             'num_detected': num_detected,
#             'gradcam_viz': gradcam_visualizations,
#             'original_image': original_image
#         }
        
#     except Exception as e:
#         st.error(f"❌ Error during prediction: {str(e)}")
#         return None

# # Main app
# def main():
#     st.title("🍽️ SmartPlate - Nutrition Balance Detector")
#     st.write("**Multi-food Detection + 4 Sehat 5 Sempurna Analysis**")
    
#     # Load model
#     model = load_model_safe()
#     if model is None:
#         st.error("❌ Failed to load model. Please check configuration.")
#         st.stop()
    
#     # Tabs
#     tab1, tab2, tab3 = st.tabs(["🏠 Home", "📤 Upload Image", "📸 Camera"])
    
#     with tab1:
#         st.header("About SmartPlate")
        
#         col_info1, col_info2 = st.columns(2)
        
#         with col_info1:
#             st.write("""
#             **SmartPlate** adalah sistem cerdas untuk mendeteksi keseimbangan gizi pada makanan 
#             berdasarkan prinsip **4 Sehat 5 Sempurna** Indonesia.
            
#             ### 🎯 Fitur Utama:
#             - **Multi-food Detection** - Deteksi semua makanan dalam 1 gambar
#             - **Grad-CAM Visualization** - Tampilkan lokasi setiap makanan
#             - **Balance Analysis** - Analisis keseimbangan gizi otomatis
#             - **Real-time Camera** - Deteksi langsung dari kamera
            
#             ### 🔬 Teknologi:
#             - Deep Learning (EfficientNet)
#             - Multi-label Classification
#             - Grad-CAM untuk visualization
#             """)
        
#         with col_info2:
#             st.write("""
#             ### 📊 5 Kategori Gizi:
#             """)
            
#             for class_name, icon in CLASS_ICONS.items():
#                 nutrition = config['nutrition_balance_formula']['classes'][class_name]
#                 st.markdown(f"""
#                 <div style="padding: 10px; background-color: #f0f0f0; border-radius: 8px; margin: 5px 0;">
#                     {icon} <strong>{class_name.title()}</strong><br/>
#                     <small>{nutrition}</small>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.write("""
#             ### ⚖️ Kriteria Seimbang:
#             - ✅ **Sangat Seimbang**: 5/5 kategori terdeteksi
#             - ✅ **Seimbang**: 4/5 kategori terdeteksi
#             - ⚠️ **Kurang Seimbang**: ≤3 kategori terdeteksi
#             """)
        
#         st.divider()
        
#         if st.button("📖 Lihat Dataset Lengkap"):
#             st.info("📊 Dataset: 5000 images, 5 classes (balanced)")
#             st.markdown("""
#             - Train: 3500 images (700/class)
#             - Validation: 750 images (150/class)
#             - Test: 750 images (150/class)
            
#             [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/andromedagroup05/data-4-sehat-5-sempurna/data)
#             """)
    
#     with tab2:
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.write("### 📤 Upload Gambar Makanan")
#             uploaded_file = st.file_uploader(
#                 "Pilih gambar (JPG, JPEG, PNG)...",
#                 type=['jpg', 'jpeg', 'png']
#             )
            
#             if uploaded_file:
#                 image = Image.open(uploaded_file)
#                 st.image(image, caption='📷 Gambar Original', use_container_width=True)
                
#                 if st.button('🔍 Analisis Keseimbangan Gizi', key='upload_analyze'):
#                     with st.spinner('🤖 AI sedang menganalisis...'):
#                         result = predict_multilabel(image, model)
                        
#                         if result:
#                             # Store result in session state
#                             st.session_state['result'] = result
        
#         # Display results if available
#         if 'result' in st.session_state:
#             result = st.session_state['result']
            
#             with col2:
#                 st.write("### 📊 Hasil Analisis")
                
#                 # Balance status card
#                 if result['is_balanced']:
#                     st.markdown(f"""
#                     <div class="balance-card">
#                         <h3>✅ GIZI SEIMBANG!</h3>
#                         <p style="font-size: 20px; margin: 0;">
#                             <strong>{result['num_detected']}/5</strong> kategori terdeteksi
#                         </p>
#                         <p style="margin: 5px 0;">Keseimbangan: <strong>{result['balance_percentage']:.0f}%</strong></p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div class="unbalanced-card">
#                         <h3>⚠️ KURANG SEIMBANG</h3>
#                         <p style="font-size: 20px; margin: 0;">
#                             <strong>{result['num_detected']}/5</strong> kategori terdeteksi
#                         </p>
#                         <p style="margin: 5px 0;">Keseimbangan: <strong>{result['balance_percentage']:.0f}%</strong></p>
#                         <p style="margin: 5px 0; font-size: 14px;">
#                             💡 Tambahkan makanan dari kategori yang belum ada
#                         </p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 # Detected foods
#                 st.write("#### 🍽️ Makanan Terdeteksi:")
                
#                 if result['detected_classes']:
#                     for det in result['detected_classes']:
#                         st.markdown(f"""
#                         <div style="padding: 10px; background-color: #e8f5e9; 
#                              border-left: 4px solid #4CAF50; margin: 5px 0; border-radius: 5px;">
#                             {det['icon']} <strong>{det['class'].title()}</strong>
#                             <span style="float: right; color: #4CAF50;">
#                                 {det['confidence']:.1f}%
#                             </span>
#                         </div>
#                         """, unsafe_allow_html=True)
#                 else:
#                     st.warning("Tidak ada makanan terdeteksi dengan confidence tinggi.")
                
#                 # Missing categories
#                 detected_names = [d['class'] for d in result['detected_classes']]
#                 missing = [c for c in CLASS_NAMES if c not in detected_names]
                
#                 if missing:
#                     st.write("#### ❌ Kategori yang Belum Ada:")
#                     for cat in missing:
#                         st.markdown(f"""
#                         <div style="padding: 8px; background-color: #ffebee; 
#                              border-left: 4px solid #f44336; margin: 5px 0; border-radius: 5px;">
#                             {CLASS_ICONS[cat]} {cat.title()}
#                         </div>
#                         """, unsafe_allow_html=True)
            
#             # Grad-CAM Visualizations
#             if result['gradcam_viz']:
#                 st.write("### 🗺️ Grad-CAM Visualization (Lokasi Makanan)")
#                 st.write("*Heatmap menunjukkan di mana AI mendeteksi setiap jenis makanan*")
                
#                 # Display in grid
#                 num_detected = len(result['gradcam_viz'])
#                 cols_per_row = min(3, num_detected)
                
#                 for i in range(0, num_detected, cols_per_row):
#                     cols = st.columns(cols_per_row)
#                     for j, col in enumerate(cols):
#                         idx = i + j
#                         if idx < num_detected:
#                             viz = result['gradcam_viz'][idx]
#                             with col:
#                                 st.image(
#                                     viz['image'],
#                                     caption=f"{viz['icon']} {viz['class'].title()} ({viz['confidence']:.1f}%)",
#                                     use_container_width=True
#                                 )
    
#     with tab3:
#         st.write("### 📸 Real-time Camera Detection")
#         st.info("💡 Gunakan kamera untuk deteksi langsung")
        
#         col_cam1, col_cam2 = st.columns([2, 1])
        
#         with col_cam1:
#             camera_image = st.camera_input("📷 Ambil foto makanan")
            
#             if camera_image:
#                 image = Image.open(camera_image)
                
#                 if st.button('🔍 Analisis Gizi', key='camera_analyze'):
#                     with st.spinner('🤖 Menganalisis...'):
#                         result = predict_multilabel(image, model)
                        
#                         if result:
#                             st.session_state['camera_result'] = result
        
#         # Display camera results
#         if 'camera_result' in st.session_state:
#             result = st.session_state['camera_result']
            
#             with col_cam2:
#                 # Balance indicator
#                 if result['is_balanced']:
#                     st.success(f"✅ Seimbang ({result['num_detected']}/5)")
#                 else:
#                     st.warning(f"⚠️ Kurang ({result['num_detected']}/5)")
                
#                 # Quick summary
#                 st.write("**Terdeteksi:**")
#                 for det in result['detected_classes']:
#                     st.write(f"{det['icon']} {det['class'].title()} - {det['confidence']:.1f}%")
                
#                 # Show missing
#                 detected_names = [d['class'] for d in result['detected_classes']]
#                 missing = [c for c in CLASS_NAMES if c not in detected_names]
                
#                 if missing:
#                     st.write("**Belum ada:**")
#                     for cat in missing:
#                         st.write(f"{CLASS_ICONS[cat]} {cat.title()}")
    
#     # Sidebar
#     st.sidebar.title("ℹ️ Info Sistem")
#     st.sidebar.write(f"""
#     **Model:** EfficientNetB3  
#     **Type:** Multi-label Classification  
#     **Input Size:** {IMG_SIZE}x{IMG_SIZE}  
#     **Classes:** {len(CLASS_NAMES)}  
#     **Threshold:** {THRESHOLD}
    
#     **Status:** {'✅ Ready' if model else '❌ Not Ready'}
#     """)
    
#     st.sidebar.divider()
    
#     st.sidebar.markdown("""
#     ### 🎯 Akurasi Model
#     - Binary Accuracy: ~75-80%
#     - Hamming Loss: <0.15
#     - Multi-label: Supported ✅
    
#     ### 🔬 Teknologi
#     - TensorFlow/Keras 3.x
#     - EfficientNetB3
#     - Grad-CAM Visualization
#     - Multi-label Classification
#     """)
    
#     # Footer
#     st.divider()
#     st.markdown("""
#     <p style='text-align: center; color: #666;'>
#         © 2024 Andromeda Team | Final Project - AI Track, Startup Campus<br/>
#         <a href='https://github.com/FAISALAKBARr/Object-Detection-for-4-Sehat-5-Sempurna-Dataset-with-CNN-TensorFlow.git' 
#            target='_blank'>
#             <img src='https://img.shields.io/badge/GitHub-Repository-black?style=flat&logo=github'/>
#         </a>
#     </p>
#     """, unsafe_allow_html=True)

# if __name__ == '__main__':
#     main()