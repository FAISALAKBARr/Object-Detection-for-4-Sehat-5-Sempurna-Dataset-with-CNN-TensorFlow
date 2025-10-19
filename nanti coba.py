import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import gc

# Lazy import tensorflow
@st.cache_resource
def get_tensorflow():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    return tf

# Set page configuration
st.set_page_config(
    page_title="SmartPlate - Nutrition Detector",
    page_icon="🍽️",
    layout="wide"
)

# Model configuration
MODEL_ID = '1DTbji3Y-JJarXD22YHCtsMWhYWcvHi5F'
IMG_SIZE = 320  # Sesuaikan dengan training
DETECTION_THRESHOLD = 0.25  # Adjustable threshold

MODEL_PATH = 'nutrition_model.keras'

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px;
        font-weight: 600;
        border-radius: 5px;
    }
    .balanced-box {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 10px 0;
    }
    .not-balanced-box {
        background: linear-gradient(90deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Class information
class_names = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']

class_info = {
    'buah': {
        'emoji': '🍎',
        'name': 'Buah-buahan',
        'desc': 'Sumber vitamin, mineral, dan serat alami',
        'color': (255, 107, 107)
    },
    'karbohidrat': {
        'emoji': '🍚',
        'name': 'Karbohidrat',
        'desc': 'Sumber energi utama untuk aktivitas',
        'color': (255, 217, 61)
    },
    'minuman': {
        'emoji': '🥤',
        'name': 'Minuman',
        'desc': 'Penting untuk hidrasi dan metabolisme',
        'color': (107, 203, 119)
    },
    'protein': {
        'emoji': '🍖',
        'name': 'Protein',
        'desc': 'Pembentuk dan pemelihara jaringan tubuh',
        'color': (77, 150, 255)
    },
    'sayur': {
        'emoji': '🥬',
        'name': 'Sayuran',
        'desc': 'Kaya vitamin, mineral, dan serat',
        'color': (149, 225, 211)
    }
}

@st.cache_resource
def load_model_safe():
    """Load model dari Google Drive"""
    try:
        tf = get_tensorflow()
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner('Downloading model from Google Drive...'):
                url = f'https://drive.google.com/uc?id={MODEL_ID}'
                try:
                    gdown.download(url, MODEL_PATH, quiet=False)
                    st.success('Model downloaded successfully!')
                except Exception as e:
                    st.error(f'Failed to download model: {str(e)}')
                    return None
        
        with st.spinner('Loading model...'):
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                st.success('Model loaded successfully!')
                return model
            except Exception as e:
                st.error(f'Error loading model: {str(e)}')
                return None
        
    except Exception as e:
        st.error(f'Unexpected error: {str(e)}')
        return None

def preprocess_image(image):
    """Preprocess image untuk prediction"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
        image = Image.fromarray(image)
    else:
        original = np.array(image)
    
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    return img_array, original

def predict_and_visualize(image, model, threshold=DETECTION_THRESHOLD):
    """Predict dengan multi-detection strategy"""
    try:
        processed_image, original_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Multi-detection
        detected = []
        for idx, prob in enumerate(predictions):
            if prob > threshold:
                detected.append({
                    'class': class_names[idx],
                    'confidence': float(prob),
                    'idx': idx
                })
        
        detected.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Visualization
        height, width = original_image.shape[:2]
        viz_image = original_image.copy()
        
        num_detected = len(detected)
        if num_detected > 0:
            cols = min(3, num_detected)
            rows = (num_detected + cols - 1) // cols
            
            box_width = width // (cols + 1)
            box_height = height // (rows + 1)
            
            for i, det in enumerate(detected):
                row = i // cols
                col = i % cols
                
                x_center = box_width * (col + 1)
                y_center = box_height * (row + 1)
                
                box_w = int(box_width * 0.7)
                box_h = int(box_height * 0.7)
                
                x1 = max(0, x_center - box_w // 2)
                y1 = max(0, y_center - box_h // 2)
                x2 = min(width, x_center + box_w // 2)
                y2 = min(height, y_center + box_h // 2)
                
                color = class_info[det['class']]['color']
                
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 3)
                
                label = f"{class_info[det['class']]['emoji']} {det['class']}: {det['confidence']*100:.1f}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                cv2.rectangle(viz_image,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0] + 10, y1),
                            color, -1)
                
                cv2.putText(viz_image, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return {
            'predictions': predictions,
            'detected': detected,
            'viz_image': viz_image,
            'original': original_image
        }
        
    except Exception as e:
        st.error(f'Prediction error: {str(e)}')
        return None

def analyze_balance(detected):
    """Analisis keseimbangan gizi"""
    detected_classes = set([d['class'] for d in detected])
    
    components = {
        'buah': 'buah' in detected_classes,
        'karbohidrat': 'karbohidrat' in detected_classes,
        'minuman': 'minuman' in detected_classes,
        'protein': 'protein' in detected_classes,
        'sayur': 'sayur' in detected_classes
    }
    
    total = sum(components.values())
    percentage = (total / 5) * 100
    is_balanced = total == 5
    missing = [k for k, v in components.items() if not v]
    
    return {
        'components': components,
        'total': total,
        'percentage': percentage,
        'is_balanced': is_balanced,
        'missing': missing
    }

def main():
    st.title("🍽️ SmartPlate - Nutrition Balance Detector")
    st.markdown("### Sistem Deteksi Keseimbangan Gizi 4 Sehat 5 Sempurna")
    
    # Load model
    model = load_model_safe()
    
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Info Sistem")
        
        st.markdown("**Kategori yang Dideteksi:**")
        for class_name, info in class_info.items():
            st.markdown(f"{info['emoji']} {info['name']}")
        
        st.divider()
        
        st.markdown("### Pengaturan")
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=DETECTION_THRESHOLD,
            step=0.05,
            help="Class terdeteksi jika confidence > threshold"
        )
        
        st.divider()
        st.markdown("### Model Info")
        st.info(f"**Input Size:** {IMG_SIZE}x{IMG_SIZE}")
        st.info(f"**Threshold:** {threshold:.2f}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Beranda", "Upload Gambar", "Camera"])
    
    with tab1:
        st.header("Selamat Datang di SmartPlate!")
        
        st.markdown("""
        **SmartPlate** adalah sistem deteksi keseimbangan gizi menggunakan AI.
        
        ### Fitur Utama:
        - Deteksi multiple jenis makanan dalam 1 gambar
        - Analisis keseimbangan gizi otomatis
        - Visualisasi interaktif dengan bounding boxes
        - Real-time detection dari kamera
        
        ### Prinsip 4 Sehat 5 Sempurna:
        1. Karbohidrat - Sumber energi
        2. Protein - Pembentuk jaringan
        3. Sayuran - Vitamin dan mineral
        4. Buah-buahan - Serat dan vitamin
        5. Minuman - Hidrasi tubuh
        """)
        
        if st.button("Lihat Dataset di Kaggle"):
            st.markdown('[Dataset Link](https://www.kaggle.com/datasets/andromedagroup05/data-4-sehat-5-sempurna/data)')
    
    with tab2:
        st.header("Analisis Gambar")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload gambar makanan", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Gambar Original', use_container_width=True)
                
                if st.button('Analisis Sekarang!'):
                    with st.spinner('Menganalisis...'):
                        result = predict_and_visualize(image, model, threshold)
                        
                        if result:
                            st.session_state['result'] = result
                            st.rerun()
        
        with col2:
            if 'result' in st.session_state:
                result = st.session_state['result']
                
                st.subheader("Hasil Deteksi")
                st.image(result['viz_image'], caption='Hasil Deteksi', use_container_width=True)
                
                if result['detected']:
                    st.success(f"Terdeteksi {len(result['detected'])} kategori makanan!")
                    
                    balance = analyze_balance(result['detected'])
                    
                    if balance['is_balanced']:
                        st.markdown('<div class="balanced-box">GIZI SEIMBANG (5/5)</div>', unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f'<div class="not-balanced-box">BELUM SEIMBANG ({balance["total"]}/5)</div>', unsafe_allow_html=True)
                    
                    st.progress(balance['percentage'] / 100)
                    st.caption(f"Kelengkapan: {balance['percentage']:.0f}%")
                    
                    st.markdown("### Komponen Gizi:")
                    cols = st.columns(5)
                    for idx, (comp, present) in enumerate(balance['components'].items()):
                        with cols[idx]:
                            info = class_info[comp]
                            if present:
                                st.markdown(f"✅ {info['emoji']}")
                                st.caption(comp)
                            else:
                                st.markdown(f"⬜ {info['emoji']}")
                                st.caption(comp)
                    
                    if balance['missing']:
                        st.warning("Yang Masih Kurang:")
                        for missing in balance['missing']:
                            info = class_info[missing]
                            st.markdown(f"- {info['emoji']} {info['name']}: {info['desc']}")
                    
                    with st.expander("Detail Deteksi"):
                        for det in result['detected']:
                            info = class_info[det['class']]
                            st.markdown(f"**{info['emoji']} {info['name']}**")
                            st.markdown(f"Confidence: {det['confidence']*100:.2f}%")
                            st.markdown(f"Deskripsi: {info['desc']}")
                            st.divider()
                    
                    with st.expander("Semua Probabilitas"):
                        for idx, prob in enumerate(result['predictions']):
                            st.write(f"{class_names[idx]}: {prob*100:.2f}%")
                            st.progress(prob)
                
                else:
                    st.warning("Tidak ada makanan terdeteksi dengan confidence tinggi")
                    st.info("Coba adjust threshold di sidebar")
    
    with tab3:
        st.header("Real-time Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            camera_image = st.camera_input("Ambil foto")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                
                with st.spinner('Analyzing...'):
                    result = predict_and_visualize(image, model, threshold)
                    
                    if result:
                        st.image(result['viz_image'], caption='Real-time Detection', use_container_width=True)
                        
                        if result['detected']:
                            balance = analyze_balance(result['detected'])
                            
                            st.metric("Kelengkapan Gizi", f"{balance['total']}/5", f"{balance['percentage']:.0f}%")
                            
                            if balance['is_balanced']:
                                st.success("Gizi Seimbang!")
                            else:
                                st.warning(f"Kurang: {', '.join(balance['missing'])}")
                            
                            st.markdown("**Terdeteksi:**")
                            for det in result['detected']:
                                info = class_info[det['class']]
                                st.write(f"{info['emoji']} {info['name']}: {det['confidence']*100:.1f}%")
        
        with col2:
            st.markdown("""
            ### Panduan Penggunaan
            
            1. Klik tombol kamera
            2. Izinkan akses kamera
            3. Arahkan ke makanan
            4. Take photo
            5. Lihat hasil
            
            ### Tips:
            - Pencahayaan cukup
            - Posisi tengah
            - Hindari blur
            - Jarak 20-50 cm
            - Background polos
            
            ### Troubleshooting:
            - Kamera tidak muncul? Cek permission
            - Error? Refresh halaman
            - Gunakan Chrome/Firefox
            """)
            
            st.divider()
            st.markdown("### Status Sistem")
            st.success("Model: Ready")
            st.info(f"Input: {IMG_SIZE}x{IMG_SIZE}")
            st.info(f"Threshold: {threshold:.2f}")
    
    # Footer
    st.divider()
    st.markdown("<p style='text-align: center;'>© 2024 Andromeda. All rights reserved.</p>", unsafe_allow_html=True)
    
    st.markdown(
        '<p style="text-align: center;">'
        '<a href="https://github.com/FAISALAKBARr/Object-Detection-for-4-Sehat-5-Sempurna-Dataset-with-CNN-TensorFlow.git" target="_blank">'
        '<img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">'
        '</a></p>',
        unsafe_allow_html=True
    )
    
    gc.collect()

if __name__ == '__main__':
    main()