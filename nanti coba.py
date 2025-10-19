import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import json
import gc

#=============================================================================
# PAGE CONFIG
#=============================================================================
st.set_page_config(
    page_title="SmartPlate - Nutrition Balance Detector",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#=============================================================================
# CONFIGURATION
#=============================================================================
class Config:
    # ✅ UPDATE INI dengan file ID dari Google Drive Anda
    MODEL_ID = '1DTbji3Y-JJarXD22YHCtsMWhYWcvHi5F'
    
    MODEL_PATH = 'nutrition_model.keras'
    CONFIG_PATH = 'model_config.json'
    
    # Default values (akan di-override dari model_config.json)
    IMG_SIZE = 320
    CONFIDENCE_THRESHOLD = 0.3
    CLASS_NAMES = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']
    
    CLASS_INFO = {
        'buah': {
            'emoji': '🍎',
            'name': 'Buah-buahan',
            'desc': 'Sumber vitamin, mineral, dan serat alami',
            'benefits': ['Vitamin C & A', 'Antioksidan', 'Serat pencernaan'],
            'color': '#FF6B6B'
        },
        'karbohidrat': {
            'emoji': '🍚',
            'name': 'Karbohidrat',
            'desc': 'Sumber energi utama untuk aktivitas',
            'benefits': ['Energi', 'Glukosa otak', 'Stamina'],
            'color': '#FFD93D'
        },
        'minuman': {
            'emoji': '🥤',
            'name': 'Minuman',
            'desc': 'Penting untuk hidrasi dan metabolisme',
            'benefits': ['Hidrasi', 'Metabolisme', 'Detoksifikasi'],
            'color': '#6BCB77'
        },
        'protein': {
            'emoji': '🍖',
            'name': 'Protein',
            'desc': 'Pembentuk dan pemelihara jaringan tubuh',
            'benefits': ['Pertumbuhan', 'Perbaikan sel', 'Imunitas'],
            'color': '#4D96FF'
        },
        'sayur': {
            'emoji': '🥬',
            'name': 'Sayuran',
            'desc': 'Kaya vitamin, mineral, dan serat',
            'benefits': ['Vitamin K', 'Folat', 'Serat tinggi'],
            'color': '#95E1D3'
        }
    }

config = Config()

#=============================================================================
# CUSTOM CSS
#=============================================================================
st.markdown("""
<style>
    .main {padding: 2rem;}
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .detection-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .balance-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
        margin: 0.5rem;
    }
    
    .balanced {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .not-balanced {
        background: linear-gradient(90deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

#=============================================================================
# TENSORFLOW LAZY LOADING
#=============================================================================
@st.cache_resource
def get_tensorflow():
    """Lazy load TensorFlow"""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    return tf

#=============================================================================
# MODEL LOADING
#=============================================================================
@st.cache_resource
def load_model_and_config():
    """Load model dan config dari Google Drive"""
    try:
        tf = get_tensorflow()
        
        # Download model if not exists
        if not os.path.exists(config.MODEL_PATH):
            with st.spinner('📥 Downloading model dari Google Drive...'):
                url = f'https://drive.google.com/uc?id={config.MODEL_ID}'
                gdown.download(url, config.MODEL_PATH, quiet=False)
                st.success('✅ Model berhasil didownload!')
        
        # Load model
        with st.spinner('🔄 Loading model...'):
            model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
            
            # Recompile
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['binary_accuracy']
            )
        
        # Load config if available
        model_config = None
        if os.path.exists(config.CONFIG_PATH):
            with open(config.CONFIG_PATH, 'r') as f:
                model_config = json.load(f)
                config.IMG_SIZE = model_config.get('img_size', config.IMG_SIZE)
                config.CONFIDENCE_THRESHOLD = model_config.get('confidence_threshold', config.CONFIDENCE_THRESHOLD)
        
        st.success('✅ Model loaded successfully!')
        return model, model_config
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

#=============================================================================
# IMAGE PREPROCESSING
#=============================================================================
def preprocess_image(image):
    """Preprocess image untuk prediction"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
        image = Image.fromarray(image)
    else:
        original = np.array(image)
    
    # Resize
    image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    return img_array, original

#=============================================================================
# GRAD-CAM VISUALIZATION
#=============================================================================
def generate_gradcam(model, image, class_idx):
    """Generate Grad-CAM heatmap"""
    try:
        tf = get_tensorflow()
        
        # Find last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create grad model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            class_channel = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
        
    except Exception as e:
        st.warning(f"Grad-CAM tidak tersedia: {str(e)}")
        return None

def overlay_gradcam(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap pada image"""
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    return superimposed

#=============================================================================
# PREDICTION WITH VISUALIZATION
#=============================================================================
def predict_and_visualize(image, model):
    """
    Predict dengan multi-label detection dan visualization
    """
    try:
        # Preprocess
        processed_image, original_image = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Detect objects above threshold
        detected = []
        for idx, prob in enumerate(predictions):
            if prob > config.CONFIDENCE_THRESHOLD:
                detected.append({
                    'class': config.CLASS_NAMES[idx],
                    'confidence': float(prob),
                    'class_idx': idx
                })
        
        # Sort by confidence
        detected.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Create visualization with pseudo bounding boxes
        height, width = original_image.shape[:2]
        viz_image = original_image.copy()
        
        # Draw boxes for detected objects
        num_detected = len(detected)
        if num_detected > 0:
            box_width = width // (num_detected + 1)
            
            for i, det in enumerate(detected):
                # Pseudo bounding box based on position
                x_center = box_width * (i + 1)
                box_size = int(min(width, height) * 0.4)
                
                x1 = max(0, x_center - box_size // 2)
                y1 = max(0, height // 2 - box_size // 2)
                x2 = min(width, x_center + box_size // 2)
                y2 = min(height, height // 2 + box_size // 2)
                
                # Color based on class
                class_info = config.CLASS_INFO[det['class']]
                color_hex = class_info['color']
                color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                
                # Draw box
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color_rgb, 3)
                
                # Label
                label = f"{class_info['emoji']} {det['class']}: {det['confidence']*100:.1f}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Background untuk text
                cv2.rectangle(viz_image,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0] + 10, y1),
                            color_rgb, -1)
                
                # Text
                cv2.putText(viz_image, label,
                           (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return {
            'predictions': predictions,
            'detected': detected,
            'viz_image': viz_image,
            'original': original_image
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

#=============================================================================
# NUTRITION BALANCE ANALYSIS
#=============================================================================
def analyze_nutrition_balance(detected):
    """
    Analisis keseimbangan gizi berdasarkan 4 Sehat 5 Sempurna
    """
    detected_classes = set([d['class'] for d in detected])
    
    # Check each component
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
    
    return {
        'components': components,
        'total': total,
        'percentage': percentage,
        'is_balanced': is_balanced,
        'missing': [k for k, v in components.items() if not v]
    }

#=============================================================================
# MAIN APP
#=============================================================================
def main():
    # Header
    st.title("🍽️ SmartPlate - Nutrition Balance Detector")
    st.markdown("### Sistem Deteksi Keseimbangan Gizi '4 Sehat 5 Sempurna'")
    
    # Load model
    model, model_config = load_model_and_config()
    
    if model is None:
        st.error("❌ Gagal memuat model. Silakan refresh halaman.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Tentang Sistem")
        
        st.markdown("""
        Sistem ini menggunakan **Deep Learning** untuk:
        - 🔍 Mendeteksi jenis makanan/minuman
        - ⚖️ Menganalisis keseimbangan gizi
        - 📊 Memberikan rekomendasi
        
        **Kategori yang Dideteksi:**
        """)
        
        for class_name, info in config.CLASS_INFO.items():
            st.markdown(f"{info['emoji']} **{info['name']}**")
        
        st.divider()
        
        st.markdown("### ⚙️ Konfigurasi Model")
        if model_config:
            st.info(f"""
            **Arsitektur:** {model_config.get('architecture', 'N/A')}\n
            **Input Size:** {config.IMG_SIZE}x{config.IMG_SIZE}\n
            **Threshold:** {config.CONFIDENCE_THRESHOLD}\n
            **Total Params:** {model_config.get('total_params', 'N/A'):,}
            """)
        else:
            st.info(f"""
            **Input Size:** {config.IMG_SIZE}x{config.IMG_SIZE}\n
            **Threshold:** {config.CONFIDENCE_THRESHOLD}
            """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Beranda",
        "📤 Upload Gambar",
        "📸 Real-time Camera",
        "📚 Panduan"
    ])
    
    #=========================================================================
    # TAB 1: HOME
    #=========================================================================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Selamat Datang di SmartPlate! 👋
            
            **SmartPlate** adalah sistem otomatis untuk menganalisis keseimbangan gizi makanan
            berdasarkan prinsip **'4 Sehat 5 Sempurna'** menggunakan teknologi AI.
            
            ### 🎯 Fitur Utama:
            - ✅ **Multi-Object Detection**: Deteksi beberapa jenis makanan sekaligus
            - ✅ **Analisis Gizi Otomatis**: Evaluasi keseimbangan nutrisi
            - ✅ **Visualisasi Interaktif**: Bounding boxes & confidence scores
            - ✅ **Real-time Processing**: Deteksi langsung dari kamera
            
            ### 🔬 Teknologi:
            - **Model**: EfficientNetB3 (Multi-label Classification)
            - **Framework**: TensorFlow/Keras
            - **Visualization**: Grad-CAM & Pseudo Bounding Boxes
            """)
            
            if st.button("📊 Lihat Dataset Lengkap"):
                st.markdown("""
                <a href="https://www.kaggle.com/datasets/andromedagroup05/data-4-sehat-5-sempurna/data" 
                   target="_blank">
                    <button style="background:#51baff; color:white; padding:10px 20px; 
                                   border:none; border-radius:5px; cursor:pointer;">
                        🔗 Buka Dataset di Kaggle
                    </button>
                </a>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🍽️ Prinsip 4 Sehat 5 Sempurna")
            
            for class_name, info in config.CLASS_INFO.items():
                with st.expander(f"{info['emoji']} {info['name']}"):
                    st.markdown(f"**{info['desc']}**")
                    st.markdown("**Manfaat:**")
                    for benefit in info['benefits']:
                        st.markdown(f"- {benefit}")
    
    #=========================================================================
    # TAB 2: UPLOAD IMAGE
    #=========================================================================
    with tab2:
        st.header("📤 Analisis Gambar")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Gambar Makanan")
            uploaded_file = st.file_uploader(
                "Pilih gambar (JPG, JPEG, PNG)",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file is not None:
                # Display original
                image = Image.open(uploaded_file)
                st.image(image, caption="📷 Gambar Original", use_container_width=True)
                
                # Analyze button
                if st.button("🔍 Analisis Sekarang!", key="analyze_upload"):
                    with st.spinner("⏳ Sedang menganalisis..."):
                        result = predict_and_visualize(image, model)
                        
                        if result:
                            # Store in session state
                            st.session_state['result'] = result
                            st.rerun()
        
        with col2:
            if 'result' in st.session_state:
                result = st.session_state['result']
                
                st.subheader("🎯 Hasil Deteksi")
                
                # Show visualization
                st.image(result['viz_image'], 
                        caption="🎨 Visualization with Bounding Boxes",
                        use_container_width=True)
                
                # Detected objects
                if result['detected']:
                    st.success(f"✅ Terdeteksi {len(result['detected'])} kategori makanan!")
                    
                    # Balance analysis
                    balance = analyze_nutrition_balance(result['detected'])
                    
                    # Balance status
                    if balance['is_balanced']:
                        st.markdown("""
                        <div class="balance-badge balanced">
                            ✅ GIZI SEIMBANG (5/5)
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="balance-badge not-balanced">
                            ⚠️ BELUM SEIMBANG ({balance['total']}/5)
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(balance['percentage'] / 100)
                    st.caption(f"Kelengkapan Gizi: {balance['percentage']:.0f}%")
                    
                    # Components checklist
                    st.markdown("### 📋 Komponen Gizi:")
                    cols = st.columns(5)
                    for idx, (comp, present) in enumerate(balance['components'].items()):
                        with cols[idx]:
                            info = config.CLASS_INFO[comp]
                            if present:
                                st.markdown(f"✅ {info['emoji']}<br>{comp}", 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f"⬜ {info['emoji']}<br>{comp}", 
                                          unsafe_allow_html=True)
                    
                    # Missing components
                    if balance['missing']:
                        st.warning("⚠️ **Yang Masih Kurang:**")
                        for missing in balance['missing']:
                            info = config.CLASS_INFO[missing]
                            st.markdown(f"- {info['emoji']} {info['name']}: {info['desc']}")
                    
                    # Detailed detections
                    with st.expander("📊 Detail Deteksi"):
                        for det in result['detected']:
                            info = config.CLASS_INFO[det['class']]
                            st.markdown(f"""
                            **{info['emoji']} {info['name']}**
                            - Confidence: {det['confidence']*100:.2f}%
                            - Deskripsi: {info['desc']}
                            """)
                    
                    # All probabilities
                    with st.expander("📈 Distribusi Probabilitas"):
                        for idx, class_name in enumerate(config.CLASS_NAMES):
                            prob = result['predictions'][idx] * 100
                            st.write(f"**{class_name.title()}**: {prob:.2f}%")
                            st.progress(prob / 100)
                
                else:
                    st.warning("⚠️ Tidak ada makanan terdeteksi dengan confidence tinggi")
                    st.info(f"💡 Coba gunakan gambar dengan pencahayaan lebih baik atau threshold lebih rendah")
    
    #=========================================================================
    # TAB 3: REAL-TIME CAMERA
    #=========================================================================
    with tab3:
        st.header("📸 Real-time Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Ambil Foto")
            camera_image = st.camera_input("📷 Aktifkan kamera")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                
                with st.spinner("🔍 Menganalisis gambar..."):
                    result = predict_and_visualize(image, model)
                    
                    if result:
                        # Show visualization
                        st.image(result['viz_image'],
                                caption="🎨 Hasil Deteksi Real-time",
                                use_container_width=True)
                        
                        # Quick summary
                        if result['detected']:
                            balance = analyze_nutrition_balance(result['detected'])
                            
                            st.metric(
                                "Kelengkapan Gizi",
                                f"{balance['total']}/5",
                                f"{balance['percentage']:.0f}%"
                            )
                            
                            if balance['is_balanced']:
                                st.success("✅ Gizi Seimbang!")
                            else:
                                st.warning(f"⚠️ Kurang: {', '.join(balance['missing'])}")
                            
                            # Detected items
                            st.markdown("**Terdeteksi:**")
                            for det in result['detected']:
                                info = config.CLASS_INFO[det['class']]
                                st.write(f"{info['emoji']} {info['name']}: {det['confidence']*100:.1f}%")
        
        with col2:
            st.markdown("""
            ### 📋 Tips Penggunaan
            
            **Untuk hasil terbaik:**
            - ✅ Pastikan pencahayaan cukup
            - ✅ Posisikan makanan di tengah
            - ✅ Hindari blur/goyang
            - ✅ Jarak ideal: 20-50 cm
            - ✅ Gunakan background polos
            
            **Troubleshooting:**
            - Kamera tidak muncul? Cek permission browser
            - Error? Refresh halaman
            - Gunakan Chrome/Firefox
            
            ### 📊 Status Sistem
            """)
            
            st.success("🟢 Model: Ready")
            st.info(f"📐 Input: {config.IMG_SIZE}x{config.IMG_SIZE}")
            st.info(f"🎯 Threshold: {config.CONFIDENCE_THRESHOLD}")
    
    #=========================================================================
    # TAB 4: PANDUAN
    #=========================================================================
    with tab4:
        st.header("📚 Panduan Lengkap")
        
        st.markdown("""
        ## 🎓 Tentang '4 Sehat 5 Sempurna'
        
        **4 Sehat 5 Sempurna** adalah pedoman gizi Indonesia yang menganjurkan konsumsi:
        
        1. **🍚 Makanan Pokok (Karbohidrat)**: Sumber energi utama
        2. **🥬 Sayur-sayuran**: Sumber vitamin dan mineral
        3. **🥩 Lauk Pauk (Protein)**: Pembentuk dan pemelihara jaringan
        4. **🍎 Buah-buahan**: Sumber vitamin dan serat
        5. **🥛 Susu/Minuman**: Pelengkap nutrisi
        
        ---
        
        ## 🔬 Cara Kerja Sistem
        
        ### 1. Multi-Label Classification
        - Model dilatih dengan 5000 gambar (3500 train, 750 val, 750 test)
        - Menggunakan EfficientNetB3 architecture
        - Output: Probability untuk setiap kategori (0-100%)
        
        ### 2. Detection Process
        ```
        Input Image → Preprocessing → Model Prediction → 
        Multi-label Detection → Balance Analysis → Visualization
        ```
        
        ### 3. Threshold-based Detection
        - Confidence > 30% = Terdeteksi
        - Dapat mendeteksi multiple foods sekaligus
        - Pseudo bounding boxes untuk visualization
        
        ---
        
        ## 💡 Tips Mendapatkan Hasil Akurat
        
        **✅ DO:**
        - Gunakan gambar dengan pencahayaan baik
        - Foto dari atas (bird's eye view)
        - Background kontras dengan makanan
        - Fokus pada objek utama
        
        **❌ DON'T:**
        - Gambar blur atau gelap
        - Terlalu banyak objek non-makanan
        - Sudut terlalu miring
        - Resolusi terlalu rendah
        
        ---
        
        ## 📊 Interpretasi Hasil
        
        ### Confidence Score
        - **> 70%**: Deteksi sangat yakin
        - **50-70%**: Deteksi cukup yakin
        - **30-50%**: Deteksi kurang yakin (mungkin mirip dengan kelas lain)
        - **< 30%**: Tidak terdeteksi
        
        ### Balance Analysis
        - **5/5**: Gizi seimbang sempurna ✅
        - **4/5**: Hampir seimbang, kurang 1 komponen ⚠️
        - **3/5 atau kurang**: Perlu tambahan ❌
        
        ---
        
        ## ❓ FAQ
        
        **Q: Apakah bisa mendeteksi makanan yang tidak ada di 5 kategori?**\n
        A: Tidak, model hanya dilatih untuk 5 kategori ini.
        
        **Q: Kenapa kadang salah deteksi?**\n
        A: Bisa karena: pencahayaan buruk, sudut tidak ideal, atau makanan terlihat mirip dengan kategori lain.
        
        **Q: Bisa deteksi porsi makanan?**\n
        A: Belum, saat ini hanya deteksi ada/tidak ada.
        
        **Q: Akurasi berapa persen?**\n
        A: Model mencapai ~90%+ accuracy pada test set.
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <p style='text-align: center; color: #666;'>
        © 2024 Andromeda - SmartPlate Nutrition Detector<br>
        Startup Campus AI Track - Final Project
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="text-align: center;">
        <a href="https://github.com/FAISALAKBARr/Object-Detection-for-4-Sehat-5-Sempurna-Dataset-with-CNN-TensorFlow.git" 
           target="_blank">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" 
                 alt="GitHub">
        </a>
    </p>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()