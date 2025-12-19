import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# -----------------------------
# CONFIGURATION PAGE
# -----------------------------
st.set_page_config(
    page_title="MASTER PRO - AI Image Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# GESTION DU THÈME
# -----------------------------
# Initialiser le thème dans session state
if 'app_theme' not in st.session_state:
    st.session_state.app_theme = 'dark'  # Par défaut sombre

# Fonction pour basculer le thème
def toggle_theme():
    st.session_state.app_theme = 'light' if st.session_state.app_theme == 'dark' else 'dark'

# -----------------------------
# CSS DYNAMIQUE POUR LES DEUX THÈMES
# -----------------------------
# CSS pour mode sombre
dark_css = """
<style>
    /* Style global - MODE SOMBRE */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #f0f0f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        text-align: center;
        font-size: 4.5rem;
        background: linear-gradient(90deg, #00dbde, #fc00ff, #00dbde);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        padding: 25px 0;
        margin-bottom: 10px;
        font-weight: 900;
        letter-spacing: 3px;
        text-transform: uppercase;
        animation: shine 3s linear infinite;
        text-shadow: 0 0 30px rgba(0, 219, 222, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0ff;
        font-size: 1.4rem;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .glass-container {
        background: rgba(25, 25, 40, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        margin: 25px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        color: #f0f0f0;
    }
    
    .section-title {
        color: #00dbde;
        font-size: 2rem;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 15px;
        padding-bottom: 15px;
        border-bottom: 2px solid rgba(0, 219, 222, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-weight: 700 !important;
    }
    
    .upload-zone {
        border: 3px dashed rgba(0, 219, 222, 0.4);
        border-radius: 15px;
        padding: 60px 40px;
        text-align: center;
        background: rgba(0, 0, 0, 0.2);
        color: #f0f0f0;
    }
    
    .test-badge {
        background: linear-gradient(135deg, #00dbde, #fc00ff);
        color: white;
    }
    
    /* Correction des couleurs du texte dans Streamlit */
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #f0f0f0 !important;
    }
    
    .stMetric {
        color: #f0f0f0 !important;
    }
    
    .stAlert {
        background: rgba(25, 25, 40, 0.9) !important;
        color: #f0f0f0 !important;
        border: 1px solid rgba(0, 219, 222, 0.3) !important;
    }
    
    .stExpander {
        background: rgba(25, 25, 40, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .streamlit-expanderHeader {
        color: #f0f0f0 !important;
    }
</style>
"""

# CSS pour mode clair
light_css = """
<style>
    /* Style global - MODE CLAIR */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        color: #1e293b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        text-align: center;
        font-size: 4.5rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        padding: 25px 0;
        margin-bottom: 10px;
        font-weight: 900;
        letter-spacing: 3px;
        text-transform: uppercase;
        animation: shine 3s linear infinite;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.4rem;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .glass-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        margin: 25px 0;
        border: 1px solid #cbd5e1;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        color: #1e293b;
    }
    
    .section-title {
        color: #3b82f6;
        font-size: 2rem;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 15px;
        padding-bottom: 15px;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-weight: 700 !important;
    }
    
    .upload-zone {
        border: 3px dashed rgba(59, 130, 246, 0.4);
        border-radius: 15px;
        padding: 60px 40px;
        text-align: center;
        background: rgba(255, 255, 255, 0.5);
        color: #1e293b;
    }
    
    .test-badge {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
    }
    
    /* Correction des couleurs du texte dans Streamlit */
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #1e293b !important;
    }
    
    .stMetric {
        color: #1e293b !important;
    }
    
    .stMetric label {
        color: #64748b !important;
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    .streamlit-expanderHeader {
        color: #1e293b !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
    }
</style>
"""

# CSS commun (animations, etc.)
common_css = """
<style>
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(59, 130, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        transition: all 0.3s;
    }
    
    .upload-zone:hover {
        transform: scale(1.02);
        transition: all 0.3s;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
"""

# Appliquer le CSS selon le thème
if st.session_state.app_theme == 'dark':
    st.markdown(dark_css + common_css, unsafe_allow_html=True)
else:
    st.markdown(light_css + common_css, unsafe_allow_html=True)

# -----------------------------
# FONCTIONS DE GÉNÉRATION D'IMAGE DE TEST
# -----------------------------
def generate_test_image_pattern(pattern_type="gradient"):
    """Génère différents types d'images de test"""
    size = 512
    
    if pattern_type == "gradient":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                img[i, j] = [
                    int((np.sin(i/50) * 0.5 + 0.5) * 255),
                    int((np.cos(j/50) * 0.5 + 0.5) * 255),
                    int(((i+j)/(size*2)) * 255)
                ]
    
    elif pattern_type == "geometric":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img.fill(30)
        
        for k in range(1, 6):
            offset = k * 40
            color = (k * 40, k * 30, 255 - k * 40)
            cv2.rectangle(img, (offset, offset), (size-offset, size-offset), color, 8)
        
        for k in range(1, 4):
            radius = k * 60
            color = (255 - k * 60, k * 60, 150)
            cv2.circle(img, (size//2, size//2), radius, color, 6)
    
    elif pattern_type == "color_bars":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        bar_width = size // 8
        colors = [
            (255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0),
            (0, 255, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255)
        ]
        
        for i, color in enumerate(colors):
            start_x = i * bar_width
            end_x = (i + 1) * bar_width
            img[:, start_x:end_x] = color
    
    else:  # test_chart
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        
        # Grille
        for i in range(0, size, 32):
            cv2.line(img, (i, 0), (i, size), (200, 200, 200), 1)
            cv2.line(img, (0, i), (size, i), (200, 200, 200), 1)
        
        # Échelles de gris
        for i in range(10):
            intensity = i * 25
            x1, x2 = 50 + i * 40, 50 + (i + 1) * 40
            cv2.rectangle(img, (x1, 50), (x2, 100), (intensity, intensity, intensity), -1)
        
        # Couleurs primaires
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, color in enumerate(colors):
            x = 50 + i * 70
            cv2.rectangle(img, (x, 150), (x + 60, 200), color, -1)
        
        cv2.putText(img, "MASTER PRO TEST CHART", (100, 250), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
    
    return Image.fromarray(img)

# -----------------------------
# FONCTIONS DE TRAITEMENT
# -----------------------------
def apply_effects(image_np, effects):
    """Applique les effets sélectionnés à l'image"""
    processed = image_np.copy()
    
    if effects['rotation'] != "Aucune":
        if effects['rotation'] == "90° Droite":
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif effects['rotation'] == "90° Gauche":
            processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif effects['rotation'] == "180°":
            processed = cv2.rotate(processed, cv2.ROTATE_180)
    
    if effects['grayscale']:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    
    if effects['blur']:
        kernel = effects['blur_intensity'] * 2 + 1
        processed = cv2.GaussianBlur(processed, (kernel, kernel), 0)
    
    if effects['edges']:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed
        processed = cv2.Canny(gray, effects['edge_low'], effects['edge_high'])
    
    if effects['invert']:
        processed = 255 - processed
    
    if effects['hdr'] and len(processed.shape) == 3:
        processed = cv2.detailEnhance(processed, sigma_s=10, sigma_r=0.15)
    
    if effects['sketch']:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        processed = cv2.divide(gray, 255 - blurred, scale=256)
    
    if effects['brightness'] != 0 or effects['contrast'] != 1.0:
        processed = cv2.convertScaleAbs(processed, alpha=effects['contrast'], beta=effects['brightness'])
    
    if effects['saturation'] != 1.0 and len(processed.shape) == 3:
        hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * effects['saturation'], 0, 255)
        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return processed

# -----------------------------
# INITIALISATION SESSION STATE
# -----------------------------
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'effects' not in st.session_state:
    st.session_state.effects = {
        'rotation': "Aucune",
        'grayscale': False,
        'blur': False,
        'blur_intensity': 3,
        'edges': False,
        'edge_low': 100,
        'edge_high': 200,
        'invert': False,
        'hdr': False,
        'sketch': False,
        'brightness': 0,
        'contrast': 1.0,
        'saturation': 1.0
    }
if 'history' not in st.session_state:
    st.session_state.history = []
if 'test_image_generated' not in st.session_state:
    st.session_state.test_image_generated = False
if 'test_image_data' not in st.session_state:
    st.session_state.test_image_data = None
if 'pattern_type' not in st.session_state:
    st.session_state.pattern_type = "gradient"

# -----------------------------
# HEADER AVEC BOUTON THÈME
# -----------------------------
col_theme, col_title, col_empty = st.columns([1, 3, 1])

with col_theme:
    # Bouton de bascule thème
    theme_icon = "🌙" if st.session_state.app_theme == 'dark' else "☀️"
    theme_text = "Mode clair" if st.session_state.app_theme == 'dark' else "Mode sombre"
    
    if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()

with col_title:
    st.markdown('<h1 class="main-title">🚀 MASTER PRO AI STUDIO 🎨</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Plateforme Professionnelle de Traitement d\'Images IA</p>', unsafe_allow_html=True)

# -----------------------------
# ONGLETS PRINCIPAUX
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🎨 STUDIO CREATIF", "📊 ANALYTICS", "🎮 GUIDE"])

with tab1:
    col_main1, col_main2 = st.columns([1, 1])
    
    with col_main1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title"><i class="fas fa-cloud-upload-alt"></i> IMPORTATION</h2>', unsafe_allow_html=True)
        
        col_upload_top = st.columns([3, 2])
        
        with col_upload_top[0]:
            uploaded_file = st.file_uploader(
                "Choisissez un fichier image",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                key="file_uploader"
            )
            
            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.test_image_generated = False
                st.success(f"✅ Fichier chargé: {uploaded_file.name}")
        
        with col_upload_top[1]:
            st.markdown("### 🎲 Image de Test")
            pattern_type = st.selectbox(
                "Type d'image",
                ["gradient", "geometric", "color_bars", "test_chart"],
                format_func=lambda x: {
                    "gradient": "🌈 Dégradé",
                    "geometric": "🔷 Formes", 
                    "color_bars": "🎨 Couleurs",
                    "test_chart": "📊 Chartre"
                }[x]
            )
            
            if st.button("🚀 GÉNÉRER IMAGE TEST", use_container_width=True, type="primary"):
                with st.spinner("Génération..."):
                    test_image = generate_test_image_pattern(pattern_type)
                    buffer = io.BytesIO()
                    test_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    
                    class FakeUploadedFile:
                        def __init__(self, buffer, name):
                            self.buffer = buffer
                            self.name = name
                            self.type = "image/png"
                        
                        def read(self):
                            return self.buffer.getvalue()
                    
                    st.session_state.test_image_data = buffer.getvalue()
                    st.session_state.uploaded_file = FakeUploadedFile(
                        buffer, f"test_image_{pattern_type}.png"
                    )
                    st.session_state.test_image_generated = True
                    st.session_state.pattern_type = pattern_type
                    st.rerun()
        
        if st.session_state.uploaded_file:
            try:
                if st.session_state.test_image_generated:
                    test_image = Image.open(io.BytesIO(st.session_state.test_image_data))
                    st.markdown("### 📸 Image de Test")
                    st.markdown('<div class="test-badge">🎲 IMAGE DE TEST</div>', unsafe_allow_html=True)
                    st.image(test_image, use_column_width=True)
                else:
                    image = Image.open(st.session_state.uploaded_file)
                    st.markdown("### 📸 Image Originale")
                    st.image(image, use_column_width=True)
                    # Informations sur l'image uploadée
                    with st.expander("📋 Informations sur l'image"):
                        img_array = np.array(image)
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Dimensions", f"{img_array.shape[1]}x{img_array.shape[0]}")
                            st.metric("Taille fichier", f"{len(st.session_state.uploaded_file.getvalue()) / 1024:.1f} KB")
                        with col_info2:
                            colors = "Niveaux de gris" if len(img_array.shape) == 2 else f"RGB ({img_array.shape[2]} canaux)"
                            st.metric("Couleurs", colors)
                            st.metric("Format", st.session_state.uploaded_file.type.split('/')[-1].upper())
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
        
        else:
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size: 4rem; margin-bottom: 20px;">
                    <i class="fas fa-cloud-upload-alt"></i>
                </div>
                <h3>Glissez-déposez une image</h3>
                <p>ou cliquez pour parcourir</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_main2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title"><i class="fas fa-magic"></i> TRANSFORMATIONS</h2>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_file:
            col_controls1, col_controls2 = st.columns(2)
            
            with col_controls1:
                st.markdown("#### 🎭 Filtres")
                st.session_state.effects['rotation'] = st.selectbox("Rotation", ["Aucune", "90° Droite", "90° Gauche", "180°"])
                st.session_state.effects['grayscale'] = st.checkbox("Niveaux de gris")
                st.session_state.effects['invert'] = st.checkbox("Inverser")
                st.session_state.effects['hdr'] = st.checkbox("Effet HDR")
                st.session_state.effects['sketch'] = st.checkbox("Effet croquis")
            
            with col_controls2:
                st.markdown("#### ⚙️ Réglages")
                st.session_state.effects['brightness'] = st.slider("Luminosité", -100, 100, 0)
                st.session_state.effects['contrast'] = st.slider("Contraste", 0.1, 3.0, 1.0, 0.1)
                st.session_state.effects['saturation'] = st.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
                
                st.session_state.effects['blur'] = st.checkbox("Flou gaussien")
                if st.session_state.effects['blur']:
                    st.session_state.effects['blur_intensity'] = st.slider("Intensité", 1, 10, 3)
                
                st.session_state.effects['edges'] = st.checkbox("Détection de contours")
                if st.session_state.effects['edges']:
                    col_edge1, col_edge2 = st.columns(2)
                    with col_edge1:
                        st.session_state.effects['edge_low'] = st.slider("Seuil bas", 1, 255, 100)
                    with col_edge2:
                        st.session_state.effects['edge_high'] = st.slider("Seuil haut", 1, 255, 200)
            
            if st.button("🚀 APPLIQUER LES TRANSFORMATIONS", use_container_width=True, type="primary"):
                with st.spinner("Traitement..."):
                    if st.session_state.test_image_generated:
                        image = Image.open(io.BytesIO(st.session_state.test_image_data))
                    else:
                        image = Image.open(st.session_state.uploaded_file)
                    
                    image_np = np.array(image)
                    start_time = time.time()
                    processed_np = apply_effects(image_np, st.session_state.effects)
                    processing_time = time.time() - start_time
                    
                    st.session_state.processed_image = processed_np
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'effects': st.session_state.effects.copy(),
                        'processing_time': processing_time,
                        'is_test_image': st.session_state.test_image_generated
                    })
                    
                    st.success(f"✅ Terminé en {processing_time:.2f}s")
            
            if st.session_state.processed_image is not None:
                st.markdown("### ✨ RÉSULTAT")
                if st.session_state.test_image_generated:
                    st.markdown('<div class="test-badge">🎲 IMAGE DE TEST</div>', unsafe_allow_html=True)
                st.image(st.session_state.processed_image, use_column_width=True)
                
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if len(st.session_state.processed_image.shape) == 2:
                        processed_pil = Image.fromarray(st.session_state.processed_image, 'L')
                    else:
                        processed_pil = Image.fromarray(st.session_state.processed_image)
                    
                    buffer = io.BytesIO()
                    processed_pil.save(buffer, format="PNG")
                    buffer.seek(0)
                    
                    filename = f"master_pro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    
                    st.download_button(
                        label="📥 TÉLÉCHARGER",
                        data=buffer,
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_export2:
                    if st.button("🔄 NOUVELLE IMAGE", use_container_width=True, type="secondary"):
                        st.session_state.processed_image = None
                        st.session_state.effects = {
                            'rotation': "Aucune",
                            'grayscale': False,
                            'blur': False,
                            'blur_intensity': 3,
                            'edges': False,
                            'edge_low': 100,
                            'edge_high': 200,
                            'invert': False,
                            'hdr': False,
                            'sketch': False,
                            'brightness': 0,
                            'contrast': 1.0,
                            'saturation': 1.0
                        }
                        st.rerun()
        
        else:
            # Message d'attente
            st.info("""
            ## 📝 Prêt à transformer !
            
            1. **Importez une image** depuis votre ordinateur
            2. **Ou générez une image de test** avec le bouton 🎲
            3. **Ajustez les paramètres** de transformation
            4. **Appliquez les effets** et téléchargez le résultat
            
            ### 🎯 Effets disponibles:
            - 🌀 **Rotation:** 90°, 180°, 270°
            - 🎨 **Filtres:** Niveaux de gris, Inversion, HDR, Croquis
            - 🔍 **Détection:** Contours avec réglages précis
            - ⚙️ **Réglages:** Luminosité, Contraste, Saturation
            - 🌫️ **Effets:** Flou gaussien paramétrable
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title"><i class="fas fa-chart-line"></i> ANALYTICS</h2>', unsafe_allow_html=True)
    
    if st.session_state.history:
        latest = st.session_state.history[-1]
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            st.metric("⏱️ Dernier traitement", f"{latest['processing_time']:.2f}s")
        with col_stats2:
            active_effects = sum(1 for k, v in latest['effects'].items() 
                               if v and k not in ['blur_intensity', 'edge_low', 'edge_high'])
            st.metric("🎯 Effets appliqués", active_effects)
        with col_stats3:
            image_type = "Image de test" if latest.get('is_test_image', False) else "Image uploadée"
            st.metric("📁 Type", image_type)
        with col_stats4:
            st.metric("📊 Total traitements", len(st.session_state.history))
        
        if len(st.session_state.history) > 1:
            times = [h['timestamp'].strftime('%H:%M') for h in st.session_state.history]
            durations = [h['processing_time'] for h in st.session_state.history]
            
            # Utiliser le thème approprié pour Plotly
            plotly_template = "plotly_white" if st.session_state.app_theme == 'light' else "plotly_dark"
            
            fig = go.Figure(data=go.Scatter(
                x=times,
                y=durations,
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=10),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                template=plotly_template,
                title="Temps de traitement",
                xaxis_title="Heure",
                yaxis_title="Secondes"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Historique")
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            with st.expander(f"Traitement #{len(st.session_state.history)-i}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Durée:** {entry['processing_time']:.2f}s")
                    st.write(f"**Type:** {'🎲 Test' if entry.get('is_test_image', False) else '📁 Upload'}")
                with col2:
                    if st.button(f"🔄 Restaurer", key=f"restore_{i}"):
                        st.session_state.effects = entry['effects']
                        st.success("Configuration restaurée!")
                        st.rerun()
        
        if st.button("🗑️ Effacer l'historique", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.rerun()
    
    else:
        st.info("Aucun traitement dans l'historique")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title"><i class="fas fa-book"></i> GUIDE</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📚 Comment utiliser MASTER PRO
    
    1. **Importez une image** ou **générez-en une** avec le bouton 🎲
    2. **Ajustez les paramètres** de transformation
    3. **Appliquez les effets** et **téléchargez** le résultat
    
    ### 🎨 Effets disponibles:
    - Rotation (90°, 180°, 270°)
    - Niveaux de gris
    - Détection de contours
    - Effets HDR et croquis
    - Inversion des couleurs
    - Flou gaussien
    - Ajustements de luminosité/contraste/saturation
    
    ### ⚡ Préréglages rapides:
    """)
    
    presets = {
        "🎨 Artistique": {'sketch': True, 'contrast': 1.5},
        "📷 Professionnel": {'hdr': True, 'contrast': 1.2},
        "⚫ Noir & Blanc": {'grayscale': True, 'contrast': 1.8},
        "🌀 Créatif": {'edges': True, 'invert': True, 'blur': True}
    }
    
    col_presets = st.columns(4)
    for idx, (name, config) in enumerate(presets.items()):
        with col_presets[idx]:
            if st.button(name, use_container_width=True):
                for key, value in config.items():
                    if key in st.session_state.effects:
                        st.session_state.effects[key] = value
                st.success(f"Préréglage appliqué: {name}")
    
    st.markdown("""
    ---
    ### 🌓 Thème de l'application
    Utilisez le bouton **🌙 Mode clair / ☀️ Mode sombre** en haut à gauche 
    pour changer le thème de l'interface selon vos préférences.
    
    **Mode sombre:** Confortable pour une utilisation nocturne
    **Mode clair:** Idéal pour une utilisation en journée
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<hr>
<div style="text-align: center; padding: 30px; color: #a0a0ff; font-size: 0.9rem;">
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 15px; flex-wrap: wrap;">
        <div><i class="fas fa-bolt"></i> Traitement Temps Réel</div>
        <div><i class="fas fa-shield-alt"></i> Sécurisé & Privé</div>
        <div><i class="fas fa-infinity"></i> Illimité</div>
        <div><i class="fas fa-rocket"></i> Haute Performance</div>
    </div>
    <p style="margin-top: 20px;">
        <strong>MASTER PRO AI STUDIO</strong> © 2025 | 
        <span style="color: #00dbde;">Version 3.1.0</span>
    </p>
    <p>
        Développé par l'équipe <span style="color: #00dbde;">FEZE Loïck - EFEMBA Manuella - EYOUM Brayan </span>
    </p>
    <p style="font-size: 0.8rem; color: #888; margin-top: 10px;">
        <i class="fas fa-cogs"></i> Powered by OpenCV 
    </p>
</div>
""", unsafe_allow_html=True)

# Effet de notification initiale
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

if st.session_state.first_visit:
    st.markdown("""
    <div style="position: fixed; top: 20px; right: 20px; background: rgba(25, 25, 40, 0.95); 
                border-left: 5px solid #00dbde; padding: 20px; border-radius: 10px; 
                box-shadow: 0 5px 20px rgba(0,0,0,0.3); z-index: 1000; max-width: 300px;
                animation: slideIn 0.5s ease-out;">
        <h4 style="color: #00dbde; margin-bottom: 10px;">
            <i class="fas fa-rocket"></i> Bienvenue dans MASTER PRO!
        </h4>
        <p style="margin: 0; color: #f0f0f0;">
            Commencez par générer une image de test 🎲 ou importez la vôtre!
        </p>
    </div>
    
    <script>
        setTimeout(() => {
            let toast = document.querySelector('[style*="slideIn"]');
            if (toast) {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100px)';
                setTimeout(() => {
                    toast.style.display = 'none';
                }, 500);
            }
        }, 5000);
    </script>
    """, unsafe_allow_html=True)
    st.session_state.first_visit = False
