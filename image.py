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
import base64

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
# CSS PERSONNALISÉ AVANCÉ
# -----------------------------
st.markdown("""
<style>
    /* Style global */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #f0f0f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Titre principal avec animation */
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
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    
    /* Sous-titre animé */
    .subtitle {
        text-align: center;
        color: #a0a0ff;
        font-size: 1.4rem;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 1px;
        animation: fadeIn 2s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Conteneurs avec effet verre morphique */
    .glass-container {
        background: rgba(25, 25, 40, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        margin: 25px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(0, 219, 222, 0.3);
    }
    
    /* Boutons néon */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    /* Bouton générer image test - STYLE SPÉCIAL */
    .test-image-btn {
        background: linear-gradient(135deg, #00dbde 0%, #fc00ff 100%) !important;
        animation: pulse 2s infinite !important;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 219, 222, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(0, 219, 222, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 219, 222, 0); }
    }
    
    /* Zone d'upload stylisée */
    .upload-zone {
        border: 3px dashed rgba(0, 219, 222, 0.4);
        border-radius: 15px;
        padding: 60px 40px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        background: rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .upload-zone:hover {
        border-color: #00dbde;
        background: rgba(0, 219, 222, 0.05);
        transform: scale(1.02);
    }
    
    /* Sections */
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
    
    /* Tabs stylisées */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(25, 25, 40, 0.8);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #a0a0a0;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00dbde, #fc00ff) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(0, 219, 222, 0.3);
    }
    
    /* Images avec effet */
    .image-frame {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 15px;
        border: 2px solid rgba(0, 219, 222, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .image-frame::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00dbde, #fc00ff);
    }
    
    /* Badge image test */
    .test-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #00dbde, #fc00ff);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        z-index: 100;
        animation: badgeGlow 1.5s infinite alternate;
    }
    
    @keyframes badgeGlow {
        from { box-shadow: 0 0 5px rgba(0, 219, 222, 0.5); }
        to { box-shadow: 0 0 15px rgba(0, 219, 222, 0.8); }
    }
    
    /* Cache les éléments Streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FONCTIONS DE GÉNÉRATION D'IMAGE DE TEST
# -----------------------------
def generate_advanced_test_image():
    """Génère une image de test professionnelle avec effets visuels"""
    size = 512
    # Créer un canevas noir
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Ajouter un dégradé radial
    center = (size // 2, size // 2)
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            max_distance = np.sqrt(center[0]**2 + center[1]**2)
            
            # Couleurs basées sur la position
            r = int((i / size) * 255)
            g = int((j / size) * 255)
            b = int((distance / max_distance) * 255)
            
            image[i, j] = [b, g, r]
    
    # Ajouter des formes géométriques complexes
    # Cercle principal avec effet 3D
    cv2.circle(image, center, 200, (255, 255, 255), 8)
    cv2.circle(image, center, 180, (0, 100, 255), -1)
    
    # Hexagone
    hexagon_points = []
    for i in range(6):
        angle = 2 * np.pi / 6 * i
        x = int(center[0] + 150 * np.cos(angle))
        y = int(center[1] + 150 * np.sin(angle))
        hexagon_points.append((x, y))
    
    pts = np.array(hexagon_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, (255, 255, 0), 4)
    
    # Lignes rayonnantes
    for i in range(12):
        angle = 2 * np.pi / 12 * i
        x = int(center[0] + 220 * np.cos(angle))
        y = int(center[1] + 220 * np.sin(angle))
        cv2.line(image, center, (x, y), (0, 255, 255), 3)
    
    # Ajouter du texte avec effet
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, 'MASTER', (120, 220), font, 2, (255, 255, 255), 5)
    cv2.putText(image, 'PRO', (180, 300), font, 2.5, (255, 255, 255), 5)
    
    cv2.putText(image, 'AI IMAGE STUDIO', (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 3)
    
    # Ajouter des particules
    for _ in range(50):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(2, 6)
        color = (np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255))
        cv2.circle(image, (x, y), radius, color, -1)
    
    # Effet de vignette
    kernel_x = cv2.getGaussianKernel(size, size/3)
    kernel_y = cv2.getGaussianKernel(size, size/3)
    kernel = kernel_x * kernel_y.T
    mask = kernel / kernel.max()
    
    for i in range(3):
        image[:, :, i] = image[:, :, i] * mask
    
    return Image.fromarray(image)

def generate_test_image_pattern(pattern_type="gradient"):
    """Génère différents types d'images de test"""
    size = 512
    
    if pattern_type == "gradient":
        # Dégradé complexe
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                img[i, j] = [
                    int((np.sin(i/50) * 0.5 + 0.5) * 255),
                    int((np.cos(j/50) * 0.5 + 0.5) * 255),
                    int(((i+j)/(size*2)) * 255)
                ]
    
    elif pattern_type == "geometric":
        # Formes géométriques
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img.fill(30)
        
        # Carrés concentriques
        for k in range(1, 6):
            offset = k * 40
            color = (k * 40, k * 30, 255 - k * 40)
            cv2.rectangle(img, (offset, offset), (size-offset, size-offset), color, 8)
        
        # Cercles
        for k in range(1, 4):
            radius = k * 60
            color = (255 - k * 60, k * 60, 150)
            cv2.circle(img, (size//2, size//2), radius, color, 6)
    
    elif pattern_type == "color_bars":
        # Barres de couleur
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
    
    else:  # pattern_type == "test_chart"
        # Chartre de test professionnelle
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
            cv2.putText(img, str(intensity), (x1 + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Couleurs primaires
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, color in enumerate(colors):
            x = 50 + i * 70
            cv2.rectangle(img, (x, 150), (x + 60, 200), color, -1)
        
        # Texte
        cv2.putText(img, "MASTER PRO TEST CHART", (100, 250), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(img, f"Resolution: {size}x{size}px", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1)
        cv2.putText(img, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (150, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
    
    return Image.fromarray(img)

# -----------------------------
# FONCTIONS DE TRAITEMENT
# -----------------------------
def apply_effects(image_np, effects):
    """Applique les effets sélectionnés à l'image"""
    processed = image_np.copy()
    
    # Rotation
    if effects['rotation'] != "Aucune":
        if effects['rotation'] == "90° Droite":
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        elif effects['rotation'] == "90° Gauche":
            processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif effects['rotation'] == "180°":
            processed = cv2.rotate(processed, cv2.ROTATE_180)
    
    # Conversion niveaux de gris
    if effects['grayscale']:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    
    # Flou gaussien
    if effects['blur']:
        kernel = effects['blur_intensity'] * 2 + 1
        if len(processed.shape) == 3:
            processed = cv2.GaussianBlur(processed, (kernel, kernel), 0)
        else:
            processed = cv2.GaussianBlur(processed, (kernel, kernel), 0)
    
    # Détection de contours
    if effects['edges']:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed
        processed = cv2.Canny(gray, effects['edge_low'], effects['edge_high'])
    
    # Inversion
    if effects['invert']:
        processed = 255 - processed
    
    # Effet HDR
    if effects['hdr'] and len(processed.shape) == 3:
        processed = cv2.detailEnhance(processed, sigma_s=10, sigma_r=0.15)
    
    # Effet croquis
    if effects['sketch']:
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        processed = cv2.divide(gray, 255 - blurred, scale=256)
    
    # Ajustements de base
    if effects['brightness'] != 0 or effects['contrast'] != 1.0:
        processed = cv2.convertScaleAbs(processed, alpha=effects['contrast'], beta=effects['brightness'])
    
    # Saturation (uniquement pour images couleur)
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
# HEADER ANIMÉ
# -----------------------------
st.markdown("""
<div style="text-align: center; margin-bottom: 40px;">
    <h1 class="main-title">
        <span style="display: inline-block; animation: float 3s ease-in-out infinite;">🚀</span>
        MASTER PRO AI STUDIO
        <span style="display: inline-block; animation: float 3s ease-in-out infinite 0.5s;">🎨</span>
    </h1>
    <p class="subtitle">
        <i class="fas fa-bolt" style="color: #00dbde;"></i>
        Plateforme Professionnelle de Traitement d'Images IA
        <i class="fas fa-brain" style="color: #fc00ff;"></i>
    </p>
</div>

<style>
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# ONGLETS PRINCIPAUX
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🎨 STUDIO CREATIF", "📊 ANALYTICS", "🎮 GUIDE & PRÉRÉGLAGES"])

with tab1:
    col_main1, col_main2 = st.columns([1, 1])
    
    with col_main1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title"><i class="fas fa-cloud-upload-alt"></i> IMPORTATION</h2>', unsafe_allow_html=True)
        
        # Section Upload avec bouton de génération
        col_upload_top = st.columns([3, 2])
        
        with col_upload_top[0]:
            uploaded_file = st.file_uploader(
                "Choisissez un fichier image",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                key="file_uploader",
                label_visibility="visible"
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
                    "gradient": "🌈 Dégradé Complexe",
                    "geometric": "🔷 Formes Géométriques", 
                    "color_bars": "🎨 Barres de Couleur",
                    "test_chart": "📊 Chartre de Test"
                }[x]
            )
            
            # BOUTON DE GÉNÉRATION FONCTIONNEL
            if st.button("🚀 GÉNÉRER UNE IMAGE DE TEST", 
                        use_container_width=True,
                        type="primary",
                        key="generate_test_button"):
                
                with st.spinner("🎨 Génération de l'image de test en cours..."):
                    # Générer l'image
                    test_image = generate_test_image_pattern(pattern_type)
                    
                    # Convertir en bytes pour le téléchargement
                    buffer = io.BytesIO()
                    test_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    
                    # Créer un faux fichier uploadé
                    class FakeUploadedFile:
                        def __init__(self, buffer, name):
                            self.buffer = buffer
                            self.name = name
                            self.type = "image/png"
                        
                        def read(self):
                            return self.buffer.getvalue()
                    
                    # Mettre à jour le session state
                    st.session_state.test_image_data = buffer.getvalue()
                    st.session_state.uploaded_file = FakeUploadedFile(
                        buffer, 
                        f"test_image_{pattern_type}_{datetime.now().strftime('%H%M%S')}.png"
                    )
                    st.session_state.test_image_generated = True
                    st.session_state.pattern_type = pattern_type
                    
                    # Forcer le rerun pour afficher l'image
                    st.rerun()
        
        # Afficher l'image chargée
        if st.session_state.uploaded_file:
            try:
                if st.session_state.test_image_generated:
                    # Afficher l'image de test générée
                    test_image = Image.open(io.BytesIO(st.session_state.test_image_data))
                    st.markdown("### 📸 Image de Test Générée")
                    
                    # Badge spécial pour image de test
                    st.markdown('<div class="test-badge">🎲 IMAGE DE TEST</div>', unsafe_allow_html=True)
                    st.image(test_image, use_column_width=True)
                    
                    # Informations sur l'image de test
                    with st.expander("📋 Informations sur l'image de test"):
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("Type", {
                                "gradient": "Dégradé Complexe",
                                "geometric": "Formes Géométriques", 
                                "color_bars": "Barres de Couleur",
                                "test_chart": "Chartre de Test"
                            }[st.session_state.pattern_type])
                            st.metric("Résolution", "512x512 px")
                        with col_info2:
                            st.metric("Couleurs", "RGB 24-bit")
                            st.metric("Taille", f"{len(st.session_state.test_image_data) / 1024:.1f} KB")
                
                else:
                    # Afficher l'image uploadée normale
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
                st.error(f"Erreur lors du chargement de l'image: {str(e)}")
        
        else:
            # Zone d'upload vide
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size: 5rem; color: #00dbde; margin-bottom: 20px;">
                    <i class="fas fa-cloud-upload-alt"></i>
                </div>
                <h3>Glissez-déposez une image ici</h3>
                <p style="color: #aaa;">ou cliquez pour parcourir vos fichiers</p>
                <p style="color: #888; font-size: 0.9rem; margin-top: 20px;">
                    Formats supportés: JPG, PNG, BMP, TIFF
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Exemples d'images de test
            st.markdown("### 🎯 Exemples d'Images de Test")
            col_examples = st.columns(4)
            example_patterns = ["gradient", "geometric", "color_bars", "test_chart"]
            example_names = ["🌈 Dégradé", "🔷 Géométrique", "🎨 Couleurs", "📊 Chartre"]
            
            for idx, (pattern, name) in enumerate(zip(example_patterns, example_names)):
                with col_examples[idx]:
                    # Générer une miniature
                    example_img = generate_test_image_pattern(pattern)
                    example_img.thumbnail((100, 100))
                    st.image(example_img, use_column_width=True)
                    st.caption(name)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_main2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title"><i class="fas fa-magic"></i> TRANSFORMATIONS</h2>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_file:
            # Interface des contrôles
            col_controls1, col_controls2 = st.columns(2)
            
            with col_controls1:
                st.markdown("#### 🌀 Transformations")
                st.session_state.effects['rotation'] = st.selectbox(
                    "Rotation",
                    ["Aucune", "90° Droite", "90° Gauche", "180°"],
                    key="rotation_control"
                )
                
                col_effects1, col_effects2 = st.columns(2)
                with col_effects1:
                    st.session_state.effects['grayscale'] = st.checkbox("Niveaux de gris", key="grayscale_control")
                    st.session_state.effects['invert'] = st.checkbox("Inverser", key="invert_control")
                    st.session_state.effects['hdr'] = st.checkbox("Effet HDR", key="hdr_control")
                
                with col_effects2:
                    st.session_state.effects['edges'] = st.checkbox("Contours", key="edges_control")
                    st.session_state.effects['sketch'] = st.checkbox("Croquis", key="sketch_control")
                    st.session_state.effects['blur'] = st.checkbox("Flou", key="blur_control")
            
            with col_controls2:
                st.markdown("#### ⚙️ Réglages")
                st.session_state.effects['brightness'] = st.slider(
                    "Luminosité", -100, 100, st.session_state.effects['brightness'],
                    key="brightness_control"
                )
                st.session_state.effects['contrast'] = st.slider(
                    "Contraste", 0.1, 3.0, st.session_state.effects['contrast'], 0.1,
                    key="contrast_control"
                )
                st.session_state.effects['saturation'] = st.slider(
                    "Saturation", 0.0, 3.0, st.session_state.effects['saturation'], 0.1,
                    key="saturation_control"
                )
                
                if st.session_state.effects['blur']:
                    st.session_state.effects['blur_intensity'] = st.slider(
                        "Intensité flou", 1, 10, st.session_state.effects['blur_intensity'],
                        key="blur_intensity_control"
                    )
                
                if st.session_state.effects['edges']:
                    col_edge1, col_edge2 = st.columns(2)
                    with col_edge1:
                        st.session_state.effects['edge_low'] = st.slider(
                            "Seuil bas", 1, 255, st.session_state.effects['edge_low'],
                            key="edge_low_control"
                        )
                    with col_edge2:
                        st.session_state.effects['edge_high'] = st.slider(
                            "Seuil haut", 1, 255, st.session_state.effects['edge_high'],
                            key="edge_high_control"
                        )
            
            # Bouton de traitement
            if st.button("🚀 APPLIQUER LES TRANSFORMATIONS", 
                        use_container_width=True,
                        type="primary",
                        key="apply_effects_button"):
                
                with st.spinner("⚡ Traitement en cours..."):
                    # Charger l'image
                    if st.session_state.test_image_generated:
                        image = Image.open(io.BytesIO(st.session_state.test_image_data))
                    else:
                        image = Image.open(st.session_state.uploaded_file)
                    
                    image_np = np.array(image)
                    
                    # Appliquer les effets
                    start_time = time.time()
                    processed_np = apply_effects(image_np, st.session_state.effects)
                    processing_time = time.time() - start_time
                    
                    # Sauvegarder le résultat
                    st.session_state.processed_image = processed_np
                    
                    # Ajouter à l'historique
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'effects': st.session_state.effects.copy(),
                        'processing_time': processing_time,
                        'is_test_image': st.session_state.test_image_generated
                    })
                    
                    st.success(f"✅ Traitement terminé en {processing_time:.2f} secondes!")
                    st.balloons()
            
            # Afficher l'image transformée
            if st.session_state.processed_image is not None:
                st.markdown("### ✨ RÉSULTAT")
                
                # Badge si c'est une image de test
                if st.session_state.test_image_generated:
                    st.markdown('<div class="test-badge">🎲 IMAGE DE TEST</div>', unsafe_allow_html=True)
                
                st.image(st.session_state.processed_image, use_column_width=True)
                
                # Boutons d'export
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    # Préparer l'image pour le téléchargement
                    if len(st.session_state.processed_image.shape) == 2:
                        processed_pil = Image.fromarray(st.session_state.processed_image, mode='L')
                    else:
                        processed_pil = Image.fromarray(st.session_state.processed_image)
                    
                    buffer = io.BytesIO()
                    processed_pil.save(buffer, format="PNG", quality=95, optimize=True)
                    buffer.seek(0)
                    
                    # Nom du fichier
                    if st.session_state.test_image_generated:
                        filename = f"master_pro_test_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    else:
                        original_name = st.session_state.uploaded_file.name
                        name_without_ext = original_name.rsplit('.', 1)[0]
                        filename = f"{name_without_ext}_processed_{datetime.now().strftime('%H%M%S')}.png"
                    
                    st.download_button(
                        label="📥 TÉLÉCHARGER L'IMAGE",
                        data=buffer,
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True,
                        key="download_button"
                    )
                
                with col_export2:
                    if st.button("🔄 TRAITER UNE NOUVELLE IMAGE",
                               use_container_width=True,
                               type="secondary",
                               key="new_image_button"):
                        # Réinitialiser pour une nouvelle image
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
                
                # Aperçu des paramètres utilisés
                with st.expander("⚙️ Paramètres appliqués"):
                    active_effects = [k for k, v in st.session_state.effects.items() 
                                    if v and k not in ['blur_intensity', 'edge_low', 'edge_high']]
                    
                    if active_effects:
                        st.write("**Effets actifs:**")
                        for effect in active_effects:
                            st.write(f"• {effect.replace('_', ' ').title()}")
                    else:
                        st.info("Aucun effet appliqué")
        
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
    st.markdown('<h2 class="section-title"><i class="fas fa-chart-line"></i> ANALYTICS & HISTORIQUE</h2>', unsafe_allow_html=True)
    
    if st.session_state.history:
        # Statistiques récentes
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
            total_treatments = len(st.session_state.history)
            st.metric("📊 Total traitements", total_treatments)
        
        # Graphique d'historique
        if len(st.session_state.history) > 1:
            st.markdown("### 📈 Évolution des temps de traitement")
            
            times = [h['timestamp'].strftime('%H:%M:%S') for h in st.session_state.history]
            durations = [h['processing_time'] for h in st.session_state.history]
            
            fig = go.Figure(data=go.Scatter(
                x=times,
                y=durations,
                mode='lines+markers',
                line=dict(color='#00dbde', width=3),
                marker=dict(size=10, color='#fc00ff'),
                fill='tozeroy',
                fillcolor='rgba(0, 219, 222, 0.1)'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f0f0f0',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Détails des traitements récents
        st.markdown("### 📋 Historique détaillé")
        
        for i, entry in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Traitement #{len(st.session_state.history)-i} - {entry['timestamp'].strftime('%H:%M:%S')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Durée:** {entry['processing_time']:.2f}s")
                    st.write(f"**Type:** {'Image de test 🎲' if entry.get('is_test_image', False) else 'Image uploadée 📁'}")
                    
                    active_in_entry = [k for k, v in entry['effects'].items() 
                                     if v and k not in ['blur_intensity', 'edge_low', 'edge_high']]
                    if active_in_entry:
                        st.write("**Effets appliqués:**")
                        for eff in active_in_entry:
                            st.write(f"• {eff.replace('_', ' ').title()}")
                
                with col2:
                    if st.button(f"🔄 Restaurer", key=f"restore_{i}"):
                        st.session_state.effects = entry['effects']
                        st.success("Configuration restaurée !")
                        st.rerun()
        
        # Bouton de nettoyage
        if st.button("🗑️ Effacer l'historique", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.success("Historique effacé !")
            st.rerun()
    
    else:
        st.info("""
        ## 📊 Analytics en attente
        
        Aucun traitement n'a encore été effectué.
        
        ### Pour commencer:
        1. Importez ou générez une image
        2. Appliquez des transformations
        3. Les statistiques apparaîtront ici automatiquement
        
        ### 📈 Vous pourrez voir:
        - Temps de traitement
        - Nombre d'effets appliqués
        - Évolution des performances
        - Historique complet des modifications
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title"><i class="fas fa-gamepad"></i> GUIDE & PRÉRÉGLAGES</h2>', unsafe_allow_html=True)
    
    col_guide1, col_guide2 = st.columns([2, 1])
    
    with col_guide1:
        st.markdown("""
        ### 🎮 GUIDE D'UTILISATION
        
        #### Étape 1: Chargement d'image
        - **Importation:** Cliquez sur "Parcourir" ou glissez-déposez
        - **Génération:** Utilisez 🎲 pour créer une image de test
        - **Formats:** JPG, PNG, BMP, TIFF supportés
        
        #### Étape 2: Transformation
        - **Filtres rapides:** Cochez les effets souhaités
        - **Réglages précis:** Utilisez les sliders pour ajuster
        - **Prévisualisation:** Les changements s'appliquent en temps réel
        
        #### Étape 3: Export
        - **Téléchargement:** PNG haute qualité
        - **Rapport:** Analytics détaillés disponibles
        - **Historique:** Suivi de tous vos traitements
        
        ### 💡 ASTUCES PRO
        
        #### Combinaisons gagnantes:
        - **Portrait pro:** HDR + Légère saturation
        - **Artistique:** Croquis + Inversion
        - **Abstrait:** Contours + Flou + Rotation
        - **Minimaliste:** Niveaux de gris + Contraste élevé
        
        #### Performances:
        - Les images < 5MB traitent plus vite
        - Sauvegardez vos réglages préférés
        - Utilisez l'historique pour retrouver vos réglages
        """)
    
    with col_guide2:
        st.markdown("### ⚡ PRÉRÉGLAGES RAPIDES")
        
        # Préréglages
        presets = {
            "🎨 Artistique": {
                'sketch': True,
                'contrast': 1.5,
                'saturation': 0.8,
                'brightness': 10
            },
            "📷 Professionnel": {
                'hdr': True,
                'contrast': 1.2,
                'saturation': 1.1,
                'brightness': 5
            },
            "⚫ Noir & Blanc Pro": {
                'grayscale': True,
                'contrast': 1.8,
                'brightness': 15
            },
            "🌀 Effet Cinéma": {
                'contrast': 1.3,
                'saturation': 0.7,
                'brightness': -10,
                'blur': True,
                'blur_intensity': 2
            },
            "🔍 Haute Définition": {
                'edges': True,
                'edge_low': 50,
                'edge_high': 150,
                'contrast': 1.4
            },
            "🌈 Psychedélique": {
                'invert': True,
                'saturation': 2.0,
                'contrast': 1.6,
                'blur': True,
                'blur_intensity': 4
            }
        }
        
        for preset_name, preset_config in presets.items():
            if st.button(preset_name, use_container_width=True, key=f"preset_{preset_name}"):
                # Appliquer le préréglage
                for key, value in preset_config.items():
                    if key in st.session_state.effects:
                        st.session_state.effects[key] = value
                
                st.success(f"Préréglage '{preset_name}' appliqué !")
                
                # Si une image est chargée, appliquer directement
                if st.session_state.uploaded_file:
                    st.info("Cliquez sur 'APPLIQUER LES TRANSFORMATIONS' pour voir le résultat")
    
    # Section démo
    st.markdown("### 🎪 DÉMONSTRATION EN DIRECT")
    
    demo_col1, demo_col2 = st.columns([2, 1])
    
    with demo_col1:
        demo_mode = st.selectbox(
            "Mode de démonstration",
            ["Simple", "Avancé", "Artiste", "Technique"],
            key="demo_mode"
        )
    
    with demo_col2:
        if st.button("🎬 LANCER LA DÉMO", use_container_width=True):
            # Animation de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 25:
                    status_text.text("📥 Chargement de l'image...")
                elif i < 50:
                    status_text.text("⚙️ Application des effets...")
                elif i < 75:
                    status_text.text("🎨 Optimisation des couleurs...")
                else:
                    status_text.text("✅ Finalisation...")
                time.sleep(0.02)
            
            progress_bar.empty()
            status_text.empty()
            
            # Effets visuels
            st.balloons()
            st.success("🎉 Démonstration terminée avec succès !")
            
            # Montrer un exemple
            if demo_mode == "Artiste":
                st.info("🎨 Mode Artiste: Combinaison parfaite pour des effets créatifs")
            elif demo_mode == "Technique":
                st.info("🔧 Mode Technique: Optimisé pour la précision et les détails")
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER AVANCÉ
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
        <strong>MASTER PRO AI STUDIO</strong> © 2024 | 
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
