import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
from datetime import datetime
from io import BytesIO
import json

# -----------------------------
# CONFIGURATION PAGE PROFESSIONNELLE
# -----------------------------
st.set_page_config(
    page_title="Master Pro | Projet1 IABD",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS POUR ADAPTATION AUTOMATIQUE AU THÈME SYSTÈME
# -----------------------------
st.markdown("""
<style>
    /* DÉTECTION AUTOMATIQUE DU THÈME SYSTÈME */
    @media (prefers-color-scheme: dark) {
        /* MODE SOMBRE SYSTÈME */
        :root {
            --primary: #8B5CF6;
            --secondary: #10B981;
            --accent: #F59E0B;
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-surface: #334155;
            --text-primary: #F1F5F9;
            --text-secondary: #94A3B8;
            --border-color: #475569;
            --button-bg: #8B5CF6;
            --button-text: #FFFFFF;
            --sidebar-bg: #1E293B;
            --card-bg: #334155;
            --metric-bg: linear-gradient(135deg, #8B5CF6, #10B981);
        }
    }
    
    @media (prefers-color-scheme: light) {
        /* MODE CLAIR SYSTÈME */
        :root {
            --primary: #0066FF;
            --secondary: #00C896;
            --accent: #FF6B35;
            --bg-primary: #FFFFFF;
            --bg-secondary: #F8FAFC;
            --bg-surface: #FFFFFF;
            --text-primary: #1E293B;
            --text-secondary: #64748B;
            --border-color: #E2E8F0;
            --button-bg: #0066FF;
            --button-text: #FFFFFF;
            --sidebar-bg: #F8FAFC;
            --card-bg: #FFFFFF;
            --metric-bg: linear-gradient(135deg, #0066FF, #00C896);
        }
    }

    /* OVERRIDE COMPLET DU CSS STREAMLIT */
    .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* FORCER LES COULEURS DE TEXTE POUR TOUT */
    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stText {
        color: var(--text-primary) !important;
    }

    /* BOUTONS - VISIBLES IMMÉDIATEMENT */
    .stButton > button {
        background: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        opacity: 0.9 !important;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* WIDGETS DE LA SIDEBAR */
    .stSelectbox > div > div {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }

    .stTextInput > div > div > input {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }

    .stSlider {
        color: var(--text-primary) !important;
    }

    .stCheckbox > label {
        color: var(--text-primary) !important;
    }

    .stRadio > label {
        color: var(--text-primary) !important;
    }

    /* HEADER PROFESSIONNEL */
    .main-header {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        padding: 4rem 2rem !important;
        border-radius: 24px !important;
        color: white !important;
        text-align: center !important;
        margin: 2rem auto !important;
        max-width: 1400px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1) !important;
        position: relative !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .main-header h1 {
        font-size: 4rem !important;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(135deg, #FFFFFF, #E2E8F0) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }

    .main-header p {
        font-size: 1.3rem !important;
        opacity: 0.9 !important;
        max-width: 800px !important;
        margin: 0 auto !important;
        color: white !important;
    }

    .badge-container {
        display: flex !important;
        justify-content: center !important;
        gap: 12px !important;
        margin-top: 2rem !important;
        flex-wrap: wrap !important;
    }

    .badge {
        background: rgba(255, 255, 255, 0.15) !important;
        padding: 10px 24px !important;
        border-radius: 50px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }

    /* MÉTRIQUES */
    .metric-card {
        background: var(--metric-bg) !important;
        color: white !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        text-align: center !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        margin: 0.5rem !important;
    }

    .metric-card h2, .metric-card div {
        color: white !important;
    }

    /* CARTES D'IMAGES */
    .image-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* CARTES DE DÉTAILS */
    .details-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* CARTES DE FONCTIONNALITÉS */
    .feature-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        height: 100% !important;
    }

    /* FOOTER */
    .footer {
        background: var(--bg-secondary) !important;
        border-top: 1px solid var(--border-color) !important;
        padding: 3rem 2rem !important;
        margin-top: 4rem !important;
        text-align: center !important;
        border-radius: 24px !important;
    }

    .footer h3 {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    .footer p {
        color: var(--text-secondary) !important;
        margin: 0.5rem 0 !important;
    }

    /* ONGLETS */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-surface) !important;
        padding: 8px !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
    }

    /* ÉLÉMENTS DE FILTRE */
    .filter-item {
        background: var(--card-bg) !important;
        border-left: 4px solid var(--primary) !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        border-radius: 12px !important;
    }

    /* TAGS */
    .filter-tag {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        padding: 6px 12px !important;
        border-radius: 20px !important;
        display: inline-block !important;
        margin: 4px !important;
    }

    /* FILE UPLOADER */
    .stFileUploader {
        border: 2px dashed var(--border-color) !important;
        border-radius: 12px !important;
        background: var(--bg-surface) !important;
    }

    /* MESSAGES */
    .stAlert, .stSuccess, .stWarning, .stError, .stInfo {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary);
    }

    /* INDICATEUR DE THÈME */
    .theme-indicator {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        padding: 8px 16px !important;
        border-radius: 50px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        z-index: 1000 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Indicateur de thème système
st.markdown("""
<div class="theme-indicator">
    <span>Thème système activé</span>
    <div style="width: 8px; height: 8px; background: var(--primary); border-radius: 50%;"></div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER PROFESSIONNEL
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>MASTER PRO</h1>
    <p>Studio Professionnel de Traitement d'Images • Projet 1 Intelligence Artificielle & Big Data</p>
    <div class="badge-container">
        <span class="badge">🎨 Photo Éditeur</span>
        <span class="badge">⚡ Traitement IA</span>
        <span class="badge">💎 Design Premium</span>
        <span class="badge">📱 Responsive</span>
        <span class="badge">🌓 Auto-Thème</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INITIALISATION SESSION
# -----------------------------
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'applied_filters' not in st.session_state:
    st.session_state.applied_filters = []
if 'processing_details' not in st.session_state:
    st.session_state.processing_details = {}

# -----------------------------
# SIDEBAR PROFESSIONNELLE
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h3 style='color: var(--primary);'>📤 IMPORTATION</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formats supportés: JPG, PNG, BMP",
        label_visibility="collapsed"
    )
    
    # Boutons d'exemple designés
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        example_landscape = st.button("🏔️ Paysage", use_container_width=True, 
                                      help="Charger un exemple de paysage")
    with col_ex2:
        example_portrait = st.button("👤 Portrait", use_container_width=True,
                                    help="Charger un exemple de portrait")
    
    st.markdown("---")
    
    # Contrôles avec design premium
    st.markdown("### 🎛️ CONTROLES")
    
    tab_basic, tab_advanced = st.tabs(["⚡ Basique", "🎨 Avancé"])
    
    with tab_basic:
        st.markdown("#### Transformation")
        rotation = st.selectbox("Rotation", ["Aucune", "90° Droite", "90° Gauche", "180°"])
        flip_h = st.checkbox("Miroir horizontal")
        flip_v = st.checkbox("Miroir vertical")
        
        st.markdown("#### Ajustements")
        brightness = st.slider("Luminosité", 0.0, 2.0, 1.0, 0.1,
                              help="Ajuster la luminosité de l'image")
        contrast = st.slider("Contraste", 0.0, 2.0, 1.0, 0.1,
                            help="Ajuster le contraste de l'image")
        
        st.markdown("#### Filtres")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            grayscale = st.checkbox("Niveaux de gris")
            blur = st.checkbox("Flou artistique")
        with col_f2:
            edges = st.checkbox("Détection contours")
            sharpen = st.checkbox("Accentuation")
        
        if blur:
            blur_amount = st.slider("Intensité flou", 1, 25, 9, 2)
    
    with tab_advanced:
        st.markdown("#### Effets Artistiques")
        effect = st.selectbox(
            "Style",
            ["Aucun", "Sépia Vintage", "Dessin au crayon", "Aquarelle", 
             "Pop Art", "Noir & Blanc Pro", "Rétro 80s", "Effet Cinéma"]
        )
        
        st.markdown("#### Corrections")
        saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
        temperature = st.slider("Température", -100, 100, 0)
        
        st.markdown("#### Effets Spéciaux")
        vignette = st.checkbox("Vignettage")
        if vignette:
            vignette_strength = st.slider("Intensité", 0.1, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    st.markdown("### 💾 EXPORT")
    
    export_format = st.selectbox(
        "Format",
        ["PNG (Haute qualité)", "JPEG (Optimisé)", "TIFF", "BMP"],
        help="Choisissez le format d'export"
    )
    
    if "JPEG" in export_format:
        quality = st.slider("Qualité", 50, 100, 95)
    
    watermark = st.checkbox("Ajouter signature")
    if watermark:
        signature = st.text_input("Signature", "MASTER IABD")
        wm_position = st.selectbox("Position", ["Bas droite", "Bas gauche", "Centre"])
    
    generate_pdf = st.checkbox("Générer rapport PDF", value=True,
                              help="Créer un rapport professionnel en PDF")
    
    st.markdown("---")
    
    # Bouton principal avec style premium
    process_clicked = st.button(
        "🚀 LANCER LE TRAITEMENT", 
        type="primary", 
        use_container_width=True,
        help="Appliquer tous les filtres et effets"
    )
    
    if process_clicked:
        st.balloons()

# -----------------------------
# FONCTIONS DE TRAITEMENT
# -----------------------------
def create_example_image(type_img):
    """Crée une image d'exemple professionnelle"""
    if type_img == "landscape":
        img = Image.new('RGB', (800, 500), color='#1E3A8A')
        draw = ImageDraw.Draw(img)
        
        # Dégradé de ciel
        for i in range(500):
            r = int(30 + (i/500)*100)
            g = int(58 + (i/500)*100)
            b = int(138 + (i/500)*50)
            draw.line([(0, i), (800, i)], fill=(r, g, b))
        
        # Montagnes
        draw.polygon([(100, 300), (400, 100), (700, 300)], fill='#064E3B')
        draw.polygon([(300, 350), (550, 150), (750, 350)], fill='#065F46')
        
        # Soleil
        draw.ellipse([600, 50, 700, 150], fill='#F59E0B')
        
        return np.array(img)
    
    else:  # portrait
        img = Image.new('RGB', (500, 700), color='#F8FAFC')
        draw = ImageDraw.Draw(img)
        
        # Dégradé d'arrière-plan
        for i in range(700):
            r = 248 - int((i/700)*20)
            g = 250 - int((i/700)*20)
            b = 252 - int((i/700)*20)
            draw.line([(0, i), (500, i)], fill=(r, g, b))
        
        # Visage
        draw.ellipse([150, 150, 350, 350], fill='#FED7AA', outline='#F97316', width=2)
        
        # Yeux
        draw.ellipse([200, 220, 240, 260], fill='#1E40AF')
        draw.ellipse([260, 220, 300, 260], fill='#1E40AF')
        
        # Sourcils
        draw.rectangle([195, 190, 245, 200], fill='#92400E')
        draw.rectangle([255, 190, 305, 200], fill='#92400E')
        
        # Nez
        draw.polygon([(250, 260), (240, 290), (260, 290)], fill='#F97316')
        
        # Bouche
        draw.arc([220, 300, 280, 330], 0, 180, fill='#DC2626', width=3)
        
        return np.array(img)

def apply_artistic_effect(image, effect_name):
    """Applique des effets artistiques professionnels"""
    if effect_name == "Aucun":
        return image
    
    elif effect_name == "Sépia Vintage":
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia = cv2.transform(image, sepia_filter)
        sepia = cv2.convertScaleAbs(sepia, alpha=0.9, beta=10)
        noise = np.random.normal(0, 10, sepia.shape)
        return np.clip(sepia + noise, 0, 255).astype(np.uint8)
    
    elif effect_name == "Dessin au crayon":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        pencil = cv2.divide(gray, 255 - blurred, scale=256.0)
        pencil = cv2.convertScaleAbs(pencil, alpha=1.2, beta=20)
        return cv2.cvtColor(pencil, cv2.COLOR_GRAY2RGB)
    
    elif effect_name == "Aquarelle":
        return cv2.stylization(image, sigma_s=60, sigma_r=0.6)
    
    elif effect_name == "Pop Art":
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 6
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        result = result.reshape(image.shape)
        return cv2.convertScaleAbs(result, alpha=1.5, beta=30)
    
    elif effect_name == "Noir & Blanc Pro":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.medianBlur(gray, 3)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    elif effect_name == "Rétro 80s":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 20) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.5, 0, 255)
        retro = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        for i in range(0, retro.shape[0], 3):
            retro[i:i+1, :] = retro[i:i+1, :] * 0.8
        return retro
    
    elif effect_name == "Effet Cinéma":
        h, w = image.shape[:2]
        bar_height = h // 6
        cinematic = image.copy()
        cinematic[:bar_height, :] = 0
        cinematic[h-bar_height:, :] = 0
        cinematic = cv2.convertScaleAbs(cinematic, alpha=0.85, beta=10)
        return cinematic
    
    return image

def add_signature_pro(image_pil, text, position):
    """Ajoute une signature professionnelle"""
    draw = ImageDraw.Draw(image_pil, 'RGBA')
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    positions = {
        "Bas droite": (image_pil.width - text_width - 40, image_pil.height - text_height - 40),
        "Bas gauche": (40, image_pil.height - text_height - 40),
        "Centre": ((image_pil.width - text_width) // 2, (image_pil.height - text_height) // 2)
    }
    
    pos = positions.get(position, positions["Bas droite"])
    
    # Ombre portée
    for offset in [(2,2), (1,1)]:
        draw.text((pos[0]+offset[0], pos[1]+offset[1]), text, 
                  font=font, fill=(0,0,0,150))
    
    # Texte avec léger dégradé
    draw.text(pos, text, font=font, fill=(255,255,255,220))
    
    return image_pil

def create_professional_pdf(original, processed, params, filters_details):
    """Crée un rapport PDF professionnel avec détails des filtres"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # En-tête avec dégradé
    c.setFillColor(HexColor('#8B5CF6'))
    c.rect(0, height-120, width, 120, fill=1, stroke=0)
    
    c.setFillColor(HexColor('#FFFFFF'))
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width/2, height-60, "Rapport Professionnel de Traitement")
    
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, height-90, "Master 2 Intelligence Artificielle & Big Data")
    
    # Informations du traitement
    c.setFillColor(HexColor('#F1F5F9'))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-150, "📋 Informations du Projet")
    
    info_y = height-180
    c.setFont("Helvetica", 12)
    
    infos = [
        ("📅 Date et heure", datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("🎨 Thème", "Mode système détecté"),
        ("📐 Dimensions originales", f"{original.width} × {original.height} px"),
        ("📏 Dimensions finales", f"{processed.width} × {processed.height} px"),
        ("🔢 Nombre de filtres", str(params.get("Nombre de filtres", 0)))
    ]
    
    for label, value in infos:
        c.drawString(60, info_y, f"• {label}: {value}")
        info_y -= 25
    
    # Détails des filtres appliqués
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, info_y-30, "🎛️ Détails des Filtres Appliqués")
    
    filter_y = info_y-60
    c.setFont("Helvetica", 11)
    
    for i, (filter_name, details) in enumerate(filters_details.items()):
        c.drawString(60, filter_y, f"✓ {filter_name}: {details}")
        filter_y -= 20
        if filter_y < 100:  # Nouvelle page si nécessaire
            c.showPage()
            filter_y = height-100
    
    # Images
    img_width = 240
    img_height = 180
    
    # Image originale
    orig_buf = BytesIO()
    original.save(orig_buf, format='PNG', quality=100)
    orig_buf.seek(0)
    
    c.drawImage(ImageReader(orig_buf), 50, filter_y-200, width=img_width, height=img_height)
    c.drawString(50, filter_y-220, "🖼️ Image Originale")
    
    # Image traitée
    proc_buf = BytesIO()
    processed.save(proc_buf, format='PNG', quality=100)
    proc_buf.seek(0)
    
    c.drawImage(ImageReader(proc_buf), width-img_width-50, filter_y-200, width=img_width, height=img_height)
    c.drawString(width-img_width-50, filter_y-220, "✨ Image Traitée")
    
    # Pied de page
    c.setFillColor(HexColor('#94A3B8'))
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width/2, 40, 
                       "AI Vision Pro - Application développée dans le cadre du Master 2 IABD")
    c.drawCentredString(width/2, 25, 
                       "© 2024 - Tous droits réservés")
    
    c.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# TRAITEMENT PRINCIPAL
# -----------------------------
image_to_process = None

if uploaded_file:
    image = Image.open(uploaded_file)
    image_to_process = np.array(image.convert('RGB'))
    image_pil = image
    
elif example_landscape or example_portrait:
    if example_landscape:
        image_to_process = create_example_image("landscape")
    else:
        image_to_process = create_example_image("portrait")
    image_pil = Image.fromarray(image_to_process)

if image_to_process is not None and (uploaded_file or example_landscape or example_portrait):
    # Simuler un traitement avec animation
    with st.spinner("🔄 Traitement en cours..."):
        time.sleep(0.8)
        
        # Sauvegarder l'original
        original_img = image_to_process.copy()
        st.session_state.original_image = original_img
        
        # Réinitialiser les filtres
        applied_filters = []
        filters_details = {}
        
        # Démarrer le traitement
        processed = original_img.copy()
        
        # 1. Rotation
        if rotation != "Aucune":
            if rotation == "90° Droite":
                processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
                applied_filters.append("Rotation 90° Droite")
                filters_details["Rotation"] = "90° vers la droite"
            elif rotation == "90° Gauche":
                processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
                applied_filters.append("Rotation 90° Gauche")
                filters_details["Rotation"] = "90° vers la gauche"
            elif rotation == "180°":
                processed = cv2.rotate(processed, cv2.ROTATE_180)
                applied_filters.append("Rotation 180°")
                filters_details["Rotation"] = "180° (retournement complet)"
        
        # 2. Miroir
        if flip_h:
            processed = cv2.flip(processed, 1)
            applied_filters.append("Miroir horizontal")
            filters_details["Miroir horizontal"] = "Image retournée horizontalement"
        if flip_v:
            processed = cv2.flip(processed, 0)
            applied_filters.append("Miroir vertical")
            filters_details["Miroir vertical"] = "Image retournée verticalement"
        
        # 3. Luminosité
        if brightness != 1.0:
            adjustment = (brightness - 1) * 60
            processed = cv2.convertScaleAbs(processed, alpha=1.0, beta=adjustment)
            applied_filters.append(f"Luminosité: {brightness}")
            filters_details["Luminosité"] = f"Multiplicateur: {brightness} (ajustement: {adjustment:+})"
        
        # 4. Contraste
        if contrast != 1.0:
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=0)
            applied_filters.append(f"Contraste: {contrast}")
            filters_details["Contraste"] = f"Multiplicateur: {contrast}"
        
        # 5. Niveaux de gris
        if grayscale:
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            applied_filters.append("Niveaux de gris")
            filters_details["Niveaux de gris"] = "Conversion RGB vers niveaux de gris"
        
        # 6. Flou
        if blur and 'blur_amount' in locals():
            ksize = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)
            applied_filters.append(f"Flou gaussien")
            filters_details["Flou gaussien"] = f"Kernel size: {ksize}×{ksize}, Sigma: 0"
        
        # 7. Accentuation
        if sharpen:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
            applied_filters.append("Accentuation")
            filters_details["Accentuation"] = "Kernel de netteté 3×3 appliqué"
        
        # 8. Contours
        if edges:
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                gray = processed
            edges_img = cv2.Canny(gray, 100, 200)
            processed = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB)
            applied_filters.append("Détection de contours")
            filters_details["Détection de contours"] = "Algorithme Canny (seuils: 100-200)"
        
        # 9. Effet artistique
        if effect != "Aucun":
            processed = apply_artistic_effect(processed, effect)
            applied_filters.append(effect)
            filters_details["Effet artistique"] = effect
        
        # 10. Saturation
        if saturation != 1.0 and len(processed.shape) == 3:
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            applied_filters.append(f"Saturation: {saturation}");
            filters_details["Saturation"] = f"Multiplicateur: {saturation}"
        
        # 11. Température
        if temperature != 0 and len(processed.shape) == 3:
            if temperature > 0:
                processed[:,:,0] = np.clip(processed[:,:,0] + temperature//2, 0, 255)
                temp_desc = f"Chaud (+{temperature})"
            else:
                processed[:,:,2] = np.clip(processed[:,:,2] + abs(temperature)//2, 0, 255)
                temp_desc = f"Froid ({temperature})"
            applied_filters.append(f"Température: {temperature}")
            filters_details["Température couleur"] = temp_desc
        
        # 12. Vignettage
        if 'vignette' in locals() and vignette:
            h, w = processed.shape[:2]
            kernel_x = cv2.getGaussianKernel(w, w/(vignette_strength*10))
            kernel_y = cv2.getGaussianKernel(h, h/(vignette_strength*10))
            kernel = kernel_y * kernel_x.T
            mask = kernel / kernel.max()
            mask = np.power(mask, 0.8)
            if len(processed.shape) == 3:
                mask = mask[:, :, np.newaxis]
            processed = (processed * mask).astype(np.uint8)
            applied_filters.append("Vignettage")
            filters_details["Vignettage"] = f"Intensité: {vignette_strength}"
        
        # Convertir en PIL
        if len(processed.shape) == 2:
            processed_pil = Image.fromarray(processed)
        else:
            processed_pil = Image.fromarray(processed)
        
        # 13. Signature
        if watermark and 'signature' in locals():
            processed_pil = add_signature_pro(processed_pil, signature, wm_position)
            applied_filters.append(f"Signature: {signature}")
            filters_details["Signature"] = f"'{signature}' - Position: {wm_position}"
        
        # Sauvegarder dans la session
        st.session_state.current_image = processed_pil
        st.session_state.applied_filters = applied_filters
        st.session_state.processing_details = filters_details
        st.session_state.processed_images.append({
            'time': datetime.now(),
            'filters': applied_filters.copy(),
            'filters_details': filters_details.copy(),
            'image': processed_pil.copy()
        })
        
        st.success("✅ Traitement terminé avec succès!")

# -----------------------------
# AFFICHAGE DES RÉSULTATS
# -----------------------------
if st.session_state.current_image is not None:
    processed_pil = st.session_state.current_image
    
    # Métriques visuelles
    st.markdown("## 📊 RÉSULTATS DU TRAITEMENT")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**LARGEUR**")
        st.markdown(f"<h2>{processed_pil.width}px</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**HAUTEUR**")
        st.markdown(f"<h2>{processed_pil.height}px</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**FORMAT**")
        st.markdown(f"<h2>{processed_pil.mode}</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**FILTRES**")
        st.markdown(f"<h2>{len(st.session_state.applied_filters)}</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Images avant/après
    st.markdown("## 🖼️ COMPARAISON")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 ORIGINALE")
        if st.session_state.original_image is not None:
            original_pil = Image.fromarray(st.session_state.original_image)
            st.image(original_pil, use_column_width=True)
            st.caption(f"Dimensions: {original_pil.width}×{original_pil.height}px • Mode: {original_pil.mode}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_img2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown("### ✨ TRAITÉE")
        st.image(processed_pil, use_column_width=True)
        st.caption(f"Filtres appliqués: {len(st.session_state.applied_filters)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # -----------------------------
    # DÉTAILS DES TRAITEMENTS
    # -----------------------------
    st.markdown("---")
    st.markdown("## 🔧 DÉTAILS DES TRAITEMENTS APPLIQUÉS")
    
    if st.session_state.applied_filters:
        st.markdown("### 📋 Chronologie des Traitements")
        
        col_details1, col_details2 = st.columns([2, 1])
        
        with col_details1:
            st.markdown('<div class="details-card">', unsafe_allow_html=True)
            st.markdown("#### 🎛️ Liste des Filtres")
            
            for i, filt in enumerate(st.session_state.applied_filters, 1):
                st.markdown(f"""
                <div class="filter-item">
                    <div style="width: 40px; height: 40px; border-radius: 10px; background: linear-gradient(135deg, var(--primary), var(--secondary)); display: flex; align-items: center; justify-content: center; margin-right: 15px; color: white; font-size: 1.2rem;">{i}</div>
                    <div>
                        <strong>{filt}</strong>
                        <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 4px;">
                            {st.session_state.processing_details.get(filt.split(':')[0], 'Appliqué avec succès')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_details2:
            st.markdown('<div class="details-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Statistiques")
            
            stats = {
                "Total filtres": len(st.session_state.applied_filters),
                "Transformations": len([f for f in st.session_state.applied_filters 
                                      if "Rotation" in f or "Miroir" in f]),
                "Ajustements": len([f for f in st.session_state.applied_filters 
                                   if "Luminosité" in f or "Contraste" in f or "Saturation" in f]),
                "Effets artistiques": len([f for f in st.session_state.applied_filters 
                                         if any(x in f for x in ["Sépia", "Crayon", "Aquarelle", "Pop Art", "Rétro", "Cinéma"])])
            }
            
            for key, value in stats.items():
                st.markdown(f"""
                <div style="margin: 10px 0; padding: 12px; background: var(--card-bg); 
                    border-radius: 10px; border-left: 4px solid var(--accent);">
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">{key}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary);">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tags des filtres
        st.markdown("### 🏷️ Tags des Filtres")
        filter_tags = " ".join([f'<span class="filter-tag">{filt}</span>' for filt in st.session_state.applied_filters[:10]])
        st.markdown(f'<div style="margin: 1rem 0;">{filter_tags}</div>', unsafe_allow_html=True)
        
        # Résumé technique
        st.markdown('<div class="details-card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Résumé Technique")
        
        tech_details = {
            "Algorithme de traitement": "OpenCV + NumPy",
            "Méthode de convolution": "Kernel-based filtering",
            "Conversion couleur": "RGB ↔ HSV ↔ GRAY",
            "Détection contours": "Algorithme de Canny",
            "Traitement batch": "Séquentiel en mémoire",
            "Optimisation": "Vectorisation NumPy"
        }
        
        for key, value in tech_details.items():
            st.markdown(f"• **{key}**: {value}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # -----------------------------
    # EXPORTATION
    # -----------------------------
    st.markdown("---")
    st.markdown("## 💾 EXPORTATION PROFESSIONNELLE")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Export image
        buffer = io.BytesIO()
        
        if "PNG" in export_format:
            processed_pil.save(buffer, format="PNG")
            mime = "image/png"
            ext = "png"
        elif "JPEG" in export_format:
            quality_val = quality if 'quality' in locals() else 95
            processed_pil.save(buffer, format="JPEG", quality=quality_val, optimize=True)
            mime = "image/jpeg"
            ext = "jpg"
        elif "TIFF" in export_format:
            processed_pil.save(buffer, format="TIFF")
            mime = "image/tiff"
            ext = "tiff"
        else:
            processed_pil.save(buffer, format="BMP")
            mime = "image/bmp"
            ext = "bmp"
        
        buffer.seek(0)
        
        st.download_button(
            label=f"📥 TÉLÉCHARGER (.{ext})",
            data=buffer,
            file_name=f"image_traitee_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}",
            mime=mime,
            use_container_width=True
        )
    
    with col_exp2:
        # Rapport PDF avec détails
        if generate_pdf and st.session_state.original_image is not None:
            original_pil_for_pdf = Image.fromarray(st.session_state.original_image)
            params = {
                "Nombre de filtres": len(st.session_state.applied_filters),
                "Rotation": rotation,
                "Effet artistique": effect,
                "Export format": export_format.split()[0],
                "Signature": "Oui" if watermark else "Non",
                "Thème": "Système détecté"
            }
            
            pdf_buffer = create_professional_pdf(
                original_pil_for_pdf, 
                processed_pil, 
                params,
                st.session_state.processing_details
            )
            
            st.download_button(
                label="📄 RAPPORT COMPLET (PDF)",
                data=pdf_buffer,
                file_name=f"rapport_traitement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with col_exp3:
        # Actions
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🔄 Nouveau", use_container_width=True):
                for key in ['current_image', 'original_image', 'applied_filters', 'processing_details']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col_act2:
            if st.button("📋 Copier détails", use_container_width=True):
                details_json = {
                    "filters": st.session_state.applied_filters,
                    "filters_details": st.session_state.processing_details,
                    "timestamp": datetime.now().isoformat(),
                    "image_size": f"{processed_pil.width}x{processed_pil.height}"
                }
                st.code(json.dumps(details_json, indent=2, ensure_ascii=False))
                st.success("✅ Détails copiés !")
    
    # Galerie d'effets
    st.markdown("---")
    st.markdown("## 🎨 GALERIE D'EFFETS")
    
    col_gal1, col_gal2, col_gal3 = st.columns(3)
    
    with col_gal1:
        if st.button("🖼️ Afficher Sépia", use_container_width=True):
            sepia_img = apply_artistic_effect(st.session_state.original_image, "Sépia Vintage")
            st.image(sepia_img, caption="Effet Sépia Vintage", use_column_width=True)
    
    with col_gal2:
        if st.button("✏️ Afficher Crayon", use_container_width=True):
            pencil_img = apply_artistic_effect(st.session_state.original_image, "Dessin au crayon")
            st.image(pencil_img, caption="Dessin au Crayon", use_column_width=True)
    
    with col_gal3:
        if st.button("🎭 Afficher Pop Art", use_container_width=True):
            popart_img = apply_artistic_effect(st.session_state.original_image, "Pop Art")
            st.image(popart_img, caption="Style Pop Art", use_column_width=True)

else:
    # -----------------------------
    # PAGE D'ACCUEIL
    # -----------------------------
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem;'>
        <h2 style='color: var(--text-primary); margin-bottom: 1.5rem;'>🚀 BIENVENUE DANS AI VISION PRO</h2>
        <p style='color: var(--text-secondary); font-size: 1.2rem; max-width: 800px; margin: 0 auto 3rem auto; line-height: 1.6;'>
        Studio professionnel de traitement d'images inspiré des standards Canva et Photoshop.
        Développé avec les dernières technologies pour le Master 2 Intelligence Artificielle & Big Data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features avec design premium
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ TRAITEMENTS DÉTAILLÉS")
        st.markdown("""
        • Chronologie complète
        • Détails techniques
        • Statistiques avancées
        • Tags visuels
        • Résumé automatique
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_f2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 💎 DESIGN CANVA/PRO")
        st.markdown("""
        • Interface premium
        • Thème auto-adaptatif
        • Animations fluides
        • Design responsive
        • UX optimisée
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_f3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📄 RAPPORTS COMPLETS")
        st.markdown("""
        • PDF professionnel
        • Détails des filtres
        • Comparaison visuelle
        • Métriques techniques
        • Historique complet
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📝 GUIDE DE DÉMARRAGE
    
    1. **Importez** une image depuis la barre latérale
    2. **Ajustez** les paramètres dans les onglets
    3. **Visualisez** les résultats avec détails complets
    4. **Exportez** avec rapports professionnels
    
    *L'interface s'adapte automatiquement au thème de votre système*
    """)

# -----------------------------
# FOOTER PROFESSIONNEL
# -----------------------------
st.markdown("""
<div class="footer">
    <h3>🎓 PROJET 1 INTELLIGENCE ARTIFICIELLE & BIG DATA</h3>
    <p>Projet1 OpenCV & Streamlit - Studio de Traitement d'Images Intelligent</p>
    <div style="margin: 1.5rem 0; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
        <span style="padding: 8px 16px; background: var(--card-bg); 
            border-radius: 10px; font-size: 0.9rem; color: var(--text-secondary);">
            🐍 Python
        </span>
        <span style="padding: 8px 16px; background: var(--card-bg); 
            border-radius: 10px; font-size: 0.9rem; color: var(--text-secondary);">
            📷 OpenCV
        </span>
        <span style="padding: 8px 16px; background: var(--card-bg); 
            border-radius: 10px; font-size: 0.9rem; color: var(--text-secondary);">
            🚀 Streamlit
        </span>
        <span style="padding: 8px 16px; background: var(--card-bg); 
            border-radius: 10px; font-size: 0.9rem; color: var(--text-secondary);">
            🎨 Pillow
        </span>
    </div>
    <p style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 1rem;">
    © 2025 Projet1 Pro • Dernière mise à jour: """ + datetime.now().strftime("%d/%m/%Y %H:%M") + """
    </p>
</div>
""", unsafe_allow_html=True)

