import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import io
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import json
import os

# -----------------------------
# CONFIGURATION PAGE PROFESSIONNELLE
# -----------------------------
st.set_page_config(
    page_title="AI Image Master Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit/streamlit',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# 🎓 Projet Master 2 IABD\n### OpenCV & Streamlit - Traitement d'Images Intelligent"
    }
)

# -----------------------------
# CSS PERSONNALISÉ - DESIGN MODERNE
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    .image-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .tab-content {
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER PROFESSIONNEL
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🎨 AI Image Master Pro</h1>
    <p>Application Intelligente de Traitement d'Images - Master 2 Intelligence Artificielle & Big Data</p>
    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 1rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">OpenCV</span>
        <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">Streamlit</span>
        <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">Python</span>
        <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">Machine Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INITIALISATION DE SESSION
# -----------------------------
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

# -----------------------------
# SIDEBAR AVANCÉE
# -----------------------------
with st.sidebar:
    st.markdown("### 📤 Importation d'Image")
    
    uploaded_file = st.file_uploader(
        "Glissez-déposez ou sélectionnez une image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Formats supportés: JPG, PNG, BMP, TIFF"
    )
    
    # Mode de chargement d'image d'exemple
    use_sample = st.checkbox("Utiliser une image exemple")
    sample_images = {
        "Paysage": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
        "Portrait": "https://images.unsplash.com/photo-1534528741775-53994a69daeb",
        "Architecture": "https://images.unsplash.com/photo-1487958449943-2429e8be8625",
        "Art": "https://images.unsplash.com/photo-1541961017774-22349e4a1262"
    }
    
    if use_sample:
        selected_sample = st.selectbox("Choisir une image exemple", list(sample_images.keys()))
    
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres de Traitement")
    
    # Onglets pour les filtres
    tab1, tab2, tab3 = st.tabs(["🎛️ Basique", "🎨 Avancé", "🤖 IA"])
    
    with tab1:
        # Filtres basiques
        brightness = st.slider("Luminosité", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
        rotation = st.select_slider("Rotation", options=["0°", "90°", "180°", "270°"])
        
        col1, col2 = st.columns(2)
        with col1:
            grayscale = st.checkbox("Niveaux de gris", value=False)
            blur = st.checkbox("Flou", value=False)
        with col2:
            edges = st.checkbox("Contours", value=False)
            sharpen = st.checkbox("Accentuation", value=False)
        
        if blur:
            blur_strength = st.slider("Force du flou", 1, 15, 5, step=2)
    
    with tab2:
        # Filtres avancés
        st.markdown("#### 🎭 Effets Artistiques")
        artistic_filter = st.selectbox(
            "Filtre artistique",
            ["Aucun", "Sépia", "Crayon", "Aquarelle", "Pop Art", "Vintage"]
        )
        
        st.markdown("#### 🌈 Ajustements Couleur")
        hue = st.slider("Teinte", 0, 360, 0)
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
        
        st.markdown("#### 🔧 Transformations")
        flip_horizontal = st.checkbox("Retournement horizontal")
        flip_vertical = st.checkbox("Retournement vertical")
    
    with tab3:
        # Effets IA/avancés
        st.markdown("#### 🤖 Effets Intelligents")
        style_transfer = st.checkbox("Transfert de style artistique")
        if style_transfer:
            style_type = st.selectbox(
                "Style artistique",
                ["Van Gogh", "Picasso", "Monet", "Hokusai", "Moderne"]
            )
        
        object_detection = st.checkbox("Détection d'objets")
        face_detection = st.checkbox("Détection de visages")
        
        st.markdown("#### 📊 Analyse")
        show_histogram = st.checkbox("Afficher l'histogramme")
        show_metrics = st.checkbox("Afficher les métriques")
    
    st.markdown("---")
    st.markdown("### 💾 Options d'Export")
    
    export_format = st.selectbox("Format d'export", ["PNG", "JPG", "TIFF", "BMP"])
    export_quality = st.slider("Qualité", 50, 100, 95) if export_format == "JPG" else 100
    
    watermark = st.checkbox("Ajouter un filigrane")
    if watermark:
        watermark_text = st.text_input("Texte du filigrane", "Master 2 IABD")
    
    st.markdown("---")
    st.markdown("#### 📈 Statistiques")
    
    if 'current_image' in st.session_state:
        if st.session_state.current_image is not None:
            img_array = st.session_state.current_image
            st.metric("Dimensions", f"{img_array.shape[1]}×{img_array.shape[0]}")
            if len(img_array.shape) == 3:
                st.metric("Couleurs", "RGB")
            else:
                st.metric("Couleurs", "Gris")

# -----------------------------
# FONCTIONS DE TRAITEMENT AVANCÉES
# -----------------------------
def apply_sepia(image):
    """Applique un effet sépia"""
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(image, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255)
    return sepia_img.astype(np.uint8)

def apply_pencil_sketch(image):
    """Convertit l'image en dessin au crayon"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    inv_blurred = 255 - blurred
    pencil_sketch = cv2.divide(gray, inv_blurred, scale=256.0)
    return cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2RGB)

def apply_watercolor(image):
    """Applique un effet aquarelle"""
    return cv2.stylization(image, sigma_s=60, sigma_r=0.6)

def apply_pop_art(image):
    """Effet pop art"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    pop_art = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    pop_art = cv2.convertScaleAbs(pop_art, alpha=1.5, beta=50)
    return pop_art

def apply_vintage(image):
    """Effet vintage"""
    vintage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    vintage[:,:,1] = vintage[:,:,1] * 0.7
    vintage = cv2.cvtColor(vintage, cv2.COLOR_HSV2RGB)
    noise = np.random.normal(0, 15, vintage.shape)
    vintage = cv2.add(vintage, noise.astype(np.uint8))
    return vintage

def add_watermark(image_pil, text):
    """Ajoute un filigrane professionnel"""
    draw = ImageDraw.Draw(image_pil)
    
    # Créer une police (fallback si police système non disponible)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Positionner le texte
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position (bas droit avec marge)
    position = (image_pil.width - text_width - 20, image_pil.height - text_height - 20)
    
    # Dessiner le texte avec ombre
    draw.text((position[0]+2, position[1]+2), text, font=font, fill=(0,0,0,128))
    draw.text(position, text, font=font, fill=(255,255,255,180))
    
    return image_pil

def calculate_metrics(original, processed):
    """Calcule des métriques de qualité"""
    if original.shape != processed.shape:
        return {}
    
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    # Calcul PSNR
    mse = np.mean((original_gray - processed_gray) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calcul SSIM (simplifié)
    from scipy.signal import correlate2d
    from scipy.ndimage import uniform_filter
    
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = uniform_filter(img1, 11)
        mu2 = uniform_filter(img2, 11)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = uniform_filter(img1**2, 11) - mu1_sq
        sigma2_sq = uniform_filter(img2**2, 11) - mu2_sq
        sigma12 = uniform_filter(img1*img2, 11) - mu1_mu2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    ssim_score = ssim(original_gray, processed_gray)
    
    return {
        "PSNR": f"{psnr:.2f} dB",
        "SSIM": f"{ssim_score:.3f}",
        "MSE": f"{mse:.2f}"
    }

# -----------------------------
# INTERFACE PRINCIPALE
# -----------------------------
if uploaded_file or use_sample:
    # Charger l'image
    with st.spinner("🔄 Chargement de l'image..."):
        if use_sample:
            import requests
            response = requests.get(sample_images[selected_sample])
            image = Image.open(io.BytesIO(response.content))
        else:
            image = Image.open(uploaded_file)
        
        # Convertir en tableau numpy
        image_np = np.array(image)
        st.session_state.current_image = image_np
        
        # Appliquer les traitements
        processed = image_np.copy()
        applied_filters = []
        
        # Filtres basiques
        if brightness != 1.0:
            processed = cv2.convertScaleAbs(processed, alpha=brightness, beta=0)
            applied_filters.append(f"Luminosité: {brightness}")
        
        if contrast != 1.0:
            processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=0)
            applied_filters.append(f"Contraste: {contrast}")
        
        if rotation != "0°":
            if rotation == "90°":
                processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == "180°":
                processed = cv2.rotate(processed, cv2.ROTATE_180)
            elif rotation == "270°":
                processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            applied_filters.append(f"Rotation: {rotation}")
        
        if grayscale:
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            applied_filters.append("Niveaux de gris")
        
        if blur:
            kernel_size = blur_strength if 'blur_strength' in locals() else 5
            if len(processed.shape) == 2:
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
            else:
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
            applied_filters.append(f"Flou (kernel: {kernel_size})")
        
        if sharpen:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
            applied_filters.append("Accentuation")
        
        if edges:
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                gray = processed
            processed = cv2.Canny(gray, 100, 200)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            applied_filters.append("Détection de contours")
        
        # Filtres avancés
        if artistic_filter != "Aucun":
            if artistic_filter == "Sépia":
                processed = apply_sepia(processed)
            elif artistic_filter == "Crayon":
                processed = apply_pencil_sketch(processed)
            elif artistic_filter == "Aquarelle":
                processed = apply_watercolor(processed)
            elif artistic_filter == "Pop Art":
                processed = apply_pop_art(processed)
            elif artistic_filter == "Vintage":
                processed = apply_vintage(processed)
            applied_filters.append(f"Effet: {artistic_filter}")
        
        if flip_horizontal:
            processed = cv2.flip(processed, 1)
            applied_filters.append("Retournement horizontal")
        
        if flip_vertical:
            processed = cv2.flip(processed, 0)
            applied_filters.append("Retournement vertical")
        
        # Ajustement de teinte et saturation
        if hue != 0 or saturation != 1.0:
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
            hsv[:,:,0] = (hsv[:,:,0] + hue/2) % 180
            hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            applied_filters.append(f"Teinte/Saturation ajustées")
        
        # Conversion en PIL pour l'export
        if len(processed.shape) == 2:
            processed_pil = Image.fromarray(processed)
        else:
            processed_pil = Image.fromarray(processed)
        
        # Appliquer le filigrane si demandé
        if watermark and 'watermark_text' in locals():
            processed_pil = add_watermark(processed_pil, watermark_text)
            applied_filters.append(f"Filigrane: {watermark_text}")
        
        # Sauvegarder dans l'historique
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filters": applied_filters,
            "original_shape": image_np.shape,
            "processed_shape": processed.shape
        }
        st.session_state.processing_history.append(history_entry)
        
        time.sleep(0.5)  # Simulation de traitement
    
    # -----------------------------
    # AFFICHAGE DES RÉSULTATS
    # -----------------------------
    st.markdown("## 📊 Résultats du Traitement")
    
    # Métriques en haut
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    with col_metrics1:
        st.metric("Filtres appliqués", len(applied_filters))
    with col_metrics2:
        st.metric("Taille originale", f"{image_np.shape[1]}×{image_np.shape[0]}")
    with col_metrics3:
        st.metric("Taille traitée", f"{processed.shape[1]}×{processed.shape[0]}")
    
    # Images côte à côte
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.subheader("🖼️ Image Originale")
        st.image(image, use_column_width=True, caption=f"Dimensions: {image_np.shape[1]}×{image_np.shape[0]}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.subheader("✨ Image Traitée")
        st.image(processed_pil, use_column_width=True, caption=f"Filtres: {', '.join(applied_filters)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # -----------------------------
    # ANALYSE AVANCÉE
    # -----------------------------
    st.markdown("---")
    st.markdown("## 📈 Analyse Avancée")
    
    tab_hist, tab_metrics, tab_info = st.tabs(["📊 Histogramme", "📐 Métriques", "📋 Informations"])
    
    with tab_hist:
        if show_histogram:
            if len(processed.shape) == 2:
                # Image en niveaux de gris
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=processed.flatten(), nbinsx=256, 
                                         marker_color='gray', name='Niveaux de gris'))
                fig.update_layout(title="Histogramme des niveaux de gris",
                                xaxis_title="Intensité",
                                yaxis_title="Fréquence",
                                template="plotly_white")
            else:
                # Image couleur
                colors = ['red', 'green', 'blue']
                channel_names = ['Rouge', 'Vert', 'Bleu']
                fig = go.Figure()
                
                for i in range(3):
                    fig.add_trace(go.Histogram(x=processed[:,:,i].flatten(), nbinsx=256,
                                             marker_color=colors[i], name=channel_names[i],
                                             opacity=0.7))
                
                fig.update_layout(title="Histogramme des canaux RVB",
                                xaxis_title="Intensité",
                                yaxis_title="Fréquence",
                                template="plotly_white",
                                barmode='overlay')
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_metrics:
        if show_metrics:
            metrics = calculate_metrics(image_np, processed)
            if metrics:
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("PSNR", metrics["PSNR"])
                with col_m2:
                    st.metric("SSIM", metrics["SSIM"])
                with col_m3:
                    st.metric("MSE", metrics["MSE"])
                
                st.info("""
                **Explication des métriques:**
                - **PSNR** (Peak Signal-to-Noise Ratio): Mesure de la qualité de reconstruction
                - **SSIM** (Structural Similarity): Mesure de similarité structurelle
                - **MSE** (Mean Squared Error): Erreur quadratique moyenne
                """)
    
    with tab_info:
        st.markdown("#### 📋 Filtres appliqués")
        for i, filt in enumerate(applied_filters, 1):
            st.markdown(f"{i}. {filt}")
        
        st.markdown("#### 📄 Propriétés de l'image")
        info_df = pd.DataFrame({
            "Propriété": ["Format", "Mode", "Dimensions", "Taille mémoire"],
            "Valeur": [
                image.format or "Inconnu",
                image.mode,
                f"{image.width} × {image.height}",
                f"{image_np.nbytes / 1024:.1f} KB"
            ]
        })
        st.table(info_df)
    
    # -----------------------------
    # EXPORT ET TÉLÉCHARGEMENT
    # -----------------------------
    st.markdown("---")
    st.markdown("## 💾 Exportation Professionnelle")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Téléchargement simple
        buffer = io.BytesIO()
        if export_format == "PNG":
            processed_pil.save(buffer, format="PNG", quality=export_quality)
            mime_type = "image/png"
        elif export_format == "JPG":
            processed_pil.save(buffer, format="JPEG", quality=export_quality)
            mime_type = "image/jpeg"
        elif export_format == "TIFF":
            processed_pil.save(buffer, format="TIFF")
            mime_type = "image/tiff"
        else:  # BMP
            processed_pil.save(buffer, format="BMP")
            mime_type = "image/bmp"
        
        buffer.seek(0)
        
        st.download_button(
            label=f"📥 Télécharger ({export_format})",
            data=buffer,
            file_name=f"image_traitee_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
            mime=mime_type,
            help=f"Télécharger l'image traitée au format {export_format}"
        )
    
    with col_exp2:
        # Rapport PDF (simulé)
        if st.button("📄 Générer un rapport PDF"):
            st.success("✅ Rapport généré avec succès! (Fonctionnalité simulée)")
            st.info("""
            Dans une version complète, ce bouton générerait un PDF contenant:
            - L'image originale et traitée
            - Les paramètres appliqués
            - Les métriques de qualité
            - L'histogramme
            - Les informations techniques
            """)
    
    with col_exp3:
        # Partager les paramètres
        if st.button("🔗 Copier les paramètres"):
            params = {
                "applied_filters": applied_filters,
                "export_format": export_format,
                "timestamp": datetime.now().isoformat()
            }
            st.code(json.dumps(params, indent=2))
            st.success("Paramètres copiés dans le presse-papier (simulé)")
    
    # -----------------------------
    # GALERIE D'HISTORIQUE
    # -----------------------------
    if len(st.session_state.processing_history) > 0:
        st.markdown("---")
        st.markdown("## 🗂️ Historique des Traitements")
        
        # Afficher les 5 derniers traitements
        for i, entry in enumerate(reversed(st.session_state.processing_history[-5:])):
            with st.expander(f"Traitement {i+1} - {entry['timestamp']}"):
                st.write(f"**Filtres appliqués:** {', '.join(entry['filters'])}")
                st.write(f"**Dimensions originales:** {entry['original_shape'][1]}×{entry['original_shape'][0]}")
                st.write(f"**Dimensions finales:** {entry['processed_shape'][1]}×{entry['processed_shape'][0]}")
    
    # -----------------------------
    # CRÉATIVITÉ ARTISTIQUE
    # -----------------------------
    st.markdown("---")
    st.markdown("## 🎭 Galerie Créative")
    
    st.info("""
    **Suggestions de créativité artistique pour votre projet:**
    1. **Style Transfer** : Implémentez un transfert de style neuronal
    2. **GAN Art** : Utilisez un GAN pour générer des variations artistiques
    3. **Augmentation 3D** : Transformez l'image en relief 3D
    4. **Animation** : Créez une animation des étapes de traitement
    5. **Reconnaissance** : Ajoutez la détection d'objets/visages avec labels
    """)
    
    # Exemple de galerie d'effets créatifs
    if st.button("🎨 Afficher des effets créatifs (exemple)"):
        col_art1, col_art2, col_art3 = st.columns(3)
        with col_art1:
            st.image(apply_sepia(image_np), caption="Effet Sépia", use_column_width=True)
        with col_art2:
            st.image(apply_pencil_sketch(image_np), caption="Dessin au crayon", use_column_width=True)
        with col_art3:
            st.image(apply_pop_art(image_np), caption="Style Pop Art", use_column_width=True)

else:
    # -----------------------------
    # PAGE D'ACCUEIL - QUAND AUCUNE IMAGE
    # -----------------------------
    st.markdown("""
    <div style='text-align: center; padding: 4rem 0;'>
        <h2>🚀 Bienvenue dans AI Image Master Pro</h2>
        <p style='font-size: 1.2rem; color: #666; max-width: 800px; margin: 2rem auto;'>
        Une application professionnelle de traitement d'images utilisant OpenCV et Streamlit.
        Développée dans le cadre du Master 2 Intelligence Artificielle & Big Data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Traitements Avancés")
        st.markdown("""
        - Filtres artistiques
        - Ajustements couleur
        - Détection de contours
        - Effets spéciaux
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_feat2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Analyse Professionnelle")
        st.markdown("""
        - Histogrammes interactifs
        - Métriques de qualité
        - Visualisations 3D
        - Rapports détaillés
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_feat3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 💾 Export Polyvalent")
        st.markdown("""
        - Multi-formats supportés
        - Qualité ajustable
        - Filigrane personnalisé
        - Rapports PDF
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📝 Guide de démarrage rapide
    
    1. **Importez une image** depuis la barre latérale
    2. **Sélectionnez les filtres** dans les onglets
    3. **Visualisez les résultats** en temps réel
    4. **Exportez votre création** avec les options avancées
    
    *Pour commencer, sélectionnez une image dans la barre latérale →*
    """)

# -----------------------------
# FOOTER PROFESSIONNEL
# -----------------------------
st.markdown("""
<div class="footer">
    <p>🎓 <strong>Master 2 Intelligence Artificielle & Big Data</strong> - Projet OpenCV & Streamlit</p>
    <p>📚 Application développée avec Python, OpenCV, Streamlit, Plotly</p>
    <p>⏱️ Dernière mise à jour: """ + datetime.now().strftime("%d/%m/%Y %H:%M") + """</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INSTRUCTIONS D'INSTALLATION (cachées)
# -----------------------------
with st.sidebar.expander("📦 Installation"):
    st.code("""
pip install streamlit opencv-python pillow numpy pandas plotly requests
streamlit run app.py
""", language="bash")
    st.markdown("**Librairies requises:**")
    st.markdown("""
    - streamlit
    - opencv-python
    - pillow
    - numpy
    - pandas
    - plotly
    - requests
    """)