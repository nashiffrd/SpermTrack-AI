import streamlit as st
import pandas as pd
import cv2
import os
import tempfile
import numpy as np
from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from models.motility_analyzer import run_motility_analysis
from models.morphology_analyzer import run_morphology_analysis

# ==========================================
# CONFIG & STYLE
# ==========================================
st.set_page_config(page_title="Sperm Analysis AI", layout="wide")

st.markdown("""
    <style>
    .main-result {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #007bff;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if 'tracks_df' not in st.session_state: st.session_state.tracks_df = None
if 'prepared_video' not in st.session_state: st.session_state.prepared_video = None
if 'motility_results' not in st.session_state: st.session_state.motility_results = None
if 'morphology_results' not in st.session_state: st.session_state.morphology_results = None

# ==========================================
# TAB DEFINITION
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Halaman Awal", 
    "⚙️ Data Loader & Processing", 
    "🔬 Analysis Process", 
    "📊 Summary Dashboard"
])

# ------------------------------------------
# TAB 1: HALAMAN AWAL
# ------------------------------------------
with tab1:
    st.title("DETEKSI ABNORMALITAS MOTILITY DAN MORFOLOGI SPERMATOZOA")
    st.subheader("Sistem Analisis Otomatis Berbasis Deep Learning")
    st.markdown("""
    ### Gambaran Umum Project
    Aplikasi ini dirancang untuk membantu klinisi dalam menganalisis kualitas semen secara objektif. 
    Menggunakan arsitektur **3D-CNN** untuk pergerakan dan **EfficientNetV2S** untuk bentuk fisik.
    
    **Cara Penggunaan:**
    1. Upload video pada tab **Data Loader**.
    2. Lakukan preprocessing dan tracking.
    3. Jalankan analisis motilitas dan morfologi pada tab **Analysis Process**.
    4. Lihat kesimpulan akhir pada tab **Summary Dashboard**.
    """)
    
# ------------------------------------------
# TAB 2: DATA LOADER & PROCESSING
# ------------------------------------------
with tab2:
    st.header("Upload & Image Processing")
    video_file = st.file_uploader("Pilih Video Sperma", type=['mp4', 'avi'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        if st.button("Jalankan Preprocessing & Tracking 🚀"):
            with st.status("Memproses data...", expanded=True) as status:
                st.write("Mengonversi Grayscale & Contrast...")
                temp_dir = tempfile.mkdtemp()
                prep_path = prepare_video_pipeline(tfile.name, temp_dir)
                st.session_state.prepared_video = prep_path
                
                # Visualisasi Step Preprocessing
                col_a, col_b, col_c = st.columns(3)
                # Ambil satu frame untuk display (dummy logic untuk simulasi visual)
                cap = cv2.VideoCapture(tfile.name)
                ret, frame_orig = cap.read()
                if ret:
                    col_a.image(frame_orig, caption="Frame Asli", use_container_width=True)
                    col_b.image(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY), caption="Grayscale", use_container_width=True)
                    # Simulasi kontras
                    col_c.image(cv2.convertScaleAbs(frame_orig, alpha=1.5, beta=0), caption="High Contrast", use_container_width=True)
                cap.release()
                
                st.write("Menjalankan Tracking (Trackpy)...")
                csv_out = os.path.join(temp_dir, "tracks.csv")
                df = tracking_pipeline(prep_path, csv_out)
                st.session_state.tracks_df = df
                status.update(label="Processing Selesai!", state="complete")

        if st.session_state.tracks_df is not None:
            st.divider()
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='metric-card'><h4>Total Partikel</h4><h2>{st.session_state.tracks_df['particle'].nunique()}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h4>Total Lintasan</h4><h2>{len(st.session_state.tracks_df)}</h2></div>", unsafe_allow_html=True)
            
            st.subheader("Final Tracks Table")
            st.dataframe(st.session_state.tracks_df.head(100), use_container_width=True)

# ------------------------------------------
# TAB 3: ANALYSIS PROCESS
# ------------------------------------------
with tab3:
    st.header("Analisis AI")
    if st.session_state.tracks_df is None:
        st.warning("Selesaikan Tab 2 terlebih dahulu.")
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("🚀 Hitung Motilitas (3D-CNN)"):
                st.session_state.motility_results = run_motility_analysis(st.session_state.prepared_video, st.session_state.tracks_df, "model_motility.h5")
                st.success("Motilitas Selesai!")
        with col_m2:
            if st.button("🔬 Hitung Morfologi (EfficientNet)"):
                st.session_state.morphology_results = run_morphology_analysis(st.session_state.prepared_video, st.session_state.tracks_df)
                st.success("Morfologi Selesai!")

# ------------------------------------------
# TAB 4: SUMMARY DASHBOARD (SESUAI WIREFRAME)
# ------------------------------------------
with tab4:
    if st.session_state.motility_results is None or st.session_state.morphology_results is None:
        st.info("Selesaikan semua proses analisis di Tab 3 untuk melihat Summary.")
    else:
        st.markdown("<div class='main-result'><h3>Main Result: FERTIL / INFERTIL</h3></div>", unsafe_allow_html=True)
        st.write("")
        
        row1_col1, row1_col2 = st.columns([2, 1])
        
        with row1_col1: # Kotak Motility
            with st.container(border=True):
                st.write("**Motility (%)**")
                m_pr = len(st.session_state.motility_results[st.session_state.motility_results['motility_label'] == 'PR'])
                m_np = len(st.session_state.motility_results[st.session_state.motility_results['motility_label'] == 'NP'])
                m_im = len(st.session_state.motility_results[st.session_state.motility_results['motility_label'] == 'IM'])
                
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Progressive", f"{m_pr}")
                mc2.metric("Non-Progressive", f"{m_np}")
                mc3.metric("Immotile", f"{m_im}")
                
        with row1_col2: # Kotak Morfologi
            with st.container(border=True):
                st.write("**Morfologi (%)**")
                n_norm = len(st.session_state.morphology_results[st.session_state.morphology_results['morphology_label'] == 'Normal'])
                n_abnorm = len(st.session_state.morphology_results[st.session_state.morphology_results['morphology_label'] == 'Abnormal'])
                st.write(f"Normal: {n_norm}")
                st.write(f"Abnormal: {n_abnorm}")

        row2_col1, row2_col2 = st.columns([2, 1])
        with row2_col1:
            with st.container(border=True):
                st.write("**Visualisasi Pergerakan Sperma**")
                st.info("Area untuk pemutaran video hasil tracking...")
                st.video(st.session_state.prepared_video)
                
        with row2_col2:
            with st.container(border=True):
                st.write("**Sampel Normal Morfologi**")
                # Ambil sampel yang dilabeli normal
                samples = st.session_state.morphology_results[st.session_state.morphology_results['morphology_label'] == 'Normal']
                if not samples.empty:
                    st.image(samples.iloc[0]['image_display'], use_container_width=True)
                else:
                    st.write("Tidak ada sampel normal ditemukan.")
