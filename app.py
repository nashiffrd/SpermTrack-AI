import os
import cv2
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

# Import modul internal dari struktur folder kamu
from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from models.motility_analyzer import run_motility_analysis
from models.morphology_analyzer import run_morphology_analysis

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sperm Analysis AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "Halaman Awal"
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "prepared_video" not in st.session_state:
    st.session_state.prepared_video = None
if "tracks_df" not in st.session_state:
    st.session_state.tracks_df = None
if "motility_results" not in st.session_state:
    st.session_state.motility_results = None
if "morphology_results" not in st.session_state:
    st.session_state.morphology_results = None

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("🧬 Sperm Analysis AI")
st.sidebar.markdown("---")
st.session_state.page = st.sidebar.radio(
    "Menu Utama",
    ["Halaman Awal", "Upload & Tracking", "Dashboard Analisis"],
    index=["Halaman Awal", "Upload & Tracking", "Dashboard Analisis"].index(st.session_state.page)
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Aplikasi"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# =====================================================
# 1. HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Sistem Analisis Spermatozoa Terintegrasi")
    st.markdown("""
    Aplikasi ini dirancang untuk mendeteksi dan menganalisis kualitas spermatozoa secara otomatis:
    
    * **Motilitas:** Menggunakan model **3D-CNN** lokal untuk klasifikasi PR, NP, dan IM.
    * **Morfologi:** Menggunakan model **EfficientNetV2S** (via Hugging Face) untuk klasifikasi Normal/Abnormal.
    """)
    
    if st.button("Mulai Analisis Sekarang ➡"):
        st.session_state.page = "Upload & Tracking"
        st.rerun()

# =====================================================
# 2. HALAMAN UPLOAD & TRACKING
# =====================================================
elif st.session_state.page == "Upload & Tracking":
    st.header("Step 1: Persiapan Video & Tracking")
    
    uploaded_file = st.file_uploader("Upload Video (Format: .mp4, .avi)", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        
        st.success("Video Berhasil Diunggah!")
        
        if st.button("Jalankan Preprocessing & Tracking 🚀"):
            with st.spinner("Sedang memproses video dan menjalankan tracking..."):
                temp_dir = tempfile.mkdtemp()
                
                # A. Pipeline Preparation
                prep_path = prepare_video_pipeline(st.session_state.video_path, temp_dir)
                st.session_state.prepared_video = prep_path
                
                # B. Pipeline Tracking
                csv_out = os.path.join(temp_dir, "final_tracks.csv")
                df_tracks = tracking_pipeline(prep_path, csv_out)
                st.session_state.tracks_df = df_tracks.reset_index(drop=True)
                
            st.success("Tracking Selesai!")
            st.session_state.page = "Dashboard Analisis"
            st.rerun()

# =====================================================
# 3. HALAMAN DASHBOARD ANALISIS
# =====================================================
elif st.session_state.page == "Dashboard Analisis":
    st.header("Step 2: Analisis Motilitas & Morfologi")
    
    if st.session_state.tracks_df is None:
        st.warning("Silakan selesaikan tahap Tracking terlebih dahulu.")
        st.stop()

    st.metric(label="Total Sperma Terdeteksi", value=st.session_state.tracks_df['particle'].nunique())
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analisis Motilitas")
        if st.button("Jalankan 3D-CNN Motility"):
            # LOKAL: Mengambil model h5 dari folder models/ di repositori
            model_mot_path = "model_motility.h5"
            
            if os.path.exists(model_mot_path):
                with st.spinner("Menganalisis pergerakan (Model Lokal)..."):
                    results_mot = run_motility_analysis(
                        st.session_state.prepared_video, 
                        st.session_state.tracks_df,
                        model_mot_path # Kirim path lokal
                    )
                    st.session_state.motility_results = results_mot
                st.success("Motilitas Selesai!")
            else:
                st.error(f"File model tidak ditemukan di {model_mot_path}")

    with col2:
        st.subheader("Analisis Morfologi")
        if st.button("Jalankan EfficientNet Morfologi"):
            with st.spinner("Mengunduh model dari Hugging Face & Analisis Bentuk..."):
                # HUGGING FACE: Model diunduh otomatis di dalam fungsi ini
                results_morf = run_morphology_analysis(
                    st.session_state.prepared_video, 
                    st.session_state.tracks_df
                )
                st.session_state.morphology_results = results_morf
            st.success("Morfologi Selesai!")

    # --- TAMPILAN HASIL ---
    st.divider()
    res_col_a, res_col_b = st.columns(2)

    with res_col_a:
        if st.session_state.motility_results is not None:
            st.write("### 📊 Hasil Motilitas")
            df_mot = st.session_state.motility_results
            st.bar_chart(df_mot['motility_label'].value_counts())
            st.dataframe(df_mot[['particle', 'motility_label', 'confidence']], use_container_width=True)

    with res_col_b:
        if st.session_state.morphology_results is not None:
            st.write("### 🔬 Hasil Morfologi")
            df_morf = st.session_state.morphology_results
            st.pie_chart(df_morf['morphology_label'].value_counts())
            
            st.write("Sampel ROI (Binary Erosion):")
            img_grid = st.columns(3)
            for i, row in df_morf.head(3).iterrows():
                img_grid[i].image(row['image_display'], caption=f"ID:{row['particle']} - {row['morphology_label']}")

    # --- DOWNLOAD REPORT ---
    if st.session_state.motility_results is not None:
        st.divider()
        final_df = st.session_state.motility_results.copy()
        if st.session_state.morphology_results is not None:
            final_df = final_df.merge(
                st.session_state.morphology_results[['particle', 'morphology_label', 'morphology_prob']], 
                on='particle', how='left'
            )
            
        st.download_button(
            "Download Report (.csv)",
            final_df.to_csv(index=False).encode('utf-8'),
            "sperm_report.csv",
            "text/csv"
        )
