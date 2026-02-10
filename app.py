import os
import cv2
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

# Import modul internal
from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from models.motility_analyzer import run_motility_analysis
from models.morphology_analyzer import run_morphology_analysis

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sperm Analysis AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
states = {
    "page": "Halaman Awal",
    "video_path": None,
    "prepared_video": None,
    "tracks_df": None,
    "motility_results": None,
    "morphology_results": None
}

for key, value in states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🧬 Sperm Analysis AI")
st.session_state.page = st.sidebar.radio(
    "Navigasi",
    ["Halaman Awal", "Data Loader", "Analysis Dashboard"],
    index=["Halaman Awal", "Data Loader", "Analysis Dashboard"].index(st.session_state.page)
)

# =====================================================
# PAGE: HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Sistem Analisis Spermatozoa Terintegrasi")
    st.info("Gunakan aplikasi ini untuk menganalisis Motilitas (3D-CNN) dan Morfologi (EfficientNetV2S) secara otomatis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analisis Motilitas")
        st.write("Mengklasifikasikan sperma menjadi PR, NP, dan IM berdasarkan pergerakan dalam 32 frame.")
    with col2:
        st.subheader("Analisis Morfologi")
        st.write("Mengidentifikasi bentuk Normal vs Abnormal menggunakan teknik Binary Erosion pada frame terbaik.")

    if st.button("Mulai Analisis Sekarang", type="primary"):
        st.session_state.page = "Data Loader"
        st.rerun()

# =====================================================
# PAGE: DATA LOADER
# =====================================================
elif st.session_state.page == "Data Loader":
    st.header("1. Upload Video")
    uploaded_file = st.file_uploader("Pilih video sperma (.mp4, .avi)", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        v_path = os.path.join(temp_dir, uploaded_file.name)
        with open(v_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.session_state.video_path = v_path
        st.success("Video berhasil diunggah!")
        
        if st.button("Proses Preprocessing & Tracking ➡"):
            with st.spinner("Membersihkan video & Tracking partikel..."):
                # 1. Preprocessing
                prep_video = prepare_video_pipeline(v_path, temp_dir)
                st.session_state.prepared_video = prep_video
                
                # 2. Tracking
                csv_out = os.path.join(temp_dir, "tracks.csv")
                df_tracks = tracking_pipeline(prep_video, csv_out)
                st.session_state.tracks_df = df_tracks.reset_index(drop=True)
                
            st.session_state.page = "Analysis Dashboard"
            st.rerun()

# =====================================================
# PAGE: ANALYSIS DASHBOARD (CORE)
# =====================================================
elif st.session_state.page == "Analysis Dashboard":
    st.header("2. Dashboard Analisis")
    
    if st.session_state.tracks_df is None:
        st.warning("Harap upload dan proses video terlebih dahulu di halaman Data Loader.")
        st.stop()

    # Tampilkan Ringkasan Tracking
    total_sperm = st.session_state.tracks_df['particle'].nunique()
    st.metric("Total Sperma Terdeteksi", total_sperm)

    st.divider()

    # --- ACTION BUTTONS ---
    colA, colB = st.columns(2)
    
    # 1. TOMBOL MOTILITY
    with colA:
        if st.button("🚀 Jalankan Analisis Motilitas", use_container_width=True):
            model_path = "model_motility.h5"
            if os.path.exists(model_path):
                with st.spinner("Menganalisis pergerakan (3D-CNN)..."):
                    st.session_state.motility_results = run_motility_analysis(
                        st.session_state.prepared_video, 
                        st.session_state.tracks_df, 
                        model_path
                    )
                st.success("Motilitas Selesai!")
            else:
                st.error("File model motility tidak ditemukan di folder models/")

    # 2. TOMBOL MORPHOLOGY
    with colB:
        if st.button("🔬 Jalankan Analisis Morfologi", use_container_width=True):
            model_path = "models/morphology_model.h5"
            if os.path.exists(model_path):
                with st.spinner("Mengekstrak ROI & Analisis Bentuk..."):
                    st.session_state.morphology_results = run_morphology_analysis(
                        st.session_state.prepared_video, 
                        st.session_state.tracks_df, 
                        model_path
                    )
                st.success("Morfologi Selesai!")
            else:
                st.error("File model morphology tidak ditemukan di folder models/")

    st.divider()

    # --- DISPLAY RESULTS ---
    if st.session_state.motility_results is not None or st.session_state.morphology_results is not None:
        res_col1, res_col2 = st.columns(2)
        
        # Tampilan Hasil Motilitas
        with res_col1:
            if st.session_state.motility_results is not None:
                st.subheader("Hasil Motilitas")
                df_mot = st.session_state.motility_results
                counts = df_mot['motility_label'].value_counts()
                st.bar_chart(counts)
                st.dataframe(df_mot[['particle', 'motility_label', 'confidence']], use_container_width=True)

        # Tampilan Hasil Morfologi
        with res_col2:
            if st.session_state.morphology_results is not None:
                st.subheader("Hasil Morfologi")
                df_morf = st.session_state.morphology_results
                counts_m = df_morf['morphology_label'].value_counts()
                st.pie_chart(counts_m) # Opsional jika ingin ganti pie
                
                # Tampilkan 5 contoh hasil binary erosion
                st.write("Sampel Hasil Binary Erosion:")
                img_cols = st.columns(3)
                for i, row in df_morf.head(3).iterrows():
                    img_cols[i].image(row['image_display'], caption=f"P-{row['particle']}: {row['morphology_label']}")

    # --- DOWNLOAD SECTION ---
    if st.session_state.motility_results is not None:
        st.divider()
        st.subheader("Export Data")
        # Gabungkan semua hasil jika ada
        final_report = st.session_state.motility_results.copy()
        if st.session_state.morphology_results is not None:
            final_report = final_report.merge(
                st.session_state.morphology_results[['particle', 'morphology_label', 'morphology_prob']], 
                on='particle', how='left'
            )
        
        st.download_button(
            label="Download Laporan Analisis (.csv)",
            data=final_report.to_csv(index=False),
            file_name="sperm_analysis_report.csv",
            mime="text/csv"
        )
