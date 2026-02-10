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
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
# Menggunakan session state agar data tidak hilang saat rerun
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
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# =====================================================
# 1. HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Sistem Analisis Spermatozoa Terintegrasi")
    st.markdown("""
    Selamat datang di aplikasi **AI Sperm Analysis**. Sistem ini mengotomatisasi pemeriksaan semen menggunakan metode:
    
    * **Tracking:** Menggunakan algoritma *Trackpy* untuk mendeteksi setiap pergerakan partikel.
    * **Motilitas:** Klasifikasi PR, NP, dan IM menggunakan model **3D-CNN**.
    * **Morfologi:** Klasifikasi Normal dan Abnormal menggunakan **EfficientNetV2S** dengan pemrosesan *Binary Erosion*.
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
        # Simpan file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        
        st.success("Video Berhasil Diunggah!")
        
        if st.button("Jalankan Preprocessing & Tracking 🚀"):
            with st.spinner("Sedang memproses video (Grayscale/Contrast) dan menjalankan tracking..."):
                # Buat folder sementara untuk output
                temp_dir = tempfile.mkdtemp()
                
                # A. Pipeline Preparation
                prep_path = prepare_video_pipeline(st.session_state.video_path, temp_dir)
                st.session_state.prepared_video = prep_path
                
                # B. Pipeline Tracking
                csv_out = os.path.join(temp_dir, "final_tracks.csv")
                tracks = tracking_pipeline(prep_path, csv_out)
                st.session_state.tracks_df = tracks.reset_index(drop=True)
                
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

    # Ringkasan Data Awal
    total_sperm = st.session_state.tracks_df['particle'].nunique()
    st.metric(label="Total Sperma Terdeteksi", value=total_sperm)

    st.divider()

    # Kolom Tombol Analisis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analisis Motilitas")
        if st.button("Jalankan 3D-CNN Motility"):
            with st.spinner("Mengekstrak clips dan melakukan prediksi motilitas..."):
                # Catatan: Fungsi ini sekarang akan mengambil model dari Hugging Face secara internal jika dikonfigurasi
                results_mot = run_motility_analysis(
                    st.session_state.prepared_video, 
                    st.session_state.tracks_df
                )
                st.session_state.motility_results = results_mot
            st.balloons()

    with col2:
        st.subheader("Analisis Morfologi")
        if st.button("Jalankan EfficientNet Morfologi"):
            with st.spinner("Mengekstrak ROI, Binary Erosion, dan Prediksi Morfologi..."):
                # Fungsi ini menarik model dari Hugging Face: nashiffrd/SpermMorpho
                results_morf = run_morphology_analysis(
                    st.session_state.prepared_video, 
                    st.session_state.tracks_df
                )
                st.session_state.morphology_results = results_morf
            st.balloons()

    # TAMPILAN HASIL (Jika sudah di-run)
    st.divider()
    
    # Grid Hasil
    res_col_a, res_col_b = st.columns(2)

    with res_col_a:
        if st.session_state.motility_results is not None:
            st.write("### 📊 Hasil Motilitas")
            df_mot = st.session_state.motility_results
            counts = df_mot['motility_label'].value_counts()
            st.bar_chart(counts)
            st.dataframe(df_mot[['particle', 'motility_label', 'confidence']], use_container_width=True)

    with res_col_b:
        if st.session_state.morphology_results is not None:
            st.write("### 🔬 Hasil Morfologi")
            df_morf = st.session_state.morphology_results
            counts_m = df_morf['morphology_label'].value_counts()
            st.pie_chart(counts_m)
            
            # Tampilkan sampel gambar erosion
            st.write("Sampel ROI (Binary Erosion):")
            img_grid = st.columns(3)
            for i, row in df_morf.head(3).iterrows():
                img_grid[i].image(row['image_display'], caption=f"ID:{row['particle']} - {row['morphology_label']}")

    # DOWNLOAD REPORT
    if st.session_state.motility_results is not None:
        st.divider()
        st.subheader("💾 Unduh Laporan")
        
        # Merge data jika kedua analisis sudah ada
        final_df = st.session_state.motility_results.copy()
        if st.session_state.morphology_results is not None:
            final_df = final_df.merge(
                st.session_state.morphology_results[['particle', 'morphology_label', 'morphology_prob']], 
                on='particle', how='left'
            )
            
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV Report",
            csv,
            "sperm_analysis_report.csv",
            "text/csv",
            key='download-csv'
        )
