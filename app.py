import os
import cv2
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from tracking.visualization import draw_locate_frame, draw_tracks
# Import analyzer baru kita
from models.motility-analyzer import run_motility_analysis

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sperm Analysis App",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "Halaman Awal"
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "prepared_video" not in st.session_state:
    st.session_state.prepared_video = None
if "tracks_df" not in st.session_state:
    st.session_state.tracks_df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("Navigasi")
st.session_state.page = st.sidebar.radio(
    "Pilih Halaman",
    ["Halaman Awal", "Data Loader", "Data Preprocessing", "Motility Analysis"],
    index=["Halaman Awal", "Data Loader", "Data Preprocessing", "Motility Analysis"].index(st.session_state.page)
)

# =====================================================
# HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Aplikasi Analisis Motilitas dan Morfologi Spermatozoa")
    st.markdown("""
    Aplikasi ini melakukan analisis sperma berbasis video melalui tahapan:
    **preprocessing**, **tracking**, dan **klasifikasi otomatis**.
    """)

    if st.button("▶ Start Analysis"):
        st.session_state.page = "Data Loader"
        st.rerun()

# =====================================================
# DATA LOADER
# =====================================================
elif st.session_state.page == "Data Loader":
    st.header("Data Loader")
    uploaded_file = st.file_uploader("Upload Video Sperma", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.video_path = video_path
        st.session_state.prepared_video = None
        st.session_state.tracks_df = None
        st.session_state.analysis_results = None
        st.success("Video berhasil diupload")

        if st.button("➡ Lanjutkan Preprocessing"):
            st.session_state.page = "Data Preprocessing"
            st.rerun()

# =====================================================
# DATA PREPROCESSING & TRACKING
# =====================================================
elif st.session_state.page == "Data Preprocessing":
    st.header("Data Preprocessing & Tracking")

    if st.session_state.video_path is None:
        st.warning("Silakan upload video terlebih dahulu.")
        st.stop()

    # Preprocessing
    if st.session_state.prepared_video is None:
        with st.spinner("Menjalankan preprocessing video..."):
            work_dir = tempfile.mkdtemp()
            st.session_state.prepared_video = prepare_video_pipeline(
                input_video_path=st.session_state.video_path,
                working_dir=work_dir
            )

    # Tracking
    if st.session_state.tracks_df is None:
        with st.spinner("Menjalankan tracking sperma..."):
            output_csv = os.path.join(os.path.dirname(st.session_state.prepared_video), "final_tracks.csv")
            tracks = tracking_pipeline(
                prepared_video_path=st.session_state.prepared_video,
                output_csv_path=output_csv
            )
            st.session_state.tracks_df = tracks.reset_index(drop=True)

    tracks_df = st.session_state.tracks_df
    
    # Visualization UI (Sama seperti sebelumnya)
    st.info(f"Ditemukan {tracks_df['particle'].nunique()} partikel untuk dianalisis.")
    
    if st.button("🚀 Jalankan Analisis Motilitas"):
        st.session_state.page = "Motility Analysis"
        st.rerun()

# =====================================================
# MOTILITY ANALYSIS (MODEL INFERENCE)
# =====================================================
elif st.session_state.page == "Motility Analysis":
    st.header("Motility Classification (3D-CNN)")

    if st.session_state.tracks_df is None:
        st.error("Data tracking tidak ditemukan.")
        st.stop()

    model_path = "model_motility.h5" # Pastikan file h5 ada di folder ini
    
    if not os.path.exists(model_path):
        st.error(f"Model file tidak ditemukan di {model_path}. Harap upload model h5 Anda.")
    else:
        if st.session_state.analysis_results is None:
            with st.spinner("Mengekstrak clips dan melakukan klasifikasi (PR/NP/IM)..."):
                try:
                    results_df = run_motility_analysis(
                        video_path=st.session_state.prepared_video,
                        tracks_df=st.session_state.tracks_df,
                        model_path=model_path
                    )
                    st.session_state.analysis_results = results_df
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat inferensi: {e}")

        if st.session_state.analysis_results is not None:
            res = st.session_state.analysis_results
            
            # --- TAMPILKAN SUMMARY ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Progressive (PR)", len(res[res['motility_label'] == 'PR']))
            col2.metric("Non-Progressive (NP)", len(res[res['motility_label'] == 'NP']))
            col3.metric("Immotile (IM)", len(res[res['motility_label'] == 'IM']))

            st.divider()
            st.subheader("Detail Hasil per Partikel")
            st.dataframe(res, use_container_width=True)
            
            # Gabungkan dengan data koordinat jika ingin download
            full_report = st.session_state.tracks_df.merge(res, on='particle', how='left')
            st.download_button(
                "Download Full Report (CSV)",
                full_report.to_csv(index=False),
                "sperm_analysis_report.csv",
                "text/csv"
            )

    if st.button("⬅ Kembali ke Preprocessing"):
        st.session_state.page = "Data Preprocessing"
        st.rerun()
