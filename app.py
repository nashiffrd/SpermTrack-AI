# app.py
import os
import tempfile
import streamlit as st
import pandas as pd

# ===== IMPORT PIPELINE =====
from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Sperm Analysis App",
    layout="wide"
)

# ===== SESSION STATE INIT =====
if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "prepared_video" not in st.session_state:
    st.session_state.prepared_video = None

if "tracks_df" not in st.session_state:
    st.session_state.tracks_df = None

# ===== SIDEBAR NAVIGATION =====
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Halaman Awal",
        "Data Loader",
        "Data Preprocessing"
    ]
)

# =========================================================
# ====================== HALAMAN AWAL =====================
# =========================================================
if page == "Halaman Awal":
    st.title("Aplikasi Analisis Motilitas dan Morfologi Spermatozoa")

    st.markdown("""
    Aplikasi ini digunakan untuk melakukan analisis sperma berbasis video
    melalui tahapan **preprocessing**, **tracking**, dan analisis lanjutan.
    """)

    st.subheader("Cara Penggunaan")
    st.markdown("""
    1. Masuk ke menu **Data Loader**
    2. Upload video sperma
    3. Jalankan preprocessing dan tracking
    4. Lanjutkan ke tahap analisis berikutnya
    """)

    st.button("▶ Start Analisis")

# =========================================================
# ======================= DATA LOADER =====================
# =========================================================
elif page == "Data Loader":
    st.header("Data Loader")

    uploaded_file = st.file_uploader(
        "Upload Video Sperma",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.video_path = video_path
        st.session_state.prepared_video = None
        st.session_state.tracks_df = None

        st.success("Video berhasil diupload")

    if st.session_state.video_path:
        st.info("Video siap diproses")

# =========================================================
# =================== DATA PREPROCESSING ==================
# =========================================================
elif page == "Data Preprocessing":
    st.header("Data Preprocessing & Tracking")

    if st.session_state.video_path is None:
        st.warning("Silakan upload video terlebih dahulu.")
        st.stop()

    # ---------- PREPROCESSING ----------
    if st.session_state.prepared_video is None:
        if st.button("⚙ Jalankan Preprocessing Video"):
            with st.spinner("Menjalankan preprocessing video..."):
                work_dir = tempfile.mkdtemp()

                prepared_video = prepare_video_pipeline(
                    input_video_path=st.session_state.video_path,
                    working_dir=work_dir
                )

                st.session_state.prepared_video = prepared_video

            st.success("Preprocessing selesai")

    else:
        st.success("Preprocessing sudah dijalankan")

    # ---------- TRACKING ----------
    if st.session_state.prepared_video and st.session_state.tracks_df is None:
        if st.button("🧬 Jalankan Tracking"):
            with st.spinner("Menjalankan tracking sperma..."):
                output_csv = os.path.join(
                    os.path.dirname(st.session_state.prepared_video),
                    "final_tracks.csv"
                )

                tracks_df = tracking_pipeline(
                    prepared_video_path=st.session_state.prepared_video,
                    output_csv_path=output_csv
                )

                st.session_state.tracks_df = tracks_df

            st.success("Tracking selesai")

    # ---------- INFO CARDS ----------
    col1, col2 = st.columns(2)

    with col1:
        total_particles = (
            len(st.session_state.tracks_df)
            if st.session_state.tracks_df is not None else "-"
        )
        st.metric("Total Partikel", total_particles)

    with col2:
        total_tracks = (
            st.session_state.tracks_df["particle"].nunique()
            if st.session_state.tracks_df is not None else "-"
        )
        st.metric("Total Tracking", total_tracks)

    st.divider()

    # ---------- VISUAL PLACEHOLDER ----------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Hasil Locate")
        st.info("Visualisasi frame hasil locate akan ditampilkan di sini")

    with colB:
        st.subheader("Hasil Link & Drift")
        st.info("Visualisasi frame hasil linking & drift akan ditampilkan di sini")

    st.divider()

    # ---------- TABLE ----------
    st.subheader("Final Tracks Data")

    if st.session_state.tracks_df is not None:
        st.dataframe(
            st.session_state.tracks_df,
            use_container_width=True
        )
    else:
        st.info("Tabel final_tracks akan muncul setelah tracking dijalankan")
