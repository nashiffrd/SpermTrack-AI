# app.py
import os
import cv2
import tempfile
import streamlit as st
import pandas as pd

from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from tracking.visualization import draw_locate_frame, draw_tracks

from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference


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

if "motility_result" not in st.session_state:
    st.session_state.motility_result = None

if "morphology_result" not in st.session_state:
    st.session_state.morphology_result = None


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("Navigasi")
st.session_state.page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Halaman Awal",
        "Data Loader",
        "Data Preprocessing",
        "Main Dashboard"
    ],
    index=[
        "Halaman Awal",
        "Data Loader",
        "Data Preprocessing",
        "Main Dashboard"
    ].index(st.session_state.page)
)

# =====================================================
# HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Aplikasi Analisis Motilitas dan Morfologi Spermatozoa")

    st.markdown("""
    Aplikasi ini melakukan analisis sperma berbasis video melalui tahapan:
    **preprocessing**, **tracking**, **motility inference**, dan **morphology inference**.
    """)

    st.subheader("Cara Penggunaan")
    st.markdown("""
    1. Klik **Start Analysis**
    2. Upload video sperma
    3. Sistem otomatis menjalankan preprocessing & tracking
    4. Hasil analisis ditampilkan pada dashboard utama
    """)

    if st.button("▶ Start Analysis"):
        st.session_state.page = "Data Loader"
        st.rerun()


# =====================================================
# DATA LOADER
# =====================================================
elif st.session_state.page == "Data Loader":
    st.header("Data Loader")

    uploaded_file = st.file_uploader(
        "Upload Video Sperma",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.video_path = video_path
        st.session_state.prepared_video = None
        st.session_state.tracks_df = None
        st.session_state.motility_result = None
        st.session_state.morphology_result = None

        st.success("Video berhasil diupload")

        if st.button("➡ Lanjutkan Preprocessing"):
            st.session_state.page = "Data Preprocessing"
            st.rerun()


# =====================================================
# DATA PREPROCESSING
# =====================================================
elif st.session_state.page == "Data Preprocessing":
    st.header("Data Preprocessing & Tracking")

    if st.session_state.video_path is None:
        st.warning("Silakan upload video terlebih dahulu.")
        st.stop()

    # ---------- PREPROCESS ----------
    if st.session_state.prepared_video is None:
        with st.spinner("Menjalankan preprocessing video..."):
            work_dir = tempfile.mkdtemp()
            st.session_state.prepared_video = prepare_video_pipeline(
                input_video_path=st.session_state.video_path,
                working_dir=work_dir
            )

    # ---------- TRACKING ----------
    if st.session_state.tracks_df is None:
        with st.spinner("Menjalankan tracking sperma..."):
            output_csv = os.path.join(
                os.path.dirname(st.session_state.prepared_video),
                "final_tracks.csv"
            )
            tracks = tracking_pipeline(
                prepared_video_path=st.session_state.prepared_video,
                output_csv_path=output_csv
            )
            st.session_state.tracks_df = tracks.reset_index(drop=True)

    tracks_df = st.session_state.tracks_df

    # ---------- INFO ----------
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Deteksi", len(tracks_df))
    with col2:
        st.metric("Total Partikel", tracks_df["particle"].nunique())

    st.divider()

    # ---------- VISUALISASI ----------
    cap = cv2.VideoCapture(st.session_state.prepared_video)
    ret, frame = cap.read()
    cap.release()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_idx = tracks_df["frame"].min()

    locate_vis = draw_locate_frame(frame_gray, tracks_df, frame_idx)
    link_vis = draw_tracks(frame_gray, tracks_df, frame_idx)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Hasil Locate")
        st.image(locate_vis, channels="BGR")

    with colB:
        st.subheader("Hasil Link & Drift")
        st.image(link_vis, channels="BGR")

    st.divider()
    st.subheader("Final Tracks Data")
    st.dataframe(tracks_df, use_container_width=True)


# =====================================================
# MAIN DASHBOARD (INFERENCE)
# =====================================================
elif st.session_state.page == "Main Dashboard":
    st.header("📊 Main Analysis Dashboard")

    if st.session_state.tracks_df is None:
        st.warning("Tracking belum tersedia.")
        st.stop()

    tracks_csv = os.path.join(
        os.path.dirname(st.session_state.prepared_video),
        "final_tracks.csv"
    )
    st.session_state.tracks_df.to_csv(tracks_csv, index=False)

    work_dir = tempfile.mkdtemp()

    # ================== MOTILITY ==================
    if st.session_state.motility_result is None:
        with st.spinner("Menjalankan motility inference..."):
            st.session_state.motility_result = run_motility_inference(
                video_path=st.session_state.prepared_video,
                tracks_csv=tracks_csv,
                model_path="model_motility.h5"
            )

    mot = st.session_state.motility_result
    pr = mot["detail"]["PR"]
    np_ = mot["detail"]["NP"]
    im = mot["detail"]["IM"]
    total = pr + np_ + im
    fertile_pct = (pr + np_) / total * 100 if total > 0 else 0
    motility_label = "Fertil" if fertile_pct > 40 else "Infertil"

    # ================== MORPHOLOGY ==================
    if st.session_state.morphology_result is None:
        with st.spinner("Menjalankan morphology inference..."):
            st.session_state.morphology_result = run_morphology_inference(
                video_path=st.session_state.prepared_video,
                tracks_df=st.session_state.tracks_df,
                model_path="model_morfologi.h5",
                work_dir=work_dir
            )

    morph = st.session_state.morphology_result["summary"]
    morph_label = "Normal" if morph["normal_pct"] > 4 else "Abnormal"

    # ================== DASHBOARD ==================
    st.markdown("## 🧬 Ringkasan Klinis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Motility")
        st.metric("Klasifikasi", motility_label)
        st.write(f"PR: {pr}")
        st.write(f"NP: {np_}")
        st.write(f"IM: {im}")
        st.write(f"PR + NP: {fertile_pct:.2f}%")

    with col2:
        st.subheader("Morphology")
        st.metric("Klasifikasi", morph_label)
        st.write(f"Normal: {morph['normal_pct']:.2f}%")
        st.write(f"Abnormal: {morph['abnormal_pct']:.2f}%")
