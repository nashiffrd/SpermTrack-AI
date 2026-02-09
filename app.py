# app.py
import os
import cv2
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
import trackpy as tp

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

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("Navigasi")
st.session_state.page = st.sidebar.radio(
    "Pilih Halaman",
    ["Halaman Awal", "Data Loader", "Data Preprocessing"],
    index=["Halaman Awal", "Data Loader", "Data Preprocessing"].index(st.session_state.page)
)

# =====================================================
# HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Aplikasi Analisis Motilitas dan Morfologi Spermatozoa")

    st.markdown("""
    Aplikasi ini melakukan analisis sperma berbasis video melalui tahapan:
    **preprocessing**, **tracking**, dan analisis lanjutan.
    """)

    st.subheader("Cara Penggunaan")
    st.markdown("""
    1. Klik **Start Analysis**
    2. Upload video sperma
    3. Sistem otomatis melakukan preprocessing & tracking
    4. Hasil ditampilkan pada halaman Data Preprocessing
    """)

    if st.button("▶ Start Analysis"):
        st.session_state.page = "Data Loader"
        st.experimental_rerun()

# =====================================================
# DATA LOADER
# =====================================================
elif st.session_state.page == "Data Loader":
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

        if st.button("➡ Lanjutkan Preprocessing"):
            st.session_state.page = "Data Preprocessing"
            st.experimental_rerun()

# =====================================================
# DATA PREPROCESSING (AUTO RUN)
# =====================================================
elif st.session_state.page == "Data Preprocessing":
    st.header("Data Preprocessing & Tracking")

    if st.session_state.video_path is None:
        st.warning("Silakan upload video terlebih dahulu.")
        st.stop()

    # ================== AUTO PREPROCESSING ==================
    if st.session_state.prepared_video is None:
        with st.spinner("Menjalankan preprocessing video..."):
            work_dir = tempfile.mkdtemp()
            st.session_state.prepared_video = prepare_video_pipeline(
                st.session_state.video_path,
                work_dir
            )

    # ================== AUTO TRACKING ==================
    if st.session_state.tracks_df is None:
        with st.spinner("Menjalankan tracking sperma..."):
            output_csv = os.path.join(
                os.path.dirname(st.session_state.prepared_video),
                "final_tracks.csv"
            )
            st.session_state.tracks_df = tracking_pipeline(
                st.session_state.prepared_video,
                output_csv
            )

    tracks_df = st.session_state.tracks_df

    # ================== INFO CARDS ==================
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Partikel", len(tracks_df))
    with col2:
        st.metric("Total Tracking", tracks_df["particle"].nunique())

    st.divider()

    # =====================================================
    # VISUALISASI FRAME LOCATE & LINK-DRIFT
    # =====================================================
    cap = cv2.VideoCapture(st.session_state.prepared_video)
    ret, frame = cap.read()
    cap.release()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_idx = tracks_df["frame"].min()
    frame_tracks = tracks_df[tracks_df["frame"] == frame_idx]

    # ---------- LOCATE VIS ----------
    locate_vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    for _, r in frame_tracks.iterrows():
        cv2.circle(
            locate_vis,
            (int(r["x"]), int(r["y"])),
            8,
            (0, 255, 0),
            1
        )

    # ---------- LINK & DRIFT VIS ----------
    link_vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    for pid, grp in tracks_df.groupby("particle"):
        pts = grp[grp["frame"] <= frame_idx][["x", "y"]].values.astype(int)
        for i in range(1, len(pts)):
            cv2.line(
                link_vis,
                tuple(pts[i - 1]),
                tuple(pts[i]),
                (255, 0, 0),
                1
            )

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Hasil Locate")
        st.image(locate_vis, channels="BGR")

    with colB:
        st.subheader("Hasil Link & Drift")
        st.image(link_vis, channels="BGR")

    st.divider()

    # ================== TABLE ==================
    st.subheader("Final Tracks Data")
    st.dataframe(tracks_df, use_container_width=True)
