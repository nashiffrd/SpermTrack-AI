# app.py
import os
import cv2
import tempfile
import streamlit as st
import pandas as pd

from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from tracking.visualization import draw_locate_frame, draw_tracks
from motility_inference.pipeline import motility_inference_pipeline

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sperm Motility Analysis",
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

if "motility_df" not in st.session_state:
    st.session_state.motility_df = None

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Navigasi")
st.session_state.page = st.sidebar.radio(
    "Pilih Halaman",
    ["Halaman Awal", "Data Loader", "Motility Analysis"]
)

# =====================================================
# HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Sperm Motility Analysis App")

    st.markdown("""
    Pipeline aplikasi:
    1. Video preprocessing  
    2. Sperm tracking (TrackPy)  
    3. Motility inference (CASA / Deep Learning)
    """)

    if st.button("▶ Start Analysis"):
        st.session_state.page = "Data Loader"
        st.rerun()

# =====================================================
# DATA LOADER
# =====================================================
elif st.session_state.page == "Data Loader":
    st.header("Upload Video")

    uploaded_file = st.file_uploader(
        "Upload video sperma",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.video_path = video_path
        st.session_state.prepared_video = None
        st.session_state.tracks_df = None
        st.session_state.motility_df = None

        st.success("Video berhasil diupload")

        if st.button("➡ Lanjutkan Analisis"):
            st.session_state.page = "Motility Analysis"
            st.rerun()

# =====================================================
# MOTILITY ANALYSIS
# =====================================================
elif st.session_state.page == "Motility Analysis":
    st.header("Tracking & Motility Analysis")

    if st.session_state.video_path is None:
        st.warning("Upload video terlebih dahulu.")
        st.stop()

    # -------- PREPROCESS --------
    if st.session_state.prepared_video is None:
        with st.spinner("Preprocessing video..."):
            work_dir = tempfile.mkdtemp()
            st.session_state.prepared_video = prepare_video_pipeline(
                input_video_path=st.session_state.video_path,
                working_dir=work_dir
            )

    # -------- TRACKING --------
    if st.session_state.tracks_df is None:
        with st.spinner("Tracking sperma..."):
            csv_path = os.path.join(
                os.path.dirname(st.session_state.prepared_video),
                "final_tracks.csv"
            )

            tracks = tracking_pipeline(
                prepared_video_path=st.session_state.prepared_video,
                output_csv_path=csv_path
            )

            st.session_state.tracks_df = tracks.reset_index(drop=True)

    tracks_df = st.session_state.tracks_df

    # -------- MOTILITY --------
    if st.session_state.motility_df is None:
        with st.spinner("Motility inference..."):
            motility_df = motility_inference_pipeline(
                video_path=st.session_state.prepared_video,
                tracks_df=tracks_df
            )
            st.session_state.motility_df = motility_df

    motility_df = st.session_state.motility_df

    # -------- METRICS --------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sperma", motility_df["particle"].nunique())
    col2.metric("Progressive (PR)", (motility_df["motility"] == "PR").sum())
    col3.metric("Non-Progressive (NP/IM)", (motility_df["motility"] != "PR").sum())

    st.divider()

    # -------- VISUALIZATION --------
    cap = cv2.VideoCapture(st.session_state.prepared_video)
    ret, frame = cap.read()
    cap.release()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_idx = tracks_df["frame"].min()

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Locate Result")
        img1 = draw_locate_frame(
            frame_gray,
            tracks_df,
            frame_idx
        )
        st.image(img1, channels="BGR")

    with colB:
        st.subheader("Tracking Result")
        img2 = draw_tracks(
            frame_gray,
            tracks_df,
            frame_idx
        )
        st.image(img2, channels="BGR")

    st.divider()

    # -------- TABLE --------
    st.subheader("Motility Result Table")
    st.dataframe(motility_df, use_container_width=True)
