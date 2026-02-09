import os
import cv2
import tempfile
import streamlit as st
import pandas as pd

from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from visualization import draw_locate_frame, draw_tracks
from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference

# =============================
# STREAMLIT CONFIG
# =============================
st.set_page_config(
    page_title="Sperm Analysis System",
    layout="wide"
)

# =============================
# SESSION STATE INIT
# =============================
for k in [
    "uploaded_video",
    "prepared_video",
    "tracks_csv",
    "tracks_df",
    "analysis_done",
    "motility_result",
    "morphology_result",
]:
    if k not in st.session_state:
        st.session_state[k] = None

# =============================
# SIDEBAR NAVIGATION
# =============================
page = st.sidebar.radio(
    "Navigation",
    [
        "Halaman Awal",
        "Data Loader",
        "Preprocessing & Tracking",
        "Main Dashboard"
    ]
)

# =============================
# HALAMAN AWAL
# =============================
if page == "Halaman Awal":
    st.title("Sperm Motility & Morphology Analysis")
    st.write(
        """
        Aplikasi ini melakukan analisis sperma berbasis video mikroskopis
        menggunakan **TrackPy**, **3D-CNN (Motility)**, dan **CNN EfficientNetV2 (Morphology)**.
        """
    )

    st.markdown("### Cara Penggunaan")
    st.markdown("""
    1. Upload video sperma  
    2. Sistem melakukan preprocessing & tracking otomatis  
    3. Model melakukan inferensi motility dan morfologi  
    4. Hasil ditampilkan dalam dashboard klinis
    """)

    if st.button("🚀 Start Analysis"):
        st.session_state["page_jump"] = "Data Loader"
        st.experimental_rerun()

# =============================
# DATA LOADER
# =============================
elif page == "Data Loader":
    st.header("Upload Video")

    video_file = st.file_uploader(
        "Upload video (.mp4 / .avi)",
        type=["mp4", "avi"]
    )

    if video_file is not None:
        tmp_dir = tempfile.mkdtemp()
        video_path = os.path.join(tmp_dir, video_file.name)

        with open(video_path, "wb") as f:
            f.write(video_file.read())

        st.session_state.uploaded_video = video_path
        st.success("✅ Video berhasil diupload")

        if st.button("➡️ Lanjutkan Preprocessing"):
            st.session_state["page_jump"] = "Preprocessing & Tracking"
            st.experimental_rerun()

# =============================
# PREPROCESSING & TRACKING
# =============================
elif page == "Preprocessing & Tracking":
    st.header("Preprocessing & Tracking")

    if st.session_state.uploaded_video is None:
        st.warning("Silakan upload video terlebih dahulu")
        st.stop()

    with st.spinner("Menjalankan preprocessing dan tracking..."):
        workdir = tempfile.mkdtemp()
        prepared_video = prepare_video_pipeline(
            st.session_state.uploaded_video,
            workdir
        )

        tracks_csv = os.path.join(workdir, "final_tracks.csv")
        tracks_df = tracking_pipeline(prepared_video, tracks_csv)

        st.session_state.prepared_video = prepared_video
        st.session_state.tracks_csv = tracks_csv
        st.session_state.tracks_df = tracks_df
        st.session_state.analysis_done = True

    st.success("✅ Tracking selesai")

    # === INFO BAR
    col1, col2 = st.columns(2)
    col1.metric("Total Partikel", tracks_df["particle"].nunique())
    col2.metric("Total Tracking Points", len(tracks_df))

    # === VISUALIZATION
    cap = cv2.VideoCapture(prepared_video)
    ret, frame = cap.read()
    cap.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Hasil Locate")
        locate_img = draw_locate_frame(gray, tracks_df, frame_idx=0)
        st.image(locate_img, channels="BGR")

    with colB:
        st.subheader("Hasil Link & Drift")
        track_img = draw_tracks(gray, tracks_df, frame_idx=50)
        st.image(track_img, channels="BGR")

    st.divider()
    st.dataframe(tracks_df.head(100))

# =============================
# MAIN DASHBOARD
# =============================
elif page == "Main Dashboard":
    st.header("Main Dashboard")

    if not st.session_state.analysis_done:
        st.warning("Lakukan preprocessing & tracking terlebih dahulu")
        st.stop()

    # =============================
    # MOTILITY INFERENCE
    # =============================
    with st.spinner("Menjalankan inferensi motility..."):
        motility = run_motility_inference(
            video_path=st.session_state.prepared_video,
            tracks_csv=st.session_state.tracks_csv,
            model_path="model_motility.h5"
        )

    pr = motility["detail"]["PR"]
    np_ = motility["detail"]["NP"]
    im = motility["detail"]["IM"]
    total = pr + np_ + im

    pct_pr = pr / total * 100 if total > 0 else 0
    pct_np = np_ / total * 100 if total > 0 else 0
    pct_im = im / total * 100 if total > 0 else 0

    motility_status = "FERTIL" if (pct_pr + pct_np) > 40 else "INFERTIL"

    # =============================
    # MORPHOLOGY INFERENCE
    # =============================
    with st.spinner("Menjalankan inferensi morfologi..."):
        morphology = run_morphology_inference(
            img_dir="roi_images",
            model_path="model_morfologi.h5"
        )

    # =============================
    # DASHBOARD DISPLAY
    # =============================
    st.markdown("## 🧬 Hasil Analisis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Motility")
        st.markdown(f"### **{motility_status}**")
        st.write(f"PR: {pct_pr:.2f}%")
        st.write(f"NP: {pct_np:.2f}%")
        st.write(f"IM: {pct_im:.2f}%")

    with col2:
        st.subheader("Morphology")
        st.markdown(f"### **{morphology['status']}**")
        st.write(f"Normal: {morphology['pct_normal']:.2f}%")
        st.write(f"Abnormal: {morphology['pct_abnormal']:.2f}%")
