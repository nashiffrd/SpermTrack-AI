import os
import cv2
import streamlit as st
import pandas as pd

from tracking.pipeline import tracking_pipeline
from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference
from visualization.draw_trajectory_video import draw_trajectory_video

# =====================
# BASIC SETUP
# =====================
st.set_page_config(
    page_title="Sperm Analysis System",
    layout="wide"
)

TEMP_DIR = "temp"
MODEL_DIR = "models"

os.makedirs(TEMP_DIR, exist_ok=True)

# =====================
# SIDEBAR
# =====================
st.sidebar.title("🧬 Sperm Analysis System")
st.sidebar.markdown("Upload video → Tracking → Motility & Morphology")

uploaded_video = st.sidebar.file_uploader(
    "Upload Video Sperma",
    type=["mp4", "avi", "mov"]
)

run_btn = st.sidebar.button("🚀 Jalankan Analisis")

# =====================
# TABS
# =====================
tab1, tab2 = st.tabs([
    "📥 Preprocessing & Tracking",
    "📊 Hasil Klasifikasi"
])

# =====================
# MAIN LOGIC
# =====================
if uploaded_video and run_btn:

    # ---------------------
    # SAVE VIDEO
    # ---------------------
    video_path = os.path.join(TEMP_DIR, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    tracks_csv = os.path.join(TEMP_DIR, "final_tracks.csv")
    trajectory_video = os.path.join(TEMP_DIR, "trajectory.mp4")

    # =====================
    # TAB 1 – TRACKING
    # =====================
    with tab1:
        st.subheader("🔍 Tracking & Preprocessing")

        with st.spinner("Menjalankan tracking sperma..."):
            tracking_result = tracking_pipeline(
                video_path=video_path,
                out_csv=tracks_csv
            )

        df_tracks = pd.read_csv(tracks_csv)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Total Partikel Terdeteksi",
                value=df_tracks["particle"].nunique()
            )

        with col2:
            st.metric(
                label="Total Frame",
                value=df_tracks["frame"].nunique()
            )

        st.markdown("### 📄 Cuplikan Final Tracks")
        st.dataframe(df_tracks.head(20), use_container_width=True)

        with st.spinner("Membuat visualisasi trajectory..."):
            draw_trajectory_video(
                video_path=video_path,
                tracks_csv=tracks_csv,
                output_path=trajectory_video
            )

        st.markdown("### 🎥 Video Trajectory")
        st.video(trajectory_video)

    # =====================
    # TAB 2 – INFERENCE
    # =====================
    with tab2:
        st.subheader("📊 Hasil Klasifikasi Motility & Morphology")

        # ---------------------
        # MOTILITY
        # ---------------------
        with st.spinner("Inference Motility..."):
            motility_result = run_motility_inference(
                video_path=video_path,
                tracks_csv=tracks_csv,
                model_path=os.path.join(MODEL_DIR, "motility_3dcnn.h5")
            )

        pr = motility_result["detail"]["PR"]
        np_ = motility_result["detail"]["NP"]
        im = motility_result["detail"]["IM"]
        total = pr + np_ + im

        motility_normal = ((pr + np_) / max(total, 1)) >= 0.4

        # ---------------------
        # MORPHOLOGY
        # ---------------------
        with st.spinner("Inference Morphology..."):
            morphology_result = run_morphology_inference(
                img_dir=os.path.join(TEMP_DIR, "morphology_binary")
            )

        total_morph = len(morphology_result)
        normal_morph = sum(
            1 for x in morphology_result if x["label"] == "Normal"
        )

        morph_normal = normal_morph >= 4

        # ---------------------
        # DASHBOARD
        # ---------------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("## 🚀 Motility")
            st.metric("PR", pr)
            st.metric("NP", np_)
            st.metric("IM", im)
            st.success(
                "Motil Normal ✅" if motility_normal else "Motil Abnormal ❌"
            )

        with col2:
            st.markdown("## 🧠 Morphology")
            st.metric("Normal", normal_morph)
            st.metric("Total", total_morph)
            st.success(
                "Morfologi Normal ✅" if morph_normal else "Morfologi Abnormal ❌"
            )

        st.markdown("---")
        st.markdown("## 🧾 Kesimpulan Akhir")

        if motility_normal and morph_normal:
            st.success("✅ Sperma DINYATAKAN NORMAL")
        else:
            st.error("❌ Sperma DINYATAKAN ABNORMAL")

else:
    st.info("⬅️ Upload video dan tekan **Jalankan Analisis**")

