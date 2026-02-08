import streamlit as st
import pandas as pd
import os

# =============================
# IMPORT PIPELINE
# =============================
from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference
from visualization.draw_trajectory_video import draw_trajectory_video

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Sperm Analysis System",
    layout="wide"
)

st.title("🧬 Integrated Sperm Motility & Morphology Analysis")

# =============================
# SIDEBAR CONFIG
# =============================
st.sidebar.header("📂 Data Configuration")

VIDEO_PATH = st.sidebar.text_input(
    "Video Path",
    value="data/video.mp4"
)

TRACKS_CSV = st.sidebar.text_input(
    "final_tracks.csv",
    value="data/final_tracks.csv"
)

MOTILITY_MODEL = st.sidebar.text_input(
    "Motility Model (.h5)",
    value="models/motility_model.h5"
)

MORPH_MODEL = st.sidebar.text_input(
    "Morphology Model (.h5)",
    value="models/morphology_model.h5"
)

RUN_BTN = st.sidebar.button("🚀 Run Full Analysis")

# =============================
# TABS
# =============================
tab1, tab2 = st.tabs([
    "📥 Data Loader & Preprocessing",
    "📊 Final Classification Dashboard"
])

# =========================================================
# TAB 1 — DATA LOADER & PREPROCESSING
# =========================================================
with tab1:
    st.header("📥 Data Loader & Preprocessing Summary")

    if os.path.exists(TRACKS_CSV):
        df = pd.read_csv(TRACKS_CSV)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Particle", df["particle"].nunique())
        col2.metric("Total Frame", df["frame"].nunique())
        col3.metric("Total Detection", len(df))

        st.subheader("📋 Preview final_tracks.csv")
        st.dataframe(df.head(20))

    else:
        st.warning("⚠ final_tracks.csv tidak ditemukan")

# =========================================================
# TAB 2 — DASHBOARD BESAR (MOTILITY + MORPHOLOGY + TRAJECTORY)
# =========================================================
with tab2:
    st.header("📊 Final Classification Result")

    if RUN_BTN:

        with st.spinner("Running full inference pipeline..."):

            # -----------------------------
            # MOTILITY INFERENCE
            # -----------------------------
            motility = run_motility_inference(
                video_path=VIDEO_PATH,
                tracks_csv=TRACKS_CSV,
                model_path=MOTILITY_MODEL
            )

            pr = motility["detail"]["PR"]
            np_ = motility["detail"]["NP"]
            im = motility["detail"]["IM"]
            total_mot = pr + np_ + im

            pr_pct = pr / total_mot * 100
            np_pct = np_ / total_mot * 100
            im_pct = im / total_mot * 100

            motility_status = (
                "NORMAL" if (pr_pct + np_pct) > 40 else "ABNORMAL"
            )

            # -----------------------------
            # MORPHOLOGY INFERENCE
            # -----------------------------
            morphology = run_morphology_inference("data/morphology/binary_erotion")

            normal_pct = morphology["normal_pct"]
            abnormal_pct = morphology["abnormal_pct"]

            morphology_status = (
                "NORMAL" if normal_pct >= 4 else "ABNORMAL"
            )

        # =============================
        # DASHBOARD DISPLAY
        # =============================
        st.markdown("## 🧠 Classification Dashboard")

        colA, colB = st.columns(2)

        with colA:
            st.success(f"🚴 MOTILITY STATUS: **{motility_status}**")
            st.write(f"**PR** : {pr_pct:.2f}%")
            st.write(f"**NP** : {np_pct:.2f}%")
            st.write(f"**IM** : {im_pct:.2f}%")

        with colB:
            st.info(f"🧬 MORPHOLOGY STATUS: **{morphology_status}**")
            st.write(f"**Normal**   : {normal_pct:.2f}%")
            st.write(f"**Abnormal** : {abnormal_pct:.2f}%")

        st.divider()

        # =============================
        # TRAJECTORY VIDEO
        # =============================
        st.subheader("🎥 Sperm Trajectory Visualization")

        output_video = "temp/trajectory.mp4"
        os.makedirs("temp", exist_ok=True)

        draw_trajectory_video(
            video_path=VIDEO_PATH,
            tracks_csv=TRACKS_CSV,
            output_path=output_video
        )

        st.video(output_video)

    else:
        st.info("⬅ Tekan **Run Full Analysis** di sidebar untuk memulai")
