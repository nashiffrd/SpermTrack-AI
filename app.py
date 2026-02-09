import os
import cv2
import streamlit as st
import pandas as pd

from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference

# ===============================
# PATH CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WORK_DIR = os.path.join(BASE_DIR, "workdir")
MODEL_DIR = os.path.join(BASE_DIR)

MOTILITY_MODEL_PATH = os.path.join(MODEL_DIR, "model_motility.h5")
MORPHOLOGY_MODEL_PATH = os.path.join(MODEL_DIR, "model_morphology.h5")

os.makedirs(WORK_DIR, exist_ok=True)

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Sperm Analysis System",
    layout="wide"
)

# ===============================
# SESSION STATE INIT
# ===============================
for key in [
    "video_path",
    "prepared_video",
    "tracks_csv",
    "tracks_df",
    "motility_result",
    "morphology_result"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ===============================
# SIDEBAR NAV
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Halaman Awal",
        "Data Loader",
        "Data Preprocessing",
        "Main Dashboard"
    ]
)

# ===============================
# HALAMAN AWAL
# ===============================
if page == "Halaman Awal":
    st.title("Automated Sperm Analysis System")
    st.markdown("""
    Sistem ini melakukan analisis **motilitas** dan **morfologi sperma**
    secara otomatis berbasis *computer vision* dan *deep learning*.
    """)

    st.markdown("### Cara Penggunaan")
    st.markdown("""
    1. Upload video sperma  
    2. Sistem melakukan preprocessing & tracking  
    3. Model melakukan inferensi motilitas & morfologi  
    4. Hasil ditampilkan pada dashboard utama
    """)

    if st.button("🚀 Start Analysis"):
        st.session_state["nav"] = "Data Loader"
        st.rerun()

# ===============================
# DATA LOADER
# ===============================
elif page == "Data Loader":
    st.title("Data Loader")

    uploaded = st.file_uploader(
        "Upload Video Sperma",
        type=["mp4", "avi"]
    )

    if uploaded:
        video_path = os.path.join(WORK_DIR, uploaded.name)
        with open(video_path, "wb") as f:
            f.write(uploaded.read())

        st.success("✅ Video berhasil diupload")
        st.session_state.video_path = video_path

        if st.button("➡️ Lanjutkan Preprocessing"):
            st.rerun()

# ===============================
# DATA PREPROCESSING
# ===============================
elif page == "Data Preprocessing":
    st.title("Data Preprocessing & Tracking")

    if st.session_state.video_path is None:
        st.warning("Silakan upload video terlebih dahulu.")
        st.stop()

    with st.spinner("⏳ Preprocessing & Tracking berjalan..."):
        prepared_video = prepare_video_pipeline(
            st.session_state.video_path,
            WORK_DIR
        )

        tracks_csv = os.path.join(WORK_DIR, "final_tracks.csv")
        tracks_df = tracking_pipeline(prepared_video, tracks_csv)

        st.session_state.prepared_video = prepared_video
        st.session_state.tracks_csv = tracks_csv
        st.session_state.tracks_df = tracks_df

    col1, col2 = st.columns(2)
    col1.metric("Total Partikel", tracks_df["particle"].nunique())
    col2.metric("Total Tracking", len(tracks_df))

    st.subheader("Final Tracks Data")
    st.dataframe(tracks_df.head(100))

# ===============================
# MAIN DASHBOARD
# ===============================
elif page == "Main Dashboard":
    st.title("📊 Main Dashboard")

    if st.session_state.tracks_csv is None:
        st.warning("Silakan selesaikan preprocessing terlebih dahulu.")
        st.stop()

    # ===============================
    # MOTILITY INFERENCE
    # ===============================
    with st.spinner("🧠 Inferensi Motilitas..."):
        motility = run_motility_inference(
            video_path=st.session_state.prepared_video,
            tracks_csv=st.session_state.tracks_csv,
            model_path=MOTILITY_MODEL_PATH
        )
        st.session_state.motility_result = motility

    detail = motility["detail"]
    total_m = sum(detail.values())

    pr = detail.get("PR", 0)
    np_ = detail.get("NP", 0)
    im = detail.get("IM", 0)

    pr_pct = pr / total_m * 100
    np_pct = np_ / total_m * 100
    im_pct = im / total_m * 100

    motility_status = (
        "FERTIL ✅" if (pr + np_) / total_m * 100 > 40 else "INFERTIL ❌"
    )

    # ===============================
    # MORPHOLOGY INFERENCE
    # ===============================
    with st.spinner("🧬 Inferensi Morfologi..."):
        morphology = run_morphology_inference(
            img_dir=os.path.join(WORK_DIR, "roi"),
            model_path=MORPHOLOGY_MODEL_PATH
        )
        st.session_state.morphology_result = morphology

    morph_df = pd.DataFrame(morphology)
    normal_cnt = (morph_df["label"] == "normal").sum()
    abnormal_cnt = (morph_df["label"] == "abnormal").sum()
    total_morph = normal_cnt + abnormal_cnt

    normal_pct = normal_cnt / total_morph * 100 if total_morph > 0 else 0
    abnormal_pct = abnormal_cnt / total_morph * 100 if total_morph > 0 else 0

    morphology_status = (
        "NORMAL ✅" if normal_pct > 4 else "ABNORMAL ❌"
    )

    # ===============================
    # DASHBOARD VIEW
    # ===============================
    st.markdown("## 🧪 Hasil Analisis Keseluruhan")

    st.markdown(f"""
    ### Motility Status: **{motility_status}**
    - PR: {pr_pct:.2f}%
    - NP: {np_pct:.2f}%
    - IM: {im_pct:.2f}%

    ### Morphology Status: **{morphology_status}**
    - Normal: {normal_pct:.2f}%
    - Abnormal: {abnormal_pct:.2f}%
    """)

    st.success("✅ Analisis selesai")
