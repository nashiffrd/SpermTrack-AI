import streamlit as st
import pandas as pd
import os
from pathlib import Path
from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference
from tracking.pipeline import tracking_pipeline
from draw_trajectory_video import draw_trajectory

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sperm Analyzer",
    layout="wide"
)

# Temp folder untuk simpan video upload & CSV
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
TRACK_CSV = UPLOAD_DIR / "final_tracks.csv"

# -----------------------------
# TAB NAVIGATOR
# -----------------------------
tabs = st.tabs(["Data Loader", "Analysis Dashboard"])

# -----------------------------
# TAB 1: Data Loader
# -----------------------------
with tabs[0]:
    st.header("Video Upload & Tracking Results")

    uploaded_file = st.file_uploader("Upload sperm video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        video_path = UPLOAD_DIR / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Video uploaded: {uploaded_file.name}")

        st.info("Running preprocessing & tracking...")

        try:
            df_tracks = tracking_pipeline(str(video_path), str(TRACK_CSV))
            st.success("Tracking completed!")

            st.subheader("Final Tracks Preview")
            st.dataframe(df_tracks.head(20))

            total_particles = df_tracks['particle'].nunique()
            st.info(f"Total sperms detected: {total_particles}")

        except Exception as e:
            st.error(f"Error during tracking: {e}")

# -----------------------------
# TAB 2: Analysis Dashboard
# -----------------------------
with tabs[1]:
    st.header("Motility & Morphology Inference")

    if TRACK_CSV.exists():
        st.info("Running inference on tracked sperms...")

        try:
            # ---------------------
            # Motility inference
            # ---------------------
            motility_results = run_motility_inference(df_tracks)
            motility_summary = pd.DataFrame(motility_results)
            st.subheader("Motility Results")
            st.dataframe(motility_summary)

            # Compute motility decision (PR + NP > 40%)
            pr_np = motility_summary[motility_summary['label'].isin(['PR','NP'])]
            percent_normal = len(pr_np)/len(motility_summary)*100
            motility_decision = "Normal" if percent_normal > 40 else "Abnormal"
            st.info(f"Motility Decision: {motility_decision} ({percent_normal:.1f}% PR+NP)")

            # ---------------------
            # Morphology inference
            # ---------------------
            morphology_results = run_morphology_inference("morfologi/unlabeled")
            morpho_summary = pd.DataFrame(morphology_results)
            st.subheader("Morphology Results")
            st.dataframe(morpho_summary)

            # Compute morphology decision (Normal >= 4)
            normal_count = sum([1 for r in morphology_results if r['label']=='Normal'])
            morpho_decision = "Normal" if normal_count >= 4 else "Abnormal"
            st.info(f"Morphology Decision: {morpho_decision} ({normal_count} normal sperms)")

            # ---------------------
            # Trajectory video
            # ---------------------
            st.subheader("Annotated Sperm Trajectory Video")
            traj_video_path = draw_trajectory(str(video_path), df_tracks)
            st.video(traj_video_path)

        except Exception as e:
            st.error(f"Error during inference: {e}")

    else:
        st.warning("No tracking CSV available. Please upload video and run tracking first.")
