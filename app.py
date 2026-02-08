import streamlit as st
from motility_inference.pipeline import run_motility_inference

st.set_page_config(page_title="Motility Test", layout="centered")

st.title("🧪 Motility Inference Test (No UI)")

VIDEO_PATH = "data/video5_prep.mp4"
TRACKS_CSV = "data/final_tracks.csv"
MODEL_PATH = "model_motility.h5"

if st.button("Run Motility Inference"):
    with st.spinner("Running pipeline..."):
        result = run_motility_inference(
            video_path=VIDEO_PATH,
            tracks_csv=TRACKS_CSV,
            model_path=MODEL_PATH
        )

    st.success("Inference selesai ✅")
    st.write("### Label Utama")
    st.write(result["label"])

    st.write("### Detail")
    st.json(result["detail"])
