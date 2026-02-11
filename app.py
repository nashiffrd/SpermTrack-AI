import streamlit as st
import pandas as pd
import cv2
import os
import tempfile
import numpy as np
from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from models.motility_analyzer import run_motility_analysis
from models.morphology_analyzer import run_morphology_analysis

# ==========================================
# 1. CONFIG & STYLE
# ==========================================
st.set_page_config(page_title="Sperm Analysis AI", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main-result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #007bff;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'tracks_df' not in st.session_state: st.session_state.tracks_df = None
if 'prepared_video' not in st.session_state: st.session_state.prepared_video = None
if 'motility_results' not in st.session_state: st.session_state.motility_results = None
if 'morphology_results' not in st.session_state: st.session_state.morphology_results = None

# ==========================================
# 3. TAB NAVIGATION
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Halaman Awal","⚙️ Data Loader & Processing","🔬 Analysis Process","📊 Summary Dashboard"])

# ------------------------------------------
# TAB 1: HALAMAN AWAL
# ------------------------------------------
with tab1:
    st.title("DETEKSI ABNORMALITAS MOTILITY DAN MORFOLOGI SPERMATOZOA")
    st.subheader("Sistem Analisis Semen Otomatis Berbasis Deep Learning")
    st.write("""
    Aplikasi ini dirancang untuk mempermudah analisis kualitas spermatozoa melalui video mikroskopis.
    Sistem bekerja secara otomatis mulai dari pembersihan video, pelacakan partikel (tracking), 
    hingga klasifikasi menggunakan model AI (3D-CNN dan EfficientNetV2S).

    **Cara Penggunaan:**
    1. Upload video pada tab **Data Loader**.
    2. Lakukan preprocessing dan tracking.
    3. Jalankan analisis motilitas dan morfologi pada tab **Analysis Process**.
    4. Lihat kesimpulan akhir pada tab **Summary Dashboard**.
    """)
    st.info("Gunakan navigasi tab di atas untuk memulai proses analisis.")

# ------------------------------------------
# TAB 2: DATA LOADER & PROCESSING
# ------------------------------------------
with tab2:
    st.header("Upload & Digital Processing")
    video_file = st.file_uploader("Pilih Video Sperma", type=['mp4', 'avi'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        if st.button("Jalankan Preprocessing & Tracking 🚀"):
            with st.status("Sedang memproses...", expanded=True) as status:
                temp_dir = tempfile.mkdtemp()
                
                # A. Preprocessing
                prep_path = prepare_video_pipeline(tfile.name, temp_dir)
                st.session_state.prepared_video = prep_path
                
                # Visualisasi (Asli > Gray > Contrast)
                cap = cv2.VideoCapture(tfile.name)
                ret, frame = cap.read()
                if ret:
                    c1, c2, c3 = st.columns(3)
                    c1.image(frame, caption="Frame Asli", use_container_width=True)
                    c2.image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), caption="Grayscale", use_container_width=True)
                    c3.image(cv2.convertScaleAbs(frame, alpha=1.5, beta=10), caption="Contrast", use_container_width=True)
                cap.release()
                
                # B. Tracking
                df = tracking_pipeline(prep_path, os.path.join(temp_dir, "tracks.csv"))
                
                # --- FIX VALUE ERROR: CEK DUPLIKASI KOLOM SEBELUM RESET INDEX ---
                if 'frame' not in df.columns:
                    df = df.reset_index()
                else:
                    df = df.reset_index(drop=True)
                
                st.session_state.tracks_df = df
                status.update(label="Tracking Selesai!", state="complete")

        if st.session_state.tracks_df is not None:
            st.divider()
            m1, m2 = st.columns(2)
            m1.markdown(f"<div class='metric-container'><h4>Total Partikel</h4><h2>{st.session_state.tracks_df['particle'].nunique()}</h2></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-container'><h4>Total Lintasan</h4><h2>{len(st.session_state.tracks_df)}</h2></div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.tracks_df.head(50), use_container_width=True)

# ------------------------------------------
# TAB 3: ANALYSIS PROCESS
# ------------------------------------------
with tab3:
    st.header("Kalkulasi Motilitas & Morfologi")
    if st.session_state.tracks_df is None:
        st.warning("Silakan selesaikan proses di Tab 2 terlebih dahulu.")
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("🚀 Jalankan Analisis Motilitas"):
                with st.spinner("Menghitung pergerakan..."):
                    st.session_state.motility_results = run_motility_analysis(
                        st.session_state.prepared_video, st.session_state.tracks_df, "model_motility.h5"
                    )
                st.success("Motilitas Selesai!")
        with col_m2:
            if st.button("🔬 Jalankan Analisis Morfologi"):
                with st.spinner("Menganalisis bentuk..."):
                    st.session_state.morphology_results = run_morphology_analysis(
                        st.session_state.prepared_video, st.session_state.tracks_df
                    )
                st.success("Morfologi Selesai!")

# ------------------------------------------
# TAB 4: SUMMARY DASHBOARD
# ------------------------------------------
with tab4:
    if st.session_state.motility_results is None or st.session_state.morphology_results is None:
        st.info("Hasil analisis akan tampil setelah Tab 3 selesai diproses.")
    else:
        # Main Result
        m_res = st.session_state.motility_results
        pr_val = len(m_res[m_res['motility_label'] == 'PR'])
        status_f = "FERTIL" if pr_val > (0.32 * len(m_res)) else "INFERTIL"
        
        st.markdown(f"<div class='main-result-card'><h1>Main Result : {status_f}</h1></div>", unsafe_allow_html=True)
        st.write("")

        # Motility & Morphology (%)
        r1c1, r1c2 = st.columns([2, 1])
        with r1c1:
            with st.container(border=True):
                st.write("**Motility (%)**")
                counts = m_res['motility_label'].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Progressive", counts.get('PR', 0))
                c2.metric("Non-Progressive", counts.get('NP', 0))
                c3.metric("Immotile", counts.get('IM', 0))

        with r1c2:
            with st.container(border=True):
                st.write("**Morfologi (%)**")
                mo_res = st.session_state.morphology_results
                mo_counts = mo_res['morphology_label'].value_counts()
                st.write(f"Normal: {mo_counts.get('Normal', 0)}")
                st.write(f"Abnormal: {mo_counts.get('Abnormal', 0)}")

        # Video & Sampel
        r2c1, r2c2 = st.columns([2, 1])
        with r2c1:
            with st.container(border=True):
                st.write("**Visualisasi Pergerakan Sperma**")
                st.video(st.session_state.prepared_video)
        with r2c2:
            with st.container(border=True):
                st.write("**Sampel Normal Morfologi**")
                norm_img = mo_res[mo_res['morphology_label'] == 'Normal']
                if not norm_img.empty:
                    st.image(norm_img.iloc[0]['image_display'], use_container_width=True)
                else:
                    st.write("Tidak ada sampel normal.")

ini kode terakhir kita , ingat2 dan aku mau revisi
