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
    .main-result-card { background-color: #ffffff; padding: 20px; border-radius: 12px; text-align: center; border: 2px solid #007bff; }
    .metric-container { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. JUDUL GLOBAL
# ==========================================
st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #ffffff; border-radius: 10px; margin-bottom: 20px; border-bottom: 4px solid #007bff;">
        <h1 style='margin: 0; color: #007bff;'>SPERM ANALYSIS AI SYSTEM</h1>
        <p style='margin: 5px; color: #666; font-size: 1.2rem;'>Laboratorium Digital: Deteksi Otomatis Motilitas & Morfologi</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE (CONTROL TABS & DATA)
# ==========================================
if 'active_tab' not in st.session_state: st.session_state.active_tab = 0
if 'tracks_df' not in st.session_state: st.session_state.tracks_df = None
if 'prepared_video' not in st.session_state: st.session_state.prepared_video = None
if 'motility_results' not in st.session_state: st.session_state.motility_results = None
if 'morphology_results' not in st.session_state: st.session_state.morphology_results = None

# ==========================================
# 4. TAB NAVIGATION (LINKED TO SESSION STATE)
# ==========================================
tab_titles = ["🏠 Halaman Awal", "⚙️ Data Loader & Processing", "🔬 Analysis Process", "📊 Summary Dashboard"]
active_tab = st.tabs(tab_titles)

# ------------------------------------------
# TAB 1: HALAMAN AWAL
# ------------------------------------------
with active_tab[0]:
    st.title("SELAMAT DATANG")
    st.write("Sistem ini akan memandu Anda melakukan analisis sperma secara otomatis menggunakan Deep Learning.")
    if st.button("Mulai Analisis ➡"):
        st.session_state.active_tab = 1
        st.rerun()

# ------------------------------------------
# TAB 2: DATA LOADER & PROCESSING
# ------------------------------------------
with active_tab[1]:
    st.header("Upload & Auto Processing")
    video_file = st.file_uploader("Pilih Video Sperma", type=['mp4', 'avi'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        # PROSES OTOMATIS TANPA TOMBOL
        if st.session_state.tracks_df is None:
            with st.status("Sedang memproses secara otomatis...", expanded=True) as status:
                temp_dir = tempfile.mkdtemp()
                prep_path = prepare_video_pipeline(tfile.name, temp_dir)
                st.session_state.prepared_video = prep_path
                
                df = tracking_pipeline(prep_path, os.path.join(temp_dir, "tracks.csv"))
                
                # Fix index issue
                if 'frame' not in df.columns: df = df.reset_index()
                else: df = df.reset_index(drop=True)
                
                st.session_state.tracks_df = df
                status.update(label="Preprocessing & Tracking Selesai!", state="complete")

        # Menampilkan Informasi Hasil
        if st.session_state.tracks_df is not None:
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='metric-container'><h4>Total Partikel</h4><h2>{st.session_state.tracks_df['particle'].nunique()}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-container'><h4>Total Lintasan</h4><h2>{len(st.session_state.tracks_df)}</h2></div>", unsafe_allow_html=True)
            
            st.divider()
            if st.button("Lanjutkan Sistem Deteksi ➡"):
                st.session_state.active_tab = 2
                st.rerun()

# ------------------------------------------
# TAB 3: ANALYSIS PROCESS
# ------------------------------------------
with active_tab[2]:
    st.header("Sistem Deteksi AI")
    if st.session_state.tracks_df is None:
        st.warning("Data belum diproses di Tab sebelumnya.")
    else:
        # SATU TOMBOL UNTUK SEMUA MODEL
        if st.button("Jalankan Deteksi Motilitas & Morfologi 🚀"):
            with st.spinner("AI sedang menganalisis motilitas (3D-CNN) & morfologi (EfficientNet)..."):
                # Motility
                st.session_state.motility_results = run_motility_analysis(
                    st.session_state.prepared_video, st.session_state.tracks_df, "model_motility.h5"
                )
                # Morphology
                st.session_state.morphology_results = run_morphology_analysis(
                    st.session_state.prepared_video, st.session_state.tracks_df
                )

        # EDA HASIL DETEKSI (Jika sudah run)
        if st.session_state.motility_results is not None and st.session_state.morphology_results is not None:
            st.divider()
            st.subheader("EDA (Exploratory Data Analysis) Hasil Deteksi")
            eda1, eda2 = st.columns(2)
            
            with eda1:
                st.write("**Distribusi Motilitas**")
                m_counts = st.session_state.motility_results['motility_label'].value_counts()
                st.bar_chart(m_counts)
                
            with eda2:
                st.write("**Distribusi Morfologi**")
                mo_counts = st.session_state.morphology_results['morphology_label'].value_counts()
                st.bar_chart(mo_counts)
            
            st.divider()
            if st.button("Lihat Summary Dashboard ➡"):
                st.session_state.active_tab = 3
                st.rerun()

# ------------------------------------------
# TAB 4: SUMMARY DASHBOARD
# ------------------------------------------
with active_tab[3]:
    if st.session_state.motility_results is None:
        st.info("Selesaikan analisis di Tab 3 untuk melihat Summary.")
    else:
        st.markdown("<h2 style='text-align: center; color: #1f77b4;'>LAPORAN HASIL ANALISIS SPERMATOZOA</h2>", unsafe_allow_html=True)
        
        m_res = st.session_state.motility_results
        pr_val = len(m_res[m_res['motility_label'] == 'PR'])
        status_f = "FERTIL" if pr_val > (0.32 * len(m_res)) else "INFERTIL"
        
        st.markdown(f"<div class='main-result-card'><h1>Main Result : {status_f}</h1></div>", unsafe_allow_html=True)
        
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

        r2c1, r2c2 = st.columns([2, 1])
        with r2c1:
            with st.container(border=True):
                st.write("**Visualisasi Pergerakan**")
                st.video(st.session_state.prepared_video)
        with r2c2:
            with st.container(border=True):
                st.write("**Sampel Normal Morfologi**")
                norm_img = mo_res[mo_res['morphology_label'] == 'Normal']
                if not norm_img.empty: st.image(norm_img.iloc[0]['image_display'], use_container_width=True)
                else: st.write("Tidak ada sampel normal.")
