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
st.set_page_config(page_title="SpermTrack AI", layout="wide", page_icon="🧬")

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
st.title("🧬 SpermTrack AI")
st.subheader("Sistem Analisis Semen Otomatis Untuk Deteksi Abnormalitas Motility dan Morfologi Spermatozoa")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Halaman Awal", 
    "⚙️ Data Loader & Processing", 
    "🔬 Analysis Process", 
    "📊 Summary Dashboard"
])

# ------------------------------------------
# TAB 1: HALAMAN AWAL
# ------------------------------------------
with tab1:
    st.write("""
    Aplikasi ini dirancang untuk mempermudah analisis kualitas spermatozoa melalui video mikroskopis berdasarkan standart WHO.
    Sistem bekerja secara otomatis mulai dari pembersihan video, pelacakan partikel (tracking), 
    hingga klasifikasi menggunakan model AI (3D-CNN dan EfficientNetV2S).

    **Cara Penggunaan:**
    1. Upload video pada tab **Data Loader & Processing**.
    2. Lakukan preprocessing dan tracking.
    3. Jalankan analisis motilitas dan morfologi pada tab **Analysis Process**.
    4. Lihat kesimpulan akhir pada tab **Summary Dashboard**.
    """)
    st.markdown("""
    <div style="
        background-color: #e8f4f9; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 3px solid #007bff;
        border-right: 3px solid #007bff;
        text-align: center;
        color: #0c5460;
        font-family: sans-serif;
        margin-bottom: 20px;
    ">
        Gunakan navigasi tab di atas untuk memulai proses analisis.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: DATA LOADER & PROCESSING
# ------------------------------------------
with tab2:
    st.header("Upload & Digital Processing")
    video_file = st.file_uploader("Pilih Video Sperma", type=['mp4', 'avi'], key="sperm_video_uploader")

    if video_file:
        current_video_id = f"{video_file.name}_{video_file.size}"
        
        if st.session_state.get('last_video_id') != current_video_id:
            st.session_state.tracks_df = None
            st.session_state.sample_frame = None
            st.session_state.last_video_id = current_video_id

        if st.session_state.tracks_df is None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            
            with st.status("Preprocessing and Tracking are Running") as status:
                temp_dir = tempfile.mkdtemp()
                cap = cv2.VideoCapture(tfile.name)
                ret, frame = cap.read()
                if ret:
                    st.session_state.sample_frame = frame
                
                prep_path = prepare_video_pipeline(tfile.name, temp_dir)
                st.session_state.prepared_video = prep_path
                
                df = tracking_pipeline(prep_path, os.path.join(temp_dir, "tracks.csv"))
                if 'frame' not in df.columns:
                    df = df.reset_index()
                else:
                    df = df.reset_index(drop=True)
                st.session_state.tracks_df = df
                status.update(label="Preprocessing & Tracking Selesai!", state="complete")

        if st.session_state.tracks_df is not None:
            if st.session_state.sample_frame is not None:
                st.write("### Visualisasi Tahap A (Preprocessing)")
                f1, f2, f3 = st.columns(3)
                img = st.session_state.sample_frame
                f1.image(img, caption="Frame Asli", use_container_width=True)
                f2.image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), caption="Grayscale", use_container_width=True)
                f3.image(cv2.convertScaleAbs(img, alpha=1.5, beta=10), caption="Contrast", use_container_width=True)

            st.write("### Visualisasi Tahap B (Tracking Data)")
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
        st.warning("Silakan selesaikan proses di Tab 2 (Upload & Tracking) terlebih dahulu.")
    else:
        if st.button("🚀 Jalankan Analisis Motility dan Morfologi"):
            with st.spinner("Analysis Process is Running"):
                st.session_state.motility_results = run_motility_analysis(
                    st.session_state.prepared_video, 
                    st.session_state.tracks_df, 
                    "model_motility.h5"
                )
                
                st.session_state.morphology_results = run_morphology_analysis(
                    st.session_state.prepared_video, 
                    st.session_state.tracks_df
                )
            st.success("Analisis Motilitas & Morfologi Selesai!")

        if st.session_state.motility_results is not None and st.session_state.morphology_results is not None:
            st.divider()
            st.subheader("📊 Exploratory Data Analysis (EDA)")
            
            eda_col1, eda_col2 = st.columns(2)
            with eda_col1:
                st.write("**Distribusi Motilitas**")
                mot_counts = st.session_state.motility_results['motility_label'].value_counts()
                st.bar_chart(mot_counts, color="#007bff")

            with eda_col2:
                st.write("**Distribusi Morfologi**")
                morf_counts = st.session_state.morphology_results['morphology_label'].value_counts()
                st.bar_chart(morf_counts, color="#ff4b4b")

            st.divider()
            st.subheader("📋 Tabel Summary Klasifikasi")
            
            # Penggabungan data dengan menyertakan kolom confidence
            df_mot = st.session_state.motility_results[['particle', 'motility_label', 'confidence']]
            df_morf = st.session_state.morphology_results[['particle', 'morphology_label', 'confidence']]
            
            summary_df = pd.merge(df_mot, df_morf, on='particle', how='inner', suffixes=('_mot', '_morf'))
            coords = st.session_state.tracks_df.groupby('particle').first().reset_index()[['particle', 'x', 'y', 'frame']]
            final_summary = pd.merge(coords, summary_df, on='particle', how='inner')
            
            # Menampilkan tabel dengan kolom confidence agar terlihat progresnya
            final_summary = final_summary[['x', 'y', 'frame', 'particle', 'motility_label', 'morphology_label', 'confidence_mot', 'confidence_morf']]
            final_summary.columns = ['X', 'Y', 'Frame', 'ID Particle', 'Motility', 'Morphology', 'Conf Motility', 'Conf Morphology']
            
            st.dataframe(final_summary, use_container_width=True)
            
# ------------------------------------------
# TAB 4: SUMMARY DASHBOARD
# ------------------------------------------
with tab4:
    if st.session_state.motility_results is None or st.session_state.morphology_results is None:
        st.info("Hasil analisis akan tampil setelah Tab 3 selesai diproses.")
    else:
        m_res = st.session_state.motility_results
        mo_res = st.session_state.morphology_results
        total_sperma = len(m_res)

        pr_val = len(m_res[m_res['motility_label'] == 'PR'])
        pr_percent = (pr_val / total_sperma) * 100 if total_sperma > 0 else 0
        
        normal_mo_val = len(mo_res[mo_res['morphology_label'] == 'Normal'])
        normal_mo_percent = (normal_mo_val / len(mo_res)) * 100 if len(mo_res) > 0 else 0

        # Diagnosis Logic
        if pr_percent < 32 and normal_mo_percent < 4:
            status_f, deskripsi, bg_color = "Asthenoteratozoospermia", "Motilitas & Morfologi Normal Rendah", "#721c24"
        elif pr_percent < 32:
            status_f, deskripsi, bg_color = "Asthenozoospermia", "Gerak Sperma Rendah", "#dc3545"
        elif normal_mo_percent < 4:
            status_f, deskripsi, bg_color = "Teratozoospermia", "Bentuk Normal Rendah", "#fd7e14"
        else:
            status_f, deskripsi, bg_color = "Normozoospermia", "Sampel Normal (Sesuai Standar WHO)", "#28a745"

        # 1. Header Diagnosis
        st.markdown(f"""
            <div style='background-color: {bg_color}; padding: 25px; border-radius: 15px 15px 0 0; text-align: center; color: white; margin-bottom: 0px;'>
                <p style='margin:0; font-size: 1.1rem; opacity: 0.9;'>Hasil Analisis Laboratorium:</p>
                <h1 style='margin:5px 0; font-size: 3rem; font-weight: 800;'>{status_f}</h1>
                <p style='margin:0; font-style: italic;'>{deskripsi}</p>
            </div>
        """, unsafe_allow_html=True)

        # 2. Body Panel
        st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 0 0 15px 15px; border: 1px solid #e9ecef; border-top: none;'>
                <div style='display: flex; justify-content: space-around; text-align: center;'>
                    <div style='flex: 1; border-right: 1px solid #dee2e6;'>
                        <p style='margin-bottom:0; color: #6c757d;'>PR Motility</p>
                        <h2 style='color:{bg_color}; margin-top:0;'>{pr_percent:.1f}%</h2>
                        <small style='color: #adb5bd;'>Threshold: 32%</small>
                    </div>
                    <div style='flex: 1;'>
                        <p style='margin-bottom:0; color: #6c757d;'>Normal Morphology</p>
                        <h2 style='color:{bg_color}; margin-top:0;'>{normal_mo_percent:.1f}%</h2>
                        <small style='color: #adb5bd;'>Threshold: 4%</small>
                    </div>
                </div>
                <hr style='margin: 20px 0; border: 0.5px solid #dee2e6;'>
                <p style='text-align: center; font-weight: bold; color: #495057; margin-bottom: 15px;'>Detail Perhitungan Partikel</p>
                <div style='display: flex; justify-content: space-between; text-align: center;'>
                    <div style='flex: 1; border-right: 1px solid #dee2e6;'><small style='color: #6c757d;'>PR</small><h4 style='margin:0;'>{m_res['motility_label'].value_counts().get('PR', 0)}</h4></div>
                    <div style='flex: 1; border-right: 1px solid #dee2e6;'><small style='color: #6c757d;'>NP</small><h4 style='margin:0;'>{m_res['motility_label'].value_counts().get('NP', 0)}</h4></div>
                    <div style='flex: 1; border-right: 1px solid #dee2e6;'><small style='color: #6c757d;'>IM</small><h4 style='margin:0;'>{m_res['motility_label'].value_counts().get('IM', 0)}</h4></div>
                    <div style='flex: 1; border-right: 1px solid #dee2e6;'><small style='color: #6c757d;'>Normal</small><h4 style='margin:0;'>{mo_res['morphology_label'].value_counts().get('Normal', 0)}</h4></div>
                    <div style='flex: 1;'><small style='color: #6c757d;'>Abnormal</small><h4 style='margin:0;'>{mo_res['morphology_label'].value_counts().get('Abnormal', 0)}</h4></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 3. AI CONFIDENCE SCORE
        conf_mot = m_res['confidence'].mean() * 100 if 'confidence' in m_res.columns else 0
        conf_mo = mo_res['confidence'].mean() * 100 if 'confidence' in mo_res.columns else 0
        sys_conf = (conf_mot + conf_mo) / 2

        st.markdown(f"""
            <div style='display: flex; flex-direction: column; align-items: center; margin-top: 15px; padding: 15px; background-color: #f8f9fa; border-radius: 12px; border: 1px dashed #ced4da;'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='color: #495057; font-weight: bold; font-size: 0.9rem;'>🤖 AI Analysis Confidence:</span>
                    <div style='width: 200px; background-color: #e9ecef; border-radius: 10px; height: 12px; overflow: hidden; border: 1px solid #dee2e6;'>
                        <div style='width: {sys_conf}%; background-color: {bg_color}; height: 100%; transition: width 0.5s;'></div>
                    </div>
                    <span style='color: {bg_color}; font-weight: 800; font-size: 1rem;'>{sys_conf:.2f}%</span>
                </div>
                <p style='margin-top: 8px; color: #6c757d; font-size: 0.75rem; text-align: center;'>
                    Skor ini menunjukkan tingkat kepastian model dalam mengklasifikasikan sel pada sampel ini.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # 4. RESET BUTTON
        st.write("")
        if st.button("🔄 Reset Analisis & Mulai Baru", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
