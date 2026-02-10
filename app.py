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
# 1. CONFIG & STYLE (MED-TECH THEME)
# ==========================================
st.set_page_config(page_title="Sperm Analysis AI", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .main-result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #007bff;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: bold;
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 HALAMAN AWAL", 
    "⚙️ DATA LOADER & PROCESSING", 
    "🔬 ANALYSIS PROCESS", 
    "📊 SUMMARY DASHBOARD"
])

# ------------------------------------------
# TAB 1: HALAMAN AWAL
# ------------------------------------------
with tab1:
    st.title("DETEKSI ABNORMALITAS MOTILITY DAN MORFOLOGI SPERMATOZOA")
    st.subheader("Sistem Analisis Semen Otomatis Berbasis Artificial Intelligence")
    
    col_intro, col_img = st.columns([1, 1])
    with col_intro:
        st.markdown("""
        ### Deskripsi Proyek
        Sistem ini mengintegrasikan teknologi **Computer Vision** untuk membantu standarisasi analisis sperma.
        
        **Fitur Utama:**
        * **Tracking Multi-Objek:** Deteksi lintasan sperma secara *real-time*.
        * **Motility Grading:** Klasifikasi PR, NP, dan IM menggunakan arsitektur **3D-CNN**.
        * **Morphology Analysis:** Identifikasi struktur Normal/Abnormal dengan **EfficientNetV2S**.
        
        ### Petunjuk Penggunaan
        1. Buka Tab **Data Loader** untuk mengunggah video sampel.
        2. Klik tombol **Preprocessing** untuk ekstraksi fitur video.
        3. Lakukan kalkulasi pada Tab **Analysis Process**.
        4. Evaluasi hasil akhir pada Tab **Summary Dashboard**.
        """)
    
    with col_img:
        st.info("Informasi Grade Motilitas (WHO Standards)")
        # Tambahkan diagram jika tersedia
        

# ------------------------------------------
# TAB 2: DATA LOADER & PROCESSING
# ------------------------------------------
with tab2:
    st.header("Upload File & Digital Processing")
    video_file = st.file_uploader("Upload Video Sampel (.mp4, .avi)", type=['mp4', 'avi'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        if st.button("Lakukan Preprocessing & Tracking 🚀"):
            with st.status("Sedang memproses video...", expanded=True) as status:
                temp_dir = tempfile.mkdtemp()
                
                # A. Pipeline Preprocessing
                st.write("🔧 Menjalankan transformasi Grayscale & Contrast...")
                prep_path = prepare_video_pipeline(tfile.name, temp_dir)
                st.session_state.prepared_video = prep_path
                
                # Visualisasi Preprocessing
                cap = cv2.VideoCapture(tfile.name)
                ret, frame_orig = cap.read()
                if ret:
                    c1, c2, c3 = st.columns(3)
                    c1.image(frame_orig, caption="Frame Asli", use_container_width=True)
                    c2.image(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY), caption="Frame Grayscale", use_container_width=True)
                    # Simulasi kontras yang ditingkatkan
                    c3.image(cv2.convertScaleAbs(frame_orig, alpha=1.5, beta=10), caption="Frame Contrast", use_container_width=True)
                cap.release()
                
                # B. Pipeline Tracking
                st.write("📍 Mendeteksi partikel dan membuat lintasan (Linker)...")
                csv_out = os.path.join(temp_dir, "tracks.csv")
                df = tracking_pipeline(prep_path, csv_out)
                
                # FIX: Reset index agar kolom 'frame' dan 'particle' terbaca sebagai kolom, bukan index
                st.session_state.tracks_df = df.reset_index()
                
                status.update(label="Preprocessing & Tracking Selesai!", state="complete")

        # Tampilkan Informasi Partikel jika tracking sudah selesai
        if st.session_state.tracks_df is not None:
            st.divider()
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class='metric-container'>
                            <p>Total Partikel Terdeteksi</p>
                            <h2>{st.session_state.tracks_df['particle'].nunique()}</h2>
                            </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='metric-container'>
                            <p>Total Baris Data Lintasan</p>
                            <h2>{len(st.session_state.tracks_df)}</h2>
                            </div>""", unsafe_allow_html=True)
            
            with st.expander("Lihat Tabel Final Tracks"):
                st.dataframe(st.session_state.tracks_df, use_container_width=True)

# ------------------------------------------
# TAB 3: PROCESS MOTILITY & MORFOLOGI
# ------------------------------------------
with tab3:
    st.header("Analisis Laboratorium AI")
    if st.session_state.tracks_df is None:
        st.warning("⚠️ Silakan selesaikan proses di Tab **Data Loader** terlebih dahulu.")
    else:
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            st.subheader("Analisis Motilitas")
            if st.button("🚀 Jalankan Hitung Motilitas"):
                # Gunakan model lokal untuk motility
                model_mot = "model_motility.h5" 
                with st.spinner("3D-CNN sedang mengklasifikasi pergerakan..."):
                    st.session_state.motility_results = run_motility_analysis(
                        st.session_state.prepared_video, 
                        st.session_state.tracks_df, 
                        model_mot
                    )
                st.success("Analisis Motilitas Berhasil!")

        with col_btn2:
            st.subheader("Analisis Morfologi")
            if st.button("🔬 Jalankan Hitung Morfologi"):
                # Morphology otomatis download dari HuggingFace di dalam fungsi
                with st.spinner("EfficientNet sedang menganalisis bentuk (ROI Erosion)..."):
                    st.session_state.morphology_results = run_morphology_analysis(
                        st.session_state.prepared_video, 
                        st.session_state.tracks_df
                    )
                st.success("Analisis Morfologi Berhasil!")

# ------------------------------------------
# TAB 4: SUMMARY DASHBOARD
# ------------------------------------------
with tab4:
    if st.session_state.motility_results is None or st.session_state.morphology_results is None:
        st.info("💡 Hasil ringkasan akan muncul di sini setelah Anda menjalankan analisis di Tab 3.")
    else:
        # Kotak Paling Atas: Main Result
        # Logika sederhana fertil (contoh: PR > 32%)
        m_results = st.session_state.motility_results
        pr_ratio = (len(m_results[m_results['motility_label'] == 'PR']) / len(m_results)) * 100
        status_fertil = "FERTIL" if pr_ratio > 32 else "INFERTIL"
        
        st.markdown(f"""<div class='main-result-card'>
                    <h1>Main Result : {status_fertil}</h1>
                    <p>Berdasarkan parameter motilitas progresif dan morfologi normal</p>
                    </div>""", unsafe_allow_html=True)

        # Baris Tengah: Motility dan Morphology (%)
        r1_c1, r1_c2 = st.columns([2, 1])
        
        with r1_c1:
            with st.container(border=True):
                st.subheader("Motility (%)")
                m_counts = m_results['motility_label'].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Progressive (PR)", m_counts.get('PR', 0))
                c2.metric("Non-Progressive (NP)", m_counts.get('NP', 0))
                c3.metric("Immotile (IM)", m_counts.get('IM', 0))
                st.bar_chart(m_counts)

        with r1_c2:
            with st.container(border=True):
                st.subheader("Morfologi (%)")
                mor_results = st.session_state.morphology_results
                mor_counts = mor_results['morphology_label'].value_counts()
                st.write(f"**Normal:** {mor_counts.get('Normal', 0)}")
                st.write(f"**Abnormal:** {mor_counts.get('Abnormal', 0)}")
                # Bar chart kecil untuk morfologi
                st.bar_chart(mor_counts)

        # Baris Bawah: Visualisasi Video dan Sampel Morfologi
        r2_c1, r2_c2 = st.columns([2, 1])
        
        with r2_c1:
            with st.container(border=True):
                st.subheader("Visualisasi Pergerakan Sperma")
                st.video(st.session_state.prepared_video)
                st.caption("Video hasil preprocessing (Grayscale/Contrast) yang digunakan untuk tracking.")

        with r2_c2:
            with st.container(border=True):
                st.subheader("Sampel Normal Morfologi")
                normal_samples = mor_results[mor_results['morphology_label'] == 'Normal']
                if not normal_samples.empty:
                    # Ambil satu gambar sampel display
                    st.image(normal_samples.iloc[0]['image_display'], caption="Contoh Struktur Normal", use_container_width=True)
                else:
                    st.warning("Tidak ditemukan sampel sperma normal.")
