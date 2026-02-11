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
    video_file = st.file_uploader("Pilih Video Sperma", type=['mp4', 'avi'])

    if video_file:
        # Membuat ID unik untuk video (nama + ukuran file)
        current_video_id = f"{video_file.name}_{video_file.size}"
        
        # Jika video yang diupload berbeda dengan yang terakhir diproses, reset session state
        if st.session_state.get('last_video_id') != current_video_id:
            st.session_state.tracks_df = None
            st.session_state.sample_frame = None
            st.session_state.last_video_id = current_video_id
        # ------------------------------------

        if st.session_state.tracks_df is None:
            # 1. Simpan video upload ke file temporary
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            
            with st.status("Preprocessing and Tracking are Running") as status:
                temp_dir = tempfile.mkdtemp()
                
                # --- VISUALISASI TAHAP A (MENGGUNAKAN TFILE) ---
                cap = cv2.VideoCapture(tfile.name)
                ret, frame = cap.read()
                if ret:
                    st.session_state.sample_frame = frame
                
                # 2. Jalankan Proses Pipeline
                prep_path = prepare_video_pipeline(tfile.name, temp_dir)
                st.session_state.prepared_video = prep_path
                
                # 3. Jalankan Tracking
                df = tracking_pipeline(prep_path, os.path.join(temp_dir, "tracks.csv"))
                if 'frame' not in df.columns:
                    df = df.reset_index()
                else:
                    df = df.reset_index(drop=True)
                st.session_state.tracks_df = df
                status.update(label="Preprocessing & Tracking Selesai!", state="complete")

        # --- TAMPILAN SETELAH SELESAI (Agar tetap muncul saat upload ulang atau pindah tab) ---
        if st.session_state.tracks_df is not None:
            # Munculkan kembali visualisasi Tahap A dari session state
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
# ------------------------------------------
# TAB 3: ANALYSIS PROCESS (REVISED WITH SUMMARY TABLE)
# ------------------------------------------
with tab3:
    st.header("Kalkulasi Motilitas & Morfologi")
    
    if st.session_state.tracks_df is None:
        st.warning("Silakan selesaikan proses di Tab 2 (Upload & Tracking) terlebih dahulu.")
    else:
        # 1. SATU TOMBOL UNTUK DUA MODEL
        if st.button("🚀 Jalankan Analisis Motility dan Morfologi"):
            with st.spinner("AI sedang menganalisis pergerakan dan bentuk spermatozoa..."):
                # A. Menjalankan Analisis Motilitas
                st.session_state.motility_results = run_motility_analysis(
                    st.session_state.prepared_video, 
                    st.session_state.tracks_df, 
                    "model_motility.h5"
                )
                
                # B. Menjalankan Analisis Morfologi
                st.session_state.morphology_results = run_morphology_analysis(
                    st.session_state.prepared_video, 
                    st.session_state.tracks_df
                )
            st.success("Analisis Motilitas & Morfologi Selesai!")

        # 2. TAMPILKAN HASIL JIKA SUDAH ADA
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

            # --- TAMBAHAN: TABEL SUMMARY KLASIFIKASI ---
            st.divider()
            st.subheader("📋 Tabel Summary Klasifikasi")
            
            # Menggabungkan hasil motility dan morphology berdasarkan ID particle
            # Kita ambil kolom x, y, frame dari tracks_df awal melalui join
            df_mot = st.session_state.motility_results[['particle', 'motility_label']]
            df_morf = st.session_state.morphology_results[['particle', 'morphology_label']]
            
            # Gabungkan kedua hasil klasifikasi
            summary_df = pd.merge(df_mot, df_morf, on='particle', how='inner')
            
            # Gabungkan dengan data koordinat (x, y, frame) dari tracks_df asli
            # Kita ambil baris pertama dari setiap particle untuk koordinat representatif
            coords = st.session_state.tracks_df.groupby('particle').first().reset_index()[['particle', 'x', 'y', 'frame']]
            
            final_summary = pd.merge(coords, summary_df, on='particle', how='inner')
            
            # Menata ulang urutan kolom sesuai instruksi
            final_summary = final_summary[['x', 'y', 'frame', 'particle', 'motility_label', 'morphology_label']]
            
            # Rename kolom agar lebih rapi di tabel
            final_summary.columns = ['X', 'Y', 'Frame', 'ID Particle', 'Klasifikasi Motility', 'Klasifikasi Morfologi']
            
            st.dataframe(final_summary, use_container_width=True)
            
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
