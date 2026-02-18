🧬 Sperm Analysis System using Deep Learning & Streamlit
Sistem analisis spermatozoa otomatis berbasis web yang dirancang untuk membantu laboratorium dalam mengklasifikasikan motilitas dan morfologi sperma secara objektif berdasarkan standar WHO Laboratory Manual.

🚀 Fitur Utama
Sistem ini terdiri dari alur kerja linear yang dibagi menjadi 4 tahap utama:

-Home: Panduan operasional dan standarisasi input data.

-Upload & Digital Processing: Pipeline pemrosesan citra digital (grayscale, contrast enhancement) dan particle tracking.

-Analysis Process: Inferensi ganda menggunakan model Deep Learning untuk klasifikasi motilitas dan morfologi.

-Summary Dashboard: Laporan klinis terintegrasi yang menyajikan indikasi penyakit otomatis (Normozoospermia, Asthenozoospermia, Teratozoospermia, atau Asthenoteratozoospermia).

📊 Parameter Analisis
Sistem melakukan evaluasi berdasarkan ambang batas klinis:
| Parameter | Threshold | Keterangan |
| :--- | :--- | :--- |
| Progressive Motility (PR) | < 32% | Indikasi Asthenozoospermia |
| Normal Morphology | < 4% | Indikasi Teratozoospermia |
| Kombinasi Motility & Morfologi | | Indikasi Asthenoteratozoospermia|

🛠️ Teknologi yang Digunakan

-Python (Core Programming)

-Streamlit (Web Interface & Dashboard)

-OpenCV (Digital Image Processing)

-TensorFlow/Keras/PyTorch (Deep Learning Models)

-Pandas & NumPy (Data Management)

🖥️ Tampilan Dashboard
Dashboard hasil akhir dirancang secara ergonomis untuk menyajikan:

-Main Diagnosis Card: Kotak diagnosis berwarna dinamis sesuai status kesehatan sampel.

-Integrated Metrics: Panel rata tengah yang menampilkan jumlah riil partikel PR, NP, IM, Normal, dan Abnormal.

-Reset Feature: Tombol untuk membersihkan seluruh sesi analisis dan memulai pengujian baru.

📝 Disclaimer
Sistem ini dikembangkan sebagai Decision Support Tool (alat bantu pendukung keputusan) untuk tenaga medis. Hasil yang dikeluarkan berupa indikasi klinis berbasis parameter sperma dan bukan merupakan diagnosis medis final atau rekomendasi terapi.
