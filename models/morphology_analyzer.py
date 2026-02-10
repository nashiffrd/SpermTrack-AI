import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Parameter sesuai training
RESIZE_TO = 224
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

def apply_binary_erosion(img_bgr):
    # Ubah ke grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Threshold (Binary Inv)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Erosi & Opening
    binary = cv2.erode(binary, KERNEL_ERODE, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_OPEN)
    
    # Ambil komponen paling tengah (target sperma)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1: return 255 * np.ones((RESIZE_TO, RESIZE_TO, 3), dtype=np.uint8)

    h, w = gray.shape
    cx_img, cy_img = w // 2, h // 2
    
    min_dist = np.inf
    target_label = 1
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2)
        if dist < min_dist:
            min_dist = dist
            target_label = i
            
    # Masking & Filling
    mask = np.zeros_like(binary)
    mask[labels == target_label] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    
    # Final: Sperma Hitam, BG Putih (3 channel untuk EfficientNet)
    final_gray = 255 - filled
    final_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
    return final_bgr

def run_morphology_analysis(video_path, tracks_df, model_path):
    # 1. Pilih frame terbaik per partikel (berdasarkan signal tertinggi)
    best_frames = (
        tracks_df.sort_values("signal", ascending=False)
          .groupby("particle")
          .first()
          .reset_index()
    )
    
    cap = cv2.VideoCapture(video_path)
    results = []
    model = load_model(model_path, compile=False) # Compile false karena pakai custom loss

    for _, row in best_frames.iterrows():
        p_id = row['particle']
        f_idx = int(row['frame'])
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # 2. Cropping ROI
        h, w = frame.shape[:2]
        half = 32 # Sesuai CROP_SIZE 64
        x, y = int(row['x']), int(row['y'])
        x1, y1 = max(0, x-half), max(0, y-half)
        x2, y2 = min(w, x+half), min(h, y+half)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0: continue
        crop_res = cv2.resize(crop, (RESIZE_TO, RESIZE_TO))
        
        # 3. Binary Erosion Process
        processed_img = apply_binary_erosion(crop_res)
        
        # 4. Predict
        # Normalisasi ke 0-1 (EfficientNetV2S biasanya butuh 0-255 jika pakai layer rescaling internal, 
        # tapi amannya kita sesuaikan dengan training gen kamu)
        img_input = np.expand_dims(processed_img.astype(np.float32) / 255.0, axis=0)
        prob = model.predict(img_input)[0][0]
        
        label = "Normal" if prob < 0.5 else "Abnormal" # Sesuaikan dengan urutan kelas trainingmu
        
        results.append({
            'particle': p_id,
            'morphology_label': label,
            'morphology_prob': prob,
            'image_display': processed_img # Untuk ditampilkan di Streamlit nanti
        })
        
    cap.release()
    return pd.DataFrame(results)
