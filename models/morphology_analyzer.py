import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download # Tambahkan ini

# Parameter sesuai training
RESIZE_TO = 224
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

def load_morphology_model_hf():
    """Mengunduh model dari Hugging Face jika belum ada di cache"""
    try:
        model_path = hf_hub_download(
            repo_id="nashiffrd/SpermMorpho", 
            filename="model_morfologi.h5"
        )
        # Load model tanpa compile karena kita hanya butuh untuk prediksi
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Gagal mengunduh model dari Hugging Face: {e}")
        return None

def apply_binary_erosion(img_bgr):
    # (Logika fungsi ini tetap sama seperti sebelumnya)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.erode(binary, KERNEL_ERODE, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_OPEN)
    
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
            
    mask = np.zeros_like(binary)
    mask[labels == target_label] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    
    final_gray = 255 - filled
    final_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
    return final_bgr

def run_morphology_analysis(video_path, tracks_df):
    """Fungsi utama dengan penarikan model dari HF"""
    # 1. Pilih frame terbaik
    best_frames = (
        tracks_df.sort_values("signal", ascending=False)
          .groupby("particle")
          .first()
          .reset_index()
    )
    
    # 2. Load Model dari Hugging Face
    model = load_morphology_model_hf()
    if model is None:
        return pd.DataFrame()

    cap = cv2.VideoCapture(video_path)
    results = []

    for _, row in best_frames.iterrows():
        p_id = row['particle']
        f_idx = int(row['frame'])
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Cropping
        h, w = frame.shape[:2]
        half = 32 
        x, y = int(row['x']), int(row['y'])
        x1, y1 = max(0, x-half), max(0, y-half)
        x2, y2 = min(w, x+half), min(h, y+half)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0: continue
        crop_res = cv2.resize(crop, (RESIZE_TO, RESIZE_TO))
        
        # Preprocessing
        processed_img = apply_binary_erosion(crop_res)
        
        # Predict
        img_input = np.expand_dims(processed_img.astype(np.float32) / 255.0, axis=0)
        prob = model.predict(img_input)[0][0]
        
        label = "Abnormal" if prob < 0.5 else "Normal"
        
        results.append({
            'particle': p_id,
            'morphology_label': label,
            'morphology_prob': prob,
            'image_display': processed_img 
        })
        
    cap.release()
    return pd.DataFrame(results)
