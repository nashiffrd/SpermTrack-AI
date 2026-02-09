import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Konfigurasi sesuai training kamu
CROP_SIZE = 64
FRAMES_PER_CLIP = 32
LABEL_MAP = {0: 'IM', 1: 'NP', 2: 'PR'}

def crop_frame_centered(frame, cx, cy, size=64):
    h, w = frame.shape[:2]
    half = size // 2
    x1, y1 = int(cx) - half, int(cy) - half
    
    # Padding jika crop keluar boundary
    pad_left = max(0, -x1)
    pad_top  = max(0, -y1)
    pad_right= max(0, (x1 + size) - w)
    pad_bot  = max(0, (y1 + size) - h)
    
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    
    crop = frame[y1_clamped: min(h, y1 + size), x1_clamped: min(w, x1 + size)]
    if any([pad_left, pad_top, pad_right, pad_bot]):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REPLICATE)
    
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size))
    return crop

def extract_particle_clips(video_path, tracks_df):
    """
    Mengambil clips per partikel langsung ke memory (numpy array)
    """
    cap = cv2.VideoCapture(video_path)
    particle_clips = {} # {particle_id: [frames]}
    
    # Ambil list partikel unik
    unique_particles = tracks_df['particle'].unique()
    
    # Sort tracks berdasarkan frame untuk pembacaan sekali jalan (efisien)
    tracks_df = tracks_df.sort_values('frame')
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Cari partikel yang muncul di frame ini
        current_frame_data = tracks_df[tracks_df['frame'] == frame_idx]
        
        for _, row in current_frame_data.iterrows():
            p_id = row['particle']
            if p_id not in particle_clips:
                particle_clips[p_id] = []
            
            # Stop jika sudah mencapai limit frames_per_clip
            if len(particle_clips[p_id]) < FRAMES_PER_CLIP:
                crop = crop_frame_centered(frame, row['x'], row['y'], CROP_SIZE)
                # Normalize 0-1
                particle_clips[p_id].append(crop.astype(np.float32) / 255.0)
        
        frame_idx += 1
    cap.release()
    
    # Post-processing: Padding untuk partikel yang durasinya kurang dari 32 frame
    final_data = []
    particle_ids = []
    
    for p_id, frames in particle_clips.items():
        if len(frames) == 0: continue
        
        # Padding dengan frame terakhir jika kurang
        while len(frames) < FRAMES_PER_CLIP:
            frames.append(frames[-1])
            
        final_data.append(np.array(frames)) # Shape: (32, 64, 64, 3)
        particle_ids.append(p_id)
        
    return np.array(final_data), particle_ids

def run_motility_analysis(video_path, tracks_df, model_path):
    """
    Fungsi utama yang dipanggil oleh app.py
    """
    # 1. Extract Clips
    clips, p_ids = extract_particle_clips(video_path, tracks_df)
    
    if len(clips) == 0:
        return pd.DataFrame()

    # 2. Load Model & Predict
    model = load_model(model_path)
    preds = model.predict(clips)
    pred_indices = np.argmax(preds, axis=1)
    
    # 3. Format Result
    results = []
    for i, p_id in enumerate(p_ids):
        results.append({
            'particle': p_id,
            'motility_label': LABEL_MAP[pred_indices[i]],
            'confidence': np.max(preds[i])
        })
        
    return pd.DataFrame(results)
