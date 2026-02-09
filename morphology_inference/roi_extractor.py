import cv2
import os
import pandas as pd
import numpy as np


def extract_morphology_rois(
    tracks_csv,
    video_dir,
    output_dir,
    crop_size=64,
    resize_to=224
):
    """
    Crop ROI sperma berdasarkan final_tracks.csv
    - pilih frame terbaik (signal tertinggi) per particle
    - flexible video matching
    - resize ke 224x224
    """

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(tracks_csv)

    # ambil list video
    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    def find_video_file(video_name_csv):
        key = os.path.splitext(video_name_csv)[0].lower()
        for vf in video_files:
            if key in vf.lower():
                return vf
        return None

    # pilih frame terbaik per particle (signal tertinggi)
    if "signal" in df.columns:
        best_frames = (
            df.sort_values("signal", ascending=False)
              .groupby("particle")
              .first()
              .reset_index()
        )
    else:
        # fallback: ambil frame pertama
        best_frames = (
            df.sort_values("frame")
              .groupby("particle")
              .first()
              .reset_index()
        )

    saved = 0

    for video_name_csv, group in best_frames.groupby("video"):
        matched_video = find_video_file(video_name_csv)
        if matched_video is None:
            continue

        video_path = os.path.join(video_dir, matched_video)
        cap = cv2.VideoCapture(video_path)

        needed_frames = set(group["frame"].astype(int))
        frames = {}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in needed_frames:
                frames[frame_idx] = frame.copy()

            frame_idx += 1

        cap.release()

        for _, row in group.iterrows():
            f = int(row["frame"])
            x, y = int(row["x"]), int(row["y"])

            if f not in frames:
                continue

            img = frames[f]
            h, w = img.shape[:2]

            half = crop_size // 2
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(w, x + half)
            y2 = min(h, y + half)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (resize_to, resize_to))

            fname = f"{matched_video}_particle{row['particle']}.png"
            cv2.imwrite(os.path.join(output_dir, fname), crop_resized)
            saved += 1

    return saved
