def create_motility_video(video_path, tracks_df, motility_results):
    # Gabungkan data tracking dengan label motilitas berdasarkan ID partikel
    # Pastikan motility_results memiliki kolom 'particle' dan 'motility_label'
    merged_df = tracks_df.merge(motility_results[['particle', 'motility_label']], on='particle', how='left')
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup Video Writer
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
    
    # Warna: BGR (OpenCV menggunakan BGR bukan RGB)
    colors = {
        'PR': (0, 255, 0),    # Hijau
        'NP': (0, 255, 255),  # Kuning
        'IM': (0, 0, 255)     # Merah
    }
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Ambil data untuk frame saat ini
        current_data = merged_df[merged_df['frame'] == frame_idx]
        
        for _, row in current_data.iterrows():
            pid = row['particle']
            label = row['motility_label']
            color = colors.get(label, (255, 255, 255)) # Putih jika tidak ada label
            
            # 1. Gambar Lingkaran di posisi sekarang
            cv2.circle(frame, (int(row['x']), int(row['y'])), 4, color, -1)
            
            # 2. Gambar Lintasan (History)
            history = merged_df[(merged_df['particle'] == pid) & (merged_df['frame'] <= frame_idx)]
            if len(history) > 1:
                points = history[['x', 'y']].values.astype(np.int32)
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=1)
                
            # 3. Opsional: Tulis ID Partikel
            # cv2.putText(frame, str(int(pid)), (int(row['x'])+5, int(row['y'])-5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    return temp_out.name
