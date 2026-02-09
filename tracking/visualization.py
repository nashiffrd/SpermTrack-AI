import cv2
import numpy as np
import pandas as pd

def draw_locate_frame(frame_gray, detections_df, frame_idx):
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    df = detections_df[detections_df["frame"] == frame_idx]

    for _, r in df.iterrows():
        cv2.circle(
            vis,
            (int(r["x"]), int(r["y"])),
            8,
            (0, 255, 0),
            1
        )
    return vis


def draw_tracks(frame_gray, tracks_df, frame_idx):
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    for pid in tracks_df["particle"].unique():
        grp = tracks_df[tracks_df["particle"] == pid]
        grp = grp.sort_values("frame")
        pts = grp[grp["frame"] <= frame_idx][["x", "y"]].values.astype(int)

        for i in range(1, len(pts)):
            cv2.line(vis, tuple(pts[i-1]), tuple(pts[i]), (255, 0, 0), 1)

    return vis
