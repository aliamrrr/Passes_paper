import supervision as sv
from tqdm import tqdm
import torch
import cv2
import streamlit as st
import numpy as np
from PIL import Image

PLAYER_ID = 0 
STRIDE = 120

def extract_player_crops(video_path, player_detection_model):
    crops = []

    frame_generator = sv.get_video_frames_generator(
        source_path=video_path, stride=STRIDE
    )

    for frame in tqdm(frame_generator, desc='Collecting crops'):
        results = player_detection_model.predict(frame, conf=0.3)
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
            class_id=results[0].boxes.cls.detach().cpu().numpy(),
            confidence=results[0].boxes.conf.detach().cpu().numpy()
        )

        detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        player_detections = detections[detections.class_id == PLAYER_ID]

        for xyxy in player_detections.xyxy:
            crop = sv.crop_image(frame, xyxy)
            crops.append(crop)
            
    return crops

def show_crops(crops, cols=5):
    if not crops:
        st.warning("Aucun crop détecté.")
        return
    rows = 3
    for i in range(rows):
        row_crops = crops[i * cols:(i + 1) * cols]

        cols_layout = st.columns(cols)
        
        for j, crop in enumerate(row_crops):
            if j < len(cols_layout):
                pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                cols_layout[j].image(pil_image, use_column_width=True, caption=f"Crop {i * cols + j + 1}")

from utils.team import TeamClassifier

def fit_team_classifier(crops, device="cpu"):
    if not crops:
        raise ValueError("La liste de crops est vide. Assurez-vous que les crops ont été correctement extraits.")
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    return team_classifier

