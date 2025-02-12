# =====================
# IMPORT LIBRARIES
# =====================

# Streamlit & Temporary File Handling
import streamlit as st
import tempfile

# Utility Functions and Models
from utils.models import load_player_detection_model, load_field_detection_model
from utils.team_classifier import show_crops, extract_player_crops, fit_team_classifier
from utils.frame_process import process_frame, draw_radar_view, passes_options, calculate_optimal_passes, calculate_realistic_optimal_passes
from utils.soccer import SoccerPitchConfiguration

# Computer Vision Libraries
import cv2
import supervision as sv
import pandas as pd

# PyTorch and Machine Learning
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# Video/Image Processing
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# System Operations
import os
import time
import json

# =====================
# INITIAL CONFIGURATION
# =====================
CONFIG = SoccerPitchConfiguration()  # Standard soccer pitch dimensions and zones

# =====================
# SIDEBAR NAVIGATION
# =====================
menu = st.sidebar.selectbox("Select an option", 
                           ["Dual Passes options", "Generate pass sequences"])

# =====================
# DUAL PASS OPTIONS MODULE
# =====================
if menu == "Dual Passes options":
    tracking_option = st.selectbox("Choose a feature", 
                                 ["View the video", "Collect, classify and view teams", 
                                  "View Pass options"])

    # --- Video Upload and Display ---
    if tracking_option == "View the video":
        uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.video(tmp_file.name)

    # --- Player Detection & Team Classification ---
    elif tracking_option == "Collect, classify and view teams":
        uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                
                # Load detection models from cloud storage
                player_model = load_player_detection_model()
                field_model = load_field_detection_model()

                if st.button("Detect players"):
                    with st.spinner("Processing video..."):
                        # Extract player crops using detection model
                        crops = extract_player_crops(tmp_file.name, player_model)
                        
                        if crops:
                            # Save detected player crops
                            os.makedirs("crops", exist_ok=True)
                            for i, crop in enumerate(crops):
                                Image.fromarray(crop).save(f"crops/crop_{i+1}.jpg")
                            
                            show_crops(crops)
                            st.write(f"Detected players: {len(crops)}")

                            # Train team classifier
                            with st.spinner("Training classifier..."):
                                try:
                                    team_classifier = fit_team_classifier(crops, device="cpu")
                                    st.session_state.team_classifier = team_classifier
                                    st.success("Team classifier ready!")
                                except Exception as e:
                                    st.error(f"Classification error: {str(e)}")
                        else:
                            st.warning("No players detected")

    # --- Pass Analysis Interface ---
    elif tracking_option == "View Pass options":
        uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())

                if "team_classifier" not in st.session_state:
                    st.warning("Complete team classification first")
                else:
                    # Video metadata extraction
                    cap = cv2.VideoCapture(tmp_file.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    # Time selection slider
                    time_seconds = st.slider("Select timestamp", 0, int(total_frames/fps), 0)
                    frame_index = int(time_seconds * fps)

                    # Frame processing
                    cap = cv2.VideoCapture(tmp_file.name)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    
                    if ret:
                        st.image(frame, channels="BGR", caption="Selected frame")

                    # Analysis mode selection
                    col1, col2, col3, col4 = st.columns(4)
                    annotated_frame = None
                    
                    with col1:
                        if st.button("Tactical view"):
                            annotated_frame = draw_radar_view(frame, CONFIG, 
                                                             st.session_state.team_classifier, 
                                                             'tactical')
                    
                    with col2:
                        if st.button("Voronoi diagram"):
                            annotated_frame = draw_radar_view(frame, CONFIG, 
                                                            st.session_state.team_classifier, 
                                                            'voronoi')
                    
                    with col3:
                        if st.button("Short passes"):
                            annotated_frame = passes_options(frame, CONFIG, 
                                                            st.session_state.team_classifier, 
                                                            'build')
                    
                    with col4:
                        if st.button("Long and safe passes"):
                            annotated_frame = passes_options(frame, CONFIG, 
                                                            st.session_state.team_classifier, 
                                                            'interception')

                    if annotated_frame is not None:
                        st.image(annotated_frame, channels="BGR", 
                                caption="Analysis results")

                    cap.release()

# =====================
# PASS SEQUENCE GENERATOR MODULE
# =====================
elif menu == "Generate pass sequences":
    tracking_option = st.selectbox("Choose a feature", 
                                 ["View the video", "Collect, classify and view teams", 
                                  "Generate pass sequences"])

    # --- Video Handling (Duplicate for module separation) ---
    if tracking_option == "View the video":
        uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.video(tmp_file.name)

    # --- Player Detection (Duplicate but necessary for module independence) ---
    elif tracking_option == "Collect, classify and view teams":
        uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                
                player_model = load_player_detection_model()
                field_model = load_field_detection_model()

                if st.button("Detect players"):
                    with st.spinner("Processing video..."):
                        crops = extract_player_crops(tmp_file.name, player_model)
                        
                        if crops:
                            os.makedirs("crops", exist_ok=True)
                            for i, crop in enumerate(crops):
                                Image.fromarray(crop).save(f"crops/crop_{i+1}.jpg")
                            
                            show_crops(crops)
                            st.write(f"Detected players: {len(crops)}")

                            with st.spinner("Training classifier..."):
                                try:
                                    team_classifier = fit_team_classifier(crops, device="cpu")
                                    st.session_state.team_classifier = team_classifier
                                    st.success("Team classifier ready!")
                                except Exception as e:
                                    st.error(f"Classification error: {str(e)}")
                        else:
                            st.warning("No players detected")

    # --- Pass Analysis Interface ---
    elif tracking_option == "Generate pass sequences":
        uploaded_file = st.file_uploader("Upload .mp4 video", type=["mp4"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())

                if "team_classifier" not in st.session_state:
                    st.warning("Complete team classification first")
                else:
                    # Video metadata extraction
                    cap = cv2.VideoCapture(tmp_file.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    # Time selection slider
                    time_seconds = st.slider("Select timestamp", 0, int(total_frames/fps), 0)
                    frame_index = int(time_seconds * fps)

                    # Frame processing
                    cap = cv2.VideoCapture(tmp_file.name)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    
                    if ret:
                        st.image(frame, channels="BGR", caption="Selected frame")

                    # Analysis mode selection
                    col1, col2, col3 = st.columns(3)
                    annotated_frame = None
                    
                    with col1:
                        if st.button("Tactical view"):
                            annotated_frame = draw_radar_view(frame, CONFIG, 
                                                             st.session_state.team_classifier, 
                                                             'tactical')
                    
                    with col2:
                        if st.button("Voronoi diagram"):
                            annotated_frame = draw_radar_view(frame, CONFIG, 
                                                            st.session_state.team_classifier, 
                                                            'voronoi')
                    
                    with col3:
                        max_passes = st.slider("Select the maximum number of passes", min_value=1, max_value=10, value=4)

                        if st.button("Generate pass sequence"):
                            passes_mode = 'build'
                            mode = 1
                            annotated_frame = calculate_realistic_optimal_passes(frame, CONFIG, st.session_state.team_classifier, max_passes=max_passes)

                    if annotated_frame is not None:
                        st.image(annotated_frame, channels="BGR", 
                                caption="Analysis results")

                    cap.release()

    


