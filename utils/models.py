import os
import torch
from ultralytics import RTDETR
import inference
from inference import get_model
import requests
import os
import requests


def load_player_detection_model(model_path="models/player_detect.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RTDETR(model_path)
    model.to(device)
    print(f"Player detection model loaded on {device}")
    return model

def load_field_detection_model(api_key="cxtZ0KX74eCWIzrKBNkM", model_id="football-field-detection-f07vi/14"):
    """
    Charge le modèle de détection de terrain depuis Roboflow en utilisant l'API key et le model ID.
    """
    field_detection_model = get_model(model_id=model_id, api_key=api_key)
    print("Field detection model loaded from Roboflow")
    return field_detection_model
