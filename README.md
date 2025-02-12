# Soccer Game Analyzer: Optimizing Goal Kick Strategies Using Computer Vision ⚽

This project aims to optimize goal kick strategies in soccer using **computer vision**. The system detects key events from broadcast video footage and provides tactical insights, helping coaches and analysts make informed decisions.

## Prerequisites

Before running the system, ensure that you have the following:

1. **Python 3.6+** installed on your machine.
2. **Streamlit** and other required libraries.

## Installation


1. **Clone the repository**:
   
   First, clone the project repository by running:

   ```bash
   git clone https://github.com/aliamrrr/Passes_paper.git
   cd Passes_paper
   pip install -r requirements.txt

2. **Download the model**
   Download the pre-trained player_detect.pt model weights using this link : https://drive.google.com/file/d/1FuibHhLGI7PvaZxSPrxhtdQxveyqdKTg/view?usp=drive_link and place them in a folder called models inside the project directory. The path should look like this:

   Passes_paper/models/player_detect.pt

4. **Run the app with test video**:
   
   You're provided by a test video (or you can use your own)

   Run this:

   ```bash
   streamlit run main.py


