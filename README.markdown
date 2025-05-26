# Emotion Recognition System (Face and Voice)

## Overview
This project implements a real-time emotion recognition system that detects emotions from both facial expressions and voice input using deep learning models. The system integrates a convolutional neural network (CNN) for facial emotion recognition and an audio-based emotion classifier, leveraging datasets like FER2013 for facial emotions and RAVDESS/CREMA-D for audio emotions.

## Features
- **Facial Emotion Recognition**: Uses a pre-trained Mini XCEPTION CNN model to classify emotions (angry, disgust, fear, happy, sad, surprised, neutral) from live video feed.
- **Voice Emotion Recognition**: Processes audio input to detect emotions using a CNN model trained on RAVDESS and CREMA-D datasets.
- **Real-Time Processing**: Combines face and voice analysis in a unified interface with probability visualizations.
- **Data Visualization**: Displays emotion probability distributions for both modalities.

## Prerequisites
- Python 3.10+
- Libraries: `tensorflow`, `keras`, `opencv-python`, `imutils`, `numpy`, `pandas`, `librosa`, `matplotlib`, `seaborn`, `pyaudio`
- Hardware: Webcam and microphone for real-time input
- Datasets:
  - FER2013 (facial emotion dataset)
  - RAVDESS and CREMA-D (audio emotion datasets)
- Pre-trained model: `_mini_XCEPTION.102-0.66.hdf5` (facial emotion classifier)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd emotion-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the following files are in place:
   - `haarcascade_frontalface_default.xml` in `haarcascade_files/`
   - Pre-trained model in `models/`
   - Datasets in appropriate directories (e.g., `/content/drive/MyDrive/ANN/` for audio data)

## Usage
1. **Train the Facial Emotion Model**:
   Run `train_emotion_classifier.py` to train or use the pre-trained Mini XCEPTION model:
   ```bash
   python train_emotion_classifier.py
   ```
2. **Train the Audio Emotion Model**:
   Execute the Jupyter notebook `Untitled18_(1edit)_(2) (1).ipynb` in a Google Colab environment with mounted Google Drive containing RAVDESS and CREMA-D datasets.
3. **Run Real-Time Emotion Detection**:
   Launch `real_time_video.py` for combined face and voice emotion detection:
   ```bash
   python real_time_video.py
   ```
   - Press `q` to exit the video stream.
   - Ensure a webcam and microphone are connected.

## Project Structure
- `real_time_video.py`: Main script for real-time face and voice emotion detection.
- `train_emotion_classifier.py`: Script to train the facial emotion classifier.
- `load_and_process.py`: Utility for loading and preprocessing FER2013 dataset.
- `Untitled18_(1edit)_(2) (1).ipynb`: Jupyter notebook for training the audio emotion classifier.
- `models/`: Directory for storing trained models.
- `haarcascade_files/`: Contains Haar cascade for face detection.

## Datasets
- **FER2013**: Facial images with 7 emotion classes (angry, disgust, fear, happy, sad, surprised, neutral).
- **RAVDESS**: Audio files with 6 emotion classes (neutral, happy, sad, angry, fearful, disgust).
- **CREMA-D**: Audio files with 6 emotion classes (same as RAVDESS).

## Notes
- The audio model excludes the "surprised" emotion from RAVDESS to align with CREMA-D classes.
- Voice emotion currently mirrors facial emotion in `real_time_video.py` due to integration simplicity.
- Ensure GPU support for faster training (e.g., Google Colab with T4 GPU).

## License
MIT License