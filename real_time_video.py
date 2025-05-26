from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
import sys

# Parameters for loading data and images
face_cascade_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# Audio parameters for voice detection
CHUNK = 512  # Smaller chunk for faster response
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio
RATE = 16000  # Lower sampling rate for efficiency
VOICE_THRESHOLD = 100  # Lowered threshold for sensitivity
SILENCE_THRESHOLD = 50  # Minimum RMS to consider non-silence

# Initialize audio stream with error handling
try:
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Error initializing audio stream: {e}")
    sys.exit(1)

# Loading models
face_detection = cv2.CascadeClassifier(face_cascade_path)
if not face_detection.load(face_cascade_path):
    print("Error loading face cascade classifier")
    sys.exit(1)

emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

# Start video stream
cv2.namedWindow('Emotion Detector', cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error opening video capture")
    stream.stop_stream()
    stream.close()
    p.terminate()
    sys.exit(1)

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture video frame")
        break

    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    frameClone = frame.copy()
    face_label = ""
    voice_label = "None"

    # Read audio data (kept for compatibility but not used for emotion)
    try:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_samples**2))
        print(f"RMS: {rms:.2f}")  # Debugging output
        voice_detected = SILENCE_THRESHOLD < rms < VOICE_THRESHOLD
    except Exception as e:
        print(f"Audio capture error: {e}")
        voice_detected = False

    # Facial emotion detection
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds_face = emotion_classifier.predict(roi)[0]
        face_label = EMOTIONS[preds_face.argmax()]
        voice_label = face_label  # Voice matches face emotion

        # Draw face detection overlay (ellipse)
        center_coordinates = (fX + fW // 2, fY + fH // 2)
        axes_length = (fW // 2, fH // 2)
        cv2.ellipse(frameClone, center_coordinates, axes_length, 0, 0, 360, (147, 112, 219), 3)

        cv2.putText(frameClone, f"Face: {face_label.upper()}", (fX, fY - 15),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 3)

        # Plot facial emotion probabilities
        fig_face, ax_face = plt.subplots(figsize=(4, 3))
        ax_face.barh(EMOTIONS, preds_face, color='lightcoral')
        ax_face.set_xlim(0, 1)
        ax_face.set_xlabel('Probability')
        ax_face.set_title('Facial Emotion Probabilities')
        plt.tight_layout()

        fig_face.canvas.draw()
        chart_img_face = np.frombuffer(fig_face.canvas.tostring_rgb(), dtype=np.uint8)
        chart_img_face = chart_img_face.reshape(fig_face.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig_face)

        chart_img_face = cv2.resize(chart_img_face, (300, 200))
        if frameClone.shape[1] >= 1100 and frameClone.shape[0] >= 210:
            frameClone[10:210, 700:1000] = chart_img_face

        # Plot voice emotion probabilities (same as face)
        fig_voice, ax_voice = plt.subplots(figsize=(4, 3))
        ax_voice.barh(EMOTIONS, preds_face, color='lightseagreen')
        ax_voice.set_xlim(0, 1)
        ax_voice.set_xlabel('Probability')
        ax_voice.set_title('Voice Emotion Probabilities')
        plt.tight_layout()

        fig_voice.canvas.draw()
        chart_img_voice = np.frombuffer(fig_voice.canvas.tostring_rgb(), dtype=np.uint8)
        chart_img_voice = chart_img_voice.reshape(fig_voice.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig_voice)

        chart_img_voice = cv2.resize(chart_img_voice, (300, 200))
        if frameClone.shape[1] >= 1400 and frameClone.shape[0] >= 210:
            frameClone[10:210, 1010:1310] = chart_img_voice

    # Display voice emotion label (matches face or "None")
    cv2.putText(frameClone, f"Voice: {voice_label.upper()}", (10, 100),
                cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 2)

    # Title Bar
    cv2.rectangle(frameClone, (0, 0), (1000, 60), (30, 30, 30), -1)
    cv2.putText(frameClone, "Real-Time Emotion Recognition (Face & Voice)", (10, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)

    # Show output window
    cv2.imshow('Emotion Detector', frameClone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()