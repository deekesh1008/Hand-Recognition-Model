#synapse_interactive.py
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import time

from gtts import gTTS
from playsound import playsound
import speech_recognition as sr


print("Available microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"  Microphone \"{name}\" found for `Microphone(device_index={index})`")

# ===============================================================
# 1. SETUP: LOAD MODELS AND CONFIGURATIONS
# ===============================================================

# --- Load the Gesture Recognition Model ---
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# --- Define Gesture-to-Speech Mappings ---
gesture_to_phrase = {
    0: 'A',
    1: 'L',
    2: 'Peace',
    3: 'Namaste',
    4: 'Home',
    5: 'Hello!',
    
}

# --- Define Speech-to-Gesture Mappings ---
phrase_to_gesture = {
    "show me a": "A.jpg",
    "show me L": "l.jpg",
    "hello": "hello.jpg",
    "home": "home.jpg",
    "namaste": "namaste.jpg",
    "peace": "peace.jpg"
}

# ===============================================================
# 2. CORE FUNCTIONS: SPEECH AND GESTURE HANDLING
# ===============================================================

def text_to_speech(text):
    """ Converts text to speech and plays it. Caches audio files. """
    try:
        # Create a sanitized filename
        filename = f"./temp_{text.replace(' ', '_').replace('!', '')}.mp3"

        # If the audio file doesn't exist, create it with gTTS
        if not os.path.exists(filename):
            print(f"Generating audio for: '{text}'")
            tts = gTTS(text=text, lang='en')
            tts.save(filename)

        # Play the audio file
        playsound(filename)

    except Exception as e:
        print(f"Error in text_to_speech: {e}")

def display_gesture_image(gesture_name):
    """ Displays an image of the recognized gesture. """
    image_path = os.path.join('gesture_images', gesture_name)
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        cv2.imshow('Spoken Gesture', img)
        cv2.waitKey(2000) # Display for 2 seconds
        cv2.destroyWindow('Spoken Gesture')
    else:
        print(f"Image not found: {image_path}")

def listen_for_commands():
    """ Runs in a separate thread to listen for voice commands. """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=1)

    with microphone as source:
        # Adjust for ambient noise once at the start
        print("Calibrating for ambient noise, please be quiet...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Calibration complete. Listening for commands...")

    while True:
        with microphone as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: {text}")

                # Check if the spoken text matches a command
                if text in phrase_to_gesture:
                    gesture_image_name = phrase_to_gesture[text]
                    print(f"Recognized command! Displaying: {gesture_image_name}")
                    display_gesture_image(gesture_image_name)

            except sr.UnknownValueError:
                
                print("DEBUG: Could not understand the audio. Please speak more clearly.")
            except sr.RequestError as e:
                print(f"Could not request results from Google API; {e}")
            except Exception as e:
                pass
        time.sleep(0.1)

# ===============================================================
# 3. MAIN APPLICATION LOOP
# ===============================================================

# --- Start the speech recognition thread ---
listener_thread = threading.Thread(target=listen_for_commands, daemon=True)
listener_thread.start()

cap = cv2.VideoCapture(0)

last_gesture_spoken = None
last_speech_time = time.time()
speech_cooldown = 3 

print("\nWebcam and gesture recognition started. Look at the camera!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_all, y_all = [], []

        if len(results.multi_hand_landmarks) == 2:
            results.multi_hand_landmarks = sorted(
                results.multi_hand_landmarks,
                key=lambda hand_landmarks: hand_landmarks.landmark[0].x
            )

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_, y_ = [], []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            min_x, min_y = min(x_), min(y_)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min_x)
                data_aux.append(y_[i] - min_y)

        
        if len(data_aux) == 42:
            data_aux.extend([0.0] * 42)

        # Make prediction and trigger speech
        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_class = int(prediction[0])
            predicted_character = gesture_to_phrase.get(predicted_class, 'Unknown')

            # --- Draw Bounding Box and Label ---
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_all.append(landmark.x)
                    y_all.append(landmark.y)

            x1, y1 = int(min(x_all) * W) - 10, int(min(y_all) * H) - 10
            x2, y2 = int(max(x_all) * W) + 10, int(max(y_all) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            current_time = time.time()
            if predicted_class != last_gesture_spoken or (current_time - last_speech_time) > speech_cooldown:
                # Run TTS in a separate thread to prevent video lag
                tts_thread = threading.Thread(target=text_to_speech, args=(predicted_character,), daemon=True)
                tts_thread.start()
                last_gesture_spoken = predicted_class
                last_speech_time = current_time

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
