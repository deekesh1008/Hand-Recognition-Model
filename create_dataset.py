#create_dataset

import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set max_num_hands to 2
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Assuming 'data' folders are structured with class labels (e.g., 0, 1, 2)
for dir_ in os.listdir(DATA_DIR):
    # Ensure it's a directory and not a file
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Only process image files
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        data_aux = []
        x_all = []
        y_all = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue  # Skip if image can't be read

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        # Sort hands based on their x-coordinate to ensure consistency
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            results.multi_hand_landmarks = sorted(
                results.multi_hand_landmarks,
                key=lambda hand_landmarks: hand_landmarks.landmark[0].x
            )

        # Check if any hands were detected
        if results.multi_hand_landmarks:

            # We will process up to 2 hands.
            for hand_landmarks in results.multi_hand_landmarks[:2]: # Process a maximum of 2 hands
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                x_all.extend(x_)
                y_all.extend(y_)

                # Normalize features for THIS HAND by its own min(x) and min(y)
                min_x = min(x_)
                min_y = min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_[i] - min_x)
                    data_aux.append(y_[i] - min_y)

            # Pad with zeros if only one hand was detected (21 landmarks * 2 coordinates/landmark = 42 features per hand)
            # The full feature vector size for two hands is 84.
            expected_feature_size = 42 * 2
            current_size = len(data_aux)

            if current_size == 42:
                # One hand detected, pad the remaining 42 features with zeros
                data_aux.extend([0.0] * 42)
            elif current_size == 0:
                 # No hands detected
                continue
            elif current_size != expected_feature_size:
                 # More than 2 hands (clipped by [:2]) or some error, skip
                continue

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
