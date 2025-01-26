import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Configuration for MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

DATA_DIR = './Data'
os.makedirs(DATA_DIR, exist_ok=True)

data, labels = [], []

# Iterate through directories and images
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    if not os.path.isdir(dir_path):
        continue
    
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        
        # Read the image
        img = cv2.imread(img_path)
        
        # Check if the image was successfully read
        if img is None:
            print(f"Failed to read image {img_path}")
            continue
        
        print(f"Processing image {img_path}")
        
        # Explicit square resize (center-cropping if the image isn't square)
        h, w = img.shape[:2]
        size = min(h, w)
        
        # Center crop
        top = (h - size) // 2
        left = (w - size) // 2
        img_square = img[top:top+size, left:left+size]
        
        # Resize to 256x256
        img_resized = cv2.resize(img_square, (256, 256))
        
        # Convert to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Print out the shape of the image to verify it's valid
        print(f"Image shape after resize: {img_rgb.shape}")
        
        # Process image with MediaPipe hands
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            print(f"Found {len(results.multi_hand_landmarks)} hands in {img_path}")
            
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_data = [
                    landmark.x * 256 for landmark in hand_landmarks.landmark
                ] + [
                    landmark.y * 256 for landmark in hand_landmarks.landmark
                ]
                
                # Append landmarks data and corresponding label
                data.append(landmarks_data)
                labels.append(dir_)
        else:
            print(f"No hands detected in {img_path}")

# Save the data and labels as a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection complete.")
hands.close()
