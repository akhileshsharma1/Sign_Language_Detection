import os
import cv2

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

camera_index = 0  # Change to 1, 2, etc., if you have multiple cameras
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Camera at index {camera_index} could not be opened.")
    exit(1)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame. Please check your camera.")
            continue

        cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame. Skipping...")
            continue

        cv2.imshow('frame', frame)

        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        print(f"Saved {image_path}")  
        counter += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Exiting data collection early.")
            break

print("Data collection completed for all classes.")

cap.release()
cv2.destroyAllWindows()
