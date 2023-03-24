import os

import cv2
import numpy as np

# data = 'positive'  # positive or negative
data = 'positive'

match data:
    case 'positive':
        folder_path = 'face_images/'
        image_name = 'face_image'
    case 'negative':
        folder_path = 'negative_images/'
        image_name = 'negative'
    case _:
        folder_path = ''
        image_name = ''


def capture_face_images():
    # Define camera index (usually 0 or 1)
    camera_index = 0

    # Initialize camera capture
    print('Opening camera...')
    cap = cv2.VideoCapture(camera_index)
    print('Camera opened')

    # Check if camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Start countdown timer and show camera feed
    images = []
    while True:
        # Read camera frame
        ret, frame = cap.read()

        # Flip camera horizontally
        frame = cv2.flip(frame, 1)

        if cv2.waitKey(1) == ord('c'):
            capturing_text = 'Capturing'
            cv2.putText(frame, capturing_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            images.append(frame)

        # Display camera feed
        cv2.imshow('Camera Feed', frame)

        # Exit loop and save images
        if cv2.waitKey(1) == ord('q'):
            break

        if len(images) % 50 == 0:
            print(len(images))

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over the list and delete each file
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)

    for i, image in enumerate(images):
        img_resized = cv2.resize(image, (224, 224))
        if not os.path.exists(folder_path):
            print(folder_path, 'created.')
            os.makedirs(folder_path)

        np.save(f'{folder_path}{image_name}{i}.npy', img_resized)

    print(len(images), 'images captured.')


if __name__ == '__main__':
    capture_face_images()
