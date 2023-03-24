from time import time

import cv2
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Define camera index (usually 0 or 1)
camera_index = 0

# Initialize camera capture
print('Opening camera...')
cap = cv2.VideoCapture(camera_index)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print('Camera opened')

# Set the resolution to 224x224
# print('Setting resolution to 224x224')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
# print('Resolution set')

# Load the model from the file
model = load_model('weights.h5')
embed = lambda x: model.predict(x, verbose=0)
face_recognition = joblib.load('svm_model.pkl')
start_time = time()
num_frames = 0

while True:
    # Read camera image
    ret, img = cap.read()

    # Flip camera horizontally
    img = cv2.flip(img, 1)

    # Update FPS variables
    num_frames += 1
    elapsed_time = time() - start_time
    fps = num_frames / elapsed_time

    # Display camera feed
    cv2.imshow('Camera Feed', img)

    # Resize the image to the expected input shape of the model
    start_resize = time()
    img = cv2.resize(img, (224, 224))
    end_resize = time()
    # Convert the image to an array and expand the dimensions to match the input shape of the model

    image_array = np.expand_dims(img, axis=0)

    start_embed = time()
    embedded_image = embed(image_array)
    end_embed = time()

    start_pred = time()
    y_pred = face_recognition.predict(embedded_image)
    end_pred = time()

    fps_text = f"\rFPS: {fps:.2f}"
    resize_time = end_resize - start_resize
    embed_time = end_embed - start_embed
    pred_time = end_pred - start_pred
    detected = "Face detected" if y_pred[0] == 1 else "No face detected"

    print(f'\r{fps_text}, Resize: {resize_time:.4f}, Embed: {embed_time:.4f}, Predict: {pred_time:.4f}, {detected}',
          end='')
    # print(y_pred)
    # print(embedded_image.shape)
    # Display the FPS on the frame

    # cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exit loop and save images
    if cv2.waitKey(1) == ord('q'):
        break
