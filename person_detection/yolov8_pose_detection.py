import cv2
import numpy as np
import onnxruntime as ort
from pprint import pprint
from torchvision.ops import nms

# Load the ONNX model
model_path = 'runs/pose/train7/weights/best.onnx'  # Replace with your model path
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Open a video capture stream (0 for the default webcam)
cap = cv2.VideoCapture('/Volumes/trainingdata/master_video_copies/3261.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image (resize, normalize, etc.)
    input_img = cv2.resize(frame, (640, 640))
    input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_img = np.transpose(input_img, (2, 0, 1))    # Change to CHW
    input_img = np.expand_dims(input_img, axis=0)     # Add batch dimension

    # Run the model
    outputs = session.run(None, {'images': input_img})


    # Extract bounding boxes and scores from the outputs
    boxes = outputs[0]

    # Filter detections with a confidence threshold
    threshold = 0.5
    filtered_boxes = []
    for detection in boxes[0]:  # Modify this loop based on your output structure

        box, score = detection[:4], detection[4]  # Modify indices as necessary
        pprint(box, score)
        if score > threshold:
            filtered_boxes.append(box)

    h, w, _ = frame.shape
#    print(h,w)
    # Draw bounding boxes on the frame
    for box in filtered_boxes[:2]:  # Only draw the first two detections
        x1, y1, x2, y2 = box
        x1 = int(x1 * w)  # Scale x1 from [0, 1] to [0, frame_width]
        y1 = int(y1 * h)  # Scale y1 from [0, 1] to [0, frame_height]
        x2 = int(x2 * w)  # Scale x2 from [0, 1] to [0, frame_width]
        y2 = int(y2 * h)  # Scale y2 from [0, 1] to [0, frame_height]

        # Check if coordinates are within frame bounds
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Print and draw the rectangle
 #       print(f"Drawing box: ({x1}, {y1}), ({x2}, {y2})")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Define a function for non-maximum suppression

    # Display the frame
    cv2.imshow('YOLOv8 Pose Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
