import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from enum import Enum

class GI_COLOR(Enum):
    WHITE = 1
    BLUE = 2
    UNKNOWN = 3

def debugShowRectangle(image, box):
    left, top, right, bottom  = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

def getCroppedPlayerArea(image, player):
    return image[player[1]:+player[3], player[0]:player[2]]

def getGiColor(grayscale_image):
    print("values >= 127: ")
    print(np.sum(grayscale >= 127))
    print("values <= 127: ")
    print(np.sum(grayscale <= 127))
    print("total values: ")
    print(np.sum(grayscale))
    return GI_COLOR.WHITE if (np.sum(grayscale >= 127) > np.sum(grayscale <= 127)) else GI_COLOR.BLUE

model = YOLO('yolov8n-pose.pt')
example_file_path = "/Volumes/trainingdata/edited/koshi guruma/13.mp4"

cap = cv2.VideoCapture(example_file_path)

while cap.isOpened():
    success, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if success:
        results = model(frame)

        for result in results:
            annotator = Annotator(frame)
            for box in result.boxes:

                converted_coords = list(map(int,box.xyxy[0]))
                debugShowRectangle(frame, converted_coords)
                player_area = getCroppedPlayerArea(frame, converted_coords)
                grayscale = cv2.cvtColor(player_area, cv2.COLOR_BGR2GRAY)
                gi_color = getGiColor(grayscale)
                print(gi_color.value)
                annotator.box_label(box.xyxy[0], f"{gi_color}")


        annotated_frame = annotator.result()
        cv2.imshow("Interference", annotated_frame)

cap.release()
cv2.destroyAllWindows()

"""
Some notes about this one:
The order that the model returns boxes at is different than before somehow? Double check that.
Also it looks like the way we are determining what gi color is doing what is just plain wrong. It works fine when people aren't overlapping, but it marks everyone as blue once the blue person is in front.
this would also be the same for whenever we have a white gi in front.

need to find a better way.
"""
