import numpy as np
import cv2
from pathlib import Path
from pprint import pprint
from enum import Enum
from ultralytics import YOLO

# so that i can log out the numpy numbers without hitting "numpy.float32 has no attribute 'write'" errors
np.set_printoptions(legacy='1.21')

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)
    else:
        print("no file at path: " + example_file_path)


def getCroppedPlayerArea(image, player):
    return image[player[1]:player[1]+player[3], player[0]:player[0]+player[2]]

def getGiColor(grayscale_image):
    # TODO: this has obvious problems with estimating gi color by average pixel count.
    # need to heavily test that this still performs correctly when players have darker or lighter
    # skin tone combinations. If would be bad if a black athlete wearing a white gi showed as wearing a Blue gi,
    # because the average pixel count happened to be slightly higher due to skin tone.
    # Can we isolate the uniform only?
    # NEED TO TEST HEAVILY
    print("values >= 127: ")
    print(np.sum(grayscale >= 127))
    print("values <= 127: ")
    print(np.sum(grayscale <= 127))
    print("total values: ")
    print(np.sum(grayscale))
    return GI_COLOR.WHITE if (np.sum(grayscale >= 127) > np.sum(grayscale <= 127)) else GI_COLOR.BLUE

def parseRows(rows, shape):
    _, confs, boxes = list(),list(),list()
    image_height, image_width, _ = shape
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    for i in range(rows):
        row = preds[0][i]
        conf = row[4]

        classes_score = row[4:]
        _,_,_, max_idx = cv2.minMaxLoc(classes_score)
        class_id = max_idx[1]
        if (classes_score[class_id] > .25):
            confs.append(conf)

            # get boxes
            x,y,w,h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)

    return confs, boxes

onnxpath = "runs/pose/train7/weights/best.onnx"
pose_onnx = "yolov8n-pose.onnx"
example_file_path = "/Volumes/trainingdata/edited/koshi guruma/13.mp4"
first_frame_path = Path("first_frame.jpg")
net = cv2.dnn.readNetFromONNX(onnxpath)
posemodel = YOLO("yolov8n-pose.pt")

class GI_COLOR(Enum):
    WHITE = 1
    BLUE = 2
    UNKNOWN = 3

# get our onnx model
if net:
    print("Found ONNX file at path: " + onnxpath)
else:
    print("No ONNX file at path: " + onnxpath)
    exit

# check if we have the first frame already
if (first_frame_path.is_file()):
    print("First frame already extracted")
else:
    print("Extracting first frame...")
    getFirstFrame(example_file_path)

image = cv2.imread(first_frame_path)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

net.setInput(blob)
output = net.forward()
preds = output.transpose((0,2,1)) # need to understand why we transpose this
rows = preds[0].shape[0]
confs, boxes = parseRows(rows, image.shape)

indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)

found_people = [
    [
        boxes[i][0],  # left
        boxes[i][1],  # top
        boxes[i][2],  # width
        boxes[i][3],  # height
        confs[i],     # conf
        GI_COLOR.UNKNOWN  # Initial GI color
    ]
    for i in indexes
]

# Draw rectangles on the image
for person in found_people:
    left, top, width, height, _, _ = person
    cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 3)

cv2.imwrite("result_image.jpg", image)

# okay now we have the boundary boxes, now we need to classify each person as either WHITE or BLUE based on gi
for found in found_people:
    # create a crop based on the pixel location to look at
    player_area=getCroppedPlayerArea(image,found)
    cv2.imwrite(f"./{found[0]}-unaltered.jpg", player_area)
    grayscale = cv2.cvtColor(player_area, cv2.COLOR_BGR2GRAY)
    found[5] = getGiColor(grayscale)
    print(f"Found player with {found[5]}")
    cv2.imwrite(f"./{found[5]}.jpg", grayscale)

    # once we have that, we need to get the pose information within each boundary box.
