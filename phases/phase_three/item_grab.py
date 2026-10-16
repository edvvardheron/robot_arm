from ultralytics import YOLO

MODEL_PATH = YOLO('phases/phase_two/yolo26n_trained.pt')

import time
from lerobot.robots.so_follower import SO101Follower
import cv2 as cv
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
import numpy as np
import math
from calibration_setup.calibration import CameraToArm

PORT = 'COM3'
ROBOT_ID = 'follower_arm'

UPPER_RED1 = np.array([5, 255, 255])
LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED2 = np.array([179, 255, 255])
LOWER_RED2 = np.array([170, 100, 100])

PAN_SCAN_SPEED = -1
PAN_SPEED = 0.1
TILT_SPEED = 0.1

L1 = 11.5 #length of the first arm segment in cm
L2 = 13.5 #length of the second arm segment in cm
L3 = 13.2 #length of the third arm segment in cm
SHOULDER_HEIGHT = 12.2 #height of the shoulder joint from the ground in cm
OBJECT_GRASP_HEIGHT = 1
GRASP_Y = OBJECT_GRASP_HEIGHT - SHOULDER_HEIGHT
TIP_ORIENTATION = -math.pi / 2
CAMERA_OFFSET = 5  # Offset from the camera to the gripper in cm


DEADBAND = 10
PAN_DIRECTION = 1
TILT_DIRECTION = -1
LOOP_DELAY = 0.01


MIN_AREA = 2000  # Minimum area of the detected object to be considered valid

JOINTS_LIMITS = {       # adjust to your calibration's safe range
    "shoulder_pan.pos": (-70, 70),
    "shoulder_lift.pos": (-90, 50),
    "elbow_flex.pos": (-40, 90),
    "wrist_flex.pos": (-60, 90),
    "wrist_roll.pos": (-130, 130),
    "gripper.pos": (5, 60),
}

STARTING_POSITIONS = {
    "shoulder_pan.pos": 0,
    "shoulder_lift.pos": -65,
    "elbow_flex.pos": 65,
    "wrist_flex.pos": 65,
    "wrist_roll.pos": 0,
    "gripper.pos": 5,
}


SHOULDER_SIGN, SHOULDER_OFFSET = -1.0, 90.0
ELBOW_SIGN,    ELBOW_OFFSET    = -1.0, -90.0
WRIST_SIGN,    WRIST_OFFSET    = -1.0, -90.0
 

model = MODEL_PATH

'''Identify an object classified by the YOLO model in the frame, use the aerial camera to identify the coordinates and angle of the object.

Convert these coordinates to the robot's coordinate system, and then use inverse kinematics to calculate the joint angles needed to reach the object.

Finally, move the robot's arm to the calculated joint angles and close the gripper at the correct angle to grab the object.'''

def find_object():
    """Find the target object in the aerial frame and return its coordinates. create a mask to identify the object based on its color and shape, and then use the YOLO model to classify the object and return its coordinates."""
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        return
    objects = model(frame)
    if not objects:
        print("No objects detected.")
        return

    #find the object from the YOLO model, setting u and v to the distance from the center of the frame to the center of the object
    frame_h, frame_w = frame.shape[:2]
    center_x, center_y = frame_w // 2, frame_h // 2
    for obj in objects:
        if obj.boxes is not None and len(obj.boxes) > 0:
            box = obj.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            u = (x1 + x2) // 2 - center_x
            v = (y1 + y2) // 2 - center_y
            print(f"Detected object at: u={u}, v={v}")
            break

    cam = CameraToArm()
    x, y, z = cam.pixel_to_arm(u, v, z_cm=1.25)   # z_cm = height of the point you're locating
    reach, pan_deg = cam.to_polar(x, y)            # -> IK solver
    cv.imshow("Object Detection", frame)
    print(frame, f"Object coordinates: x={x:.2f}, y={y:.2f}, z={z:.2f}, reach={reach:.2f}, pan={pan_deg:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv.waitKey(1)
    time.sleep(1)

    

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Could not open camera')
def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.waitKey(1)
        cv.imshow("Object Detection", frame)



        find_object()

if __name__ == "__main__":
    main()

