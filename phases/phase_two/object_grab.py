from ultralytics import YOLO

MODEL_PATH = YOLO('phases/phase_two/yolo26n_trained.pt')

import time
from lerobot.robots.so_follower import SO101Follower
import cv2 as cv
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
import numpy as np
import math

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
 
ELBOW_UP = True

CENTERED_FRAMES_REQUIRED = 5
centered_count = 0

def pan_scan(object_coords):
    """Scan the pan joint back and forth until an object is found."""
    for angle in range(65, -65, PAN_SCAN_SPEED):
        robot.send_action({"shoulder_pan.pos": angle})
        time.sleep(LOOP_DELAY)
        object_coords = find_object()
        if object_coords is not None:

            print(f"Object found at angle {angle}")
            return object_coords


def find_object():
    """Find the target object in the camera frame and return its coordinates."""
    model = MODEL_PATH

    
    CONF_THRESHOLD = 0.5  # Minimum confidence threshold for detection
    frame = cap.read()[1]
    if frame is None:
        return None
    results = model(frame, verbose=False)[0]
    best_box = None
    best_conf = CONF_THRESHOLD
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if cls_name == TARGET_CLASS and conf > best_conf:
            best_conf = conf
            best_box = box
    if best_box is None:
        return None
    frame_h, frame_w = frame.shape[:2]
    frame_cx, frame_cy = frame_w // 2, frame_h // 2
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    cv.imshow('frame', frame)
    cv.waitKey(1)
    coord_x = cx - frame_cx
    coord_y = cy - frame_cy
    return frame, coord_x - 80, coord_y + 110
    




def track_object(object_coords, pan, tilt):
    """Track the object by adjusting pan and tilt based on its position."""
    frame, coord_y, coord_x = object_coords #swapped coord_x and coord_y to match the correct axes
    print(f"Tracking object at coords: x={coord_x}, y={coord_y}")
    if abs(coord_x) > DEADBAND or abs(coord_y) > DEADBAND:
        pan += PAN_DIRECTION * PAN_SPEED * coord_x * 0.1
        tilt += TILT_DIRECTION * TILT_SPEED * coord_y * 0.1
        #apply joint limits
        pan = max(JOINTS_LIMITS["shoulder_pan.pos"][0], min(JOINTS_LIMITS["shoulder_pan.pos"][1], pan))
        tilt = max(JOINTS_LIMITS["wrist_flex.pos"][0], min(JOINTS_LIMITS["wrist_flex.pos"][1], tilt))

        #send action to robot
        robot.send_action({"shoulder_pan.pos": pan, "wrist_flex.pos": tilt})
        time.sleep(LOOP_DELAY)
        centered = False
        return pan, tilt, centered
    else:
        centered = True
        return pan, tilt, centered




def position_arm(current, targets, gripper_open=25):
    shoulder_t, elbow_t, wrist_t = targets
 
    for i in range(100):
        f = (i + 1) / 100
        robot.send_action({
            "shoulder_lift.pos": current["shoulder_lift.pos"] + (shoulder_t - current["shoulder_lift.pos"]) * f,
            "elbow_flex.pos":    current["elbow_flex.pos"]    + (elbow_t    - current["elbow_flex.pos"])    * f,
            "wrist_flex.pos":    current["wrist_flex.pos"]    + (wrist_t    - current["wrist_flex.pos"])    * f,
            "gripper.pos":       gripper_open,
        })
        time.sleep(LOOP_DELAY)

    time.sleep(0.5)  # let it settle before the gripper closes
    return gripper_open
    
    

def calculate_depth(tilt):
    '''calculate depth of the object based on wrist flex angle'''
    degree_angle = (100 - tilt)
    result_deg = math.tan(math.radians(degree_angle)) * 21 + 1.8
    print(f"Calculated depth of the object: {result_deg} cm, with wrist flex angle: {tilt} degrees")
    return result_deg


def solve_arm_angles(reach_x, elbow_up=ELBOW_UP):
    """
    Given the horizontal distance from the shoulder axis to the object,
    return (shoulder_lift_pos, elbow_flex_pos, wrist_flex_pos) in robot
    units, or None if the pose is unreachable.
 
    reach_x: cm, horizontal distance shoulder axis -> object, in the
             vertical plane already selected by shoulder_pan.
    """
    # 1. Target tip pose. y is fixed because the table is flat.
    tip_x, tip_y = reach_x, GRASP_Y
 
    # 2. Back off along the final link to get the wrist joint position.
    #    With the tip pointing straight down, the wrist sits directly
    #    above the tip by L3.
    wx = tip_x - L3 * math.cos(TIP_ORIENTATION)   # = tip_x
    wy = tip_y - L3 * math.sin(TIP_ORIENTATION)   # = tip_y + L3
 
    LOFF = 4.0
    BRACKET_SIDE = -1   # +1 or -1, whichever matches the visualiser

    Le = math.hypot(L1, LOFF)
    phi = math.atan2(BRACKET_SIDE * LOFF, L1)

    r = math.hypot(wx, wy)
    if r > (Le + L2) or r < abs(Le - L2):
        return None

    cos_q2 = max(-1.0, min(1.0, (r*r - Le*Le - L2*L2) / (2 * Le * L2)))
    q2e = -math.acos(cos_q2) if elbow_up else math.acos(cos_q2)
    qe = math.atan2(wy, wx) - math.atan2(L2 * math.sin(q2e), Le + L2 * math.cos(q2e))

    q1 = qe - phi    # motor angle: direction of arm 1
    q2 = q2e + phi   # motor angle: L2 relative to arm 1
    q3 = TIP_ORIENTATION - q1 - q2
 
    # 5. Convert to the robot's joint convention.
    shoulder = SHOULDER_SIGN * math.degrees(q1) + SHOULDER_OFFSET
    elbow    = ELBOW_SIGN    * math.degrees(q2) + ELBOW_OFFSET
    wrist    = 90 + WRIST_SIGN    * math.degrees(q3) + WRIST_OFFSET
 
    return shoulder, elbow, wrist


def check_limits(angles):
    """Return True only if every solved angle is inside its safe range."""
    if angles is None:
        return False
    shoulder, elbow, wrist = angles
    return (JOINTS_LIMITS["shoulder_lift.pos"][0] <= shoulder <= JOINTS_LIMITS["shoulder_lift.pos"][1] and
            JOINTS_LIMITS["elbow_flex.pos"][0] <= elbow <= JOINTS_LIMITS["elbow_flex.pos"][1] and
            JOINTS_LIMITS["wrist_flex.pos"][0] <= wrist <= JOINTS_LIMITS["wrist_flex.pos"][1])


def tip_position(q1, q2, q3):
    """Where the tip actually ends up, in the shoulder frame (radians in)."""
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + L3 * math.cos(q1 + q2 + q3)
    y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2) + L3 * math.sin(q1 + q2 + q3)
    return x, y


def close_gripper(gripper_pos):
    """Close the gripper to pick up the object."""
    for i in range(gripper_pos, 12, -1):
        robot.send_action({"gripper.pos": i})
        time.sleep(0.01)


def move_to_hand_position():
    """Move the arm to a position suitable for handing over the object."""
    handover_positions = {
        "shoulder_pan.pos": 65,
        "shoulder_lift.pos": -70,
        "elbow_flex.pos": 60,
        "wrist_flex.pos": 72,
    }

    current_positions = robot.get_observation()

    for i in range(100):
        intermediate_positions = {
    "shoulder_pan.pos": current_positions["shoulder_pan.pos"] + (handover_positions["shoulder_pan.pos"] - current_positions["shoulder_pan.pos"]) * (i + 1) / 100,
    "shoulder_lift.pos": current_positions["shoulder_lift.pos"] + (handover_positions["shoulder_lift.pos"] - current_positions["shoulder_lift.pos"]) * (i + 1) / 100,
    "elbow_flex.pos": current_positions["elbow_flex.pos"] + (handover_positions["elbow_flex.pos"] - current_positions["elbow_flex.pos"]) * (i + 1) / 100,
    "wrist_flex.pos": current_positions["wrist_flex.pos"] + (handover_positions["wrist_flex.pos"] - current_positions["wrist_flex.pos"]) * (i + 1) / 100,
}
        robot.send_action(intermediate_positions)
        time.sleep(0.011)
    time.sleep(1)  # Allow time for the gripper to close

    #release the object by opening the gripper
    robot.send_action({"gripper.pos": 30})
    time.sleep(1)  # Allow time for the gripper to open


    


robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
robot = SO101Follower(robot_config)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    cap = cv.VideoCapture(0)

def main():
    try:
        robot.connect()
        robot.send_action(STARTING_POSITIONS)
        pan = STARTING_POSITIONS["shoulder_pan.pos"]
        tilt = STARTING_POSITIONS["wrist_flex.pos"]
        time.sleep(1)  # Allow time for the robot to reach starting positions
        while True:
            coords = find_object()
            if coords is None:
                #pan_scan(coords)
                continue
            pan, tilt, centered = track_object(coords, pan, tilt)
            centered_count = centered_count + 1 if centered else 0
            print(f"Centered count: {centered_count}")
            if centered_count < CENTERED_FRAMES_REQUIRED:
                continue
            depth = calculate_depth(tilt)
            
            #reach_x = depth + CAMERA_OFFSET
            angles = solve_arm_angles(depth)
            if angles is None:
                continue
            if not check_limits(angles):
                print("Calculated angles are out of limits. Skipping this position.")
                continue
            print(f"Calculated angles: Shoulder: {angles[0]}, Elbow: {angles[1]}, Wrist: {angles[2]}")

            current_positions = robot.get_observation()
            gripper_open = position_arm(current_positions, angles)
            time.sleep(1)  # Allow time for the arm to reach the calculated position
            close_gripper(gripper_open)
            time.sleep(1)  # Allow time for the gripper to close
            move_to_hand_position()
            centered_count = 0

            robot.send_action(STARTING_POSITIONS)
            time.sleep(2)  # Allow time for the robot to reach starting positions
            pan, tilt = STARTING_POSITIONS["shoulder_pan.pos"], STARTING_POSITIONS["wrist_flex.pos"]

            if cv.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()
        robot.disconnect() 
        print("Disconnected.")


if __name__ == "__main__":
    TARGET_CLASS = 'black ball'  # Change this to the class you want to detect (tipp_ex, car, black ball)
    main()
    