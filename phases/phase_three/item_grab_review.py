from ultralytics import YOLO

MODEL_PATH = YOLO('phases/phase_three/yolo26n-seg_trained.pt')  # Load the YOLO model from the specified path

import time
from lerobot.robots.so_follower import SO101Follower
import cv2 as cv
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
import numpy as np
import math
from calibration_setup.calibration import CameraToArm

PORT = 'COM3'
ROBOT_ID = 'follower_arm'

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
    "shoulder_lift.pos": -90,
    "elbow_flex.pos": 50,
    "wrist_flex.pos": 65,
    "wrist_roll.pos": 0,
    "gripper.pos": 5,
}
OPEN_GRIPPER_POSITION = 30  # Position to open the gripper

SHOULDER_SIGN, SHOULDER_OFFSET = -1.0, 90.0
ELBOW_SIGN,    ELBOW_OFFSET    = -1.0, -90.0
WRIST_SIGN,    WRIST_OFFSET    = -1.0, -90.0

# --- extra geometry used by solve_arm_angles --------------------------------
LOFF = 4.0           # fixed 90-degree bracket between arm 1 and the elbow axis, cm
BRACKET_SIDE = -1     # +1 / -1. Must match the "Bracket on +90 deg side" tickbox in the visualiser
PAN_AXIS_X = -8  # pan axis relative to the shoulder_lift axis, cm (+x = reach direction, so
                     # negative = pan axis sits behind the shoulder)
ELBOW_UP = True      # which of the two IK solutions to use

# shoulder_pan.pos = PAN_SIGN * pan_deg + PAN_OFFSET            (calibrate against cam.to_polar)
PAN_SIGN,  PAN_OFFSET  = -1.0, -10.0
# wrist_roll.pos = ROLL_SIGN * (object angle relative to arm heading) + ROLL_OFFSET
ROLL_SIGN, ROLL_OFFSET = 1.0, 0.0


GRIPPER_OPEN_WIDTH = 7.5

model = MODEL_PATH

'''Identify an object classified by the YOLO model in the frame, use the aerial camera to identify the coordinates and angle of the object.

Convert these coordinates to the robot's coordinate system, and then use inverse kinematics to calculate the joint angles needed to reach the object.

Finally, move the robot's arm to the calculated joint angles and close the gripper at the correct angle to grab the object.'''

def find_object(frame):
    """Find the target object in the aerial frame and return its coordinates. create a mask to identify the object based on its color and shape, and then use the YOLO model to classify the object and return its coordinates."""
    results = model(frame)
    CONF_THRESHOLD = 0.7  # Confidence threshold for object detection
    best_box = None
    best_conf = CONF_THRESHOLD
    for i, box in enumerate(results[0].boxes):
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box
            best_index = i
        if best_box is None:
            return None
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)
    print(f"Detected object at pixel coordinates: u={u}, v={v}")

    # attain object rotation angle using the minAreaRect function from OpenCV

    mask = results[0].masks.xy[best_index]
    pts = np.array(mask, dtype=np.int32)
    rect = cv.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    if w < h:
        angle -= 90
        w, h = h, w
    box = cv.boxPoints(rect)
    box = np.intp(box)
    cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
    cv.putText(frame, f"Angle: {angle:.2f}", (int(cx), int(cy)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



    x, y, z = cam.pixel_to_arm(u, v, z_cm=1.5)   # z_cm = height of the point you're locating
    reach, pan_deg = cam.to_polar(x, y)            # -> IK solver
    print(f"Object coordinates in robot frame: x={x:.2f}, y={y:.2f}, z={z:.2f}, reach={reach:.2f}, pan_deg={pan_deg:.2f}, angle={-angle:.2f}")
    cv.imshow("Object Detection", frame)
    return best_box, x, y, z, reach, pan_deg, -angle



def solve_arm_angles(target_location, elbow_up=ELBOW_UP):
    """
    Inverse kinematics for the SO-101 with a vertical gripper.

    reach   : horizontal distance from the PAN AXIS to the object, cm (from cam.to_polar)
    pan_deg : bearing of the object about the pan axis, degrees (from cam.to_polar)
    z       : height of the grasp point above the table, cm
    angle   : object long-axis angle in the image, degrees (from find_object)

    Returns a dict of joint positions (robot units) ready for robot.send_action(),
    or None if the pose is unreachable or breaks a joint limit.
    The gripper is not included - open and close it separately.
    """
    # 1. Move into the shoulder_lift frame the planar IK is solved in.
    #    The pan axis sits PAN_AXIS_X from the shoulder axis, so an object `reach`
    #    from the pan axis is PAN_AXIS_X + reach from the shoulder axis.
    reach = target_location[4]
    pan_deg = target_location[5]
    z = target_location[3]
    angle = target_location[6]
    tip_x = reach + PAN_AXIS_X
    tip_y = z - SHOULDER_HEIGHT

    # 2. Gripper points straight down, so the wrist joint sits L3 above the tip.
    wx = tip_x - L3 * math.cos(TIP_ORIENTATION)
    wy = tip_y - L3 * math.sin(TIP_ORIENTATION)

    # 3. The 90-degree bracket makes shoulder->elbow one effective link of length Le,
    #    bent phi away from arm 1. Solve two-link IK on that.
    Le = math.hypot(L1, LOFF)
    phi = math.atan2(BRACKET_SIDE * LOFF, L1)

    r = math.hypot(wx, wy)
    if r > (Le + L2) or r < abs(Le - L2):
        print(f"Unreachable: wrist target {r:.1f} cm from shoulder, "
              f"allowed {abs(Le - L2):.1f}-{Le + L2:.1f} cm")
        return None

    cos_q2 = max(-1.0, min(1.0, (r * r - Le * Le - L2 * L2) / (2 * Le * L2)))
    q2e = -math.acos(cos_q2) if elbow_up else math.acos(cos_q2)
    qe = math.atan2(wy, wx) - math.atan2(L2 * math.sin(q2e), Le + L2 * math.cos(q2e))

    # 4. Rotate back from the effective link to the real motor angles.
    q1 = qe - phi            # direction of arm 1
    q2 = q2e + phi           # L2 relative to arm 1
    q3 = TIP_ORIENTATION - q1 - q2   # wrist takes up whatever keeps the tip vertical

    # 5. Convert to the robot's joint convention.
    targets = {
        "shoulder_pan.pos":  PAN_SIGN * pan_deg + PAN_OFFSET,
        "shoulder_lift.pos": (SHOULDER_SIGN * math.degrees(q1) + SHOULDER_OFFSET),
        "elbow_flex.pos":    (ELBOW_SIGN * math.degrees(q2) + ELBOW_OFFSET),
        "wrist_flex.pos":   90 + (WRIST_SIGN * math.degrees(q3) + WRIST_OFFSET),
    }

    # 6. Wrist roll: align the gripper with the object. The gripper is symmetric
    #    under 180 degrees, so wrap to [-90, 90).
    relative = angle - pan_deg
    relative = ((relative + 90.0) % 180.0) - 90.0
    targets["wrist_roll.pos"] = ROLL_SIGN * relative + ROLL_OFFSET

    # 7. Refuse anything outside the safe joint ranges.
    for joint, value in targets.items():
        lo, hi = JOINTS_LIMITS[joint]
        if not (lo <= value <= hi):
            print(f"{joint} = {value:.1f} is outside its limit ({lo} to {hi})")
            return None

    return targets


def move_arm_to_targets(robot, targets):
    """Move the robot's arm to the specified joint targets."""

    current = robot.get_observation()

    if targets is None:
        print("No valid targets to move to.")
        return

    for i in range(100):
        f = (i + 1)/100
        itermediate_positions = {
            'shoulder_pan.pos': current['shoulder_pan.pos'] + f * (targets['shoulder_pan.pos'] - current['shoulder_pan.pos']),
            'shoulder_lift.pos': current['shoulder_lift.pos'] + f * (targets['shoulder_lift.pos'] - current['shoulder_lift.pos']),
            'elbow_flex.pos': current['elbow_flex.pos'] + f * (targets['elbow_flex.pos'] - current['elbow_flex.pos']),
            'wrist_flex.pos': current['wrist_flex.pos'] + f * (targets['wrist_flex.pos'] - current['wrist_flex.pos']),
            'wrist_roll.pos': current['wrist_roll.pos'] + f * (targets['wrist_roll.pos'] - current['wrist_roll.pos']),
            'gripper.pos': OPEN_GRIPPER_POSITION
        }
        robot.send_action(itermediate_positions)
        time.sleep(0.01)  # Small delay to allow the arm to move smoothly
    time.sleep(1)  # Wait for the arm to reach the position

def move_to_start():
    '''Move the robot to the starting position defined in STARTING_POSITIONS.'''

    current = robot.get_observation()
    targets = STARTING_POSITIONS
    for i in range(100):
        f = (i + 1)/100
        itermediate_positions = {
            'shoulder_pan.pos': current['shoulder_pan.pos'] + f * (targets['shoulder_pan.pos'] - current['shoulder_pan.pos']),
            'shoulder_lift.pos': current['shoulder_lift.pos'] + f * (targets['shoulder_lift.pos'] - current['shoulder_lift.pos']),
            'elbow_flex.pos': current['elbow_flex.pos'] + f * (targets['elbow_flex.pos'] - current['elbow_flex.pos']),
            'wrist_flex.pos': current['wrist_flex.pos'] + f * (targets['wrist_flex.pos'] - current['wrist_flex.pos']),
            'wrist_roll.pos': current['wrist_roll.pos'] + f * (targets['wrist_roll.pos'] - current['wrist_roll.pos']),
            'gripper.pos': current['gripper.pos'] + f * (targets['gripper.pos'] - current['gripper.pos'])
        }
        robot.send_action(itermediate_positions)
        time.sleep(0.01)  # Small delay to allow the arm to move smoothly
    time.sleep(1)  # Wait for the arm to reach the position


def close_gripper(gripper_pos):
    """Close the gripper to pick up the object."""
    #for i in range(12, gripper_pos, -1):
        #robot.send_action({"gripper.pos": i})
    robot.send_action({"gripper.pos": 12})  # Close the gripper to the specified position
        #time.sleep(0.01)

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



def gripper_block(target_location, frame):
    '''this function checks whether, given the angle of pick up, whether the gripper will be blocked by the object. If it is, it returns True, otherwise False.'''
    x = target_location[1]
    y = target_location[2]
    angle = target_location[6]


    half_width = GRIPPER_OPEN_WIDTH / 2
    perp = (-math.sin(math.radians(angle)), math.cos(math.radians(angle)))  # perpendicular vector to the object angle
    top_finger = (x + half_width * perp[0], y + half_width * perp[1])
    bottom_finger = (x - half_width * perp[0], y - half_width * perp[1])

    # Check if the gripper fingers are within the bounding box of another object
    results = model(frame)
    tx1, ty1, tx2, ty2 = target_location[0].xyxy[0].tolist()
    tu, tv = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    MARGIN_CM = 1.0  # extra clearance beyond the fingers' own thickness

    for i, box in enumerate(results[0].boxes):
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        bu, bv = (bx1 + bx2) / 2, (by1 + by2) / 2
        if abs(bu - tu) < 2 and abs(bv - tv) < 2:
            continue  # this is the target object itself, not a neighbour

        mask = results[0].masks.xy[i]
        poly_cm = np.array([cam.pixel_to_arm(u, v, z_cm=1.5)[:2] for u, v in mask], dtype=np.float32)
        for finger in (top_finger, bottom_finger):
            dist = cv.pointPolygonTest(poly_cm, (float(finger[0]), float(finger[1])), True)
            if dist > -MARGIN_CM:
                return True  # a finger would hit, or nearly hit, this neighbouring object

    return False








robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
robot = SO101Follower(robot_config)
cam = CameraToArm()
cap = cv.VideoCapture(0)


if not cap.isOpened():
    raise RuntimeError('Could not open camera')
def main():
    try:
        print("Connecting to robot...")
        robot.connect()
        print("Connected.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            move_to_start()  # Move the robot to the starting position before attempting to grab an object
            target_location = find_object(frame)  # Get the best bounding box of the identified object
            #gripper_positions = gripper_block(target_location)  # Check if the gripper will be blocked by the object
            if target_location is not None:
                # If an object is detected, move the arm to the target location and grab it
                joint_targets = solve_arm_angles(target_location)  # Calculate the joint angles needed to reach the object
                move_arm_to_targets(robot, joint_targets)
                close_gripper(OPEN_GRIPPER_POSITION)  # Close the gripper to pick up the object
                time.sleep(1)  # Allow time for the gripper to close
                move_to_hand_position()
            else:
                print("No object detected.")
                continue
    finally:
        cap.release()
        cv.destroyAllWindows()
        robot.disconnect()

if __name__ == "__main__":
    main()
