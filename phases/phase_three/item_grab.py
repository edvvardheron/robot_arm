from ultralytics import YOLO

MODEL_PATH = YOLO('phases/phase_three/yolo26n-seg_trained.pt')  # Load the YOLO model from the specified path

import time
from lerobot.robots.so_follower import SO101Follower
import cv2 as cv
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
import numpy as np
import math
from calibration_setup.calibration import CameraToArm, open_camera

PORT = 'COM3'
ROBOT_ID = 'follower_arm'

L1 = 11.5 #length of the first arm segment in cm
L2 = 13.5 #length of the second arm segment in cm
L3 = 13.2 #length of the third arm segment in cm
SHOULDER_HEIGHT = 12 #height of the shoulder joint from the ground in cm
OBJECT_GRASP_HEIGHT = 1
GRASP_Y = OBJECT_GRASP_HEIGHT - SHOULDER_HEIGHT
TIP_ORIENTATION = -math.pi / 2


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
    "shoulder_lift.pos": -65,
    "elbow_flex.pos": 65,
    "wrist_flex.pos": 65,
    "wrist_roll.pos": 0,
    "gripper.pos": 5,
}

HOVER_HEIGHT = 5.0  # Height above the object to hover before grabbing, in cm
OPEN_GRIPPER_POSITION = 30  # Position to open the gripper
CLOSED_GRIPPER_POSITION = 14  # Position to close the gripper

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

# Height of the objects' top face above the table, cm. The masks are labelled on
# the top face, so this is the height used to convert mask pixels to arm coordinates.
# Measure your objects - this is NOT the hover or grasp height.
OBJECT_TOP_HEIGHT = {"brick": 2, "tube": 4, "ball": 3.5}

model = MODEL_PATH

'''Identify an object classified by the YOLO model in the frame, use the aerial camera to identify the coordinates and angle of the object.

Convert these coordinates to the robot's coordinate system, and then use inverse kinematics to calculate the joint angles needed to reach the object.

Finally, move the robot's arm to the calculated joint angles and close the gripper at the correct angle to grab the object.'''

def grasp_from_contour(contour_px, z_cm, camera=None):
    """Mask outline (image pixels) -> grasp geometry in the ARM frame.

    Every outline point is converted to arm coordinates first, and the rectangle
    is fitted there. That removes the camera's rotation relative to the arm, the
    flipped image y-axis and the lens distortion, so the angle needs no sign fix.

    Returns (x, y, long_angle_deg, width_cm, box_cm):

    """
    camera = camera or cam
    pts = np.array([camera.pixel_to_arm(float(u), float(v), z_cm)[:2]
                    for u, v in np.asarray(contour_px).reshape(-1, 2)], dtype=np.float32)
    rect = cv.minAreaRect(pts)
    box = cv.boxPoints(rect)

    # Take the angle from the rectangle's edges, not from minAreaRect's angle
    # output, whose convention changed between OpenCV versions.
    e1, e2 = box[1] - box[0], box[2] - box[1]
    long_e, short_e = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    long_angle = math.degrees(math.atan2(long_e[1], long_e[0]))
    long_angle = ((long_angle + 90.0) % 180.0) - 90.0   # an axis has no direction: fold to [-90, 90)

    length_cm = np.linalg.norm(long_e)
    x, y = rect[0]
    return float(x), float(y), long_angle, float(np.linalg.norm(short_e)), box, length_cm


def find_object(frame):
    """Find the target object in the aerial frame and return its coordinates.
    
    create a mask to identify the object based on its color and shape, and then use the YOLO model to classify the object and return its coordinates.
    """
    results = model(frame)
    CONF_THRESHOLD = 0.6  # Confidence threshold for object detection
    best_box = None
    best_conf = CONF_THRESHOLD
    for i, box in enumerate(results[0].boxes):
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"Detected object: {cls_name} with confidence {conf:.2f}")
        if conf > best_conf:
            best_conf = conf
            best_box = box
            best_index = i
            best_cls = cls_name
    if best_box is None:
        return None  # No object detected with sufficient confidence
    
    # Position, angle and width all come from the mask, measured in the ARM frame.
    contour = results[0].masks.xy[best_index]
    top_height = OBJECT_TOP_HEIGHT[best_cls]
    x, y, long_angle, width_cm, box_cm, length_cm = grasp_from_contour(contour, top_height)
    reach, pan_deg = cam.to_polar(x, y)
    if reach < 7: #skip objects that are too close to the robot
        return None

    # Draw the arm-frame rectangle back onto the image.

    box_px = cam.arm_to_pixel([[bx, by, top_height] for bx, by in box_cm]).astype(np.int32)
    cv.polylines(frame, [box_px], True, (0, 255, 0), 2)
    u, v = cam.arm_to_pixel([[x, y, top_height]])[0].astype(int)
    cv.putText(frame, f"{long_angle:.0f} deg, {width_cm:.1f} cm", (int(u), int(v)),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    z = HOVER_HEIGHT   # IK target height for the first approach (not used for locating)
    cv.imshow("Object Detection", frame)
    cv.waitKey(1)  # Display the frame for a brief moment
    time.sleep(1)  # Allow time for the frame to be displayed
    angle = gripper_clear(x, y, candidate_angles(long_angle, length_cm), best_index, results)
    if angle is not None:
        return [x, y, top_height, z, reach, pan_deg, angle]
    return None  # No clear angle found for the gripper




def solve_arm_angles(target_location, elbow_up=ELBOW_UP):
    """
    Inverse kinematics for the SO-101 with a vertical gripper.

    reach   : horizontal distance from the PAN AXIS to the object, cm (from cam.to_polar)
    pan_deg : bearing of the object about the pan axis, degrees (from cam.to_polar)
    z       : height of the grasp point above the table, cm
    angle   : object long-axis angle in the ARM frame, degrees (from find_object)

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
    relative = ((relative + 90.0) % 180.0)
    targets["wrist_roll.pos"] = ROLL_SIGN * relative + ROLL_OFFSET
    if targets["wrist_roll.pos"] > 90:
        targets["wrist_roll.pos"] -= 180.0


    # 7. Refuse anything outside the safe joint ranges.
    for joint, value in targets.items():
        lo, hi = JOINTS_LIMITS[joint]
        if not (lo <= value <= hi):
            print(f"{joint} = {value:.1f} is outside its limit ({lo} to {hi})")
            return None

    return targets


def move_arm_to_targets(robot, targets, gripper_position=OPEN_GRIPPER_POSITION):
    """
    Move the robot's arm to the specified joint targets.
    """

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
            'gripper.pos': gripper_position
        }
        robot.send_action(itermediate_positions)
        time.sleep(0.01)  # Small delay to allow the arm to move smoothly
    time.sleep(1)  # Wait for the arm to reach the position

def move_to_start():
    """
    Move the robot to the starting position defined in STARTING_POSITIONS.
    """

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


def close_gripper():
    """
    Close the gripper to pick up the object.
    """
    #for i in range(12, gripper_pos, -1):
        #robot.send_action({"gripper.pos": i})
    robot.send_action({"gripper.pos": CLOSED_GRIPPER_POSITION})  # Close the gripper to the specified position
        #time.sleep(0.01)

def move_to_hand_position():
    """
    Move the arm to a position suitable for handing over the object.
    """
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
    "elbow_flex.pos": current_positions["elbow_flex.pos"] + (handover_positions["elbow_flex.pos"] - current_positions["elbow_flex.pos"]) * (i + 1) / 100, #move elbow faster to avoid collision with other objects
    "wrist_flex.pos": current_positions["wrist_flex.pos"] + (handover_positions["wrist_flex.pos"] - current_positions["wrist_flex.pos"]) * (i + 1) / 100,
}
        robot.send_action(intermediate_positions)
        time.sleep(0.011)
    time.sleep(1)  # Allow time for the gripper to close

    #release the object by opening the gripper
    robot.send_action({"gripper.pos": 30})
    time.sleep(1)  # Allow time for the gripper to open




def gripper_clear(x, y, angles, target_index, results):
    MARGIN_CM = 2.0  # Margin in cm to consider for gripper clearance
    print(f'available angles: {angles}')
    neighbours = []
    for i, box in enumerate(results[0].boxes):
        mask = results[0].masks.xy[i]
        if i == target_index or len(mask) < 3:
            continue
        z = OBJECT_TOP_HEIGHT[model.names[int(box.cls[0])]]
        neighbours.append(np.array([cam.pixel_to_arm(float(u), float(v), z)[:2] for u, v in np.asarray(mask).reshape(-1, 2)], dtype=np.float32))

    half_width = GRIPPER_OPEN_WIDTH / 2.0
    for angle in angles:
        print(f"Checking gripper clearance at angle {angle} degrees.")
        px, py = -math.sin(math.radians(angle)) * half_width, math.cos(math.radians(angle)) * half_width
        fingers = np.array([[x + px, y + py], [x - px, y - py]], dtype=np.float32)
        if all(cv.pointPolygonTest(poly, f, True) <= -MARGIN_CM for poly in neighbours for f in fingers):
            return angle
    print(f"Gripper blocked at angle {angle} degrees.")
    return None  # No clear angle found

def candidate_angles(long_angle, length_cm):
    angles = [long_angle]
    if length_cm < GRIPPER_OPEN_WIDTH:
        angles.append(long_angle + 90)
    return angles


robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
robot = SO101Follower(robot_config)
cam = CameraToArm()
cap = open_camera()


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
            for i in range(5):  # Try to find the object multiple times
                cap.grab()  # Grab a frame from the camera
            ret, frame = cap.read()
            hover_location = find_object(frame)  # Get the best bounding box of the identified object
            #gripper_positions = gripper_block(hover_location)  # Check if the gripper will be blocked by the object
            if hover_location is not None:
                # If an object is detected, move the arm to the target location and grab it
                hover_joints = solve_arm_angles(hover_location)  # Calculate the joint angles needed to reach the object
                if hover_joints is None:
                    print("Target location is unreachable or breaks joint limits.")
                    continue
                move_arm_to_targets(robot, hover_joints)
                # drop the arm down to the object to grab it
                item_location = list(hover_location)
                item_location[3] = 1.5  # Set the z coordinate to 1.5 cm above the object
                item_location = solve_arm_angles(item_location)  # Calculate the joint angles needed to reach the object at the new z coordinate
                move_arm_to_targets(robot, item_location)  # Move the arm to the calculated joint angles
                close_gripper()  # Close the gripper to pick up the object
                time.sleep(1)  # Allow time for the gripper to close

                return_hover_location = list(hover_location)
                return_hover_location[3] = HOVER_HEIGHT + 5  # Set the z coordinate to the hover
                return_hover_joints = solve_arm_angles(return_hover_location)  # Calculate the joint angles needed to reach the hover position
                
                move_arm_to_targets(robot, return_hover_joints, gripper_position=CLOSED_GRIPPER_POSITION) # Move the arm back to the hover position after picking up the object

                time.sleep(1)  # Allow time for the arm to move back to the hover position



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