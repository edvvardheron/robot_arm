"""
Phase 1b: Visual tracking with eye-in-hand camera.

Keeps the detected red object centered in the camera's view by continuously
nudging the arm's pan/tilt joints. This is a simple proportional (P) controller
operating directly in image space -- no 3D calibration needed yet.

Run the color-detection logic standalone first (no arm) to confirm your HSV
thresholds are solid before combining with arm movement.
"""

import time

import cv2
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# --- CONFIG ---
PORT = "COM3"
ROBOT_ID = "follower_arm"
CAMERA_INDEX = 0

# HSV range for "red" -- red wraps around 0/180 in OpenCV's hue range,
# so you often need two ranges. Tune these against your actual lighting.
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([5, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

# Controller gains -- START SMALL. If the arm oscillates or overshoots,
# lower these. If it barely moves, raise them slightly.
KP_PAN = 0.02
KP_TILT = 0.02

# Flip these to -1 if the arm moves the WRONG direction on your first test.
# You'll find out immediately -- if the object drifts further off-center
# instead of toward it, flip the sign.
PAN_SIGN = 1
TILT_SIGN = -1

# Safety limits
MAX_STEP = 2.0        # max degrees-equivalent change per loop iteration
DEADBAND_PX = 15       # if within this many pixels of center, stop moving
LOOP_DELAY = 0.1       # seconds between control updates (10 Hz)

JOINT_LIMITS = {       # adjust to your calibration's safe range
    "shoulder_pan.pos": (-45, 45),
    "wrist_flex.pos": (-30, 30),
}
# ----------------


def find_red_object(frame):
    """Returns (cx, cy, area) of the largest red contour, or None if not found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200:  # ignore tiny noise blobs
        return None

    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    return cy, cx, area


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(robot_config)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Connecting to robot...")
    robot.connect()
    print("Connected. Starting tracking loop. Press Ctrl+C to stop.")

    try:
        # Start from current position and adjust from there
        current = robot.get_observation()
        pan = current.get("shoulder_pan.pos", 0.0)
        tilt = current.get("wrist_flex.pos", 0.0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, skipping frame")
                continue

            h, w = frame.shape[:2]
            frame_cx, frame_cy = w // 2, h // 2

            result = find_red_object(frame)

            if result is not None:
                obj_cx, obj_cy, area = result

                # Pixel error: positive = object is right/below center
                error_x = obj_cx - frame_cx
                error_y = obj_cy - frame_cy

                # Draw for visual debugging
                cv2.circle(frame, (obj_cx, obj_cy), 8, (0, 255, 0), -1)
                cv2.circle(frame, (frame_cx, frame_cy), 4, (255, 0, 0), -1)

                if abs(error_x) > DEADBAND_PX or abs(error_y) > DEADBAND_PX:
                    pan_step = clamp(PAN_SIGN * KP_PAN * error_x, -MAX_STEP, MAX_STEP)
                    tilt_step = clamp(TILT_SIGN * KP_TILT * error_y, -MAX_STEP, MAX_STEP)

                    pan = clamp(pan + pan_step, *JOINT_LIMITS["shoulder_pan.pos"])
                    tilt = clamp(tilt + tilt_step, *JOINT_LIMITS["wrist_flex.pos"])

                    robot.send_action({
                        "shoulder_pan.pos": pan,
                        "wrist_flex.pos": tilt,
                    })
                    print(f"Tracking: error=({error_x},{error_y}) -> pan={pan:.1f} tilt={tilt:.1f}")
                else:
                    print("Object centered, holding position")
            else:
                print("No object detected")

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) == ord("q"):
                break

            time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()