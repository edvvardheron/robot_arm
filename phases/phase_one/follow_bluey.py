import time
from lerobot.robots.so_follower import SO101Follower
import cv2 as cv
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
import numpy as np

#step 1: config for robot, colour, controller speed, controller direction
#step 2: find red object in camera frame
#step 3: connect to robot and camera
#step 4: get current position of robot and start tracking loop
#step 5: calculate error between object and center of frame
#step 6: if error is greater than deadband, calculate new pan and tilt values
#step 7: send new pan and tilt values to robot


#step 1
PORT = "COM3"
ROBOT_ID = "follower_arm"

UPPER_RED1 = np.array([5, 255, 255])
LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED2 = np.array([179, 255, 255])
LOWER_RED2 = np.array([170, 100, 100])


PAN_SPEED = 0.1
TILT_SPEED = 0.1

PAN_DIRECTION = 1
TILT_DIRECTION = -1
DEADBAND = 20
LOOP_DELAY = 0.01

JOINT_LIMITS = {       # adjust to your calibration's safe range
    "shoulder_pan.pos": (-45, 45),
    "wrist_flex.pos": (-75, 75),
}


#step 2
def find_red_object(frame, frame_cx, frame_cy):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv.bitwise_or(mask1, mask2)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rect = cv.minAreaRect(largest_contour)
        box = cv.boxPoints(rect).reshape(-1, 2).astype(np.int32)
        cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        # Calculate the distance from the center of the frame to the center of the red object
        coord_x = cx - frame_cx
        coord_y = cy - frame_cy
        area = cv.contourArea(largest_contour)
        if area < 1000:  # ignore tiny noise blobs
            return None

        #show blob center
        cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        return coord_x, coord_y, area

#step 3
def main():
    robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(robot_config)

    try:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Could not open camera')
        print("Connecting to robot...")
        robot.connect()
        print("Connected. Starting tracking loop. Press Ctrl+C to stop.")
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

            result = find_red_object(frame, frame_cx, frame_cy)
            if result is not None:
                print(f"Red object detected at: {result}")
                coord_y, coord_x, area = result
                if abs(coord_x) > DEADBAND or abs(coord_y) > DEADBAND:
                    pan += PAN_DIRECTION * PAN_SPEED * coord_x * 0.1
                    tilt += TILT_DIRECTION * TILT_SPEED * coord_y * 0.1
                    # Apply joint limits
                    pan = max(JOINT_LIMITS["shoulder_pan.pos"][0], min(JOINT_LIMITS["shoulder_pan.pos"][1], pan))
                    tilt = max(JOINT_LIMITS["wrist_flex.pos"][0], min(JOINT_LIMITS["wrist_flex.pos"][1], tilt))
                    robot.send_action({"shoulder_pan.pos": pan})
                    robot.send_action({"shoulder_pan.pos": pan, "wrist_flex.pos": tilt})
                else:
                    print("Object is within deadband, no movement needed")
            else:
                print("No red object detected, no movement needed")

            cv.imshow("Tracking", frame)   
            if cv.waitKey(1) == ord("q"):
                break

            time.sleep(LOOP_DELAY)
    finally:
        cap.release()
        cv.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")


def horizontal_tracking():
    robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(robot_config)

    try:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Could not open camera')
        print("Connecting to robot...")
        robot.connect()
        print("Connected. Starting tracking loop. Press Ctrl+C to stop.")
        current = robot.get_observation()
        pan = current.get("shoulder_pan.pos", 0.0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, skipping frame")
                continue
            h, w = frame.shape[:2]
            frame_cx, frame_cy = w // 2, h // 2
            result = find_red_object(frame, frame_cx, frame_cy)
            if result is not None:
                print(f"Red object detected at: {result}")
                coord_y, coord_x, area = result
                if abs(coord_x) > DEADBAND:
                    pan += PAN_DIRECTION * PAN_SPEED * coord_x * 0.1
                    # Apply joint limits
                    pan = max(JOINT_LIMITS["shoulder_pan.pos"][0], min(JOINT_LIMITS["shoulder_pan.pos"][1], pan))
                    robot.send_action({"shoulder_pan.pos": pan})
                else:
                    print("Object is within deadband, no movement needed")
            else:
                print("No red object detected, no movement needed")

            cv.imshow("Tracking", frame)   
            if cv.waitKey(1) == ord("q"):
                break

            time.sleep(LOOP_DELAY)


    finally:
        cap.release()
        cv.destroyAllWindows()
        robot.disconnect()
        print("Disconnected.")



if __name__ == "__main__":
    main()
    #horizontal_tracking()