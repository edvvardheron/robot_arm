"""
Hello World: SO-101 follower arm movement test.

This confirms your arm is wired, calibrated, and controllable from Python
BEFORE you touch any camera or CV code. Get this working first.

Update the port and id below to match your setup (from `lerobot-find-port`
and whatever id you used in `lerobot-calibrate`).
"""

import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# --- CONFIG: update these two lines for your arm ---
PORT = "COM3"          # from `lerobot-find-port`
ROBOT_ID = "follower_arm"   # whatever id you used in `lerobot-calibrate`
# ----------------------------------------------------

robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
robot = SO101Follower(robot_config)

print("Connecting to robot...")
robot.connect()
print("Connected.")

# All joint values are in normalized units per LeRobot's calibration,
# NOT raw degrees. 0.0 is roughly the calibrated "middle" position for
# most joints. Keep movements small and slow for this first test.

home_position = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}

# A second, slightly different position so you can SEE it move.
# Small offsets only — this is a "does it move" test, not a real motion.
test_position = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -65.0,
    "elbow_flex.pos": 65.0,
    "wrist_flex.pos": 65.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 30.0,
}

try:
    print("Moving to home position...")
    robot.send_action(home_position)
    time.sleep(2)

    print("Reading current joint positions:")
    print(robot.get_observation())

    print("Moving to test position...")
    robot.send_action(test_position)
    time.sleep(2)

    '''print("Moving back to home...")
    robot.send_action(home_position)
    time.sleep(2)'''

    print("Success — arm is responding to commands.")

finally:
    robot.disconnect()
    print("Disconnected.")