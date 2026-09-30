from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

config = SO101FollowerConfig(port="COM3", id="follower_arm")
robot = SO101Follower(config)
robot.connect()
robot.bus.disable_torque()          # arm goes limp, safe to move by hand
with robot.bus.torque_disabled():   # or just leave it disabled manually
    while True:
        obs = robot.get_observation()   # reads Present_Position via sync_read
        print(obs)