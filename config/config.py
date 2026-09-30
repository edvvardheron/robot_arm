from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader

config = SO101LeaderConfig(
    port="COM3",
    id="leader_arm",
)

leader = SO101Leader(config)
leader.connect(calibrate=False)
leader.calibrate()
leader.disconnect()