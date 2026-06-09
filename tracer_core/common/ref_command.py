class RefCommand(object):
    """
    Simulator-independent command representation.

    For now, vx/vy/yaw_rate/height_rate are normalized joystick-axis commands.
    Later, this can be changed to physical units for Isaac Sim or a real robot.
    """
    def __init__(self, vx=0.0, vy=0.0, yaw_rate=0.0, height_rate=0.0,
                 walk=False, emergency_stop=False):
        self.vx = vx
        self.vy = vy
        self.yaw_rate = yaw_rate
        self.height_rate = height_rate
        self.walk = walk
        self.emergency_stop = emergency_stop
