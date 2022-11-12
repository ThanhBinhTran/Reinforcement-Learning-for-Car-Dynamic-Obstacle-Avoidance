import math
from enum import Enum

class RobotType(Enum):
    circle = 0
    rectangle = 1



class Car_base:
    def __init__(self, vision_range=20, robot_type=RobotType.circle, robot_radius=0.2):
        self.vel_MAX =  5.0                              # [m/s]
        self.vel_MIN = -5.0                              # [m/s]
        self.vel_resolution = 0.1                        # [m/s]

        self.steer_MAX = math.pi/4                       # max left yaw
        self.steer_MIN = -math.pi/4                      # max right yaw
        self.steer_resolution = math.pi/18                # radian

        self.accel_max = 10                               # times
        self.accel_resolution = 0.5                       # acceleration
        
        self.dt = 0.5                                       # [s] Time tick for motion prediction
        self.robot_type = robot_type
        self.vision_range = vision_range                    # the range of input vision
        self.vision_range_MAX_left = math.pi/4
        self.vision_range_MAX_right = math.pi/4
        self.back_range = vision_range/2                    # the range of input vision
        self.back_range_MAX_left = math.pi/16
        self.back_range_MAX_right = math.pi/16
        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.radius = robot_radius                          # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.width = 0.1                              # [m] for collision check
        self.length = 0.2                             # [m] for collision check
    def set_width(self, width):
        self.width =width
    def set_length(self, length):
        self.length = length
    def set_width_length(self, width, length):
        self.set_width(width=width)
        self.set_length(length=length)
        
    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value