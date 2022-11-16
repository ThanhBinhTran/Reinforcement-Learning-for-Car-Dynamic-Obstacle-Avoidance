"""
This is a part of the autonomous car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

import math
from enum import Enum

class RobotType(Enum):
    circle = 0
    rectangle = 1

class Robot_mode(Enum):
    learning = 0
    running = 1

class Robot_status(Enum):
    none = 0
    time_out = 1
    hit_obstacles =2
    out_of_boundary =3

class Car_base:
    def __init__(self, vision_range=20, robot_type=RobotType.circle, robot_radius=0.2):
        self.vel_MAX =  5.0                              # [m/s]
        self.vel_MIN = -5.0                              # [m/s]
        self.vel_resolution = 0.4                        # [m/s]

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
        self.back_range = 1.5                               # the range of back vision
        self.back_range_MAX_left = math.pi/16
        self.back_range_MAX_right = math.pi/16
        
        # if robot_type == RobotType.circle
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