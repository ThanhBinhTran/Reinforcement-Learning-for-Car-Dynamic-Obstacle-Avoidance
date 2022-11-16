"""
This is a part of the autonomous car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

import numpy as np
import math as math
from car_configuration import *

class Obstacles:
    def __init__(self) -> None:
        self.generate()
        self.dt = 0.1       # [s] Time tick for motion

    def generate(self):
        self.obstacles = np.random.uniform(low=WORKING_SPACE_X_MIN + 3, high = WORKING_SPACE_X_MAX -1.5, size=(OBSTACLES_NUM,2))
        self.obstacles_radius = np.random.uniform(low=0.1, high = 0.2, size=(OBSTACLES_NUM,1))
        self.obstacles_velocity = np.random.uniform(low=0, high = 1, size=(OBSTACLES_NUM,1))
        self.obstacles_yaw =  np.random.uniform(low=-math.pi, high = math.pi, size=(OBSTACLES_NUM,1))

    def motion(self):
        direction = np.array([(math.cos(yaw), math.sin(yaw)) for yaw in self.obstacles_yaw]).reshape((OBSTACLES_NUM, 2))
        distance = direction*self.obstacles_velocity*self.dt
        self.obstacles += distance
        
        
        # whenever hitting boundary, it reswpam at oposite side
        X_SPACE = WORKING_SPACE_X_MAX - WORKING_SPACE_X_MIN
        Y_SPACE = WORKING_SPACE_Y_MAX - WORKING_SPACE_Y_MIN
        boundary_MIN = self.obstacles < [WORKING_SPACE_X_MIN, WORKING_SPACE_Y_MIN]
        boundary_MAX = self.obstacles > [WORKING_SPACE_X_MAX, WORKING_SPACE_Y_MAX]
        
        self.obstacles += boundary_MIN*np.array([X_SPACE,Y_SPACE])
        self.obstacles -= boundary_MAX*np.array([X_SPACE,Y_SPACE])
    def save(self, file_name):
        np.savez(file_name, self.obstacles, self.obstacles_radius, self.obstacles_velocity, self.obstacles_yaw)
    
    def load(self, filename):
        data = np.load(filename)
        self.obstacles = data["arr_0"]
        self.obstacles_radius = data["arr_1"]
        self.obstacles_velocity = data["arr_2"]
        self.obstacles_yaw = data["arr_3"]

    def print (self):
        print("obstacles coordinate ", self.obstacles)
        print("obstacles radius ", self.obstacles_radius)
        print("obstacles velocity ", self.obstacles_velocity)
        print("obstacles yaw ", self.obstacles_yaw)
if __name__ == '__main__':
    obstacles = Obstacles()
    #obstacles.print()
    obstacles.motion()
