import numpy as np
import math as math
from car_configuration import *

class Obstacles:
    def __init__(self) -> None:
        self.obstacles = np.random.uniform(low=WORKING_SPACE_X_MIN + 3, high = WORKING_SPACE_X_MAX -1.5, size=(OBSTACLES_NUM,2))
        self.obstacles_radius = np.random.uniform(low=0.1, high = 0.2, size=(OBSTACLES_NUM,1))
        self.obstacles_velocity = np.random.uniform(low=0, high = 1, size=(OBSTACLES_NUM,1))
        self.obstacles_yaw =  np.random.uniform(low=-math.pi, high = math.pi, size=(OBSTACLES_NUM,1))
        self.dt = 0.1       # [s] Time tick for motion

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

    def print (self):
        print("obstacles coordinate ", self.obstacles)
        print("obstacles radius ", self.obstacles_radius)
        print("obstacles velocity ", self.obstacles_velocity)
        print("obstacles yaw ", self.obstacles_yaw)
if __name__ == '__main__':
    obstacles = Obstacles()
    #obstacles.print()
    obstacles.motion()
