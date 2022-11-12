"""
This is a part of the autonomous driving car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

EPSILON = 0.000001

''' LiDAR front vision '''
LIDAR_SECTORS = 6            # number of sector in front of car
LIDAR_PLUSES_PER_SECTOR=5    # number of laser pluses in a sector
LIDAR_PLUSES = LIDAR_SECTORS*LIDAR_PLUSES_PER_SECTOR # total number of laser pluses

''' Back of the car vision '''
BACK_VISION_LINES = 4
BACK_VISION_SAFE_DISTANCE = 1   # safety distance for the back of the car

''' 
STATES
LIDAR STATE (EACH SECTOR HAS DANGER LEVEL +1 STATE)
BOUNDARY X-AXIS STATE (EACH SECTOR HAS 3 STATES: LEFTSIDE, RIGHTSIDE, INSIDE)
BOUNDARY Y-AXIS STATE (EACH SECTOR HAS 3 STATES: UPSIDE, DOWNSIDE, INSIDE)
YAW STATE (EACH SECTOR HAS 4 SECTOR (UP, DOWN, RIGHT, LEFT))
BACK VISION STATE (EACH SECTOR HAS 2 STATES: CLOSE-BACK-OBSTACLES, FAR-AWAY-BACK_OBSTACLES)
'''
DANGER_LEVEL = 4            # = distance to obstacles = range(0,4)
BOUNDARY_X_STATE = 3
BOUNDARY_Y_STATE = 3
YAW_STATE = 4
BACK_VISION_STATE = 2

''' Obstacles number '''
OBSTACLES_NUM = 30          # number of generated obstacles

''' Working space configuration '''
WORKING_SPACE_X_MIN = 0
WORKING_SPACE_X_MAX = 10
WORKING_SPACE_Y_MIN = 0
WORKING_SPACE_Y_MAX = 10

''' For save q table '''
Q_TALBE_FILE='q_table_car'  # q table files
