"""
This is a part of the autonomous car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

from car_base import *
import numpy as np
from obstacles import *
from car_configuration import *
from math_lib import *

class Car (Car_base):
    def __init__(self, start = [1.1, 1.1], vision_range=4.0, robot_type=RobotType.rectangle, robot_radius=0.2):
        super().__init__(vision_range, robot_type, robot_radius)
        self.start =np.array(start, dtype=float)
        self.reset()
        if self.robot_type == RobotType.rectangle:
            self.radius = max(self.length/2, self.width/2)
        self.lidar_start()
        
    def reset(self):
        self.coordinate = np.array(self.start)
        self.pre_coordinate = self.coordinate
        self.yaw = math.pi/4
        self.vel = 0
        self.steer = 0
        self.trajectory = np.array([self.coordinate])
        self.car_status = Robot_status.none
        #("car start at: ", self.coordinate, " with yaw ", math.degrees(self.yaw))

    ''' emit lidar '''
    def lidar_start(self):
        lidar_pluses = np.array([self.vision_range,0]*LIDAR_PLUSES).reshape((LIDAR_PLUSES,2))
        step = (self.vision_range_MAX_left + self.vision_range_MAX_right)/(LIDAR_PLUSES)
        start = self.yaw - self.vision_range_MAX_right
        end = self.yaw + self.vision_range_MAX_left
        yaws = np.arange(start, end, step, dtype=float)

        self.lidar_pluses = [matrix_rotation(lidar_pluse, yaw=yaw) for lidar_pluse, yaw in zip(lidar_pluses, yaws)]
        self.lidar_pluses = vector_translation(vector=self.lidar_pluses, translate_vec=self.coordinate)

    ''' find visibility lines from robot point to obstacles'''
    def find_visibility_lines(self, obstacles:Obstacles, radius, lines_pt):
        ptA = self.coordinate
        inrange_obstacles, inrange_obstacles_radius = self.in_range_obstacles(obstacles=obstacles, in_range_radius=radius)

        for ptB in lines_pt:
            line_sengment = (ptA, ptB)
            lidar_closer_pt = None
            lidar_min_distance = radius
            for obs_coordinate, obs_radius in zip(inrange_obstacles, inrange_obstacles_radius):
                x, y = obs_coordinate
                r = obs_radius
                is_pt = intersection(x=x, y=y, radius=r, line_segment=line_sengment)
                if is_pt is not None:
                    pdist_0 = point_dist(self.coordinate, is_pt[0])
                    pdist_1 = point_dist(self.coordinate, is_pt[1])
                    if pdist_0> pdist_1:
                        closer_pt = is_pt[1]
                        closer_dist = pdist_1
                    else:
                        closer_pt = is_pt[0]
                        closer_dist = pdist_0

                    if closer_dist < lidar_min_distance:
                        if inside_line_segment(closer_pt, (self.coordinate, ptB)):
                            lidar_min_distance = closer_dist
                            lidar_closer_pt = closer_pt
            if lidar_closer_pt is not None:
                ptB[0] = lidar_closer_pt[0]
                ptB[1] = lidar_closer_pt[1]

    ''' find visibility lines in boundary area'''
    def find_visibility_lines_boundary(self, lines_pt):
        linesegmentA = (WORKING_SPACE_X_MIN, WORKING_SPACE_Y_MIN), (WORKING_SPACE_X_MIN, WORKING_SPACE_Y_MAX)
        linesegmentB = (WORKING_SPACE_X_MAX, WORKING_SPACE_Y_MIN), (WORKING_SPACE_X_MAX, WORKING_SPACE_Y_MAX)
        linesegmentC = (WORKING_SPACE_X_MIN, WORKING_SPACE_Y_MIN), (WORKING_SPACE_X_MAX, WORKING_SPACE_Y_MIN)
        linesegmentD = (WORKING_SPACE_X_MIN, WORKING_SPACE_Y_MAX), (WORKING_SPACE_X_MAX, WORKING_SPACE_Y_MAX)
        for pt in lines_pt:
            linesegment1 = self.coordinate, pt
            is_pt= line_across(line1=linesegment1, line2=linesegmentA)
            if is_pt is not None:
                pt[0] = is_pt[0]
                pt[1] = is_pt[1]
                continue
            is_pt= line_across(line1=linesegment1, line2=linesegmentB)
            if is_pt is not None:
                pt[0] = is_pt[0]
                pt[1] = is_pt[1]
                continue
            is_pt= line_across(line1=linesegment1, line2=linesegmentC)
            if is_pt is not None:
                pt[0] = is_pt[0]
                pt[1] = is_pt[1]
                continue
            is_pt= line_across(line1=linesegment1, line2=linesegmentD)
            if is_pt is not None:
                pt[0] = is_pt[0]
                pt[1] = is_pt[1]
                continue


    ''' check for obstacle detection'''
    def lidar_scan(self, obstacles:Obstacles):
        self.lidar_start()
        # check for obstacles 
        self.find_visibility_lines(obstacles=obstacles, radius=self.vision_range, lines_pt=self.lidar_pluses)

        # check for boundary 
        self.find_visibility_lines_boundary(lines_pt=self.lidar_pluses)

    ''' emit back_vision '''
    def back_vision_start(self):
        back_vision = np.array([self.back_range,0]*BACK_VISION_LINES).reshape((BACK_VISION_LINES,2))
        step = (self.back_range_MAX_left + self.back_range_MAX_right)/BACK_VISION_LINES
        start = self.yaw - self.back_range_MAX_right + math.pi
        end = self.yaw + self.back_range_MAX_left + math.pi
        yaws = np.arange(start, end, step, dtype=float)

        self.back_lines = [matrix_rotation(back_line, yaw=yaw) for back_line, yaw in zip(back_vision, yaws)]
        self.back_lines = vector_translation(vector=self.back_lines, translate_vec=self.coordinate)
    
    def back_vision_scan(self, obstacles:Obstacles):
        self.back_vision_start()
        # check for obstacles 
        self.find_visibility_lines(obstacles=obstacles, radius=self.back_range, lines_pt=self.back_lines)
        
        # check for boundary 
        self.find_visibility_lines_boundary(lines_pt=self.back_lines)
    ''' 
        return section where car's direction belong to 
        0: in range [-pi/4,  pi/4]
        1: in range [ pi/4, 3pi/4]
        2: in range [3pi/4, 5pi/4]
        3: in range [5pi/4, 7pi/4]
    '''

    def get_yaw_state(self):
        if self.yaw < math.pi/4 or self.yaw >= 7*math.pi/4: return 0
        elif self.yaw >= math.pi/4 and self.yaw < 3*math.pi/4: return 1
        elif self.yaw >= 3*math.pi/4 and self.yaw <5*math.pi/4: return 2
        elif self.yaw >= 5*math.pi/4 and self.yaw < 7*math.pi/4: return 3
        else: 
            print ("yaw value invalids", self.yaw)
            return -1

    def get_boundary_state(self):
        # get position state
        position_x_state = 0
        if self.coordinate[0] < WORKING_SPACE_X_MIN + 0.11:
            position_x_state = 1
        elif self.coordinate[0] > WORKING_SPACE_X_MAX - 0.11:
            position_x_state = 2
        
        position_y_state = 0
        if self.coordinate[1] < WORKING_SPACE_Y_MIN + 0.11:
            position_y_state = 1
        elif self.coordinate[1] > WORKING_SPACE_Y_MAX - 0.11:
            position_y_state = 2
        return position_x_state, position_y_state

    def get_lidar_state(self):
        # get lidar state
        danger_distance = self.vision_range/DANGER_LEVEL
        lidar_distances = [point_dist(self.coordinate, pt) for pt in self.lidar_pluses]
        lidar_danger_data = [np.min(lidar_distances[i*LIDAR_PLUSES_PER_SECTOR:(i+1)*LIDAR_PLUSES_PER_SECTOR]) for i in range (LIDAR_SECTORS)]
        lidar_danger_data = np.array(lidar_danger_data)
        for i in range (DANGER_LEVEL+1):
            mask = np.logical_and(lidar_danger_data > i*danger_distance -EPSILON, lidar_danger_data < (i+1)*danger_distance -EPSILON)
            lidar_danger_data[mask] = i
        return lidar_danger_data
    
    def get_back_vision_state(self):
        back_distances = [point_dist(self.coordinate, pt) for pt in self.back_lines]
        back_state = np.min(back_distances) > (BACK_VISION_SAFE_DISTANCE - EPSILON)
        return back_state

    def get_state(self, obstacles:Obstacles):
        # lidar (front vision) and back vision
        self.lidar_scan(obstacles=obstacles)
        self.back_vision_scan(obstacles=obstacles)

        # state = [lidar, x position, y position, yaw, back_vision]
        lidar_state = self.get_lidar_state()
        x_state, y_state = self.get_boundary_state()
        yaw_state = self.get_yaw_state()
        back_state = self.get_back_vision_state()
        self.state = np.hstack((lidar_state, x_state, y_state, yaw_state, back_state))
        
        #print (self.state)
        return tuple(self.state.astype(int))

    ''' return obstacles which are inside lidar area'''
    def in_range_obstacles(self, obstacles:Obstacles, in_range_radius:float):
        obstacles_len = len(obstacles.obstacles_radius)
        dist_to_obstacles = [point_dist(self.coordinate, pt) for pt in obstacles.obstacles]
        in_range_obstacles_mask = dist_to_obstacles < in_range_radius + obstacles.obstacles_radius.T
        in_range_obstacles_mask = in_range_obstacles_mask.reshape((obstacles_len, 1))

        inrange_obstacles_radius = obstacles.obstacles_radius[in_range_obstacles_mask]
        in_range_obstacles_mask = np.hstack( (in_range_obstacles_mask, in_range_obstacles_mask) )

        inrange_obstacles = obstacles.obstacles[in_range_obstacles_mask]
        in_range_obstacles_len = int(len(inrange_obstacles)/2)
        inrange_obstacles = inrange_obstacles.reshape((in_range_obstacles_len,2))

        return inrange_obstacles, inrange_obstacles_radius   

    ''' return obstacles which are inside robot radius from robot center,
        in other words, robot hit obstacles
    '''
    def collision_detection(self, obstacles:Obstacles):
        conllision_obstacles, conllision_obstacles_radius = self.in_range_obstacles(obstacles=obstacles, in_range_radius=self.radius)
        return conllision_obstacles, conllision_obstacles_radius

    ''' motion equation '''
    def motion(self):
        self.yaw += self.steer * self.dt
        if self.yaw > 2*math.pi:
            self.yaw -= math.pi*2
        elif self.yaw < 0:
            self.yaw += math.pi*2

        self.coordinate += self.vel * np.array([math.cos(self.yaw),math.sin(self.yaw)]) * self.dt
        self.trajectory = np.concatenate((self.trajectory, np.array([self.coordinate])), axis=0)
        #print ("yaw {0}, cooordinate {1}".format(math.degrees(self.yaw), self.trajectory))

    def take_action(self, action):
        steer_increase = 0
        vel_increase = 0
        if action == 0:
            steer_increase, vel_increase =  0, 1
        elif action == 1:
            steer_increase, vel_increase =  1, 1
        elif action == 2:
            steer_increase, vel_increase = -1, 1
        elif action == 3:
            steer_increase, vel_increase =  0,-1
        elif action == 4:
            steer_increase, vel_increase =  1,-1
        elif action == 5:
            steer_increase, vel_increase = -1,-1
        #elif action == 6:
        #    steer_increase, vel_increase =  0, 0
        elif action == 6:
            steer_increase, vel_increase =  1, 0
        elif action == 7:
            steer_increase, vel_increase = -1, 0
        else:
            print ("No predefined action")

        
        self.steer = self.steer_resolution*steer_increase
        self.vel = self.vel_resolution*vel_increase

        if self.steer > self.steer_MAX:
            self.steer = self.steer_MAX
        elif self.steer < self.steer_MIN:
            self.steer = self.steer_MIN

        if self.vel > self.vel_MAX:
            self.vel = self.vel_MAX
        elif self.vel < self.vel_MIN:
            self.vel = self.vel_MIN

        self.motion()
        

    def is_out_of_boundary(self):
        self.car_status = Robot_status.out_of_boundary
        return self.coordinate[0] < WORKING_SPACE_X_MIN or self.coordinate[0] > WORKING_SPACE_X_MAX or \
        self.coordinate[1] < WORKING_SPACE_Y_MIN or self.coordinate[1] > WORKING_SPACE_Y_MAX
    
    def is_hit_obstacles(self, obstacles:Obstacles):
        obs, _ = self.collision_detection(obstacles=obstacles)
        if len(obs)> 0:
            self.car_status = Robot_status.hit_obstacles
            return True
        return False

if __name__ == '__main__':
    obstacles = Obstacles()
    car = Car()
    car.get_state(obstacles=obstacles)
    car.find_visibility_lines_boundary(lines_pt=car.back_lines)