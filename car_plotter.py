"""
This is a part of the autonomous car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import numpy as np
import math
import sys

from math_lib import *
from car import *
from obstacles import *
from car_configuration import *
from program_configuration import * 

car = Car()

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    steer_increase = 0
    vel_increase = 0
    if event.key == 'a':
        steer_increase = -1
    elif event.key == 'd':
        steer_increase = 1
    elif event.key == 'w':
        vel_increase =  1
    elif event.key == 'x':
        vel_increase =  -1
    elif event.key == 'escape':
        exit(0)
    car.steer += car.steer_resolution*steer_increase
    car.vel += car.vel_resolution*vel_increase

    if car.steer > car.steer_MAX:
        car.steer = car.steer_MAX
    elif car.steer < car.steer_MIN:
        car.steer = car.steer_MIN

    if car.vel > car.vel_MAX:
        car.vel = car.vel_MAX
    elif car.vel < car.vel_MAX:
        car.vel = car.vel_MAX
    
    car.motion()

class Plotter:
    def __init__(self, size=(7,7), title="Autonomous Robot"):
        self.plt   = plt
        self.fig, self.ax = plt.subplots(figsize=size)
        self.fig.canvas.manager.set_window_title(title)
        

    show = lambda self: self.plt.show()
    pause = lambda self, x: self.plt.pause(x)
    clear = lambda self: self.plt.cla()
    set_equal = lambda self: self.plt.axis("equal")
    show_grid = lambda self: self.plt.grid(True)
    set_axis = lambda self, x0, y0, x1, y1: self.plt.axis([x0, x1, y0, y1])

    press_key = lambda self: self.fig.canvas.mpl_connect('key_press_event', on_press) 

    ''' lidar front vision arc '''
    def lidar_arc(self, robot:Car, cl="g", ls_ts="-"):
        arc_patches = []

        # center_ox: a point starts from center and follows X-axis direction 
        center_ox = np.add(robot.coordinate, [1,0] )
        theta1radian = unsigned_angle(robot.coordinate, center_ox, robot.lidar_pluses[0])
        theta2radian = unsigned_angle(robot.coordinate, center_ox, robot.lidar_pluses[-1])
        theta1 = math.degrees(theta1radian)
        theta2 = math.degrees(theta2radian)
        for i in range (1,DANGER_LEVEL+1):     
            wedge = patches.Wedge(robot.coordinate, robot.vision_range*i/DANGER_LEVEL, theta1=theta1, theta2=theta2, width=0.01)
            arc_patches.append(wedge)
        collection = PatchCollection(arc_patches, facecolor=cl, linestyle=ls_ts, edgecolor='g', alpha=0.1)
        self.ax.add_collection(collection)

    ''' back vision arc '''
    def back_vision_arc(self, robot:Car, cl="g", ls_ts="-"):
        arc_patches = []

        # center_ox: a point starts from center and follows X-axis direction 
        center_ox = np.add(robot.coordinate, [1,0] )
        theta1radian = unsigned_angle(robot.coordinate, center_ox, robot.back_lines[0])
        theta2radian = unsigned_angle(robot.coordinate, center_ox, robot.back_lines[-1])
        theta1 = math.degrees(theta1radian)
        theta2 = math.degrees(theta2radian)
        wedge = patches.Wedge(robot.coordinate, robot.back_range, theta1=theta1, theta2=theta2)
        arc_patches.append(wedge)
        collection = PatchCollection(arc_patches, facecolor=cl, linestyle=ls_ts, edgecolor='y', alpha=0.1)
        self.ax.add_collection(collection)

    ''' draw a circle with center, radius '''
    def draw_a_circle(self, center, radius, ls="-", color="blue"):
        """ draw a circle that limits the vision of robot """
        circle = self.plt.Circle(center, radius, color=color, linestyle=ls, fill=True, alpha=0.5)
        self.plt.gcf().gca().add_artist(circle)

    ''' plot obstacles as circles'''
    def obstacles_circle(self, obstacles:Obstacles, ls="-", color="blue"):
        for center, radius, vel, yaw in zip(obstacles.obstacles, obstacles.obstacles_radius,\
                                    obstacles.obstacles_velocity, obstacles.obstacles_yaw):
            self.draw_a_circle(center=center, radius=radius, ls=ls, color=color)
            if show_obstacles_direction:
                direction = [math.cos(yaw), math.sin(yaw)]*vel
                self.plt.arrow(center[0], center[1], direction[0], direction[1], head_width = 0.03, width = 0.01, ec ='green', alpha=0.6)
    
    ''' plot collision obstacles '''
    def obstacles_circle_collision(self, obstacles_coordinate, obstacles_radius, ls="-", color="red"):
        for center, radius, in zip(obstacles_coordinate, obstacles_radius):
            self.draw_a_circle(center=center, radius=radius, ls=ls, color=color)

    ''' plot a robot '''
    def robot(self, robot:Car):  # pragma: no cover
        if robot.robot_type == RobotType.rectangle:
            l_2 = robot.length / 2
            w_2 = robot.width / 2
            outline = np.array([[-l_2, l_2, l_2, -l_2, -l_2], [w_2, w_2, -w_2, -w_2, w_2]])
            outline = matrix_rotation(matrix=outline, yaw= robot.yaw)
            outline = matrix_translation(matrix=outline, vector=np.array([robot.coordinate]))
            self.plt.plot(outline[0, :], outline[1, :], "-k")

        elif robot.robot_type == RobotType.circle:
            self.draw_a_circle(center=robot.coordinate, radius=robot.radius, ls='-', color='k')
    
    ''' plot trajectory '''
    def plot_trajectory(self, robot:Car):
        self.plt.plot(robot.trajectory[:, 0] , robot.trajectory[:, 1], ":r")

    ''' plot_bunch_of_line_segment, 1 point of linesegment is root center, others are ptsq'''
    def plot_bunch_line_segment(self, robot:Car, pts, color='g', lw = 0.15):
        pts_len = len(pts)
        #print ("pts_len", pts_len)
        if pts_len > 0:
            center = np.array([robot.coordinate]*pts_len).reshape((pts_len,2))
            self.plt.plot((center[:,0],pts[:,0]), (center[:,1],pts[:,1]), lw = lw, color=color)

    ''' plot lidar pluses vision'''
    def plot_Lidar(self, robot:Car):
        distances = [point_dist(robot.coordinate, pt) for pt in robot.lidar_pluses]
        distance_mask = np.array(distances)< robot.vision_range-EPSILON

        # obstacles line segments
        pts = robot.lidar_pluses[distance_mask]
        self.plot_bunch_line_segment(robot=robot, pts=pts, color='red')
        # free line sengments
        pts = robot.lidar_pluses[np.logical_not(distance_mask)]
        self.plot_bunch_line_segment(robot=robot, pts=pts, color='green')
        
        if show_lidar_vision:   # show lidar vision
            self.lidar_arc(robot=robot)

    ''' plot back vision '''
    def plot_back_vision(self, robot:Car):
        distances = [point_dist(robot.coordinate, pt) for pt in robot.back_lines]
        distance_mask = np.array(distances)< robot.back_range-EPSILON

        # obstacles line segments
        pts = robot.back_lines[distance_mask]
        self.plot_bunch_line_segment(robot=robot, pts=pts, color='red')
        # free line sengments
        pts = robot.back_lines[np.logical_not(distance_mask)]
        self.plot_bunch_line_segment(robot=robot, pts=pts, color='green')
        
        #if show_back_vision:   # show back vision
        #    self.back_vision_arc(robot=robot)
            
    def display_all(self, car:Car, obstacles:Obstacles, goal):
        self.clear()
        self.set_equal()
        self.set_axis(x0=WORKING_SPACE_X_MIN, y0=WORKING_SPACE_Y_MIN, 
                      x1=WORKING_SPACE_X_MAX, y1=WORKING_SPACE_Y_MAX)


        detected_collision, detected_collision_radius = car.collision_detection(obstacles=obstacles)

        self.plt.plot(goal[0], goal[1], "*r")
        self.robot(robot=car)
        self.plot_Lidar(robot=car)
        self.plot_back_vision(robot=car)
        self.plot_trajectory(robot=car)
        self.obstacles_circle(obstacles=obstacles)
        self.obstacles_circle_collision(obstacles_coordinate=detected_collision, \
                obstacles_radius=detected_collision_radius, ls="-", color='red')
        self.pause(0.1)

    ''' save figure '''
    def save_figure(self, mode:Robot_mode, learning_rate=None, episode=None, number_of_obstacles=None, \
                runtimes=0, reach_goal=False, robot_status=Robot_status.none, dpi=300):
        if mode == Robot_mode.learning:
            fig_name = "Robot_Learning_Episode{0}_LearningRate{1}_Obstacles{2}".format(episode, learning_rate, number_of_obstacles)
        elif mode == Robot_mode.running:
            fig_name = "Robot_Running_times{0}_Reach_goal_{1}_{2}".format(runtimes, reach_goal, robot_status)

        
        #file_extension_pgf = ".pgf"
        #plt.savefig(fig_name + file_extension_pgf, bbox_inches ="tight", dpi=dpi)
        #print ("saved: {0}{1}".format(fig_name, file_extension_pgf))

        file_extension_png = ".png"
        plt.savefig(fig_name + file_extension_png, bbox_inches ="tight", dpi=dpi)
        print ("saved: {0}{1}".format(fig_name, file_extension_png))
        
if __name__ == '__main__':
    
    car = Car()
    obstacles = Obstacles()
    plotter = Plotter(title="reinforcement learning sterring tutorial")
    
    
    run_times = 0
    
    plotter.press_key()
    while run_times >-1:
        
        if dynamic_obstacles:
            obstacles.motion()
        car.get_state(obstacles=obstacles)
        
        plotter.display_all(car=car, obstacles=obstacles, goal=(9,9))
      
        run_times += 1
    print ("DONE")
    plotter.show()
    