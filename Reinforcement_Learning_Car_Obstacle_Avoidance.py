"""
This is a part of the autonomous driving car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

import numpy as np
from car import *
from obstacles import *
from car_plotter import *
from car_configuration import *
from math_lib import *

import argparse

'''
state = lidar , boundary x, boundary y , yaw, back vision
action = combination of steer and velocity
    steer values: 0 unchange, 1 more left, -1 more right
    veloctity values: 0 unchange, 1 increase speed, -1 decrease speed
    exception case: vel = 0. steer = 0 
'''

NUMBER_OF_ACTIONS = 8
#NUMBER_OF_STATES = [DANGER_LEVEL+1]*LIDAR_SECTORS + [BOUNDARY_X_STATE, BOUNDARY_Y_STATE, YAW_STATE] 
NUMBER_OF_STATES = [DANGER_LEVEL+1]*LIDAR_SECTORS + [BOUNDARY_X_STATE, BOUNDARY_Y_STATE, YAW_STATE, BACK_VISION_STATE] 

LEARNING_RATE = 0.2
DISCOUNT = 0.90
EPISODES = 25000
SHOW_EVERY = 1000


# calculate reward base on goal and car coordination
def calculate_reward(car:Car, obstacles:Obstacles, cost_to_goal, goal):
    done = False
    reward = cost_to_goal - point_dist(car.coordinate, goal)
    if point_dist(goal,car.coordinate)<2:
        reward = reward*1000
    else:
        reward = reward*10

    if car.is_hit_obstacles(obstacles=obstacles) or car.is_out_of_boundary():
        reward = -10000000.0
        done = True

    return reward, done

def RL_main(mode=None, car=Car, obstacles=Obstacles, plotter=Plotter, goal=None, q_table_name=None):
    reach_goal = False
    done = False
    max_step = 1000

    # q table for reinforcement learning
    q_table = np.random.uniform(size=(NUMBER_OF_STATES + [NUMBER_OF_ACTIONS]) )

    if mode == 'l': # learning mode

        # start to train
        for episode in range(EPISODES):
            print ("Episode: ", episode)

            render = episode % SHOW_EVERY == 0
            
            # reset all avariable
            car.reset()
            state = car.get_state(obstacles=obstacles)
            done = False
            max_step = 1000

            while not done:
                max_step -= 1

                # get current action
                action = np.argmax(q_table[state])
                # calculate distance for car to goal for later rewarding 
                cost_to_goal = point_dist(car.coordinate, goal)

                # car take action to move to new coordinate
                car.take_action(action)

                # get new state at new coordinate
                new_state = car.get_state(obstacles=obstacles)

                # give award score for the move
                reward, done = calculate_reward(car=car, obstacles=obstacles,cost_to_goal=cost_to_goal, goal=goal)

                # update q score in q_table
                max_future_q = np.max(q_table[new_state])
                current_q = q_table[state + (action,)]

                new_q = current_q  + LEARNING_RATE*(reward + DISCOUNT*max_future_q - current_q)
                q_table [state + (action,)] = new_q
                #print ("new q_values at state {0}".format(q_table[state]))
                if point_dist(car.coordinate, goal) < car.radius: # reach goal
                    q_table[state + (action,)] = 10000000.0
                    done = True
                    reach_goal = True
                
                state = new_state
                if max_step == 0 :
                    done = True
                

                if render:
                    plotter.display_all(car=car, obstacles=obstacles, goal=goal)

            if reach_goal:
                    break

        np.save(Q_TALBE_FILE, q_table)

    elif mode == 'r': # running mode
        q_table = np.load(q_table_name)
        car.reset()
        state = car.get_state(obstacles=obstacles)
        max_step = 500
        while not done:
            max_step -= 1
            # get current action
            action = np.argmax(q_table[state])
            
            # car take action to move to new coordinate
            car.take_action(action)
                
            

            # get new state at new coordinate
            state = car.get_state(obstacles=obstacles)
        
            if car.is_hit_obstacles(obstacles=obstacles) or car.is_out_of_boundary():
                done = True

            if point_dist(car.coordinate, goal) < car.radius: # reach goal
                done = True
                reach_goal = True
            
            if max_step == 0 :
                done = True
            plotter.display_all(car=car, obstacles=obstacles, goal=goal)
    
    # display for the last
    plotter.display_all(car=car, obstacles=obstacles, goal=goal)
    plotter.show()
    print ("DONE") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinformance Learning for Dynamic Obstacles Avoidance.')
    parser.add_argument('-m', metavar="running mode", type=str, help='running mode', default='r')
    parser.add_argument('-q', metavar="q talbe", type=str, help='running mode', default='q_table_car_OKIE.npy')
    args = parser.parse_args()
    mode = args.m
    q_table_name = args.q

    # car, obstacles and plotter delaration
    car = Car()
    obstacles = Obstacles()
    plotter = Plotter(title= "Reinforcement Learning")
    # declare the goal
    goal = np.array([6,9])

    RL_main(mode=mode, car=car, obstacles=obstacles, plotter=plotter, goal=goal, q_table_name=q_table_name)