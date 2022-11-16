"""
This is a part of the autonomous car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

import numpy as np
from car import *
from obstacles import *
from car_plotter import *
from car_configuration import *
from math_lib import *

import argparse             # for user input
from result_class import *  # for record result
import pickle               # for save/load object
from datetime import datetime
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
EPISODES = 2500
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

def RL_main(mode:None, car:Car, obstacles:Obstacles, plotter:Plotter, goal:None, q_table_fname:None,\
            run_times:int):
    reach_goal = False
    done = False

    # q table
    q_table = np.random.uniform(size=(NUMBER_OF_STATES + [NUMBER_OF_ACTIONS]) )
    #q_table = np.zeros((NUMBER_OF_STATES + [NUMBER_OF_ACTIONS]) )
    
    if mode == Robot_mode.learning: # learning mode
        # start to train
        for episode in range(EPISODES):
            print ("Episode: ", episode)
            
            q_table_fname = "file_q_table_{0}".format(datetime.now().strftime("%m_%d_%H_%M_%S"))
            render = episode % SHOW_EVERY == 0
            
            # reset all avariable
            car.reset()
            state = car.get_state(obstacles=obstacles)
            done = False

            while not done:
                # get current action
                action = np.argmax(q_table[state])

                # calculate distance for car to goal for later rewarding 
                cost_to_goal = point_dist(car.coordinate, goal)

                if dynamic_obstacles:
                    obstacles.motion()

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

                if  car.is_reach_goal(goal=goal): # reach goal
                    q_table[state + (action,)] = 10000000.0
                    done = True
                
                if car.time_out() :
                    done = True
                
                state = new_state

                if render and show_animation:
                    plotter.display_all(car=car, obstacles=obstacles, goal=goal)
            
            if car.reach_goal:
                if not show_animation and save_figure:
                    plotter.display_all(car=car, obstacles=obstacles, goal=goal)
                    plotter.save_figure(mode=mode, learning_rate=LEARNING_RATE, \
                    episode=episode, number_of_obstacles=len(obstacles.obstacles))

                # generate new obstacles to train for new knowledge
                obstacles.new_data()
                break


        np.save(q_table_fname, q_table)

    elif mode == Robot_mode.running: # running mode

        # load pretrain q table
        dtime = "11_16_13_44_33"
        q_table = np.load("file_q_table_{0}.npy".format(dtime))

        # log result for comparison
        result_log = Result_Log()
        reached_goal_stat = 0

        for i in range (run_times):
            result_item = []
            car.reset()
            obstacles.new_data()

            if dynamic_obstacles:
                obstacles.motion()

            state = car.get_state(obstacles=obstacles)
            
            done = False

            while not done:
                # get current action
                action = np.argmax(q_table[state])
                
                # car take action to move to new coordinate
                car.take_action(action)

                if dynamic_obstacles:
                    obstacles.motion()
                
                # get new state at new coordinate
                state = car.get_state(obstacles=obstacles)
            
                # stop if car hit obstacles, is out of boundary or reach goal or time out (out of max_step)
                if car.is_hit_obstacles(obstacles=obstacles) or car.is_out_of_boundary() or \
                    car.is_reach_goal(goal=goal) or car.time_out():
                    done = True

                if show_animation:
                    plotter.display_all(car=car, obstacles=obstacles, goal=goal)

            if (car.reach_goal):
                reached_goal_stat += 1

            if not show_animation and save_figure:
                plotter.display_all(car=car, obstacles=obstacles, goal=goal)
                plotter.save_figure(mode=mode, learning_rate=LEARNING_RATE, \
                                    runtimes=i, reach_goal=reach_goal, robot_status=car.status)

            result_item.append(reach_goal)
            result_item.append(car.status)
            result_log.add_result(result=result_item)
        result_log.write_csv("result_log.csv")
        print ("reached_goal {0} out of {1}".format(reached_goal_stat, run_times))
    
    # display for the last
    plotter.display_all(car=car, obstacles=obstacles, goal=goal)
    if show_animation:
        plotter.show()
    print ("DONE") 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reinformance Learning for Dynamic Obstacles Avoidance.')
    parser.add_argument('-m', metavar="running mode", type=str, help='running mode', default='r')
    parser.add_argument('-q', metavar="q talbe", type=str, help='running mode', default='q_table_car_OKIE.npy')
    parser.add_argument('-n', metavar="run times for testing", type=int, help='run times for testing', default=100)
    args = parser.parse_args()
    mode = args.m
    run_times = args.n
    q_table_name = args.q

    if mode == 'l':
        mode = Robot_mode.learning
    elif mode == 'r':
        mode = Robot_mode.running
    else:
        mode = Robot_mode.running

    # car, obstacles and plotter delaration
    car = Car()
    obstacles = Obstacles()
    plotter = Plotter(title= "Reinforcement Learning")
    # declare the goal
    goal = np.array([6,9])

    RL_main(mode=mode, car=car, obstacles=obstacles, plotter=plotter, goal=goal, q_table_fname=q_table_name, run_times=run_times)