import numpy as np
from car import *
from obstacles import *
from car_plotter import *
from car_configuration import *
from math_lib import *

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
SHOW_EVERY = 100000

# declare the goal
goal = np.array([6,9])

# q table for reinforcement learning
q_table = np.random.uniform(size=(NUMBER_OF_STATES + [NUMBER_OF_ACTIONS]) )

# car, obstacles and plotter delaration
car = Car()
obstacles = Obstacles()
plotter = Plotter(title= "Reinforcement Learning")

# calculate reward base on goal and car coordination
def calculate_reward(car:Car, obstacles:Obstacles, cost_to_goal):
    done = False
    obs, _ = car.collision_detection(obstacles=obstacles)
    reward = cost_to_goal - point_dist(car.coordinate, goal)
    if point_dist(goal,car.coordinate)<2:
        reward = reward*1000
    else:
        reward = reward*10
    if len(obs) > 0:
        #print ("check for hitting obstacles")
        reward = -10000000.0
        done = True

    # out of boundary
    if car.coordinate[0] < WORKING_SPACE_X_MIN or car.coordinate[0] > WORKING_SPACE_X_MAX or \
        car.coordinate[1] < WORKING_SPACE_Y_MIN or car.coordinate[1] > WORKING_SPACE_Y_MAX:
        #print ("out of boundary")
        done = True
        reward = -10000000.0

    return reward, done

reach_goal = False

# start to train
for episode in range(EPISODES):
    print ("Episode: ", episode)

    render = episode % SHOW_EVERY == 0

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
        reward, done = calculate_reward(car=car, obstacles=obstacles,cost_to_goal=cost_to_goal)
        #print ("action {0}, reward {1}, done {2}, state{3}".format(action, reward, done, state)) 
        #print ("q_values at state {0}".format(q_table[state]))
        #print ("current car coordinate ", car.coordinate)

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

# display for the last
plotter.display_all(car=car, obstacles=obstacles, goal=goal)
plotter.show()
print ("DONE")