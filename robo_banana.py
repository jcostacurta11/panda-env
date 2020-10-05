from env_julia import SimpleEnv
import numpy as np
import time
import pybullet as p


env = SimpleEnv()
state = env.reset()
start_time = time.time()
curr_time = time.time() - start_time
T1 = 3

currentPose = p.getLinkState(env.panda.panda, 11)
currentPosition = currentPose[0]
action = [0.8, 0, 0.005]
move = np.array(action) - np.array(currentPosition)
print(move)
v = np.abs(move/T1)
curr_state = 0
#
while True:
    if curr_state == 0:
        curr_time = time.time() - start_time
        currentPose = p.getLinkState(env.panda.panda, 11)
        currentPosition = currentPose[0]
        next_state, reward, done, info = env.position(action,True,v,move)
        #print(currentPosition[2])
        if currentPosition[2] <= 1.01*action[2]:
            curr_state+=1
            start1_time = time.time() - start_time
        print(curr_state)

    elif curr_state == 1:

        # curr_time = time.time() - start_time
        action = [0, 0, 0]
        v = [0,0,0]
        # next_state, reward, done, info = env.step(action)
        next_state, reward, done, info = env.position(action, False, v, move)
        curr_time = time.time() - start_time
        grip = False
        #next_state, reward, done, info = env.grip(grip)
        if curr_time-start1_time > 1:
            curr_state += 1
            currentPose = p.getLinkState(env.panda.panda, 11)
            currentPosition = currentPose[0]
            action = [0.5, -0.5, 0.4]
            move = np.array(action) - np.array(currentPosition)
            v = np.abs(move / T1)
        # action = [0.75, 0, 1]
        # currentPose = p.getLinkState(env.panda.panda, 11)
        # currentPosition = currentPose[0]
        # action = [0.75, 0, 0.02]
        # move = np.array(action) - np.array(currentPosition)
        # v = np.abs(move / T1)
        print(curr_state)

    elif curr_state == 2:
        next_state, reward, done, info = env.position(action, False, v, move)
        curr_time = time.time() - start_time
        currentPose = p.getLinkState(env.panda.panda, 11)
        currentPosition = currentPose[0]
        if currentPosition[1] <= 0.99*action[1]:
            curr_state+=1

    elif curr_state == 3:
        action = [0, 0, 0]
        v = [0, 0, 0]
        next_state, reward, done, info = env.position(action, True, v, move)
# while curr_time < 4:
#     curr_time = time.time() - start_time
#     action = [0, -0.1*curr_time, -0.02*curr_time]
#     next_state, reward, done, info = env.step(action)
# while curr_time >= 4:
#     curr_time = time.time() - start_time
#     action = [0, 0, 0]
#     next_state, reward, done, info = env.step(action)
