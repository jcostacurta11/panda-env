from env_julia import SimpleEnv
import numpy as np
import time
import pybullet as p


env = SimpleEnv()

currentPosition = p.getLinkState(env.panda.panda, 11)
currentPose = currentPosition[0]
goalPose = [0.8, 0, 0.005]
delta_t = 1

state = env.reset()
start_time = time.time()
curr_time = time.time() - start_time
while curr_time < 4*np.pi:
    curr_time = time.time() - start_time
    action = [0.01*np.cos(curr_time), 0.01*np.sin(curr_time), 0]
    next_state, reward, done, info = env.step(action)
    # img = env.render()
    if done:
        break
env.close()
