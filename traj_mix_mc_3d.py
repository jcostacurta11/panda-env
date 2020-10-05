import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import deepcopy
from env_julia import SimpleEnv
import numpy as np
import time
import pybullet as p

num_agents = 1
theta = np.array([0, 6, 6, 8]) #TODO: ask mengxi about these variables

num_correction = 4
T = 5
gamma = 0.  # human effort
beta = 1.  # procedure trajctory cost
num_samples = 1000
seq_decay = 0.9
seq_smooth = False

#distance function for two points
def dist(u, v):
    return np.sqrt(np.sum((u-v) * (u-v)))


class TrajEnv(object):
    """docstring for task_env"""

    def __init__(self, init_obs, delta_t, num_agents=1):
        # init_obs for each agent is 4d array [[x0, y0, vx0, vy0],[x1, y1, vx1, vy1]...]
        super(TrajEnv, self).__init__() #inherit all properties from input object
        self.init_obs = init_obs
        self.delta_t = delta_t
        self.num_agents = num_agents

    def _v(self, actions):
        # actions is n*2*T dimensional array [ax^0_0, ax^0_1, ..., ax^0_T, ay^0_0, ay^0_1, ..., a^0y_T, ax^1_0, ax^1_1, ..., ]
        # returned velocity is n*2*(T+1) dimensional array [vx^0_0,vx^0_1, ..., vx^0_{T+1}, vy^0_0, vy^0_1, ..., vy^0_{T+1},vx^1_0,v^0_1, ...]
        if len(actions.shape) > 1:
            actions = self.flat_action(actions) #working with flat action vector
        act_len = len(actions) // 3 #account for number of agents and x,y actions
        vx = []
        vy = []
        vz = []
        # for i in range(self.num_agents): #complete for each agent
        ax, ay, az = actions[0:act_len], actions[act_len:2*act_len], actions[2*act_len:3*act_len] #picking out which actions apply to the agent (won't need to do this exactly)
        tmp_vx = np.hstack([self.init_obs[3], np.cumsum(
            ax) * self.delta_t + self.init_obs[3]]) #TODO: I think this is getting the resulting velocities from the actions? ask mengxi what the actions are
        tmp_vy = np.hstack([self.init_obs[4], np.cumsum(
            ay) * self.delta_t + self.init_obs[4]])
        tmp_vz = np.hstack([self.init_obs[5], np.cumsum(
            ay) * self.delta_t + self.init_obs[5]])
        vx = ax
        vy = ay
        vz = az
        return np.stack([vx, vy, vz]) #TODO reshape: look into this for 3d

    def _pos(self, actions):
        # actions is n*2*T dimensional array [ax^0_0, ax^0_1, ..., ax^0_T, ay^0_0, ay^0_1, ..., a^0y_T, ax^1_0, ax^1_1, ..., ]
        # returned position is n*2*(T+1) dimensional array [x^0_0, x^0_1, ..., x^0_{T+1}, y^0_0, y^0_1, ..., y^0_{T+1}, x^1_0, x^1_1, ...,]
        if len(actions.shape) > 1:
            actions = self.flat_action(actions)
        act_len = len(actions) // 3
        x = []
        y = []
        z = []
        v = self._v(actions)

        ax, ay, az = actions[0:act_len], actions[act_len:2*act_len], actions[2*act_len:3*act_len]
        # tmp_x = np.hstack([self.init_obs[0], self.init_obs[0] + np.cumsum(v[0][:act_len]) * self.delta_t + 0.5 * np.cumsum(ax) * self.delta_t ** 2])
        # tmp_y = np.hstack([self.init_obs[1], self.init_obs[1] + np.cumsum(
        #     v[1][:act_len]) * self.delta_t + 0.5 * np.cumsum(ay) * self.delta_t ** 2])
        # tmp_z = np.hstack([self.init_obs[2], self.init_obs[2] + np.cumsum(
        #     v[2][:act_len]) * self.delta_t + 0.5 * np.cumsum(az) * self.delta_t ** 2])

        tmp_x = np.hstack([self.init_obs[0], self.init_obs[0] + np.cumsum(ax) * self.delta_t])
        tmp_y = np.hstack([self.init_obs[1], self.init_obs[1] + np.cumsum(ay) * self.delta_t])
        tmp_z = np.hstack([self.init_obs[2], self.init_obs[2] + np.cumsum(az) * self.delta_t])

        x = tmp_x
        y = tmp_y
        z = tmp_z
        return np.stack([x, y, z]) #TODO: reshape check dimensions

    def flat_action(self, action): #TODO: check dimensions
        assert len(action.shape) == 2
        assert action.shape[0] == 3
        # convert from shape [num_agent, 2, T] to flatten
        return action.reshape([-1])

    def unflat_action(self, action, shape): #TODO: check dimensions
        assert shape[0] == 3
        return action.reshape([shape[0], shape[1]])

    def vis_traj(self, actions, task_obj, fig=None, ax=None): #plotter (would be nice to adapt this but not priority)
        pos = self._pos(actions)
        x = pos[0, :]
        y = pos[1, :]
        z = pos[2, :]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(x, y, z, marker="o")
        #ax.set_xlim(-5,5)
        #ax.set_zlim(-5,5)
        #ax.set_aspect('equal', adjustable='datalim')
        return fig, ax

class Navigation:#the first thing defined in mengxi's main loop, takes in theta
    def __init__(self, currentPose, goalPose, theta):
        self.num_agents = num_agents
        #self.num_obstacles = 1

        # limit
        self.u_max = 5.0
        self.v_max = 5.0

        # setting initial position and velocities for up to 3 agents
        self.mass = np.array(1) #change to 1d
        self.x_init = np.array(currentPose[0])
        self.y_init = np.array(currentPose[1])
        self.z_init = np.array(currentPose[2])
        self.vx_init = np.array(0)
        self.vy_init = np.array(0)
        self.vz_init = np.array(0)
        self.colors = ['b', 'g', 'y']

        #TODO: what are these?? ask mengxi
        self.x_formation = self.x_init
        self.y_formation = self.y_init
        self.z_formation = self.z_init

        #setting goal end states
        self.x_end = np.array(goalPose[0])
        self.y_end = np.array(goalPose[1])
        self.z_end = np.array(goalPose[2])
        self.vx_end = np.array(0)
        self.vy_end = np.array(0)
        self.vz_end = np.array(0)

        self.theta = theta

class OptimizeMultiTraj(object):
    # T: time horizon of action for a single agent
    def __init__(self, task_env, task_obj, T, gamma=1., beta=1., seed=0, seq_decay=0, seq_smooth=False):
        self.actions = np.zeros(3*T) #looks like we'll be creating the actions in this class
        self.task_env = task_env
        self.num_agents = task_env.num_agents
        self.task_obj = task_obj
        self.T = T
        self.gamma = gamma
        self.beta = beta
        self.seed(seed)
        self.epsilon = 1e-6
        self.seq_decay = seq_decay
        self.seq_smooth = seq_smooth

        self.constraints = [{'type': 'ineq', 'fun': self.a_con}, #TODO: look into this syntax, I think it's needed for the optimize command
                            {'type': 'ineq', 'fun': self.v_con}]
        # {'type': 'ineq', 'fun': self.goal_con_ineq}]

        self.options = {'maxiter': 150000, 'disp': True}

    def seed(self, n):
        self.np_random = np.random.RandomState(n) #start at random set of actions

    def init_actions(self):
        self.actions = np.zeros_like(self.actions) #clear actions to all zero
        theta = deepcopy(self.task_obj.theta) #TODO: what is deepcopy? look into this
        self.task_obj.theta[:-2] = 0
        self.task_obj.theta[-2] = 1
        self.task_obj.theta[-1] = 2 #resetting some theta values
        #TODO: what is theta?
        res = minimize(self.objective, self.actions,
                       constraints=self.constraints, options=self.options)
        self.actions = res.x + self.np_random.randn(len(res.x))
        self.task_obj.theta = theta #putting theta values back?

    def get_traj_cost(self, x, y, z, vx, vy, vz, task_obj):
        center_x = x
        center_y = y
        center_z = z
        cost_formation = 0


        #cost based on how long the path is (need this!)
        cost_length = 0
        #for i in range(task_obj.num_agents):
        dx = x[:-1] - x[1:]
        dy = y[:-1] - y[1:]
        dz = z[:-1] - z[1:]
        tmp_l = np.sum(np.sqrt(dx * dx + dy * dy + dz * dz))
        cost_length += tmp_l

        #cost based on how far you are from the goal
        cost_goal = 0
        #for i in range(task_obj.num_agents):
        pos = np.array([x[-1], y[-1], z[-1]])
        goal = np.array([task_obj.x_end, task_obj.y_end, task_obj.z_end])
        cost_goal += dist(pos, goal)
        v = np.array([vx[-1], vy[-1], vz[-1]])
        vgoal = np.array([task_obj.vx_end, task_obj.vy_end, task_obj.vz_end])
        cost_goal += dist(v, vgoal)

        #weight all the costs together with theta!!
        cost = cost_formation * \
            task_obj.theta[0] + cost_length * \
            task_obj.theta[2] + cost_goal * task_obj.theta[3]
        return cost

    def objective(self, actions, task_obj=None):
        # actions is all decision variables, (n*T*2, ) array
        #gives the cost
        if len(actions.shape) > 1:
            actions = self.task_env.flat_action(actions)
        if task_obj is None:
            task_obj = self.task_obj
        pos = self.task_env._pos(actions)
        v = self.task_env._v(actions)
        x = pos[0, :]
        y = pos[1, :]
        z = pos[2, :]
        vx = v[0, :]
        vy = v[1, :]
        vz = v[2, :]
        return self.get_traj_cost(x, y, z, vx, vy, vz, task_obj)

    #these two functions are used somehow in the constraints
    def a_con(self, actions, epsilon=0): #bounds the size of actions
        return self.task_obj.u_max + epsilon - np.abs(actions)

    def v_con(self, actions, epsilon=0): #bounds size of velocities
        v = self.task_env._v(actions)
        return self.task_obj.v_max + epsilon - np.abs(v.reshape([-1]))

    def optimize(self):
        self.init_actions()
        res = minimize(self.objective, self.actions,
                       constraints=self.constraints, options=self.options)
        self.actions = res.x
        self.actions = self.task_env.unflat_action(
            self.actions, [3, self.T])
        return self.actions, self.objective(res.x)

if __name__ == "__main__":
    env = SimpleEnv()

    env.reset()
    currentPosition = p.getLinkState(env.panda.panda, 11)
    currentPose = currentPosition[0]
    goalPose = [0.7, 0, 0.1]
    task_obj = Navigation(currentPose=currentPose, goalPose=goalPose, theta=theta)
    init_obs = [task_obj.x_init, task_obj.y_init, task_obj.z_init,
                 task_obj.vx_init, task_obj.vy_init, task_obj.vz_init]
    task_env = TrajEnv(init_obs=init_obs, delta_t=1, num_agents=1)

    optimizer = OptimizeMultiTraj(
        task_env=task_env, task_obj=task_obj, T=T, gamma=gamma, beta=beta, seq_decay=seq_decay, seq_smooth=seq_smooth)
    actions, _ = optimizer.optimize()
    print(actions)
    print(task_env._pos(actions)[:,-1])
    print(task_env._v(actions)[:,-1])
    # fig_0, ax_0 = task_env.vis_traj(actions, task_obj)
    # plt.show()

    #state = env.reset()
    curr_state = 0

    while True:
        if curr_state == 0:
            start_time = time.time()
            curr_time = time.time() - start_time
            while curr_time < 5:
                actionnum = int(np.floor(curr_time))
                #print(actionnum)
                action = actions[:,actionnum]
                #print(action)
                next_state, reward, done, info = env.step(action)
                # img = env.render()
                #time.sleep(1)
                if done:
                    break
                curr_time = time.time() - start_time
            curr_state += 1
        elif curr_state == 1:
            start_time = time.time()
            curr_time = time.time() - start_time
            while curr_time < 2:
                v = [0,0,0]
                next_state, reward, done, info = env.step(action)
                curr_time = time.time() - start_time
            curr_state += 1
        elif curr_state == 2:
            print(p.getLinkState(env.panda.panda, 11)[0])
            env.close()
    # for i in range(actions.shape[0]):
    #     start_time = time.time()
    #     curr_time = time.time() - start_time
    #     while start_time-curr_time<1: #do this action for dt
    #         curr_time = time.time() - start_time
    #         action = list(actions[:,i])
    #         next_state, reward, done, info = env.step(action)
    #         # img = env.render()
    #         if done:
    #             break
    # env.close()
