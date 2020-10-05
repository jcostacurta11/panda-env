import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import deepcopy


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

#distance function for point and segment
# def dist_point_segment(u, v, p):
#     l2 = np.sum((u - v) * (u - v))
#     if (l2 == 0.0):
#         return dist(p, v)
#     t = max(0, min(1, np.sum((p-v) * (u-v)) / l2))
#     v_proj = v + t * (u-v)
#     return dist(v_proj, p)


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
        act_len = len(actions) // (2 * self.num_agents) #account for number of agents and x,y actions
        vx = []
        vy = []
        for i in range(self.num_agents): #complete for each agent
            ax, ay = actions[i * 2 * act_len: (i * 2 + 1) * act_len], actions[(
                i * 2 + 1) * act_len:(i * 2 + 2) * act_len] #picking out which actions apply to the agent (won't need to do this exactly)
            tmp_vx = ax
            tmp_vy = ay
            # tmp_vx = np.hstack([self.init_obs[i][2], np.cumsum(
            #     ax) * self.delta_t + self.init_obs[i][2]]) #TODO: I think this is getting the resulting velocities from the actions? ask mengxi what the actions are
            # tmp_vy = np.hstack([self.init_obs[i][3], np.cumsum(
            #     ay) * self.delta_t + self.init_obs[i][3]])
            vx.append(tmp_vx)
            vy.append(tmp_vy)
        vx = np.array(vx)
        vy = np.array(vy)
        return np.stack([vx, vy]).transpose(1, 0, 2) #reshape: look into this later

    def _pos(self, actions):
        # actions is n*2*T dimensional array [ax^0_0, ax^0_1, ..., ax^0_T, ay^0_0, ay^0_1, ..., a^0y_T, ax^1_0, ax^1_1, ..., ]
        # returned position is n*2*(T+1) dimensional array [x^0_0, x^0_1, ..., x^0_{T+1}, y^0_0, y^0_1, ..., y^0_{T+1}, x^1_0, x^1_1, ...,]
        if len(actions.shape) > 1:
            actions = self.flat_action(actions)
        act_len = len(actions) // (2 * self.num_agents)
        x = []
        y = []
        v = self._v(actions)

        for i in range(self.num_agents):
            ax, ay = actions[i * 2 * act_len: (i * 2 + 1) * act_len], actions[(
                i * 2 + 1) * act_len:(i * 2 + 2) * act_len]
            #TODO: make vector of all x positions (is this over time? ask mengxi)
            tmp_x = np.hstack([self.init_obs[i][0], self.init_obs[i][0] + np.cumsum(
                ax) * self.delta_t ])
            tmp_y = np.hstack([self.init_obs[i][1], self.init_obs[i][1] + np.cumsum(
                ay) * self.delta_t ])
            x.append(tmp_x)
            y.append(tmp_y)
        x = np.array(x)
        y = np.array(y)
        return np.stack([x, y]).transpose(1, 0, 2)

    def flat_action(self, action):
        assert len(action.shape) == 3
        assert action.shape[1] == 2
        # convert from shape [num_agent, 2, T] to flatten
        return action.reshape([-1])

    def unflat_action(self, action, shape):
        assert shape[1] == 2
        return action.reshape([shape[0], shape[1], shape[2]])

    def vis_traj(self, actions, task_obj, fig=None, ax=None): #plotter (would be nice to adapt this but not priority)
        pos = self._pos(actions)
        x = pos[:, 0, :]
        y = pos[:, 1, :]
        if ax is None:
            fig, ax = plt.subplots()
        # for i in range(task_obj.num_obstacles):
        #     ax.add_artist(plt.Circle(
        #         (task_obj.obs_x[i], task_obj.obs_y[i]), task_obj.obs_r_min[i], color='#000033', alpha=0.5))
        for i in range(task_obj.num_agents):
            ax.plot(x[i], y[i], marker="o", color=task_obj.colors[i])
        ax.set_aspect('equal', adjustable='datalim')
        return fig, ax

class Navigation:#the first thing defined in mengxi's main loop, takes in theta
    def __init__(self, theta):
        self.num_agents = num_agents
        #self.num_obstacles = 1

        # limit
        self.u_max = 5.0
        self.v_max = 5.0

        # setting initial position and velocities for up to 3 agents
        self.mass = np.array([1, 1, 1])
        self.x_init = np.array([0, -1, 1])
        self.y_init = np.array([2, 0, 0])
        self.vx_init = np.array([0, 0, 0])
        self.vy_init = np.array([0, 0, 0])
        self.colors = ['b', 'g', 'y']

        #TODO: what are these?? ask mengxi
        self.x_formation = self.x_init[:self.num_agents] - \
            self.x_init[:self.num_agents].mean()
        self.y_formation = self.y_init[:self.num_agents] - \
            self.y_init[:self.num_agents].mean()

        #setting goal end states
        self.x_end = np.array([0, -1, 1])
        self.y_end = np.array([12, 10, 10])
        self.vx_end = np.array([0, 0, 0])
        self.vy_end = np.array([0, 0, 0])

        #obstacle placement and radius
        # self.obs_x = np.array([0.5])
        # self.obs_y = np.array([5])
        # # self.obs_r_max = np.array([2.5])
        # self.obs_r_min = np.array([2])
        self.theta = theta

class OptimizeMultiTraj(object):
    # T: time horizon of action for a single agent
    def __init__(self, task_env, task_obj, T, gamma=1., beta=1., seed=0, seq_decay=0, seq_smooth=False):
        self.actions = np.zeros(2*T*task_obj.num_agents) #looks like we'll be creating the actions in this class
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

    def get_traj_cost(self, x, y, task_obj):
        center_x = np.zeros_like(x[0])
        center_y = np.zeros_like(y[0])
        for i in range(task_obj.num_agents):
            center_x = center_x + x[i]
            center_y = center_y + y[i]
        #average position of agents?
        center_x = center_x / task_obj.num_agents
        center_y = center_y / task_obj.num_agents

        #does this cost have to do with the number of agents?
        cost_formation = np.zeros_like(center_x)
        for i in range(task_obj.num_agents):
            cost_formation += np.square(x[i] - center_x - task_obj.x_formation[i]) + \
                np.square(y[i] - center_y - task_obj.y_formation[i])
        cost_formation = np.mean(cost_formation)
        # cost_formation = 0

        #costs for colliding with obstacle (won't need this)
        cost_collision = 0
        # for i in range(task_obj.num_agents):
        #     for j in range(task_obj.num_obstacles):
        #         distance = np.sqrt(
        #             np.square(x[i] - task_obj.obs_x[j]) + np.square(y[i] - task_obj.obs_y[j]))
        #         min_ind = np.argmin(distance)
        #         min_dist = distance[min_ind]
        #         p = np.array([task_obj.obs_x[j], task_obj.obs_y[j]])
        #         if min_ind > 0:
        #             u = np.array([x[i][min_ind-1], y[i][min_ind-1]])
        #             v = np.array([x[i][min_ind], y[i][min_ind]])
        #             new_dist = dist_point_segment(u=u, v=v, p=p)
        #             if new_dist < min_dist:
        #                 min_dist = new_dist
        #         if min_ind + 1 < len(x[i]):
        #             u = np.array([x[i][min_ind+1], y[i][min_ind+1]])
        #             v = np.array([x[i][min_ind], y[i][min_ind]])
        #             new_dist = dist_point_segment(u=u, v=v, p=p)
        #             if new_dist < min_dist:
        #                 min_dist = new_dist
        #         raw_cost = -np.minimum(0, min_dist - task_obj.obs_r_min[j])
        #         cost_collision += raw_cost

        #cost based on how long the path is (need this!)
        cost_length = 0
        for i in range(task_obj.num_agents):
            dx = x[i][:-1] - x[i][1:]
            dy = y[i][:-1] - y[i][1:]
            tmp_l = np.sum(np.sqrt(dx * dx + dy * dy))
            cost_length += tmp_l

        #cost based on how far you are from the goal
        cost_goal = 0
        for i in range(task_obj.num_agents):
            pos = np.array([x[i][-1], y[i][-1]])
            goal = np.array([task_obj.x_end[i], task_obj.y_end[i]])
            cost_goal += dist(pos, goal)

        #weight all the costs together
        cost = cost_formation * \
            task_obj.theta[0] + cost_collision * \
            task_obj.theta[1] + cost_length * \
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
        x = pos[:, 0, :]
        y = pos[:, 1, :]
        return self.get_traj_cost(x, y, task_obj)

    #these two functions are used somehow in the constraints
    def a_con(self, actions, epsilon=0):
        return self.task_obj.u_max + epsilon - np.abs(actions)

    def v_con(self, actions, epsilon=0):
        v = self.task_env._v(actions)
        return self.task_obj.v_max + epsilon - np.abs(v.reshape([-1]))

    def optimize(self):
        self.init_actions()
        res = minimize(self.objective, self.actions,
                       constraints=self.constraints, options=self.options)
        self.actions = res.x
        self.actions = self.task_env.unflat_action(
            self.actions, [self.num_agents, 2, self.T])
        return self.actions, self.objective(res.x)

if __name__ == "__main__":
    task_obj = Navigation(theta=theta)
    init_obs = [[task_obj.x_init[i], task_obj.y_init[i],
                 task_obj.vx_init[i], task_obj.vy_init[i]] for i in range(num_agents)]
    task_env = TrajEnv(init_obs=init_obs, delta_t=1, num_agents=num_agents)

    optimizer = OptimizeMultiTraj(
        task_env=task_env, task_obj=task_obj, T=T, gamma=gamma, beta=beta, seq_decay=seq_decay, seq_smooth=seq_smooth)
    actions, _ = optimizer.optimize()
    print(task_env._pos(actions))
    print(task_env._v(actions))
    fig_0, ax_0 = task_env.vis_traj(actions, task_obj)
    plt.show()