import numpy as np
import datetime
import gym

class TwoAgentsMatrixGame():
    def __init__(self, payoff_table, replay_dir='./replay_dir'):
        self.payoff_table = np.array(payoff_table, dtype=np.float)
        self._init_replay()
        self.current_episode = 0

        self.replay_dir = replay_dir

        self.n_actions = 3
        self.n_agents = 2
        self.state_shape = 1
        self.obs_shape = 1
        self.episode_limit = 1

        self.env_info = {"n_actions": self.n_actions,
                         "n_agents": self.n_agents,
                         "state_shape": self.state_shape,
                         "obs_shape": self.obs_shape,
                         "episode_limit": self.episode_limit,
                         "env_name": "SingleStateMatrixGame"
                         }

    def step(self, actions):
        ''' action是一个list '''
        reward = self.payoff_table[actions[0], actions[1]]
        terminated = True
        info = {}

        # replay
        self.replay[self.current_episode]["obs"].append([1., 1.])
        self.replay[self.current_episode]["state"].append(1.)
        self.replay[self.current_episode]["actions"].append(actions)
        self.replay[self.current_episode]["reward"].append(reward)
        self.replay[self.current_episode]["episode_length"] += 1

        return reward, terminated, info

    def get_obs(self):
        ''' 返回一个lst，所有agent的obs '''
        return [np.array([0.]), np.array([0.])]

    def get_state(self):
        return np.array([0.])

    def get_avail_actions(sefl):
        ''' 一个lst，每个agent哪些动作可以做 '''
        return [np.array([1, 1, 1]), np.array([1, 1, 1])]

    def get_avail_agent_actions(self, agent_id):
        ''' 获得某个agent的可选动作 '''
        return np.array([1, 1, 1])

    def reset(self):
        if self.replay[self.current_episode]["episode_length"] != 0:
            self.replay.append({"obs": [], "state": [], "actions": [], "reward": [], "episode_length": 0})
            self.current_episode += 1
        return

    def close(self):
        self._init_replay()
        self.current_episode = 0
        return

    def _init_replay(self):
        self.replay = []
        self.replay.append({"obs": [], "state": [], "actions": [], "reward": [], "episode_length": 0})

    def save_replay(self):
        d = datetime.datetime.today()
        dateStr = d.strftime('%Y-%m-%d_%H %M %S')
        np.save(self.env_info["env_name"] + dateStr, self.replay)


    def get_env_info(self):
        return self.env_info

    def get_episodes(self):
        flattened_payoff = self.payoff_table.flatten()  # (9,)
        n_episodes = len(flattened_payoff)  # 9
        o = np.ones((n_episodes, self.episode_limit, self.n_agents, self.obs_shape), dtype=np.float)
        s = np.ones((n_episodes, self.episode_limit, self.state_shape), dtype=np.float)

        # generate actions
        u = np.zeros((n_episodes, self.episode_limit, self.n_agents, 1), dtype=np.long)   # (9,1,2,1)
        u_index = np.arange(0, self.n_actions)  # (3,)三个动作0，1，2
        cartesian_product = np.array(np.meshgrid(*[u_index] * self.n_agents)).T.reshape(-1, self.n_agents)  # (9, n_agents)两个agent的action空间做笛卡尔积(得到每种的动作组合)
        for i in range(n_episodes):
            u[i, 0] = cartesian_product[i].reshape(self.n_agents, -1)

        u_onehot = np.zeros((n_episodes, self.episode_limit, self.n_agents, self.n_actions), dtype=np.long)    # (9,1,2,3)
        for i in range(n_episodes):
            for agent_id in range(self.n_agents):
                action_onehot = np.zeros(self.n_actions)
                action_id = u[i, 0, agent_id, 0]
                action_onehot[action_id] = 1
                u_onehot[i, 0, agent_id] = action_onehot

        r = flattened_payoff.reshape(n_episodes, self.episode_limit, 1)

        terminate = np.ones([n_episodes, self.episode_limit, 1])
        padded = np.zeros([n_episodes, self.episode_limit, 1])
        avail_u = np.ones([n_episodes, self.episode_limit, self.n_agents, self.n_actions])

        episodes = dict(o=o.copy(),
                        s=s.copy(),
                        u=u.copy(),
                        r=r.copy(),
                        avail_u=avail_u.copy(),
                        o_next=o.copy(),
                        s_next=s.copy(),
                        avail_u_next=avail_u.copy(),
                        u_onehot=u_onehot.copy(),
                        padded=padded.copy(),
                        terminated=terminate.copy()
                        )
        return episodes


class MultiAgentSimpleEnv2(gym.Env):  # Matrix game
    def __init__(self, n_predator=1):
        self.state = [1]
        self.action_dim = 3
        self.state_dim = 1

        self.payoff2 = np.array([[8., -12., -12.], [-12., 0., 0.], [-12., 0., 0.]])

    def reset(self):
        self.state = [1]

        return self.state

    def step(self, action):
        info = {'n': []}
        reward = []
        done = []
        reward.append(self.payoff2[action[0], action[1]])
        self.state = [3]
        done.append(True)

        return self.state, reward, done, info

    def call_action_dim(self):
        return self.action_dim

    def call_state_dim(self):
        return self.state_dim



if __name__ == '__main__':
    env = TwoAgentsMatrixGame()

    terminated, win_tag, step, episode_reward = False, False, 0, 0


    env_info = env.get_env_info()
    episode_limit = env_info["episode_limit"]
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]

    last_actions = np.zeros((n_agents, n_actions))

    while not terminated and step < episode_limit:
        obs = env.get_obs()  # 一个lst，有所有agent的obs
        state = env.get_state()  # state
        actions, actions_onehot = [], []  # 存所有agent的action组成lst
        avail_actions = env.get_avail_actions()  # mask向量

        for agent_id in range(n_agents):
            action = np.random.randint(0, 3)
            action_onehot = np.zeros(n_actions)
            action_onehot[action] = 1

            actions.append(action)
            actions_onehot.append(action_onehot)
            last_actions[agent_id] = action_onehot  # 更新上一次动作

        reward, terminated, info = env.step(actions)  # env step

        win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
