import numpy as np

class RolloutWorker():
    '''
    rollout需要：
        env：交互的环境
        agents：所有agents，与环境交互得到轨迹

    返回：交互环境产生的episodes(n_episodes, episode_limit, n_agents, features)
    '''
    def __init__(self, env, mac, args):
        self.env = env
        self.mac = mac
        self.episode_limit = args.episode_limit # 最大长度
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def init_last_actions(self):
        """ 初始化一个全0的动作，在一个episode开始时使用，因为最开始的上一次动作不存在 """
        return np.zeros((self.args.n_agents, self.args.n_actions))  # (n_agents, n_actions)

    def generate_episodes(self, n_episodes=1, evaluate=False, random_select=False):
        steps_tot = 0
        wins_tag = []
        episodes_reward = []

        # save replay set. if to save, need to env.close() to throw previous replay
        if self.args.replay_dir != '' and evaluate:  # prepare for save replay of evaluation
            self.env.close()

        # generate n episodes
        episodes = None # store n episodes
        for num_episode in range(n_episodes):
            self.env.reset()

            # initial one episode hidden_states for generate data
            self.mac.init_hidden(1)

            # epsilon
            epsilon = 0 if evaluate else self.epsilon  # 若是测试时，为确定性策略，epsilon为0
            if self.args.epsilon_anneal_scale == 'episode':     # 退火epsilon，每一个episode减去固定值
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon



            # vars for controlling & storing
            terminated, win_tag, step, episode_reward = False, False, 0, 0
            last_actions = self.init_last_actions()
            o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []   # 存一个episode中的数据，其他lst存当前timestep的数据

            # generate one episode
            while not terminated and step < self.episode_limit:
                obs = self.env.get_obs()  # 一个lst，有所有agent的obs
                state = self.env.get_state()  # state
                actions, actions_onehot = [], [] # 存所有agent的action组成lst
                avail_actions = self.env.get_avail_actions()    # mask向量


                for agent_id in range(self.n_agents):
                    if random_select:
                        action = np.random.randint(0, self.n_actions-1)
                        while avail_actions[agent_id][action] == 0:
                            action = np.random.randint(0, self.n_actions - 1)
                    else:
                        if self.args.RTW:
                            action = self.mac.choose_action(obs[agent_id], last_actions[agent_id], agent_id, avail_actions, epsilon, evaluate)
                        else:
                            action = self.mac.choose_action(obs[agent_id], last_actions[agent_id], agent_id, avail_actions[agent_id], epsilon, evaluate)
                    action_onehot = np.zeros(self.n_actions)
                    action_onehot[action] = 1

                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    last_actions[agent_id] = action_onehot # 更新上一次动作

                # action = self.mac.choose_action(np.stack(obs, axis=0), last_actions, np.stack(avail_actions, axis=0), epsilon, evaluate)

                reward, terminated, info = self.env.step(actions)   # env step

                win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
                o.append(obs)  # 所有agent的obs
                s.append(state)
                u.append(np.reshape(actions, [self.n_agents, 1]))  # 所有agent的action
                u_onehot.append(actions_onehot)
                avail_u.append(avail_actions)
                r.append([reward])  # 团队的reward
                terminate.append([terminated])
                padded.append([0.])  # paded==0表示该位置没有padding
                episode_reward += reward
                step += 1

                if self.args.epsilon_anneal_scale == 'step':  # 根据step调整epsilon
                    epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            # last obs（获得episode中结束后的一次数据（因为env.step()没有给出下一步的观测信息））
            obs = self.env.get_obs()
            state = self.env.get_state()
            o.append(obs)
            s.append(state)
            o_next = o[1:]  # o_next相当于截断第一个元素的o
            s_next = s[1:]
            o = o[:-1]  # o相当于截断最后一个元素的o
            s = s[:-1]
            # get avail_action for last obs，because target_q needs avail_action in training（获得最后obs的可用动作）
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_action)
            avail_u.append(avail_actions)
            avail_u_next = avail_u[1:]  # avail_u_next相当于截断第一个元素的avail_u
            avail_u = avail_u[:-1]  # avail_u相当于截断最后一个元素的avail_u

            # if step < self.episode_limit，padding（若一个episode没有达到最大长度，那么pad 0，是为了保证多个episodes拼接等长的）
            for i in range(step, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])  # paded==1表示该位置是padding
                terminate.append([1.])

            episode = dict(o=o.copy(),
                           s=s.copy(),
                           u=u.copy(),
                           r=r.copy(),
                           avail_u=avail_u.copy(),
                           o_next=o_next.copy(),
                           s_next=s_next.copy(),
                           avail_u_next=avail_u_next.copy(),
                           u_onehot=u_onehot.copy(),
                           padded=padded.copy(),
                           terminated=terminate.copy()
                           )
            # add episode dim, so the dim of one key is (1, episode_limit, n_agents, features)
            for key in episode.keys():
                episode[key] = np.array([episode[key]])  # 在最开始增加一个维度，并且为1，表示episode个数

            # concat n episodes at dim=0, then the dim of one key is (n_episodes, episode_limit, n_agents, features)
            if num_episode == 0:
                episodes = episode.copy()
            else:
                for key in episodes.keys():
                    episodes[key] = np.concatenate((episodes[key], episode[key]), axis=0)


            steps_tot += step
            wins_tag.append(win_tag)
            episodes_reward.append(episode_reward)

            # if evaluate then save n episodes replay
            if evaluate and num_episode == n_episodes - 1 and self.args.replay_dir != '':  # 若是测试，保存测试的录像
                self.env.save_replay()
                self.env.close()

            # update(if not evaluate) epsilon for next generating episodes(放在最后更新是因为有episode和step两种退火方式，两种方式都能照顾到)
            if not evaluate:
                self.epsilon = epsilon

        # return shape:   dict and dict[key].shape = arr(n_episodes, episode_limit, n_agents, features), lst(n_episodes,), lst(n_episodes,), int
        return episodes, episodes_reward, wins_tag, steps_tot

    # def cut_max_episode_len(self, episodes):
    #     # calculate the max_episode_len of n episodes for cutting redundant transitions
    #     n_episodes = episodes["o"].shape[0]
    #     max_episode_len = 0
    #     for episode_idx in range(n_episodes):
    #         for transition_idx in range(self.episode_limit):
    #             if episodes['terminated'][episode_idx][transition_idx][0] == 1:
    #                 if transition_idx + 1 >= max_episode_len:
    #                     max_episode_len = transition_idx + 1
    #                 break
    #     if max_episode_len == 0:  # 防止所有episode都是最大长度
    #         max_episode_len = self.episode_limit
    #
    #     # cut redundant transitions
    #     for key in episodes.keys():
    #         episodes[key] = episodes[key][:, :max_episode_len]  # 截取max_episode_len前面的数据
    #     return episodes