import numpy as np
import os
from rollout import RolloutWorker
from controller.share_params import SharedMAC, SeparatedMAC, SharedMACWithState, RTWMAC
from common.replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
from algorithm.q_learner import QLearner
from algorithm.qtran_learner import QTRANLearner
from algorithm.RTW_q_learner import RTWQLearner
from algorithm.q_learner_state import QLearnerWithState
from utils.logging import Logger


class Runner():
    # buffer在Runner中
    def __init__(self, env, logger, args):
        self.env = env

        # reuse the network
        if args.reuse_network:
            if args.RTW:
                self.mac = RTWMAC(args)
            else:
                self.mac = SharedMAC(args)
        else:
            self.mac = SeparatedMAC(args)

        self.rolloutWorker = RolloutWorker(env, self.mac, args)

        self.buffer = ReplayBuffer(args)    # off-line policy needs ReplayBuffer
        self.args = args
        self.eval_win_rates = []  # 存测试胜率
        self.eval_episode_rewards = []  # 存测试rewards

        # 用来保存plt(测试曲线)和pkl(测试曲线的数据)
        if self.args.env == 'smac':
            self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        else:
            raise ValueError("env {} dose not exist!".format(args.env))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        logger.setup_tb(self.save_path + '/tb/other')
        self.logger = logger

        if args.alg.find('vdn') > -1 or args.alg.find('qmix') > -1 or args.alg.find('qplex') > -1:
            if args.RTW:

                self.learner = RTWQLearner(self.mac, self.logger, args)
            else:
                self.learner = QLearner(self.mac, args)
        elif args.alg.find('qtran_base') > -1 or args.alg.find('qtran_alt') > -1:
            self.learner = QTRANLearner(self.mac, args)
        else:
            raise ValueError('learner {} cannot find!'.format(args.alg))

        # load model
        if args.load_model:
            self.learner.load_models()

    def run(self, num):
        '''
        参数：
            num:run的编号，通常训练n次，算n次的平均效果，或者保存其中最好的模型

        功能：整个训练流程，包括：采集数据，存储buffer，训练，评估，保存评估数据，保存模型
        '''
        time_steps, train_steps, evaluate_steps = 0, 0, -1  # train_steps+1表示增加一个episode（而不是steps）
        while time_steps < self.args.n_steps:  # 总训练步数
            print('Run {}, time_steps {}'.format(num, time_steps))

            # evaluate
            if time_steps // self.args.evaluate_cycle > evaluate_steps:  # 每固定步数评估一次(evaluate_steps=-1表示第0步也要测试)
                win_rate, episode_reward = self.evaluate()
                print('win_rate is ', win_rate)
                self.eval_win_rates.append(win_rate)
                self.eval_episode_rewards.append(episode_reward)
                self.plt(num)  # 存测试数据，画测试曲线，num是runner编号
                self.logger.log_stat("test_win_rate", win_rate, time_steps)
                self.logger.log_stat("test_episode_reward", episode_reward, time_steps)
                evaluate_steps += 1

            # generate n episodes for training
            # TODO Random_select
            episodes, rewards, win_tags, steps = self.rolloutWorker.generate_episodes(n_episodes=self.args.n_episodes, random_select=False)  # episodes中每个key的value的shape为(n_episodes, episode_limit, n_agents, features)
            time_steps += steps
            self.logger.log_stat("episode_length", steps, time_steps)
            self.logger.log_stat("train_win_rate", sum(win_tags)/self.args.n_episodes, time_steps)
            self.logger.log_stat("train_episode_reward", sum(rewards)/self.args.n_episodes, time_steps)

            # store n episodes into buffer
            self.buffer.store_episode(episodes)

            # sample batch from buffer and train
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))    # sample mini batch
                loss = self.learner.train(mini_batch, train_steps)     # train_steps用于更新target网络
                train_steps += 1

            # log
            self.logger.log_stat("total_loss", loss, time_steps)

            # save model for k episodes
            if train_steps > 0 and train_steps % self.args.save_cycle == 0:
                self.learner.save_models(train_steps)


        # at end of training, eval one time in the last
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)  # final win rate
        self.eval_win_rates.append(win_rate)
        self.eval_episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        ''' 评估模型m次，算平均胜率与reward '''
        if self.args.evaluate_epoch == 0:   # 若不测试，直接返回0，0
            return 0, 0
        # 测试M次，算平均reward与胜率
        _, episodes_reward, win_tags, _ = self.rolloutWorker.generate_episodes(n_episodes=self.args.evaluate_epoch, evaluate=True)
        return sum(win_tags) / self.args.evaluate_epoch, sum(episodes_reward) / self.args.evaluate_epoch

    def plt(self, num):
        ''' 显示第num次训练的整个测试曲线，并保存数据与曲线 '''
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.eval_win_rates)), self.eval_win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.eval_episode_rewards)), self.eval_episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.eval_win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.eval_episode_rewards)
        plt.close()

