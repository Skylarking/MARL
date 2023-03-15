import numpy as np
import os
from rollout import RolloutWorker
from controller.share_params import SharedMAC, SeparatedMAC
from common.replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
from algorithm.q_learner import QLearner
from algorithm.qtran_learner import QTRANLearner
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
from env.single_state_matrix_game import TwoAgentsMatrixGame

def evaluate(args, rolloutWorker):
    ''' 评估模型m次，算平均胜率与reward '''
    if args.evaluate_epoch == 0:  # 若不测试，直接返回0，0
        return 0, 0
    # 测试M次，算平均reward与胜率
    _, episodes_reward, win_tags, _ = rolloutWorker.generate_episodes(n_episodes=args.evaluate_epoch, evaluate=True)
    return sum(win_tags) / args.evaluate_epoch, sum(episodes_reward) / args.evaluate_epoch


def plt_save(args, eval_episode_rewards, save_path):
    ''' 显示第num次训练的整个测试曲线，并保存数据与曲线 '''
    plt.figure()
    plt.ylim([0, 105])
    plt.cla()

    plt.plot(range(len(eval_episode_rewards)), eval_episode_rewards)
    plt.xlabel('Iteration')
    plt.ylabel('episode_rewards')

    plt.savefig(save_path + '/plt.png', format='png')
    np.save(save_path + '/episode_rewards', eval_episode_rewards)
    plt.close()

def run():
    args = get_common_args()
    get_mixer_args(args)
    args.alg = 'qtran_base'
    args.lr = 0.001
    tot_epoch = 20000
    args.cuda = True

    payoff_table1 = [[8,-12,-12],
                     [-12,0,0],
                     [-12,0,0]]

    payoff_table2 = [[8,-12,-12],
                     [-12,6,0],
                     [-12,0,6]]

    payoff_table3 = [[8,3,2],
                     [-12,-13,-14],
                     [-12,-13,-14]]

    env = TwoAgentsMatrixGame(payoff_table=payoff_table1)
    env_info = env.get_env_info()

    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    save_path = './result/' + args.alg + '/' + 'MatrixGame'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mac = SharedMAC(args)
    rolloutWorker = RolloutWorker(env, mac, args)
    if args.alg == 'qtran_base':
        learner = QTRANLearner(mac, args)
    else:
        learner = QLearner(mac, args)

    eval_episode_rewards = []
    time_steps, train_steps = 0, 0  # train_steps+1表示增加一个episode（而不是steps）
    while time_steps < tot_epoch:  # 总训练轮次
        # evaluate
        _, episode_reward = evaluate(args, rolloutWorker)
        eval_episode_rewards.append(episode_reward)
        plt_save(args, eval_episode_rewards, save_path=save_path)

        # generate n episodes for training
        # 这里获取的数据是固定9个episode，每个episode包含一种不同的joint action(共9种不同的joint action)，这样做是省去了探索
        episodes, rewards, win_tags, steps = env.get_episodes(), 0, False, 1
        time_steps += steps

        # train
        # 直接利用产生的数据训练
        loss = learner.train(episodes, train_steps)
        if time_steps % 100 == 0:
            print('Iteration: {a}   MSE loss: {b}'.format(a=time_steps, b=loss))
        train_steps += 1

    # at end of training, eval one time in the last
    _, episode_reward = evaluate(args, rolloutWorker)
    eval_episode_rewards.append(episode_reward)
    plt_save(args, eval_episode_rewards, save_path=save_path)

    # get q table输出q table并保存
    q_tot_table, q_table_i, q_table_j = learner.get_q_and_q_tot_table()
    print("=====================================================================")
    print("+++++++++++++++++++++++++++++q_tot_table+++++++++++++++++++++++++++++")
    print(q_tot_table)
    r, c = divmod(q_tot_table.argmax(), q_tot_table.shape[1])
    print("greedy joint-action is: ", [r, c])
    print("++++++++++++++++++++++++++++++q_i_table++++++++++++++++++++++++++++++")
    print(q_table_i)
    print("agent row greedy action is: ", [q_table_i.argmax()])
    print("++++++++++++++++++++++++++++++q_j_table++++++++++++++++++++++++++++++")
    print(q_table_j)
    print("agent col greedy action is: ", [q_table_j.argmax()])
    print("=====================================================================")


if __name__ == '__main__':
    run()


