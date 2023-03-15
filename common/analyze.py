import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean():
    path = []
    alg_num = 4
    win_rates = [[] for _ in range(alg_num)]
    game_map = '2s3z'
    path.append('../result/vdn/' + game_map)
    path.append('../result/qmix/' + game_map)
    path.append('../result/qtran_base/' + game_map)
    # path.append('../result/qtran_alt/' + game_map)
    path.append('../result/qplex/' + game_map)


    # 设置哪些算法显示
    alg_index = []

    num_run = 1 # TODO 默认8
    for i in range(alg_num):
        for j in range(num_run):
            win_rates[i].append(np.load(path[i] + '/win_rates_{}.npy'.format(j)))


    # win_rates = np.array(win_rates).mean(axis=1)    # (alg_num, steps*5000)算num次训练结果的均值
    # new_win_rates = [[] for _ in range(alg_num)]
    # average_cycle = 1   # 每average_cycle算一次平均
    # for i in range(alg_num):
    #     rate = 0
    #     time = 0
    #     for j in range(len(win_rates[0])):
    #         rate += win_rates[i, j]
    #         time += 1
    #         if time % average_cycle == 0:
    #             new_win_rates[i].append(rate / average_cycle)
    #             time = 0
    #             rate = 0
    # new_win_rates = np.array(new_win_rates)
    # new_win_rates[:, 0] = 0
    # win_rates = new_win_rates



    plt.figure()
    plt.ylim(0, 1.0)
    plt.plot(range(len(win_rates[0][0])), win_rates[0][0], c='b', label='vdn')
    plt.plot(range(len(win_rates[1][0])), win_rates[1][0], c='r', label='qmix')
    plt.plot(range(len(win_rates[2][0])), win_rates[2][0], c='g', label='qtran_base')
    # plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='y', label='qtran_alt')
    plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='m', label='qplex')

    plt.legend()
    plt.xlabel('steps * 5000')
    plt.ylabel('win_rate')
    plt.savefig('../result/overview_{}.png'.format(game_map))
    plt.show()


if __name__ == '__main__':
    plt_win_rate_mean()