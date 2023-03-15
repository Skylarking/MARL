import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean():
    win_rates = []
    win_rates.append(np.load('win_rates_0.npy'))
    win_rates.append(np.load('win_rates_state.npy'))
    win_rates.append(np.load('win_rates_state_1.npy'))


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
    plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='qmix')
    plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='qmix_state')
    plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='qmix_state_1')
    # plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='y', label='qtran_alt')
    # plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='m', label='qplex')

    plt.legend()
    plt.xlabel('steps * 5000')
    plt.ylabel('win_rate')
    plt.savefig('../result/overview.png')
    plt.show()


if __name__ == '__main__':
    plt_win_rate_mean()