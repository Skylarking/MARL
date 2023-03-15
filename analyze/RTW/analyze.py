import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean():
    win_rates = []
    win_rates.append(np.load('win_rates_0.npy')[0:125])    # pred-i
    win_rates.append(np.load('win_rates_all.npy')[0:125])  # pred All
    win_rates.append(np.load('win_rates_qmix.npy')[0:125]) #qmix


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
    plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='RTW-Qmix -s')
    plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='RTW-Qmix -a')
    plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='Qmix')
    # plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='y', label='qtran_alt')
    # plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='m', label='qplex')

    plt.legend()
    plt.xlabel('steps * 5000')
    plt.ylabel('win_rate')
    plt.savefig('./overview_qmix_RTW.png')
    plt.show()


if __name__ == '__main__':
    plt_win_rate_mean()