import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean():
    win_rates = []
    win_rates.append(np.load('win_rates_0.npy'))    # Qmix
    maic_qmix = [
        0.0,
        0.0,
        0.125,
        0.625,
        0.78125,
        0.6875,
        0.75,
        0.65625,
        0.75,
        0.6875,
        0.1875,
        0.375,
        0.25,
        0.375,
        0.53125,
        0.375,
        0.21875,
        0.25,
        0.625,
        0.46875,
        0.75,
        0.53125,
        0.6875,
        0.40625,
        0.5625,
        0.875,
        0.59375,
        0.875,
        0.875,
        0.84375,
        0.875,
        0.84375,
        0.9375,
        0.875,
        0.6875,
        0.875,
        0.90625,
        0.9375,
        0.9375,
        0.90625,
        0.96875,
        1.0,
        0.96875,
        0.90625,
        0.9375,
        0.9375,
        0.90625,
        0.90625,
        0.90625,
        1.0,
        1.0,
        0.9375,
        0.9375,
        0.96875,
        0.9375,
        1.0,
        0.9375,
        1.0,
        0.78125,
        0.96875,
        1.0,
        1.0,
        0.90625,
        0.9375,
        1.0,
        0.96875,
        0.96875,
        0.96875,
        1.0,
        0.96875,
        1.0,
        1.0,
        1.0,
        0.96875,
        0.9375,
        0.9375,
        0.96875,
        1.0,
        0.9375,
        0.96875,
        0.96875,
        0.9375,
        0.96875,
        0.96875,
        1.0,
        0.9375
    ]
    maic_qmix = np.array(maic_qmix).repeat(2)
    win_rates.append(maic_qmix)

    rtw_qmix = [0.0,
        0.0,
        0.0,
        0.34375,
        0.34375,
        0.375,
        0.34375,
        0.34375,
        0.25,
        0.375,
        0.25,
        0.34375,
        0.25,
        0.3125,
        0.1875,
        0.4375,
        0.46875,
        0.375,
        0.53125,
        0.75,
        0.6875,
        0.65625,
        0.71875,
        0.65625,
        0.78125,
        0.75,
        0.84375,
        0.53125,
        0.71875,
        0.84375,
        0.6875,
        0.5625,
        0.75,
        0.78125,
        0.875,
        0.75,
        0.75,
        0.90625,
        0.90625,
        0.78125,
        0.8125,
        0.75,
        0.875,
        0.8125,
        0.9375,
        1.0,
        0.9375,
        0.9375,
        1.0,
        0.90625,
        0.9375,
        0.9375,
        1.0,
        1.0,
        0.96875,
        1.0,
        0.96875,
        0.96875,
        0.9375,
        0.875,
        1.0,
        0.96875,
        1.0,
        0.96875,
        0.9375,
        0.96875,
        0.90625,
        0.96875,
        0.96875,
        0.96875,
        0.96875,
        1.0,
        1.0,
        0.90625,
        0.9375,
        1.0,
        1.0,
        0.90625,
        1.0,
        0.96875,
        0.96875
    ]
    rtw_qmix = np.array(rtw_qmix).repeat(2)
    win_rates.append(rtw_qmix)

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
    plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='maic-qmix')
    plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='rtw-qmix')
    # plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='y', label='qtran_alt')
    # plt.plot(range(len(win_rates[3][0])), win_rates[3][0], c='m', label='qplex')

    plt.legend()
    plt.xlabel('steps * 5000')
    plt.ylabel('win_rate')
    plt.savefig('./overview_qmix_maic_rtw.png')
    plt.show()


if __name__ == '__main__':
    plt_win_rate_mean()