#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, early access, 2019, DOI:10.1109/TMC.2019.2928811.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio  # import scipy.io for .mat file I/
import numpy as np  # import numpy

# for tensorflow2
from memoryTF2 import MemoryDNN
from optimization import bisection

import time


def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
    #    rolling_intv = 20

    plt.plot(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values),
                     np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color='b', alpha=0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()


def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)


if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N = 10  # number of users
    n = 30000  # number of time frames
    # K越大，计算时间越长
    K = N  # initialize K = N
    decoder_mode = 'OP'  # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024  # capacity of memory structure
    Delta = 32  # Update interval for adaptive K

    print(
        '#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d' % (N, n, K, decoder_mode, Memory, Delta))
    # Load data
    channel = sio.loadmat('./data/data_%d' % N)['input_h']
    rate = sio.loadmat('./data/data_%d' % N)[
        'output_obj']  # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    # channel每个元素加倍
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    # channel长度x0.8
    split_idx = int(.8 * len(channel))
    # 0.2*len(channel)、0.2*n
    num_test = min(len(channel) - split_idx, n - int(.8 * n))  # training data size

    # learning_rate 为adam训练器训练速率
    # 训练间隔 δ 为 training_interval
    mem = MemoryDNN(net=[N, 120, 80, N],
                    learning_rate=0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    # DNN生成k个mode（第一阶段），选择最大的继续训练（第二阶段）
    for i in range(n):
        # 每隔10次输出一次数据
        if i % (n // 10) == 0:
            print("%0.1f" % (i / n))
        if i > 0 and i % Delta == 0:
            # 更新max_k，最终目的是更新K值
            # index counts from 0
            if Delta > 1:
                # 取k_idx_his后32位中最大的一位
                max_k = max(k_idx_his[-Delta:-1]) + 1
            else:
                # 取k_idx_his最后一位。按照设置的Delta值，不会走到这一分支
                max_k = k_idx_his[-1] + 1
            # 获得一个K值，训练周期够长就足够抵消 K 值小的错误。4.4节进行阐述。
            # 为提高性能，动态更新k值。delta=1则每一个时间帧就更新K，若delta区域∞，则永远不更新k值。
            # K <= N + 1
            K = min(max_k + 1, N)

        # 前80%的数据
        if i < n - num_test:
            # training（训练数据）split_idx = n - num_test
            i_idx = i % split_idx
        # 后20%的数据
        else:
            # test（测试数据）
            i_idx = i - n + num_test + split_idx
        # 确定i_idx值，根据推断始终为i
        # channel为2维数组（源数据），取第i_idx行数据
        h = channel[i_idx, :]

        # the action selection must be either 'OP' or 'KNN'
        # 初始设置的decoder_mode值
        # m_list
        # K值越大，解的质量越好，计算复杂度越高，反之亦然
        m_list = mem.decode(h, K, decoder_mode)

        r_list = []
        for m in m_list:
            # channel中的元素*1000000，递归操作，h是channel中的一行
            # 资源分配子问题，计算得出 a 和 tao
            r_list.append(bisection(h / 1000000, m)[0])

        # encode the mode with the largest reward
        mem.encode(h, m_list[np.argmax(r_list)])
        # 将计算结果存入 memory
        # the main code for DROO training ends here
        # 训练结束

        # 记录一些需要展示的数据
        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of the largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])

    total_time = time.time() - start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio)

    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1]) / num_test)
    print('Total time consumed:%s' % total_time)
    print('Average time per channel:%s' % (total_time / n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
