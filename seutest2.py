import tensorflow as tf
import os
from creat_model import LeNet
import numpy as np
import matplotlib.pyplot as plt
import seu
import tqdm
import copy
import fault_injection
import time
# test_time 表示对于注入概率p，取多少次结果
# t 注入批次
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"    # 使用第一, 三块GPU
def get_acc(model, weights, x_test, y_test ,test_times, p, t, bitlen, TMR, avevalid):
    # weights = model.get_weights()
    weights2 = copy.deepcopy(weights)
    test_loss_seu, test_acc_seu = 0, 0
    loss = []
    acc = []
    for j in tqdm.tqdm(range(test_times)):
        # start_time = time.time()
        for i in range(len(weights)):
            # print(len(weights[i]), type(weights[i]), type(weights[i][0]))
            # print("----------level {} ------------".format(i))
            # x, bitlen, lamda, TMR, p, batchnum, avevalid
            lamda = 1
            if (0):  #是否对卷积层注入

                if (i > 3):
                    weights2[i], cnt = fault_injection.seu(wehts[i], bitlen, lamda, TMR, p, t, avevalid)
            else:
                weights2[i], cnt = fault_injection.seu(weights[i], bitlen, lamda, TMR, p, t, avevalid)
        # print(len(weights), len(weights2))
        end_time = time.time()
        # print(start_time - end_time)
        model.set_weights(weights2)

        # start_time = time.time()
        test_loss_seu1, test_acc_seu1 = model.evaluate(x_test[0:3000], y_test[0:3000], batch_size=64, verbose=0)

        test_loss_seu += test_loss_seu1
        test_acc_seu += test_acc_seu1
        # print(test_loss_seu1, test_acc_seu1)
        loss.append(test_loss_seu1)
        acc.append(test_acc_seu1)

        # end_time = time.time()
        # print(end_time - start_time)
    # test_acc_seu /= test_times
    # test_loss_seu /= test_times
    # loss.append()
    return loss, acc


# test_loss_seu, test_acc_seu = get_acc(model, 20, 0.001)
# print(type(test_loss))
# print(weights2[0])
# print(test_loss_seu)
# print(test_acc_seu)

# test_loss_seu, test_acc_seu = model.evaluate(x_test, y_test)
# print()
# print("noseu_test_loss: ", test_loss, "noseU_test_acc: ", test_acc)
# print("seu_test_loss: ", sum(test_loss_seu) / len(test_loss_seu), "seU_test_acc: ", sum(test_acc_seu) / len(test_acc_seu))

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = LeNet()
    checkpoint_save_path_noseu = "./mnist/LeNet.ckpt"
    checkpoint_save_path_seu = "./mnist_seu/LeNet.ckpt"
    checkpoint_save_path = checkpoint_save_path_noseu
    result_savepath_noseu = "./pic/noseu_noquant_test_3.png"
    result_savepath_seu = "./pic/seu_noquant_test.png"
    result_savepath = result_savepath_noseu
    sel = 0
    if sel == 0:
        checkpoint_save_path = checkpoint_save_path_noseu
        result_savepath = result_savepath_noseu
    elif sel == 1:
        checkpoint_save_path = checkpoint_save_path_seu
        result_savepath = result_savepath_seu
    print(checkpoint_save_path)
    if os.path.exists(checkpoint_save_path + ".index"):
        print("*******load the model******")
        model.load_weights(checkpoint_save_path)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    weights = model.get_weights()
    # print(len(model.get_weights()))

    with open("seu_mnist_lenet_weights.txt", "w") as f:
        for v in model.trainable_variables:
            f.write(str(v.name) + "\n")
            f.write(str(v.shape) + "\n")
            f.write(str(v.numpy()) + "\n")



    # p = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003]
    cyc = 10
    pointnum = 40
    maxyear = 10
    p = np.linspace(0, maxyear / 2 * 0.001, cyc)
    loss = []
    accx = []
    accy_noave_TMR0 = []
    t = 0

    bitlen = 32
    TMR = 3
    avevalid = 1
    timescale = 2000

    xstick = []
    good = []
    bad = []
    verybad = []
    goodlimit = 0.85
    badlimit = 0.3
    for pp in tqdm.tqdm(p):
        # t = t + 1
        t = 1
        avevalid = 0
        TMR = 0
        lossl0, accl0 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, TMR, avevalid)

        for j in range(len(accl0)):
            accx.append(pp * timescale)
            accy_noave_TMR0.append(accl0[j])

        g = 0
        b = 0
        vb = 0
        xstick.append("{:.1f}".format(pp*timescale))
        for j in range(len(accl0)):
            if (accl0[j] > goodlimit):
                g += 1
            elif(accl0[j] > badlimit):
                b += 1
            else:
                vb += 1
        good.append(round(g / pointnum, 2))
        bad.append(round(b / pointnum, 2))
        verybad.append(round(1 - round(g / pointnum, 2) - round(b / pointnum, 2), 2))

    acc_counts = {
        'Benign':np.array(good),
        'Poor': np.array(bad),
        'critical': np.array(verybad),
    }
    width = 0.6

    bottom = np.zeros(cyc)

    for condition, acc_count in acc_counts.items():
        p = plt.bar(xstick, acc_count, width, label=condition, bottom=bottom)

        for j in range(len(p)):
            if (acc_count[j] > 0.002):
                # plt.bar_label(p[j], label_type='center')
                m = round((maxyear / (cyc - 1)) * j,2)
                plt.text(j, (acc_count[j]) / 2 + bottom[j], str(acc_count[j]),verticalalignment='center', horizontalalignment='center')
                print(".........")
                print(m, (acc_count[j]) / 2 + bottom[j], str(acc_count[j]))
        bottom += acc_count
        # break

    result_savepath = "./pic"
    # plt.figure()
    # plt.scatter(accx, accy_noave_TMR0, c="blue", s = 30, alpha=0.5, label='accuracy')
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("proportion")
    plt.legend()
    plt.savefig(result_savepath + "/noave_TMR001.png")
    # plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# # 创建柱状图数据
# x = np.arange(5)
# y = np.array([2, 4, 6, 8, 10])
#
# # 绘制柱状图
# bars = plt.bar(x, y)

# # 给柱状图的一部分打标签
# for i in range(len(bars)):
#     if i < 3:  # 只给前三个柱子打标签
#         plt.text(x[i], y[i] / 2, str(y[i]), ha='center', va='bottom')
#
# # 显示图形
# plt.show()
