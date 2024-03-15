import tensorflow as tf
import os
from creat_seumodel import LeNet
import numpy as np
import matplotlib.pyplot as plt
import seu
import tqdm
import copy

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # 使用第一, 三块GPU

# test_time 表示对于注入概率p，取多少次结果
# t 注入批次
def get_acc(model, weights, x_test, y_test ,test_times, p, t):
    # weights = model.get_weights()
    weights2 = copy.deepcopy(weights)
    test_loss_seu, test_acc_seu = 0, 0
    loss = []
    acc = []
    for j in tqdm.tqdm(range(test_times)):
        for i in range(len(weights)):
            # print(len(weights[i]), type(weights[i]), type(weights[i][0]))
            # print("----------level {} ------------".format(i))
            if (0):  # 是否对卷积层注入
                if (i > 3):
                    weights2[i], cnt = seu.seu(weights[i], 32, 0, 0, p, t)
            else:
                weights2[i], cnt = seu.seu(weights[i], 32, 1, 0, p, t, 1)
        # print(len(weights), len(weights2))
        model.set_weights(weights2)
        test_loss_seu1, test_acc_seu1 = model.evaluate(x_test, y_test, verbose=1)
        test_loss_seu += test_loss_seu1
        test_acc_seu += test_acc_seu1
        # print(test_loss_seu1, test_acc_seu1)
        loss.append(test_loss_seu1)
        acc.append(test_acc_seu1)
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
    checkpoint_save_path_seu = "./.model/mnist_seu/LeNet.ckpt"
    checkpoint_save_path = checkpoint_save_path_noseu
    result_savepath_noseu = "./pic/noseu_noquant_test_1.png"
    result_savepath_seu = "./pic/seu_noquant_test_2.png"
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
    print(len(x_test))
    print(test_loss, test_acc)
    weights = model.get_weights()
    # print(len(model.get_weights()))

    with open("seu_mnist_lenet_weights.txt", "w") as f:
        for v in model.trainable_variables:
            f.write(str(v.name) + "\n")
            f.write(str(v.shape) + "\n")
            f.write(str(v.numpy()) + "\n")



    # p = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003]
    cyc = 20

    p = np.linspace(0, 0.02, cyc)
    loss = []
    accx = []
    accy = []
    t = 0
    for pp in tqdm.tqdm(p):
        t = t + 1
        lossl, accl = get_acc(model, weights, x_test, y_test, 40, pp, t)
        for j in (accl):
            accx.append(pp)
            accy.append(j)


    plt.figure()
    plt.scatter(accx, accy, c="blue", s = 30, alpha=0.5)
    plt.savefig(result_savepath)
    plt.show()