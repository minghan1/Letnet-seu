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
import random
from scipy.optimize import curve_fit

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"    # 禁用gpu

# 自定义拟合函数
def func(x, a, b, c):
    return c - b * np.exp(-1 * a * x)

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
    result_savepath_noseu = "./pic/weights_distri.png"
    result_savepath_seu = "./pic/seu_weights_distri.png"
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
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    weights = model.get_weights()
    # print(test_loss, test_acc)
    # model.summary()
    # print()
    extreme_num = [1.7e34,1.6e36, -1.8e33, -2.4e36]

    # indexx = []
    kk = 8
    la = [1,3,5,6,7]  #weight layer
    for kk in range(0,10,2):
        acc = []
        num = []
        cyc = 10
        nnn = 50
        for j in tqdm.tqdm(range(nnn + 1)):
            accj = []
            index = [np.random.randint(kk,kk+2) for k in range(j)]
            # print(index)
            for mm in range(cyc):
                weights2 = copy.deepcopy(weights)
                for i in range(j):
                    y = np.ndarray.flatten(weights[index[i]])
                    t = np.random.randint(0, y.shape[0])
                    y[t] = extreme_num[int(random.randint(0, 3))]
                    weights2[index[i]] = np.reshape(y, weights[index[i]].shape)
                model.set_weights(weights2)
                ls, accc = model.evaluate(x_test[0:3000], y_test[0:3000], verbose=0, batch_size=64)
                accj.append(accc)
            acc.append(accj)
            num.append(j)
        # print(acc)
        for ii in range(len(acc)):
            # print("{}: \n".format(i), acc[i])
            t = 0
            print(acc[ii])
            for k in acc[ii]:
                if k < 0.4:
                    t += 1
            acc[ii] = t
            acc[ii] /= cyc
        plt.plot(num, acc, label="layer[{}]".format(la[kk//2]))
        # popt, pcov = curve_fit(func, num, acc)
        with open("./data/extrem_num.txt", "w") as f:
            for x, y in zip(num, acc):
                f.write("{} {}\n".format(x, y))
        x_fit = np.linspace(1, nnn, nnn)
        # y_fit = func(x_fit, *popt)
        # plt.plot(x_fit,y_fit,color="green")
        plt.xlabel("number of extreme weights")
        plt.ylabel("The probability of critical model")
        plt.legend()
        plt.savefig("./pic/model_extreme_num" + "{}.png".format(kk))
    # plt.show()