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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"    # 使用第一, 三块GPU

def pTMR(p):
    return 3 * (p**2) - 2 * (p**3)


# injectsel:  0: alllay - TMR
#             1: lay0-1 - TMR
#             2: lay2-3 - TMR  3: lay4-5 - TMR 4: lay6-7 - TMR   5: lay8-9 - TMR
#             6:

#
def get_acc(model, weights, x_test, y_test ,test_times, p, t, bitlen, TMR, avevalid, injectsel):
    # weights = model.get_weights()
    weights2 = copy.deepcopy(weights)
    test_loss_seu, test_acc_seu = 0, 0
    loss = []
    acc = []
    TMR = 0
    for j in tqdm.tqdm(range(test_times)):
        # start_time = time.time()
        for i in range(len(weights)):
            # print(len(weights[i]), type(weights[i]), type(weights[i][0]))
            # print("----------level {} ------------".format(i))
            # x, bitlen, lamda, TMR, p, batchnum, avevalid
            lamda = 1
            p1 = p
            if (injectsel == 0):  #alllay
                p1 = pTMR(p)
                # weights2[i], cnt = fault_injection.seu(weights[i], bitlen, lamda, TMR, p1, t, avevalid)
            elif(injectsel == 1):
                if ((i == 0) or (i == 1)):
                    p1 = pTMR(p)
            elif (injectsel == 2):
                if ((i == 2) or (i == 3)):
                    p1 = pTMR(p)
            elif (injectsel == 3):
                if ((i == 4) or (i == 5)):
                    p1 = pTMR(p)
            elif (injectsel == 4):
                if ((i == 6) or (i == 7)):
                    p1 = pTMR(p)
            elif (injectsel == 5) :
                if ((i == 8) or (i == 9)):
                    p1 = pTMR(p)
            elif (injectsel == 6):
                if (i < 4):
                    p1 = pTMR(p)
            elif (injectsel == 7):
                if (i > 3):
                    p1 = pTMR(p)
            elif (injectsel == 8):
                if ((i == 1) or (i == 2) or (i == 8) or (i == 9)):
                    p1 = pTMR(p)
            weights2[i], cnt = fault_injection.seu(weights[i], bitlen, lamda, TMR, p1, t, avevalid)

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
    cyc = 20
    pointnum = 40
    p = np.linspace(0, 0.065, cyc)
    loss = []
    accx = []

    accy_ave_TMR0 = []
    accy_ave_TMR1 = []
    accy_ave_TMR2 = []
    accy_ave_TMR3 = []
    accy_ave_TMR4 = []
    accy_ave_TMR5 = []

    accy_ave_TMR6 = []
    accy_ave_TMR7 = []
    accy_ave_TMR8 = []
    t = 0

    bitlen = 32
    injectsel = 0
    avevalid = 1
    for pp in tqdm.tqdm(p):
        # t = t + 1
        t = 1
        # injectsel = 0
        # lossl0, accl0 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid, injectsel)
        # injectsel = 1
        # lossl1, accl1 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid,injectsel)
        # injectsel = 2
        # lossl2, accl2 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid,injectsel)
        # injectsel = 3
        # lossl3, accl3 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid,injectsel)
        # injectsel = 4
        # lossl4, accl4 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid,injectsel)
        # injectsel = 5
        # lossl5, accl5 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid, injectsel)
        #
        injectsel = 6
        lossl6, accl6 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid, injectsel)
        injectsel = 7
        lossl7, accl7 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid,injectsel)
        injectsel = 8
        lossl8, accl8 = get_acc(model, weights, x_test, y_test, pointnum, pp, t, bitlen, 0, avevalid,injectsel)
        timescale = 2000

        for j in range(len(accl6)):
            accx.append(pp * timescale)
            #
            # accy_ave_TMR0.append(accl0[j])
            # accy_ave_TMR1.append(accl1[j])
            # accy_ave_TMR2.append(accl2[j])
            # accy_ave_TMR3.append(accl3[j])
            # accy_ave_TMR4.append(accl4[j])
            # accy_ave_TMR5.append(accl5[j])

            accy_ave_TMR6.append(accl6[j])
            accy_ave_TMR7.append(accl7[j])
            accy_ave_TMR8.append(accl8[j])

    # 将数据写入txt文件
    # with open('./data/data3.txt', 'a') as f:
    #     for item1, item2, item3 ,item4, item5, item6 in \
    #     zip(
    #         accy_ave_TMR0,
    #         accy_ave_TMR1,
    #         accy_ave_TMR2,
    #         accy_ave_TMR3,
    #         accy_ave_TMR4,
    #         accy_ave_TMR5):
    #         f.write(f"{item1} {item2} {item3} {item4} {item5} {item6} \n")

    with open('./data/data4.txt', 'w') as f:
        for item1, item2, item3 in \
        zip(
            accy_ave_TMR6,
            accy_ave_TMR7,
            accy_ave_TMR8):
            f.write(f"{item1} {item2} {item3}  \n")

    result_savepath = "./pic3"
    plt.figure()
    plt.scatter(accx, accy_ave_TMR6, c="blue", s = 30, alpha=0.5, label='accuracy')
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(result_savepath + "/ave_TMR6.png")


    plt.figure()
    plt.scatter(accx, accy_ave_TMR7, c="blue", s = 30, alpha=0.5, label='accuracy')
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(result_savepath + "/ave_TMR7.png")

    plt.figure()
    plt.scatter(accx, accy_ave_TMR8, c="blue", s=30, alpha=0.5, label='accuracy')
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(result_savepath + "/ave_TMR8.png")




    # plt.show()