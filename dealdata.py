# import tensorflow as tf
# from creat_model import  LeNet
import numpy as np
import matplotlib.pyplot as plt
import re




def fic1():
    f = open("./data/data3.txt", "r")
    f2 = open("./data/data4.txt", "r")
    accy_ave_TMR0 = []
    accy_ave_TMR1 = []
    accy_ave_TMR2 = []
    accy_ave_TMR3 = []
    accy_ave_TMR4 = []
    accy_ave_TMR5 = []
    accy_ave_TMR6 = []
    accy_ave_TMR7 = []
    accy_ave_TMR8 = []

    for line in f:
        # print(type(line), line)
        data = re.findall(r"\d+\.\d+", line)
        for dd in range(len(data)):
            data[dd] = float(data[dd])
        accy_ave_TMR0.append(data[0])
        accy_ave_TMR1.append(data[1])
        accy_ave_TMR2.append(data[2])
        accy_ave_TMR3.append(data[3])
        accy_ave_TMR4.append(data[4])
        accy_ave_TMR5.append(data[5])

    for line in f2:
        data = re.findall(r"\d+\.\d+", line)
        for dd in range(len(data)):
            data[dd] = float(data[dd])
        accy_ave_TMR6.append(data[0])
        accy_ave_TMR7.append(data[1])
        accy_ave_TMR8.append(data[2])

    print(accy_ave_TMR0)
    goodlimit = 0.85
    badlimit = 0.3
    pointnum = 40
    cyc = 20
    year = np.linspace(0, 130, cyc)
    print(year)

    zerotmr_noave = []
    fulltmr_noave = []
    zerotmr_ave = []

    zerotmr_noave2 = []
    fulltmr_noave2 = []
    zerotmr_ave2 = []
    fulltmr_ave2 = []

    mm0 = []
    mm1 = []
    mm2 = []
    mm3 = []
    mm4 = []
    mm5 = []
    mm6 = []
    mm7 = []
    mm8 = []
    xstick = []
    #
    for i in range(cyc):
        # if (not((i == 0) or (i == 2) or (i == 10) or (i == 15) or (i == 19))):
        #     continue
        if (not (i % 2)):
            continue
        xstick.append("{:.0f}".format(year[i]))
        t0 = 0
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        t5 = 0
        t6 = 0
        t7 = 0
        t8 = 0
        bia = 1200
        # t11,t21,t31 = 0,0,0
        t41 = 0
        for j in range(pointnum):
            index  = i * pointnum + j
            index2 = index + bia
            if (accy_ave_TMR0[index2] > goodlimit):
                t0+=1
            if (accy_ave_TMR1[index2] > goodlimit):
                t1 += 1
            if (accy_ave_TMR2[index2] > goodlimit):
                t2 += 1
            if (accy_ave_TMR3[index2] > goodlimit):
                t3 += 1
            if (accy_ave_TMR4[index2] > goodlimit):
                t4 += 1
            if (accy_ave_TMR5[index2] > goodlimit):
                t5 += 1
            if (accy_ave_TMR6[index] > goodlimit):
                t6 += 1
            if (accy_ave_TMR7[index] > goodlimit):
                t7 += 1
            if (accy_ave_TMR8[index] > goodlimit):
                t8 += 1
        # print(t0,t1,t2,t3,t4,t5,t6,t7,t8)
        mm0.append(t0 / pointnum)
        mm1.append(t1 / pointnum)
        mm2.append(t2 / pointnum)
        mm3.append(t3 / pointnum)
        mm4.append(t4 / pointnum)
        mm5.append(t5 / pointnum)
        mm6.append(t6 / pointnum)
        mm7.append(t7 / pointnum)
        mm8.append(t8 / pointnum)
    print(xstick)
    # name = ("{}".format())
    penguin_means = {
        'All-layer-TMR':mm0,
        'IO-layer-TMR': mm8,
        'Conv-layer-TMR': mm6,
        'Dense-layer-TMR': mm7,

    }

    width = 0.2  # the width of the bars
    multiplier = 0
    x = np.arange(len(mm0))
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute)
        # plt.bar_label(rects, padding=3)
        multiplier += 1
    plt.legend(loc='center right')
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("probability of benign model [%]")
    plt.xticks(x + width * 2 , xstick)

    plt.subplot(122)
    width = 0.15  # the width of the bars
    multiplier = 0
    penguin_means2 = {
        'First-layer-TMR': mm1,
        'Third-layer-TMR': mm2,
        'Fifth-layer-TMR': mm3,
        'Sixth-layer-TMR': mm4,
        'Seventh-layer-TMR': mm5,
    }

    for attribute, measurement in penguin_means2.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute)
        # plt.bar_label(rects, padding=3)
        multiplier += 1
    plt.legend(loc='center right')
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("probability of benign model [%]")
    plt.xticks(x + width * 2 , xstick)


    # plt.figure(figsize=(10,5))
    # plt.subplot(121)
    # plt.plot(year, mm0, label="all")
    # plt.plot(year, mm1, label="1lay")
    # plt.plot(year, mm2, label="2lay")
    # plt.plot(year, mm3, label="3lay")
    # plt.plot(year, mm4, label="4lay")
    # plt.plot(year, mm5, label="5lay")
    #
    # plt.plot(year, mm6, label="covlay")
    # plt.plot(year, mm7, label="denslay")
    # plt.plot(year, mm8, label="iolay")
    #
    # plt.xlabel("duration in orbit [year]")
    # plt.ylabel("probability of benign model [%]")
    # plt.legend()
    #
    # plt.subplot(122)
    # plt.plot(year, mm6, label="covlay")
    # plt.plot(year, mm7, label="denslay")
    # plt.plot(year, mm8, label="iolay")
    # plt.xlabel("duration in orbit [year]")
    # plt.ylabel("probability of benign model [%]")
    # plt.legend()


    plt.savefig("./pic/benigncompare3.png")

    plt.show()

def fic2():
    f = open("./data/data2.txt", "r")
    accy_ave_TMR0 = []
    accy_ave_TMR1 = []
    accy_ave_TMR2 = []
    accy_ave_TMR3 = []
    accy_ave_TMR4 = []
    accy_ave_TMR5 = []
    accy_ave_TMR6 = []
    accy_ave_TMR7 = []
    accy_ave_TMR8 = []

    for line in f:
        # print(type(line), line)
        data = re.findall(r"\d+\.\d+", line)
        for dd in range(len(data)):
            data[dd] = float(data[dd])
        accy_ave_TMR0.append(data[0])
        accy_ave_TMR1.append(data[1])
        accy_ave_TMR2.append(data[2])
        accy_ave_TMR3.append(data[3])
        accy_ave_TMR4.append(data[4])
        accy_ave_TMR5.append(data[5])

    print(accy_ave_TMR0)
    goodlimit = 0.85
    badlimit = 0.3
    pointnum = 40
    cyc = 20
    year = np.linspace(0, 130, cyc)
    print(year)

if __name__ == "__main__":
    fic1()