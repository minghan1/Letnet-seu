import tensorflow as tf
# from creat_model import  LeNet
import numpy as np
import matplotlib.pyplot as plt
import re



f = open("./data/data2.txt", "r")
accy_noave_TMR0 = []
accy_noave_TMR1 = []
accy_noave_TMR2 = []
accy_noave_TMR3 = []
accy_ave_TMR0   = []
accy_ave_TMR1   = []
accy_ave_TMR2   = []
accy_ave_TMR3   = []

for line in f:
    # print(type(line), line)
    data = re.findall(r"\d+\.\d+", line)
    for dd in range(len(data)):
        data[dd] = float(data[dd])
    accy_noave_TMR0.append(data[0])
    accy_noave_TMR1.append(data[1])
    accy_noave_TMR2.append(data[2])
    accy_noave_TMR3.append(data[3])
    accy_ave_TMR0.append(data[4])
    accy_ave_TMR1.append(data[5])
    accy_ave_TMR2.append(data[6])
    accy_ave_TMR3.append(data[7])
print(accy_noave_TMR0)
goodlimit = 0.85
badlimit = 0.3
pointnum = 40
cyc = 20
year = np.linspace(0,60,cyc)
print(year)

def fic1():
    zerotmr_noave = []
    fulltmr_noave = []
    zerotmr_ave = []

    zerotmr_noave2 = []
    fulltmr_noave2 = []
    zerotmr_ave2 = []
    fulltmr_ave2 = []
    #
    for i in range(cyc):
        t1 = 0
        t2 = 0
        t3 = 0
        t11,t21,t31 = 0,0,0
        t41 = 0
        for j in range(pointnum):
            index  = i * pointnum + j
            if (accy_noave_TMR0[index] < badlimit):
                t1+=1
            if (accy_noave_TMR3[index] < badlimit):
                t2 += 1
            if (accy_ave_TMR0[index] < badlimit):
                t3 += 1
            if (accy_noave_TMR0[index] > goodlimit):
                t11 += 1
            if (accy_noave_TMR3[index] > goodlimit):
                t21 += 1
            if (accy_ave_TMR0[index] > goodlimit):
                t31 += 1
            if (accy_ave_TMR3[index] > goodlimit):
                t41 += 1
        zerotmr_noave.append(t1 / pointnum)
        fulltmr_noave.append(t2 / pointnum)
        zerotmr_ave.append(t3 / pointnum)

        zerotmr_noave2.append(t11 / pointnum)
        fulltmr_noave2.append(t21 / pointnum)
        zerotmr_ave2.append(t31 / pointnum)
        fulltmr_ave2.append(t41 / pointnum)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(year, zerotmr_noave, label="Nolimit-ZeroTMR")
    plt.plot(year, fulltmr_noave, label="Nolimit-FullTMR")
    plt.plot(year, zerotmr_ave, label="Limit-ZeroTMR")
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("probability of critical model [%]")
    plt.legend()

    plt.subplot(122)
    plt.plot(year, zerotmr_noave2, label="Nolimit-ZeroTMR")
    plt.plot(year, fulltmr_noave2, label="Nolimit-FullTMR")
    plt.plot(year, zerotmr_ave2, label="Limit-ZeroTMR")
    plt.plot(year, fulltmr_ave2, label="Limit-FullTMR")
    plt.xlabel("duration in orbit [year]")
    plt.ylabel("probability of benign model [%]")

    plt.legend()
    plt.savefig("./pic/criticalcompare.png")

    plt.show()

if __name__ == "__main__":
    fic1()