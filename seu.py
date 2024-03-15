#using falut_injection file 
#dont use this !!!!!!!!!!

import numpy as np
import random
import struct


def seu_random(p):
    if random.random() < p:
        return 1
    else:
        return 0
def float32_to_bin(num):
    # 将32位浮点数打包为二进制数据
    packed_data = struct.pack('!f', num)

    # 将二进制数据转换为字符串表示
    bin_str = ''.join(format(byte, '08b') for byte in packed_data)

    return bin_str
def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]


# 对naddry x进行辐射注入， x的最小单元元素位宽为bitlen
# 每bit每个空间粒子翻转概率 为pro1, 两位是pro2，三位是pro3
# pro1,2,3 都是一维列表， 索引值为粒子能量 [3, 7, 23, 33, 36, 39, 82]
#  pardistri 粒子空间分布
#  time 在轨时长(p)
# lamda 是 空载的缩放系数
# TMR 表示是否进行关键位三模  强分别为0,1,2,3度
# batchnum 批次 几批注完
# avevalid
def seu(x, bitlen, lamda, TMR, p, batchnum, avevalid):
    y = np.ndarray.flatten(x)
    cnt = 0
    for batch in range(batchnum):
        for i in range(y.shape[0]):
            if (bitlen == 32):
                t = list(float32_to_bin(y[i]))
            else:
                t = list(bin(int(y[i]))[2:].rjust(bitlen).replace(" ", "0"))

            # print(t)
            for tt in range(len(t)):
                if (seu_random(p / batchnum)):
                    t[tt] = "0" if (t[tt] == "1") else "1"
                    cnt += 1

            if (bitlen == 32):
                y[i] = bin_to_float("".join(t))
                if(np.isnan(y[i])):
                    y[i] = 0.1
                elif(y[i] > 1):
                    y[i] = 1
                elif (y[i] < -1):
                    y[i] = -1
                if (np.isnan(y[i])):
                    print("-------------error {}, {}---------------".format(y[i], t))
            else:
                y[i] = int("".join(t))
    return np.reshape(y, x.shape), cnt

if __name__ == "__main__":
    a = np.array([[0.01, 0.2, 0.0003], [-0.4,-0.0005,-0.126]])
    b = seu(a, 32, 0, 0, 0.01,1)
    print(a)
    print(b)

    # tt = float32_to_bin(1.4e100)
    # print(tt, bin_to_float(tt))
    t = float(1.4e100)
    print(t)