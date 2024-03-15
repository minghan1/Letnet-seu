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

def pTMR(p):
    return 3 * (p**2) - 2 * (p**3)

def gg(x):
    y = x
    y[0] = 2
    return y

#inject fault to one layer
def float_fault_injection(x, lamda, TMR, p, batchnum, avevalid):    # p 和 batchnum 改成数量
    y = np.ndarray.flatten(x)
    cnt = 0
    for batch in range(batchnum):
        for i in range(y.shape[0]):
            t = list(float32_to_bin(y[i]))
            for tt in range(len(t)):
                if TMR == 0 or (TMR == 1 and tt != 0) or (TMR == 2 and tt not in [0, 1, 9]):
                    if seu_random(p / batchnum):
                        t[tt] = "0" if t[tt] == "1" else "1"
                        cnt += 1
                elif (TMR == 1 or TMR == 2 or TMR == 3):
                    if seu_random(pTMR(p / batchnum)):
                        # print(p/batchnum, pTMR(p))
                        t[tt] = "0" if t[tt] == "1" else "1"
                        cnt += 1

            y[i] = bin_to_float("".join(t))
            if np.isnan(y[i]):
                y[i] = 0.1
            if avevalid:
                if np.isnan(y[i]):
                    y[i] = 0.1
                elif y[i] > 1:
                    y[i] = 1
                elif y[i] < -1:
                    y[i] = -1
            if np.isnan(y[i]):
                print("-------------error {}, {}, {}---------------".format(y[i], avevalid, TMR))
    return np.reshape(y, x.shape), cnt


# 对naddry x进行辐射注入， x的最小单元元素位宽为bitlen
# 每bit每个空间粒子翻转概率 为pro1, 两位是pro2，三位是pro3
# pro1,2,3 都是一维列表， 索引值为粒子能量 [3, 7, 23, 33, 36, 39, 82]
#  pardistri 粒子空间分布
#  time 在轨时长(p)
# lamda 是空载的缩放系数
# TMR 表示是否进行关键位三模  强度分别为0,1,2,3
# batchnum 批次 几批注完
# avevalid 限幅加固
def seu(x, bitlen, lamda, TMR, p, batchnum, avevalid):
    if (bitlen == 32):
        return float_fault_injection(x, lamda, TMR, p, batchnum, avevalid)
 

if __name__ == "__main__":
    # a = np.array([[0.01, 0.2, 0.0003], [-0.4,-0.0005,-0.126]])
    # b = seu(a, 32, 0, 0, 0.01,1)
    # print(a)
    # print(b)

    # tt = float32_to_bin(1.4)
    # print(tt, bin_to_float(tt))
    # t = float(1.4e100)
    # print(tt)

    # x = []
    # x.append(5)
    # y =  gg(x)
    # print(x, y)

    # for i in range(100):
    #     print(np.random.randint(0, 10))

    t = float32_to_bin(-0.4)
    print(t, bin_to_float("11111110110011001100110011001101"))