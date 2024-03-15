import math
import fault_injection
import numpy as np
import copy

def generate_normal_distribution(n):
    mean = 0  # 正态分布的均值
    std = 1  # 正态分布的标准差
    result = np.random.normal(mean, std, n)
    return result.tolist()

def get_p(a):
    t = fault_injection.float32_to_bin(a)
    # tt = [copy.deepcopy(t) for i in range(len(t))]
    cnt = 0
    for i in range(len(t)):
        t_now = copy.deepcopy(t)
        t_now = list(t_now)
        t_now[i] = '0' if (t_now[i] == '1') else '1'
        k = fault_injection.bin_to_float("".join(t_now))
        if ((k == np.inf) or (np.isnan(a)) or (k > 2**20) or (k < -1 * 2**20)):
            cnt += 1
            print(k)
    cnt /= 32.0
    return cnt
def C(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
#
def calculate_formula(N, N1, p1, p2):
    result = 0
    for i in range(N1 + 1):
        result += C(N, i) * (p1 * p2) ** i * (1 - p1 * p2) ** (N - i)
    return 1 - result

# def calculate_formula(N, N1, p1, p2):
#     result = 0
#     result = C(N, N1) * (p1 * p2) ** N1 * (1 - p1 * p2) ** (N - N1)
#     return result

def pTMR3(p):
    return 2 * p**2 - 3 * p**3

def p1(p, bitlen):
    return 1 - C(bitlen, 0)*(1-p)**bitlen

p11 = [0.8 * x / 15 for x in range(15)]

if __name__ == "__main__":
    year = 10
    pp = year / 2 * 0.001
    p = calculate_formula(int(5e4), 10, p1(pTMR3(year / 2 * 0.001), 32),  0.025) * 1
    pP = calculate_formula(int(5e4), 10, p1(pp,32), 0.025) * 1
    # # #
    print(p, pP)
    pP2 = 0
    # print(1e38,1e35,-1.7e37,-6.9e300 + 1e300)
    # ans = 0
    # for x in range(15):
    #     pP = calculate_formula(int(5e4), x, year / 2 * 0.001, 0.025)
    #     # print(pP)
    #     pP2 += pP
    #     print(pP2)
    #     ans += pP * p11[x]
    # print(ans + 0.9 * 0.1)
    # a = generate_normal_distribution(10000)
    # cnt = 0
    # for i in a:
    #     t = get_p(i)
    #     print(t)
    #     cnt += t
    # print(cnt / 10000)
    #
