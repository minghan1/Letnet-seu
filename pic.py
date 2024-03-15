import numpy as np
import matplotlib.pyplot as plt

f = open("./data/extrem_num.txt", "r")

for line in f:
    print(type(line), line)