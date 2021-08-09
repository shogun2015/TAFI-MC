import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_300 = pd.read_table("priority_recordernew_300_4_l.txt", header=None)
data_600 = pd.read_table("priority_recordernew_600_4_l.txt", header=None)
data_900 = pd.read_table("priority_recordernew_900_4_l.txt", header=None)
data_1200 = pd.read_table("priority_recordernew_1200_4_l.txt", header=None)
data_1800 = pd.read_table("priority_recordernew_1800_l.txt", header=None)
data_random = pd.read_table("priority_recordernew_random_density_single_intersection.txt", header=None)

# line = data_300[0][0].split(' ')
# print(float(line[0]))
for data in [data_300, data_600, data_900, data_1200, data_1800, data_random]:
    mean_list = []
    max_list = []
    min_list = []
    for i in range(len(data[0])):
        line = data[0][i].split(' ')
        mean_list.append(float(line[0]))
        max_list.append(float(line[1]))
        min_list.append(float(line[2]))

    x_list = list(range(len(mean_list)))

    plt.plot(x_list, mean_list, label='mean')
    plt.plot(x_list, max_list, label='max')
    plt.plot(x_list, min_list, label='min')
    plt.legend()
    plt.show()


