import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def np_move_avg(a, n, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode)


json_files = os.listdir("result_jsons/")
length = 500000
# 对比虚拟车道和实际车道中不同临近车数量的选择对碰撞车辆数量的影响
data_c = ["Actual-1", "Actual-3", "Actual-6", "Virtual-1", "Virtual-3", "Virtual-6"]
color = ["burlywood", "orange", "peru", "cyan", "skyblue", "dodgerblue"]
plt.figure(0)
for c in data_c:
    c = c.lower()
    json_f = None
    for j_f in json_files:
        if c in j_f:
            json_f = j_f
            break
    if json_f is not None:
        print("plotting %s" % c)
        json_path = os.path.join("result_jsons", json_f)
        collisions_veh_numbers = json.load(open(json_path, 'r'))
        collisions_veh_numbers_new_x = []
        collisions_veh_numbers_new_y = []
        for item in collisions_veh_numbers:
            if item[1] < length:
                collisions_veh_numbers_new_y.append(item[2])
                collisions_veh_numbers_new_x.append(item[1])
        plt.plot(collisions_veh_numbers_new_x, collisions_veh_numbers_new_y)
plt.legend(data_c)
plt.xlabel("steps")
plt.ylabel("the number of collisions")
plt.savefig("exp_result_imgs/collisions_num.png", dpi=600)
plt.close()

# for json_f in json_files:
#     json_path = os.path.join("result_jsons", json_f)
#     collisions_veh_numbers = json.load(open(json_path, 'r'))
#     collisions_veh_numbers_new_x = []
#     collisions_veh_numbers_new_y = []
#     for item in collisions_veh_numbers:
#         if item[1] < length:
#             collisions_veh_numbers_new_y.append(item[2])
#             collisions_veh_numbers_new_x.append(item[1])
#     plt.figure(1)
#     plt.plot(collisions_veh_numbers_new_x[
#              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)],
#              np_move_avg(collisions_veh_numbers_new_y, 50, mode="same")[
#              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
#     plt.xlabel("steps")
#     y_label = json_f[10:].split(".")[0]
#     plt.ylabel(y_label)
#     plt.title("The %s varies with the number of training steps" % y_label, fontsize='small')
#     plt.savefig("exp_result_imgs/%s.png" % y_label, dpi=600)
#     plt.close()
#
plt.figure(0)
sns.set(style="darkgrid", font_scale=1)
# co_jsons = ["run_.-tag-collision_veh_numbers.virtual-6.json"]
# dd_jsons = ["run_.-tag-collision_veh_numbers.ddpg-2.json"]
co_jsons = ["run_.-tag-actor_loss.maddpg-1.json"]
dd_jsons = ["run-.-tag-actor_loss.ddpg-2.json"]
cls = [co_jsons, dd_jsons]
col = ["r", "b"]
legend = ["CoMADDPG", "DDPG"]
for seq, jsons in enumerate(cls):
    y = []
    time = []
    for m_j in jsons:
        ma_json_path = os.path.join("result_jsons", m_j)
        collisions_veh_numbers = json.load(open(ma_json_path, 'r'))
        new_y = []
        new_x = []
        for item in collisions_veh_numbers:
            # if item[1] < length:
            new_y.append(item[2])
            new_x.append(item[1])
        y.append(np_move_avg(new_y, 50, mode="same")[
                 0:len(new_y) - int(len(new_y) / 4)])
        y.append(np_move_avg(new_y, 500, mode="same")[
                 0:len(new_y) - int(len(new_y) / 4)])
        y.append(np_move_avg(new_y, 100, mode="same")[
                 0:len(new_y) - int(len(new_y) / 4)])
        y.append(np_move_avg(new_y, 200, mode="same")[
                 0:len(new_y) - int(len(new_y) / 4)])
    # for i in range(len(y)):
    #     y[i] = new_y[:5] + list(y[i])
    time = new_x[0:len(new_y) - int(len(new_y) / 4)]
    # y.append(np_move_avg(collisions_veh_numbers_new_y, 10, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(collisions_veh_numbers_new_y, 50, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(collisions_veh_numbers_new_y, 1000, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(collisions_veh_numbers_new_y, 100, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(collisions_veh_numbers_new_y, 300, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(collisions_veh_numbers_new_y, 500, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(new_y, 10, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(new_y, 50, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(new_y, 1000, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(new_y, 100, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(new_y, 300, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    # y.append(np_move_avg(new_y, 500, mode="same")[
    #              0:len(collisions_veh_numbers_new_x) - int(len(collisions_veh_numbers_new_x) / 35)])
    sns.tsplot(time=time, data=y, color=col[seq])
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")
plt.legend(legend)
plt.xlabel("steps")
plt.ylabel("the number of collisions")
plt.savefig("exp_result_imgs/test.png", dpi=600)
# plt.close()
plt.show()
#
