import scipy.io as scio
import matplotlib.pyplot as plt
import time
from traffic_interaction_scene import TrafficInteraction
mat_path = "/media/jiangmingzhi/Data/tie-matlab-emulator/arvTimeNewVeh_600.mat"

data = scio.loadmat(mat_path)  # 加载.mat数据
arrive_time = data["arvTimeNewVeh"]
env = TrafficInteraction(arrive_time, 150)
plt.figure(1)
px = [[] for i in range(4)]
py = [[] for i in range(4)]
for i in range(6000):
    for k in range(4):
        px[k].clear()
        py[k].clear()
    for lane in range(4):
        for veh_id, veh in enumerate(env.veh_info[lane]):
            env.step(lane, veh_id, 0)
    env.scene_update()
    for lane in range(4):
        for veh_id, veh in enumerate(env.veh_info[lane]):
            if lane == 0:
                px[0].append(veh["p"])
                py[0].append(-10)
            elif lane == 1:
                px[1].append(veh["p"])
                py[1].append(10)
            elif lane == 2:
                px[2].append(10)
                py[2].append(veh["p"])
            else:
                px[3].append(-10)
                py[3].append(veh["p"])
    print(px, py)
    plt.xlim((-160, 160))
    plt.ylim((-160, 160))
    time.sleep(0.1)
    if i % 10 == 0:
        print("-----")
        plt.plot(px[0], py[0], c='r', ls='', marker='4')  # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.plot(px[1], py[1], c='r', ls='', marker='3')  # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.plot(px[2], py[2], c='r', ls='', marker='2')  # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.plot(px[3], py[3], c='r', ls='', marker='1')  # 画出当前 ax 列表和 ay 列表中的值的图形

        plt.xlim((-160, 160))
        plt.ylim((-160, 160))
        plt.show()
    # plt.pause(0.001)  # 暂停一秒
# plt.ioff()  # 关闭画图的窗口
