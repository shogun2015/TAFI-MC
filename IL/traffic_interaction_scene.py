import numpy as np
import matplotlib.pyplot as plt
import os
import copy as cp


class TrafficInteraction:
    # vm = 0; % minimum
    # velocity
    # v0 = 10; % initial
    # velocity
    # vM = 13; % maximum
    # velocity
    # am = -3; % minimum
    # acceleration
    # aM = 3; % maximum
    def __init__(self, arrive_time, dis_ctl, args, deltaT=0.1, vm=5, vM=13, am=-3, aM=3, v0=10, diff_max=220, lane_cw=5,
                 loc_con=True, show_col=False, virtual_l=True):
        # 坐标轴,车道0 从左到右, 车道1,从右到左 车道2,从下到上 车道3 从上到下
        #           dis_ctl
        #  -dis_ctl    0     dis_ctl
        #           -dis_ctl
        self.virtual_l = virtual_l
        self.show_col = show_col
        self.loc_con = loc_con
        self.collision_thr = args.collision_thr
        # self.choose_veh = [i for i in range(10, 20)]
        self.choose_veh = [5, 20]
        self.choose_veh_info = [[] for i in range(len(self.choose_veh))]
        self.veh_info_record = [[] for i in range(4)]
        self.vm = vm
        self.vM = vM
        self.am = am
        self.aM = aM
        self.v0 = v0
        self.thr = pow(self.vM - self.vm, 2) / 4 / self.aM + 2.2
        self.safe_distance = 10
        self.lane_cw = lane_cw
        self.closer_veh_num = args.o_agent_num
        self.c_mode = args.c_mode
        self.merge_p = [
            [0, 0, self.lane_cw, -self.lane_cw],
            [0, 0, -self.lane_cw, self.lane_cw],
            [-self.lane_cw, self.lane_cw, 0, 0],
            [self.lane_cw, -self.lane_cw, 0, 0]
        ]
        self.arrive_time = arrive_time
        self.current_time = 0
        self.passed_veh = 0
        self.passed_veh_step_total = 0
        self.jerk_total = 0
        self.total_moe = 0
        self.virtual_lane = []
        self.virtual_lane_4 = [[] for i in range(4)]
        self.closer_cars = []
        self.closer_same_l_car = [-1, -1]
        self.deltaT = deltaT
        self.dis_control = dis_ctl
        self.veh_num = [0, 0, 0, 0]  # 每个车道车的数量
        self.veh_rec = [0, 0, 0, 0]  # 每个车道车的总数量
        self.veh_info = [[] for i in range(4)]
        self.diff_max = diff_max
        self.collision = False
        self.id_seq = 0
        self.delete_veh = []
        self.args = args
        self.bad_scene = 0
        self.k = [[-0.679439, 0.029665, -0.000276, 1.49e-06]
            , [0.135273, 0.004808, -2.05e-05, 5.54e-08]
            , [0.015946, 8.33e-05, 9.37e-07, -2.48e-08]
            , [-0.001189, -6.13e-05, 3.04e-07, -4.47e-09]]
        init = True
        while init:
            for i in range(4):
                if self.veh_num[i] > 0:
                    init = False
            if init:
                self.scene_update()

    def scene_update(self):
        self.current_time += self.deltaT
        collisions = 0
        estm_collisions = 0
        re_state = []
        reward = []
        collisions_per_veh = []
        actions = []
        ids = []
        self.delete_veh.clear()
        for i in range(4):
            self.virtual_lane_4[i].clear()
            for _itr in self.virtual_lane:
                self.virtual_lane_4[i].append([_itr[0] + 2 * self.merge_p[_itr[1]][i], _itr[1], _itr[2]])
            self.virtual_lane_4[i] = sorted(self.virtual_lane_4[i], key=lambda item: item[0])  # 对虚拟车道的车辆重新通过距离进行排序
            for j, item in enumerate(self.veh_info[i]):
                try:
                    self.veh_info[i][j]["collision"] = 0
                    self.veh_info[i][j]["estm_collision"] = 0
                except Exception:
                    continue
                t_distance = 2
                d_distance = 100
                if self.veh_info[i][j]["control"]:
                    self.veh_info_record[i][item["seq_in_lane"]].append(
                        [self.current_time, item["p"], item["v"], item["a"]]
                    )
                    sta = self.get_state(i, j)
                    self.veh_info[i][j]["state"] = cp.deepcopy(sta)
                    re_state.append(np.array(sta))
                    actions.append([state[3] for state in sta])
                    actions[-1] += [self.veh_info[i][j]["rem_a"]]
                    ids.append([i, j])
                    self.veh_info[i][j]["count"] += 1
                    # reward.append(
                    #     self.veh_info[i][j]["step"] * (-1) - self.veh_info[i][j][
                    #         "collision"] * 50 + self.passed_veh * 10)

                    dis_front = 100
                    id_seq = [item[1:3] for item in self.virtual_lane_4[i]]
                    cur_index = id_seq.index([i, j])
                    if cur_index == 0:
                        self.veh_info[i][j]["header"] = True
                    else:
                        dis_front = self.virtual_lane_4[i][cur_index][0] - \
                                    self.virtual_lane_4[i][cur_index - 1][0]
                        if i == self.virtual_lane_4[i][cur_index - 1][1] and \
                                self.virtual_lane_4[i][cur_index][0] > 30:
                            self.veh_info[i][j]["dis_front"] = dis_front
                        else:
                            self.veh_info[i][j]["dis_front"] = 50
                    acc_reward = 0
                    if dis_front < self.safe_distance:
                        acc_reward = min(np.log(pow(min(dis_front / self.safe_distance, 1.5), 12) + 0.0000001) * \
                                         (self.veh_info[i][j]["a"] + 0.1), 120)
                    self.veh_info[i][j]["a_r"] = self.veh_info[i][j]["a_r"] * 0.8 + acc_reward * 0.2
                    # reward.append(-1 * self.veh_info[i][j][
                    #     "collision"] * 50 - 10 * abs(sta[0][2]))
                    closer_car = self.closer_cars[0]
                    if closer_car[0] >= 0:
                        self.veh_info[i][j]["closer_p"] = self.veh_info[closer_car[0]][closer_car[1]]["p"] + 2 * \
                                                          self.merge_p[closer_car[0]][i]
                        d_distance = self.veh_info[i][j]["p"] + self.merge_p[i][closer_car[0]] - (
                                self.veh_info[closer_car[0]][closer_car[1]]["p"] + self.merge_p[closer_car[0]][i])
                        if not self.virtual_l:
                            if i != closer_car[0]:
                                d_distance = np.sqrt(
                                    np.power(self.veh_info[i][j]["p"] + self.merge_p[i][closer_car[0]], 2) + np.power(
                                        self.veh_info[closer_car[0]][closer_car[1]]["p"] + self.merge_p[closer_car[0]][
                                            i], 2))
                        if d_distance != 0:
                            t_distance = (self.veh_info[i][j]["p"] + self.merge_p[i][closer_car[0]] - (
                                    self.veh_info[closer_car[0]][closer_car[1]]["p"] + self.merge_p[closer_car[0]][
                                i])) / (self.veh_info[i][j]["v"] - self.veh_info[closer_car[0]][closer_car[1]][
                                "v"] + 0.0001)
                    else:
                        self.veh_info[i][j]["closer_p"] = 150
                    # reward.append(
                    #     np.log(pow(d_distance / 10, 5) + 0.00001) + 0.1 * np.log(
                    #         pow(t_distance / (10 * self.deltaT), 2) + 0.00001) + (self.veh_info[i][j]["v"] - self.v0))
                    # reward.append(
                    #     min(20, max(-20, np.log(pow(d_distance / 10, 5) + 0.00001) + (1 / np.tanh(-t_distance)))))
                    # d_safe_thr = 10
                    # same_l_dis = -20
                    # if self.closer_same_l_car[0] != -1:
                    #     same_l_dis = self.veh_info[i][j]["p"] - \
                    #                  self.veh_info[self.closer_same_l_car[0]][self.closer_same_l_car[1]]["p"]
                    #     if max(1, d_distance) / same_l_dis < -0.1 and self.veh_info[i][j]["p"] > 50:
                    #         t_distance = same_l_dis / (self.veh_info[i][j]["v"] -
                    #                                    self.veh_info[self.closer_same_l_car[0]][
                    #                                        self.closer_same_l_car[1]][
                    #                                        "v"] + 0.0001)
                    #         d_distance = same_l_dis
                    if abs(d_distance) < 10:
                        r_ = -3
                        if abs(d_distance) < 3:
                            r_ = -10
                        if t_distance > 0:
                            r_ -= 5
                    elif 10 > t_distance > 0:
                        r_ = -3
                    else:
                        r_ = self.veh_info[i][j]["v"]
                    reward.append(r_ / 10)
                    if 0 < t_distance < 1:
                        t_reward = -pow(1.5 / np.tanh(-t_distance), 2)
                    else:
                        t_reward = 2
                    d_reward = np.log(pow(d_distance / 10, 10) + 0.0001)
                    self.veh_info[i][j]["t_r"] = self.veh_info[i][j]["t_r"] * 0.8 + t_reward * 0.2
                    self.veh_info[i][j]["d_r"] = self.veh_info[i][j]["d_r"] * 0.8 + d_reward * 0.2
                    # if self.veh_info[i][j]["seq_in_lane"] in self.choose_veh:
                    #     self.choose_veh_info[self.choose_veh.index(self.veh_info[i][j]["seq_in_lane"])].append(
                    #         [self.current_time, self.veh_info[i][j]["p"], self.veh_info[i][j]["v"],
                    #          self.veh_info[i][j]["a"]])
                    re_b = min(30, max(-30, 0.2 * acc_reward + d_reward + t_reward))
                    if self.veh_info[i][j]["seq_in_lane"] in self.choose_veh:
                        self.choose_veh_info[self.choose_veh.index(self.veh_info[i][j]["seq_in_lane"])].append(
                            [self.current_time, self.veh_info[i][j]["d_r"], re_b,
                             self.veh_info[i][j]["a_r"]])
                    self.veh_info[i][j]["reward"] = re_b
                    if 0 <= closer_car[0] != i and self.virtual_l:
                        d_distance = np.sqrt(
                            np.power(self.veh_info[i][j]["p"] + self.merge_p[i][closer_car[0]], 2) + np.power(
                                self.veh_info[closer_car[0]][closer_car[1]]["p"] + self.merge_p[closer_car[0]][
                                    i], 2))
                        # print(i, closer_car[0], "distance: ", self.veh_info[i][j]["p"],
                        #       self.veh_info[closer_car[0]][closer_car[1]]["p"], "bias: ",
                        #       self.merge_p[i][closer_car[0]], self.merge_p[closer_car[0]][i], "dis: ", d_distance)
                    if abs(d_distance) < self.collision_thr:
                        self.veh_info[i][j]["collision"] += 1  # 发生碰撞
                    # if self.veh_info[i][j]["estm_collision"] == 0:
                    #     reward[-1] += 5
                    if self.veh_info[i][j]["finish"]:
                        self.veh_info[i][j]["control"] = False
                    #     reward[-1] = 20
                    collisions += self.veh_info[i][j]["collision"]
                    estm_collisions += self.veh_info[i][j]["estm_collision"]
                    collisions_per_veh.append([self.veh_info[i][j]["collision"], self.veh_info[i][j]["estm_collision"]])
                if (i == 0 or i == 2) and not self.loc_con:
                    if self.veh_info[i][j]["p"] > self.dis_control or self.veh_info[i][j]["collision"] > 0:
                        # 驶出交通路口, 删除该车辆
                        self.veh_info[i][j]["Done"] = True
                        self.delete_veh.append([i, j])
                    elif self.veh_info[i][j]["p"] > self.lane_cw and self.veh_info[i][j]["control"]:
                        # 驶出控制区
                        self.veh_info[i][j]["Done"] = True
                        self.veh_info[i][j]["finish"] = True
                        self.veh_info[i][j]["control"] = False
                        self.passed_veh += 1
                        self.passed_veh_step_total += self.veh_info[i][j]["step"]
                        self.jerk_total += self.veh_info[i][j]["jerk"] / self.veh_info[i][j]["step"]

                else:
                    if self.veh_info[i][j]["p"] < -self.dis_control or self.veh_info[i][j]["collision"] > 0:
                        # 驶出交通路口, 删除该车辆
                        self.delete_veh.append([i, j])
                        self.veh_info[i][j]["Done"] = True
                        if self.veh_info[i][j]["control"]:
                            reward[-1] = -10
                    elif self.veh_info[i][j]["p"] < - self.lane_cw and self.veh_info[i][j]["control"]:
                        reward[-1] = 10
                        self.veh_info[i][j]["Done"] = True
                        self.veh_info[i][j]["finish"] = True
                        self.veh_info[i][j]["control"] = False
                        self.passed_veh += 1
                        self.passed_veh_step_total += self.veh_info[i][j]["step"]
                        self.bad_scene += self.veh_info[i][j]["bad_scene"]
                        self.total_moe += self.veh_info[i][j]["MOE"]
                        self.jerk_total += self.veh_info[i][j]["jerk"] / self.veh_info[i][j]["step"]
            # 添加新车
            self.add_new_veh(i)
            # if self.show_col:
            #     print("add new car:", i, self.veh_num[i] - 1)
        self.virtual_lane.clear()
        return ids, re_state, reward, actions, collisions, estm_collisions, collisions_per_veh

    def add_new_veh(self, i):
        if self.current_time >= self.arrive_time[self.veh_rec[i]][i]:
            state = []
            state_total = np.zeros((self.closer_veh_num + 1, (self.closer_veh_num + 1) * 4))
            j = self.veh_num[i]
            if (i == 0 or i == 2) and not self.loc_con:
                p = -self.dis_control
            else:
                p = self.dis_control
            self.closer_cars.clear()
            # if self.virtual_l:
            self.virtual_lane_search_closer(i, j, mode=self.c_mode, veh_num=self.closer_veh_num)
            # else:
            #     if i in [0, 2]:
            #         self.closer_cars.append(self.get_other_state(i, p, i, j))
            #         self.closer_cars.append(self.get_other_state((i + 2) % 4, p, i, j))
            #         self.closer_cars.append(self.get_other_state((i + 3) % 4, p, i, j))
            #     else:
            #         self.closer_cars.append(self.get_other_state(i, p, i, j))
            #         self.closer_cars.append(self.get_other_state((i + 1) % 4, p, i, j))
            #         self.closer_cars.append(self.get_other_state((i + 2) % 4, p, i, j))
            for num, car in enumerate(self.closer_cars):
                if car[0] != -1:
                    car_temp = self.veh_info[car[0]][car[1]]
                    state += [car_temp["p"] + 2 * self.merge_p[car[0]][i], car_temp["v"], car_temp["action"],
                              car[0] == i]
                    state_total[num + 1] = np.array(self.veh_info[car[0]][car[1]]["state"][0][:])
                else:
                    state += [0, 0, 0, 0]
                    state_total[num + 1] = np.array([0 for m in range(4 * (self.closer_veh_num + 1))])
            state = [p, self.v0, 0, 0] + state[:]
            state_total[0] = np.array(state[:])
            self.veh_info[i].append(
                {
                    "p": p,
                    "jerk": 0,
                    "a_diff": 0,
                    "buffer": [],
                    "a_r": 0,
                    "d_r": 0,
                    "t_r": 0,
                    "rem_a": 0,
                    "count": 0,
                    "bad_scene": 0,
                    "v": self.v0,
                    "Done": False,
                    "a": 0,
                    "MOE": 0,
                    "action": 0,
                    "lane": i,
                    "seq_in_lane": self.veh_rec[i],
                    "control": True,
                    "state": state_total,
                    "step": 0,
                    "closer_p": 150,
                    "reward": 10,
                    "dis_front": 100,
                    "collision": 0,
                    "finish": False,
                    "estm_collision": 0,
                    "estm_arrive_time": abs(p / self.v0),
                    "id_info": [self.id_seq, self.veh_num[i]],
                })
            # "id_info":[在所有车中的出现次序,在当前车道中的出现次序]
            self.veh_num[i] += 1
            self.veh_rec[i] += 1
            self.veh_info_record[i].append([])
            self.id_seq += 1

    def delete_vehicle(self):
        # 删除旧车
        self.delete_veh = sorted(self.delete_veh, key=lambda item: -item[1])
        for d_i in self.delete_veh:
            if len(self.veh_info[d_i[0]]) > d_i[1]:
                self.veh_info[d_i[0]].pop(d_i[1])
                if self.veh_num[d_i[0]] > 0:
                    self.veh_num[d_i[0]] -= 1
            else:
                print("except!!!")

    def get_state(self, i, j):
        state = []
        state_total = np.zeros((self.closer_veh_num + 1, (self.closer_veh_num + 1) * 4))
        p = self.veh_info[i][j]["p"]
        # if self.virtual_l:
        self.virtual_lane_search_closer(i, j, mode=self.c_mode, veh_num=self.closer_veh_num)
        # else:
        #     if i in [0, 2]:
        #         self.closer_cars.append(self.get_other_state(i, p, i, j))
        #         self.closer_cars.append(self.get_other_state((i + 2) % 4, p, i, j))
        #         self.closer_cars.append(self.get_other_state((i + 3) % 4, p, i, j))
        #     else:
        #         self.closer_cars.append(self.get_other_state(i, p, i, j))
        #         self.closer_cars.append(self.get_other_state((i + 1) % 4, p, i, j))
        #         self.closer_cars.append(self.get_other_state((i + 2) % 4, p, i, j))
        for num, car in enumerate(self.closer_cars):
            if car[0] != -1:
                car_temp = self.veh_info[car[0]][car[1]]
                state += [car_temp["p"] + 2 * self.merge_p[car[0]][i], car_temp["v"], car_temp["a"], car_temp["action"]]
                state_total[num + 1] = np.array(self.veh_info[car[0]][car[1]]["state"][0][:])
            else:
                state += [0, 0, 0, 0]
                state_total[num + 1] = np.array([0 for m in range(4 * (self.closer_veh_num + 1))])
        state = [p, self.veh_info[i][j]["v"], self.veh_info[i][j]["a"], self.veh_info[i][j]["action"]] + state[:]
        state_total[0] = np.array(state[:])
        return state_total

    def virtual_lane_search_closer(self, i, j, mode="front", veh_num=3):
        id_seq = [item[1:] for item in self.virtual_lane_4[i]]
        if [i, j] not in id_seq:
            index = -1
        else:
            index = id_seq.index([i, j])
        self.closer_cars.clear()
        self.closer_same_l_car = [-1, -1]
        if index >= 0:
            if mode == "front":  # 搜寻前车
                for k in range(index - 1, -1, -1):
                    veh_info = id_seq[k]
                    lane_id = veh_info[0]  # 获取车道id
                    if i + lane_id not in [1, 5]:  # 不添加临近车道
                        self.closer_cars.append(veh_info)
                    if len(self.closer_cars) >= veh_num:
                        break
            elif mode == "front-back":
                for k in range(index - 1, -1, -1):
                    veh_info = id_seq[k]
                    lane_id = veh_info[0]  # 获取车道id
                    if i + lane_id not in [1, 5]:  # 不添加临近车道
                        self.closer_cars.append(veh_info)
                    if len(self.closer_cars) >= veh_num - int(veh_num / 2):
                        break
                for k in range(index + 1, len(id_seq)):
                    veh_info = id_seq[k]
                    lane_id = veh_info[0]  # 获取车道id
                    if i + lane_id not in [1, 5]:  # 不添加临近车道
                        self.closer_cars.append(veh_info)
                    if len(self.closer_cars) >= veh_num / 2:
                        break
            elif mode == "closer":
                virtual_lane_abs = []
                for _itr in self.virtual_lane_4[i]:
                    flag = 1
                    # if _itr[1] != i:
                    #     flag += abs(_itr[0] - self.virtual_lane_4[i][index][0])/5.0
                    virtual_lane_abs.append(
                        [abs(_itr[0] - self.virtual_lane_4[i][index][0]) * flag, _itr[1], _itr[2], _itr[0]])
                virtual_lane_abs = sorted(virtual_lane_abs,
                                          key=lambda item: item[0])  # 对虚拟车道的车辆重新通过距离进行排序
                for _id, _itr in enumerate(virtual_lane_abs):
                    if [_itr[1], _itr[2]] != [i, j] and len(self.closer_cars) < veh_num and i + _itr[1] not in [1, 5]:
                        if _itr[1] != i and abs(
                                _itr[3] / self.veh_info[_itr[1]][_itr[2]]["v"] - self.virtual_lane_4[i][index][0] /
                                self.veh_info[i][j]["v"]) > 0.5:
                            continue
                        self.closer_cars.append([_itr[1], _itr[2], _itr[3]])
                        if _itr[1] == i and self.closer_same_l_car[0] == -1:
                            self.closer_same_l_car = [_itr[1], _itr[2]]
        for k in range(veh_num - len(self.closer_cars)):
            self.closer_cars.append([-1, -1])

    def get_other_state(self, i, dis, z_i, z_j):
        """
        获取最密切车辆的状态
        :param i: 车道序号
        :param dis: 本车的距离交叉口中心点的距离
        :param z_i, z_j: 当前车的序号
        :return: 位置(取几何坐标还是距离交叉口中心的距离(绝对值)??, 暂定后者) 速度 加速度 车道序号
        """
        diff = 9999
        seq = -1
        dis_temp = 9999
        estm_collision = False
        for ind, veh in enumerate(self.veh_info[i]):
            if veh["control"]:
                if i != z_i:  # 不是同一车道
                    dis_temp = np.sqrt(np.power(dis, 2) + np.power(veh["p"], 2))
                    if z_j < self.veh_num[z_i]:
                        arv_time_diff = abs(
                            self.veh_info[z_i][z_j]["estm_arrive_time"] - self.veh_info[i][ind]["estm_arrive_time"])
                        if arv_time_diff < 1 * self.deltaT:
                            # 迅速情况下会发生碰撞
                            estm_collision = True
                elif ind != z_j:  # 同一车道但是不是同一辆车
                    dis_temp = abs(abs(dis) - abs(veh["p"]))
                    if z_j < self.veh_num[z_i]:
                        if ind > z_j:  # 后进入
                            if (self.veh_info[z_i][z_j]["estm_arrive_time"] - self.veh_info[i][ind][
                                "estm_arrive_time"]) > -1 * self.deltaT:
                                # 迅速情况下会发生碰撞
                                estm_collision = True
                        else:  # 该车先进入
                            if (self.veh_info[z_i][z_j]["estm_arrive_time"] - self.veh_info[i][ind][
                                "estm_arrive_time"]) < 1 * self.deltaT:
                                # 迅速情况下会发生碰撞
                                estm_collision = True
                if dis_temp < diff:
                    diff = dis_temp
                    seq = ind
                if estm_collision:
                    self.veh_info[z_i][z_j]["estm_collision"] += 1
                    if self.show_col:
                        print("estimate collision occurred!!", [z_i, z_j], [i, ind])
                    estm_collision = False
                if z_j < self.veh_num[z_i]:
                    if diff < 2:
                        # 发生碰撞!!!!
                        self.veh_info[z_i][z_j]["collision"] += 1
                        if self.show_col:
                            print("collision occurred!!", [z_i, z_j], [i, seq])
        if diff < self.diff_max and seq != -1:
            return [i, seq]
        else:
            return [-1, -1]

    def judge_fb(self, i, j):
        #  函数功能：判断最邻近车辆在后面还是前面
        back = True
        closer_p = self.veh_info[i][j]["closer_p"]
        if closer_p < self.veh_info[i][j]["p"]:
            back = False
        return back

    def step(self, i, j, eval_a, thr=3):
        self.veh_info[i][j]["action"] = eval_a
        action_a = self.veh_info[i][j]["a"] + eval_a
        rcd_a = self.veh_info[i][j]["reward"] / float(thr)
        back = self.judge_fb(i, j)
        if back:
            rcd_a = abs(rcd_a)
        eval_a = rcd_a
        if self.veh_info[i][j]["reward"] < -2:
            if self.veh_info[i][j]["control"]:
                self.veh_info[i][j]["bad_scene"] += 1
            # if rcd_a * action_a > 0:
            #     eval_a = max(rcd_a, action_a)
        if self.args.priori_knowledge:
            # eval_a = rcd_a
            if self.veh_info[i][j]["p"] > 50:
                if j > 0 and self.veh_info[i][j - 1]["p"] > 0 and self.veh_info[i][j]["p"] - \
                        self.veh_info[i][j - 1]["p"] < self.thr:
                    # if self.veh_info[i][j + 1]["p"] - self.veh_info[i][j]["p"] < 6:
                    if self.veh_num[i] > j + 1 and self.veh_info[i][j + 1]["p"] - self.veh_info[i][j][
                        "p"] < self.thr:
                        eval_a = 0
                    else:
                        eval_a = self.am
                elif self.veh_num[i] > j + 1 and self.veh_info[i][j + 1]["p"] - self.veh_info[i][j][
                    "p"] < self.thr:
                    eval_a = self.aM
        rule_a = eval_a
        if self.args.type == "train" and self.args.model == "IL":
            thr = -1

        if thr == -1:
            eval_a = action_a
        else:
            eval_a = action_a * 0.5 + eval_a * 0.5
        # eval_a = (eval_a * 0.4 + self.veh_info[i][j]["a"] * 0.6)
        # if self.veh_info[i][j]["reward"] > -2:
        #     eval_a = 0
        # if self.veh_inf                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               o[i][j]["v"] == self.vM and eval_a > 0:
        #     eval_a = 0
        target_a = min(self.aM, max(self.am, eval_a))
        if len(self.virtual_lane_4[i]) > 0 and self.virtual_lane_4[i][0][1:3] == [i, j]:
            target_a = self.aM
        self.veh_info[i][j]["jerk"] += ((target_a - self.veh_info[i][j]["a"]) / self.deltaT) ** 2
        self.veh_info[i][j]["a_diff"] = (target_a - self.veh_info[i][j]["a"]) ** 2
        self.veh_info[i][j]["a"] = target_a
        if (i == 0 or i == 2) and not self.loc_con:
            self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] + self.veh_info[i][j]["v"] * self.deltaT + 0.5 * \
                                       self.veh_info[i][j]["a"] * pow(self.deltaT, 2)
        else:
            self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] - self.veh_info[i][j]["v"] * self.deltaT - 0.5 * \
                                       self.veh_info[i][j]["a"] * pow(self.deltaT, 2)
        self.veh_info[i][j]["v"] = min(self.vM,
                                       max(self.veh_info[i][j]["v"] + self.veh_info[i][j]["a"] * self.deltaT, self.vm))
        self.veh_info[i][j]["estm_arrive_time"] = abs(self.veh_info[i][j]["p"] / self.veh_info[i][j]["v"])
        self.veh_info[i][j]["step"] += 1
        if not self.veh_info[i][j]["control"]:
            self.veh_info[i][j]["a"] = self.aM  # 出交叉口之后所有车的速度都变回初始速度
        else:
            for m in range(4):
                for n in range(4):
                    self.veh_info[i][j]["MOE"] += self.k[m][n] * pow(self.veh_info[i][j]["v"], m) * pow(
                        self.veh_info[i][j]["a"], n)
            self.virtual_lane.append([self.veh_info[i][j]["p"], i, j])
        self.veh_info[i][j]["rem_a"] = rule_a


class Visible:
    def __init__(self, lane_w=5, control_dis=150, l_mode="actual", c_mode="front-back"):
        plt.figure(1)
        self.px = [[] for i in range(4)]
        self.py = [[] for i in range(4)]
        self.lane_w = lane_w
        self.color_m = np.zeros((4, 433)) - 1
        self.l_mode = l_mode
        self.c_mode = c_mode
        self.control_dis = control_dis

    def show(self, env, i):
        c_c = ["g", "b", "y", "brown", "gray", "deeppink"]
        point = []
        for k in range(4):
            self.px[k].clear()
            self.py[k].clear()
        if self.l_mode == "actual":
            for lane in range(4):
                for veh_id, veh in enumerate(env.veh_info[lane]):
                    if lane == 0:
                        self.px[0].append(-veh["p"])
                        self.py[0].append(-self.lane_w)
                    elif lane == 1:
                        self.px[1].append(veh["p"])
                        self.py[1].append(self.lane_w)
                    elif lane == 2:
                        self.px[2].append(self.lane_w)
                        self.py[2].append(-veh["p"])
                    else:
                        self.px[3].append(-self.lane_w)
                        self.py[3].append(veh["p"])
            plt.plot(self.px[0], self.py[0], c='r', ls='', marker='4')  # 画出当前 ax 列表和 ay 列表中的值的图形
            plt.plot(self.px[1], self.py[1], c='r', ls='', marker='3')  # 画出当前 ax 列表和 ay 列表中的值的图形
            plt.plot(self.px[2], self.py[2], c='r', ls='', marker='2')  # 画出当前 ax 列表和 ay 列表中的值的图形
            plt.plot(self.px[3], self.py[3], c='r', ls='', marker='1')  # 画出当前 ax 列表和 ay 列表中的值的图形
        elif self.l_mode == "virtual":
            for lane in range(2, 3):
                for item in env.virtual_lane_4[lane]:
                    if item[1] != 1 and [item[1], item[2]] not in point:
                        if item[1] == lane:
                            count = item[2]
                            color = c_c[count % len(c_c)]
                            env.virtual_lane_search_closer(item[1], item[2], mode=self.c_mode, veh_num=6)
                            dis = 99
                            if env.closer_cars[0][0] != -1:
                                id_seq = [item[1:] for item in env.virtual_lane_4[lane]]
                                index = id_seq.index(env.closer_cars[0])
                                # dis = abs(item[0] - env.virtual_lane_4[0][index][0])
                                dis = np.sqrt(
                                    np.power(abs(env.veh_info[item[1]][item[2]]["p"]) + env.merge_p[item[1]][
                                        env.closer_cars[0][0]], 2) + np.power(
                                        abs(env.veh_info[env.closer_cars[0][0]][env.closer_cars[0][1]]["p"]) +
                                        env.merge_p[env.closer_cars[0][0]][item[1]], 2))
                                if item[1] == env.virtual_lane_4[lane][index][1]:
                                    dis = abs(item[0] - env.virtual_lane_4[lane][index][0])
                                if dis < 1:
                                    color = "r"
                                plt.plot(self.lane_w, -env.virtual_lane_4[lane][index][0], c=color, ls='',
                                         marker=str(4 - env.virtual_lane_4[lane][index][1]))  # 画出当前 ax 列表和 ay 列表中的值的图
                                point.append(id_seq[index])
                            plt.plot(self.lane_w, -item[0], c=color, ls='',
                                     marker=str(4 - item[1]))  # 画出当前 ax 列表和 ay 列表中的值的图形
                            plt.text(self.lane_w + 10, -item[0], "%0.1f" % dis)
                        else:
                            plt.plot(self.lane_w, -item[0], c='black', ls='',
                                     marker=str(4 - item[1]))  # 画出当前 ax 列表和 ay 列表中的值的图形
                            # self.px[item[1]].append(-item[0])
                            # self.py[item[1]].append(-self.lane_w)
        plt.plot([-self.control_dis, self.control_dis], [0, 0], c='y', ls='--')
        plt.plot([0, 0], [self.control_dis, -self.control_dis], c='y', ls='--')
        plt.plot([-self.control_dis, self.control_dis], [2 * self.lane_w, 2 * self.lane_w], c='b', ls='-')
        plt.plot([2 * self.lane_w, 2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
        plt.plot([-self.control_dis, self.control_dis], [-2 * self.lane_w, -2 * self.lane_w], c='b', ls='-')
        plt.plot([-2 * self.lane_w, -2 * self.lane_w], [self.control_dis, -self.control_dis], c='b', ls='-')
        # plt.plot(self.px[0], self.py[0], c='r', ls='', marker='4')  # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.plot(self.px[1], self.py[1], c='r', ls='', marker='3')  # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.plot(self.px[2], self.py[2], c='r', ls='', marker='2')  # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.plot(self.px[3], self.py[3], c='r', ls='', marker='1')  # 画出当前 ax 列表和 ay 列表中的值的图形

        plt.xlim((-self.control_dis + 5, self.control_dis + 5))
        plt.ylim((-self.control_dis + 5, self.control_dis + 5))
        if not os.path.exists("results_img"):
            os.makedirs("results_img")
        plt.savefig("results_img/%s.png" % i)
        plt.close()
        # plt.show()
