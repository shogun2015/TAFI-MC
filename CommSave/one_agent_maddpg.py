# 模型训练的主代码
import csv
import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import argparse
import cv2
from shutil import copyfile
import matplotlib.pyplot as plt
from traffic_interaction_scene import TrafficInteraction
from traffic_interaction_scene import Visible
import time
from model_agent_maddpg import MADDPG
from replay_buffer import ReplayBuffer


def create_init_update(oneline_name, target_name, tau=0.99):
    """
    :param oneline_name: the online model name
    :param target_name: the target model name
    :param tau: The proportion of each transfer from the online model to the target model
    :return:
    """
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                     zip(online_var, target_var)]  # 按照比例用online更新target

    return target_init, target_update


def get_agents_action(sta, sess, agent, noise_range=0.0):
    """
    :param sta: the state of the agent
    :param sess: the session of tf
    :param agent: the model of the agent
    :param noise_range: the noise range added to the agent model output
    :return: the action of the agent in its current state
    """
    agent1_action = agent.action(state=[sta], sess=sess) + np.random.randn(1) * noise_range
    return agent1_action


def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update,
                agent_critic_target_update, sess, summary_writer, args):
    batch, w_id, eid, m_p = agent_memory.getBatch(
        args.batch_size)
    if not batch:
        return
    agent_num = args.o_agent_num + 1
    total_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))
    rew_batch = np.zeros((args.batch_size,))
    total_act_batch = np.zeros((args.batch_size, agent_num))
    total_next_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))
    next_state_mask = np.zeros((args.batch_size,))
    for k, (s0, a, r, s1, done) in enumerate(batch):
        total_obs_batch[k] = s0
        rew_batch[k] = r
        total_act_batch[k] = a
        if not done:
            total_next_obs_batch[k] = s1
            next_state_mask[k] = 1
    other_act_next = []
    other_act = []
    act_batch = np.array(total_act_batch[:, 0])  # 获取本agent动作集
    act_batch = act_batch.reshape(act_batch.shape[0], 1)
    for n in range(1, agent_num):
        other_act.append(total_act_batch[:, n])
        other_act_next.append(agent_ddpg_target.action(total_next_obs_batch[:, n, :], sess))
    other_act_batch = np.vstack(other_act).transpose()
    other_act_next = np.hstack(other_act_next)
    e_id = eid
    obs_batch = total_obs_batch[:, 0, :]  # 获取本agent当前状态集
    next_obs_batch = total_next_obs_batch[:, 0, :]  # 获取本agent下一状态集
    target = rew_batch.reshape(-1, 1) + args.gamma * agent_ddpg_target.Q(
        state=next_obs_batch, action=agent_ddpg_target.action(next_obs_batch, sess), other_action=other_act_next,
        sess=sess)
    td_error = abs(agent_ddpg_target.Q(
        state=obs_batch, action=act_batch, other_action=other_act_batch, sess=sess) - target)
    agent_memory.update_priority(e_id, td_error)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess,
                            summary_writer=summary_writer, lr=args.critic_lr)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess, summary_writer=summary_writer,
                           lr=args.actor_lr)
    sess.run([agent_actor_target_update, agent_critic_target_update])  # 从online模型更新到target模型


def train_agent_seq(agent_ddpg, agent_ddpg_target, agent_memory, agent_memory_sample, agent_actor_target_update,
                    agent_critic_target_update, sess, summary_writer, args):
    batch, w_id, eid, m_p = agent_memory.getBatch(
        args.batch_size)
    if not batch:
        return
    agent_num = args.o_agent_num + 1
    total_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))
    rew_batch = np.zeros((args.batch_size,))
    total_act_batch = np.zeros((args.batch_size, agent_num + 1))
    total_next_obs_batch = np.zeros((args.batch_size, agent_num, agent_num * 4))
    next_state_mask = np.zeros((args.batch_size,))
    for k, (s0, a, r, s1, done) in enumerate(batch):
        total_obs_batch[k] = s0
        rew_batch[k] = r
        total_act_batch[k] = a
        if not done:
            total_next_obs_batch[k] = s1
            next_state_mask[k] = 1
    other_act = []
    act_batch = np.array(total_act_batch[:, 0])  # 获取本agent动作集
    act_batch = act_batch.reshape(act_batch.shape[0], 1)
    for n in range(1, agent_num):
        other_act.append(total_act_batch[:, n])
    rem_actions = np.array(total_act_batch[:, -1])
    rem_actions = rem_actions.reshape(rem_actions.shape[0], 1)
    other_act_batch = np.vstack(other_act).transpose()
    e_id = eid
    obs_batch = total_obs_batch[:, 0, :]  # 获取本agent当前状态集

    target = rew_batch.reshape(-1, 1)

    # td_error = abs(agent_ddpg_target.Q(
    #     state=obs_batch, action=act_batch, other_action=other_act_batch, sess=sess) - target)
    # agent_memory.update_priority(e_id, td_error)

    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess,
                            summary_writer=summary_writer, lr=args.critic_lr)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, rem_actions=rem_actions, sess=sess,
                           summary_writer=summary_writer,
                           lr=args.actor_lr)

    # 更改经验生成
    actor_loss_error = abs(agent_ddpg.actor_loss_error(action=act_batch, rem_actions=rem_actions
                                                       , state_input=obs_batch, sess=sess))
    agent_memory.update_priority(e_id, actor_loss_error)
    agent_memory_sample.update_priority(e_id, actor_loss_error)

    sess.run([agent_actor_target_update, agent_critic_target_update])  # 从online模型更新到target模型

    batch, w_id, eid, m_p = agent_memory_sample.getBatch(1000)
    return m_p


def parse_args():
    parser = argparse.ArgumentParser("MADDPG experiments for multiagent traffic interaction environments")
    # Environment
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes")  # episode次数
    parser.add_argument("--o_agent_num", type=int, default=6, help="other agent numbers")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="learning rate for Adam optimizer")  # 学习率
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate for Adam optimizer")  # 学习率
    parser.add_argument("--gamma", type=float, default=0.90, help="discount factor")  # 折扣率
    parser.add_argument("--trans_r", type=float, default=0.998, help="transfer rate for online model to target model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of episodes to optimize at the same time")  # 经验采样数目
    parser.add_argument("--learn_start", type=int, default=20000,
                        help="learn start step")  # 经验采样数目
    parser.add_argument("--num_units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--collision_thr", type=float, default=2, help="the threshold for collision")
    parser.add_argument("--thr", type=float, default=-1, help="the threshold for collision")
    parser.add_argument("--actual_lane", action="store_true", default=False, help="")
    parser.add_argument("--seq_data", action="store_true", default=True, help="")
    parser.add_argument("--c_mode", type=str, default="closer",
                        help="the way of choosing closer cars, front ,front-end or closer")

    parser.add_argument("--model", type=str, default="MADDPG",
                        help="the model for training, MADDPG or DDPG")
    parser.add_argument("--thr_ratio", type=float, default=0.1,
                        help="the ratio of thr p")
    parser.add_argument("--init_thr", type=float, default=-2,
                        help="The initial of the thr")
    parser.add_argument("--thr_mean", action="store_true", default=False)

    # Checkpointing
    parser.add_argument("--exp_name", type=str, default="test", help="name of the experiment")  # 实验名
    parser.add_argument("--type", type=str, default="test", help="type of experiment train or test")
    parser.add_argument("--mat_path", type=str, default="./arvTimeNewVeh_300.mat", help="the path of mat file")
    parser.add_argument("--save_dir", type=str, default="model_data",
                        help="directory in which training state and model should be saved")  # 模型存储
    parser.add_argument("--save_rate", type=int, default=1,
                        help="save model once every time this many episodes are completed")  # 存储模型的回合间隔
    parser.add_argument("--itr", type=int, default=1,
                        help="the num of training after one step")  # 存储模型的回合间隔
    parser.add_argument("--load_dir", type=str, default="",
                        help="directory in which training state and model are loaded")  # 模型加载目录
    parser.add_argument("--video_name", type=str, default="",
                        help="if it not empty, program will generate a result video (.mp4 format defaultly)with the result imgs")
    parser.add_argument("--visible", action="store_true", default=False, help="visible or not")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)  # 恢复之前的模型，在 load-dir 或 save-dir
    parser.add_argument("--priori_knowledge", action="store_true", default=True)  # 用保存的模型跑测试
    parser.add_argument("--display", action="store_true", default=False)  # 将训练完成后的测试过程显示出来
    parser.add_argument("--benchmark", action="store_true", default=False)  # 用保存的模型跑测试
    parser.add_argument("--benchmark_iters", type=int, default=3000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")  # 训练曲线的目录
    return parser.parse_args()


def benchmark(model, arrive_time, sess):
    collisions_count = 0
    for mat_file in ["arvTimeNewVeh_new_2100.mat"]:
        data = scio.loadmat(mat_file)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        env = TrafficInteraction(arrive_time, 150, args, show_col=False, virtual_l=not args.actual_lane)
        for i in range(args.benchmark_iters):
            for lane in range(4):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"]
                    agent1_action = [[0]]
                    if veh["control"]:
                        agent1_action = get_agents_action(o_n[0], sess, model, noise_range=0)  # 模型根据当前状态进行预测
                    env.step(lane, ind, agent1_action[0][0])  # 环境根据输入的动作返回下一时刻的状态和奖励
                    # env.step(lane, ind, 0)  # 环境根据输入的动作返回下一时刻的状态和奖励
            ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh = env.scene_update()
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    collisions_count += 1
            if i % 50 == 0:
                print("i: %s collisions_rate: %s reward std: %s reward mean: %s" % (
                    i, float(collisions_count) / env.id_seq, np.std(reward), np.mean(reward)))

            env.delete_vehicle()
        print(
            "vehicle number: %s; collisions occurred number: %s; collisions rate: %s, MOE: %0.4f,"
            " pT-m: %0.4f s avg_jerk:%0.4f bad_scene:%0.4f" % (
                env.id_seq, collisions_count, float(collisions_count) / env.id_seq,
                env.total_moe / (env.passed_veh + 0.00001),
                float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT,
                env.jerk_total / (env.passed_veh + 0.00001), env.bad_scene / (env.passed_veh + 0.00001)))
    return float(collisions_count) / env.id_seq, float(env.passed_veh_step_total) / (
            env.passed_veh + 0.0001) * env.deltaT, env.jerk_total / (env.passed_veh + 0.00001)


def train():
    # data = scio.loadmat(args.mat_path)  # 加载.mat数据
    # arrive_time = data["arvTimeNewVeh"]
    # 建立Agent，Agent对应两个DDPG结构，一个是eval-net，一个是target-net
    agent1_ddpg = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr, nb_other_aciton=args.o_agent_num,
                         num_units=args.num_units, model=args.model)
    agent1_ddpg_target = MADDPG('agent1_target', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                nb_other_aciton=args.o_agent_num, num_units=args.num_units, model=args.model)
    saver = tf.train.Saver()  # 为存储模型预备
    agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1actor', 'agent1_targetactor',
                                                                              tau=args.trans_r)
    agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic',
                                                                                tau=args.trans_r)
    count_n = 0
    col = tf.Variable(0, dtype=tf.int8)
    collisions_op = tf.summary.scalar('collisions', col)
    etsm_col = tf.Variable(0, dtype=tf.int8)
    etsm_collisions_op = tf.summary.scalar('estimate_collisions', etsm_col)
    v_mean = tf.Variable(0, dtype=tf.float32)
    v_mean_op = tf.summary.scalar('v_mean', v_mean)
    collision_rate = tf.Variable(0, dtype=tf.float32)
    collision_rate_op = tf.summary.scalar('collision_rate', collision_rate)
    acc_mean = tf.Variable(0, dtype=tf.float32)
    acc_mean_op = tf.summary.scalar('acc_mean', acc_mean)
    reward_mean = tf.Variable(0, dtype=tf.float32)
    reward_mean_op = tf.summary.scalar('reward_mean', reward_mean)
    collisions_mean = tf.Variable(0, dtype=tf.float32)
    collisions_mean_op = tf.summary.scalar('collisions_mean', collisions_mean)
    estm_collisions_mean = tf.Variable(0, dtype=tf.float32)
    estm_collisions_mean_op = tf.summary.scalar('estm_collisions_mean', estm_collisions_mean)
    collisions_veh_numbers = tf.Variable(0, dtype=tf.int32)
    collisions_veh_numbers_op = tf.summary.scalar('collision_veh_numbers', collisions_veh_numbers)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init])
    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name)))
        print("load cptk file from " + tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name)))

    summary_writer = tf.summary.FileWriter(os.path.join(args.save_dir, args.exp_name), graph=tf.get_default_graph())

    # 设置经验池最大空间
    agent1_memory = ReplayBuffer(500000, args.batch_size, args.learn_start, 50000)
    agent1_memory_seq = ReplayBuffer(500000, args.batch_size, args.learn_start, 50000)
    agent1_memory_seq_sample = ReplayBuffer(500000, 1000, args.learn_start, 50000)
    # collisions_memory = ReplayBuffer(10000)
    reward_list = []
    collisions_list = []
    estm_collisions_list = []
    statistic_count = 0
    mean_window_length = 50
    state_now = []
    collisions_count = 0
    c_rate = 1.0
    rate_latest = 1.0
    test_rate_latest = 1.0
    # visible = Visible(lane_w=5)
    time_total = []
    seq_max_step = 32
    last_c_rate = 1
    last_pt_m = 100
    last_jerk = 1000
    model_count = 0
    txt_writer = None
    file_name = "priority_recorder" + args.mat_path[14:-4] + "_" + str(args.thr_ratio) + ".txt"
    txt_writer = open(file_name, "w")
    temp_m_p = []
    count_total_exp = 0
    count_update_exp = 0
    for epoch in range(args.num_episodes):
        collisions_count_last = collisions_count
        data = scio.loadmat(args.mat_path)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        env = TrafficInteraction(arrive_time, 150, args, vm=6, virtual_l=not args.actual_lane)
        if model_count > 5:
            saver.restore(sess, os.path.join(args.save_dir, args.exp_name, 'test_best.cptk'))
            print("load cptk file from " + os.path.join(args.save_dir, args.exp_name, 'test_best.cptk'))
        for i in range(20000):
            state_now.clear()
            for lane in range(4):
                for ind, veh in enumerate(env.veh_info[lane]):
                    o_n = veh["state"]
                    agent1_action = [[0]]
                    if veh["control"]:
                        count_n += 1
                        agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg,
                                                          noise_range=max(0.1, 1 - count_n * 1e-6))  # 模型根据当前状态进行预测
                        state_now.append(o_n)
                    env.step(lane, ind, agent1_action[0][0], thr=args.thr)
            ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh = env.scene_update()
            if args.seq_data:
                for seq, car_index in enumerate(ids):
                    env.veh_info[car_index[0]][car_index[1]]["buffer"].append(
                        [state_now[seq], actions[seq], reward[seq], state_next[seq],
                         env.veh_info[car_index[0]][car_index[1]]["Done"]])
                    if env.veh_info[car_index[0]][car_index[1]]["Done"] or env.veh_info[car_index[0]][car_index[1]][
                        "count"] > seq_max_step:
                        seq_data = env.veh_info[car_index[0]][car_index[1]]["buffer"]
                        if env.veh_info[car_index[0]][car_index[1]]["Done"]:
                            r_target = seq_data[-1][2]
                        else:
                            other_act_next = []
                            for n in range(1, args.o_agent_num + 1):
                                other_act_next.append(agent1_ddpg_target.action([seq_data[-1][3][n]], sess)[0][0])
                            r_target = seq_data[-1][2] + args.gamma * agent1_ddpg_target.Q(state=[seq_data[-1][3][0]],
                                                                                           action=agent1_ddpg_target.action(
                                                                                               [seq_data[-1][3][0]],
                                                                                               sess), other_action=[
                                    other_act_next], sess=sess)[0][0]
                        # 判断经验值大小加入

                        if temp_m_p:
                            actor_loss_error = abs(agent1_ddpg.actor_loss_error(
                                action=[[seq_data[-1][1][0]]], rem_actions=[[seq_data[-1][1][-1]]]
                                , state_input=seq_data[-1][0], sess=sess))
                            count_total_exp += 1
                            _m_p = temp_m_p[-1]
                            _m_p.sort()
                            # _thr = (temp_m_p[1] - temp_m_p[2]) * args.thr_ratio + temp_m_p[2]
                            if args.thr_mean:
                                _thr = int(_m_p[0])
                            else:
                                _thr = int(_m_p[int(len(_m_p) * args.thr_ratio)])
                            if actor_loss_error[0] > _thr or args.thr_ratio == 0:
                                agent1_memory_seq.add(np.array(seq_data[-1][0]), np.array(seq_data[-1][1]), r_target,
                                                      np.array(seq_data[-1][3]), False, args.init_thr)
                                agent1_memory_seq_sample.add(np.array(seq_data[-1][0]), np.array(seq_data[-1][1]), r_target,
                                                      np.array(seq_data[-1][3]), False, args.init_thr)
                                count_update_exp += 1
                        else:
                            agent1_memory_seq.add(np.array(seq_data[-1][0]), np.array(seq_data[-1][1]), r_target,
                                                  np.array(seq_data[-1][3]), False, args.init_thr)
                            agent1_memory_seq_sample.add(np.array(seq_data[-1][0]), np.array(seq_data[-1][1]), r_target,
                                                  np.array(seq_data[-1][3]), False, args.init_thr)
                            count_total_exp += 1
                            count_update_exp += 1
                        beta = 1
                        start = 0
                        tmp = 0
                        for cur_data in reversed(seq_data[:-1]):
                            r_target = cur_data[2] + args.gamma * beta * r_target
                            tmp += 1
                            if tmp > start or env.veh_info[car_index[0]][car_index[1]]["Done"]:
                                if temp_m_p:
                                    actor_loss_error = abs(agent1_ddpg.actor_loss_error(
                                        action=[[cur_data[1][0]]], rem_actions=[[cur_data[1][-1]]]
                                        , state_input=cur_data[0], sess=sess
                                    ))
                                    count_total_exp += 1
                                    _m_p = temp_m_p[-1]
                                    _m_p.sort()
                                    # _thr = (temp_m_p[1] - temp_m_p[2]) * args.thr_ratio + temp_m_p[2]
                                    if args.thr_mean:
                                        _thr = int(_m_p[0])
                                    else:
                                        _thr = int(_m_p[int(len(_m_p) * args.thr_ratio)])
                                    if actor_loss_error[0] > _thr or args.thr_ratio == 0:
                                        agent1_memory_seq.add(np.array(cur_data[0]), np.array(cur_data[1]), r_target,
                                                              np.array(cur_data[3]), False, args.init_thr)
                                        agent1_memory_seq_sample.add(np.array(cur_data[0]), np.array(cur_data[1]), r_target,
                                                              np.array(cur_data[3]), False, args.init_thr)
                                        count_update_exp += 1
                                else:
                                    agent1_memory_seq.add(np.array(cur_data[0]), np.array(cur_data[1]), r_target,
                                                          np.array(cur_data[3]), False, args.init_thr)
                                    agent1_memory_seq_sample.add(np.array(cur_data[0]), np.array(cur_data[1]), r_target,
                                                          np.array(cur_data[3]), False, args.init_thr)
                                    count_total_exp += 1
                                    count_update_exp += 1
                        # if len(env.veh_info[car_index[0]][car_index[1]]["buffer"]) > start:
                        #     env.veh_info[car_index[0]][car_index[1]]["buffer"] = \
                        #         env.veh_info[car_index[0]][car_index[1]]["buffer"][start:]
                        #     env.veh_info[car_index[0]][car_index[1]]["count"] = len(
                        #         env.veh_info[car_index[0]][car_index[1]]["buffer"])
                        # else:
                        env.veh_info[car_index[0]][car_index[1]]["buffer"] = []
                        env.veh_info[car_index[0]][car_index[1]]["count"] = 0
            reward_list += reward
            if len(collisions_per_veh) > 0:
                collisions_list += list(np.array(collisions_per_veh)[:, 0])
                estm_collisions_list += list(np.array(collisions_per_veh)[:, 1])
            reward_list = reward_list[-mean_window_length:]
            collisions_list = collisions_list[-mean_window_length:]
            estm_collisions_list = estm_collisions_list[-mean_window_length:]
            for k in range(len(actions)):
                if collisions_per_veh[k][0] > 0:
                    # collisions_memory.add(np.array(state_now[k]), np.array(actions[k]), reward[k],
                    #                       np.array(state_next[k]), False)
                    collisions_count += 1
                if not args.seq_data:
                    agent1_memory.add(np.array(state_now[k]), np.array(actions[k]), reward[k], np.array(state_next[k]),
                                      False)
            if count_n > args.learn_start:
                statistic_count += 1
                time_t = time.time()
                for s_t in range(args.itr):
                    if args.seq_data:
                        m_p = train_agent_seq(agent1_ddpg, agent1_ddpg_target, agent1_memory_seq, agent1_memory_seq_sample,
                                              agent1_actor_target_update, agent1_critic_target_update, sess,
                                              summary_writer,
                                              args)
                        temp_m_p = m_p
                        if m_p:
                            if txt_writer and isinstance(m_p[0], float):
                                txt_writer.write("%0.5f %0.5f %0.5f %s %s\n" % (
                                    m_p[0], m_p[1], m_p[2], count_total_exp, count_update_exp))
                    else:
                        train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory,
                                    agent1_actor_target_update, agent1_critic_target_update, sess, summary_writer, args)
                time_total.append(time.time() - time_t)
                if len(actions) > 0:
                    summary_writer.add_summary(sess.run(collisions_op, {col: collisions}), statistic_count)
                    summary_writer.add_summary(sess.run(etsm_collisions_op, {etsm_col: estm_collisions}),
                                               statistic_count)
                    summary_writer.add_summary(sess.run(v_mean_op, {v_mean: np.mean(np.array(state_next)[:, 0, 1])}),
                                               statistic_count)
                    summary_writer.add_summary(
                        sess.run(acc_mean_op, {acc_mean: np.mean(np.array(state_next)[:, 0, 2])}),
                        statistic_count)
                summary_writer.add_summary(sess.run(reward_mean_op, {reward_mean: np.mean(reward_list)}),
                                           statistic_count)
                summary_writer.add_summary(sess.run(collisions_mean_op, {collisions_mean: np.mean(collisions_list)}),
                                           statistic_count)
                summary_writer.add_summary(
                    sess.run(estm_collisions_mean_op, {estm_collisions_mean: np.mean(estm_collisions_list)}),
                    statistic_count)
                summary_writer.add_summary(
                    sess.run(collisions_veh_numbers_op, {collisions_veh_numbers: collisions_count}), statistic_count)
                if i % 100 == 0:
                    # train_summary _summary(train_summary, statistic_count)  # 调用train_writer的add_summary方法将训练过程以及训练步数保存
                    # summary_writer.add_summary(sess.run([merged], options=run_options, run_metadata=run_metadata),
                    #                            statistic_count)
                    print(
                        "reward mean: %s;epoch: %s;i: %s;count: %s;collisions_count: %s latest_c_rate: %s;"
                        "test best c_rate: %s;a-lr: %0.6f; c-lr: %0.6f; time_mean: %s" % (
                            np.mean(reward_list), epoch, i, count_n, collisions_count, rate_latest, test_rate_latest,
                            args.actor_lr, args.critic_lr, np.mean(time_total)))
            env.delete_vehicle()
        # if epoch % 10 == 0:
        if epoch % args.save_rate == 0:
            print('update model to ' + os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk'))
            saver.save(sess, os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk'))
            if rate_latest > (collisions_count - collisions_count_last) / float(env.id_seq):
                rate_latest = (collisions_count - collisions_count_last) / float(env.id_seq)
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.data-00000-of-00001'),
                    os.path.join(args.save_dir, args.exp_name, 'best.cptk.data-00000-of-00001'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.index'),
                    os.path.join(args.save_dir, args.exp_name, 'best.cptk.index'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.meta'),
                    os.path.join(args.save_dir, args.exp_name, 'best.cptk.meta'))
            summary_writer.add_summary(sess.run(collision_rate_op, {
                collision_rate: (collisions_count - collisions_count_last) / float(env.id_seq)}),
                                       epoch)
        if epoch % 1 == 0 and args.benchmark:
            bak_model = False
            c_rate, pt_m, jerk = benchmark(agent1_ddpg, arrive_time, sess)
            if c_rate < last_c_rate:
                last_c_rate = c_rate
                bak_model = True
            if c_rate == 0:
                if pt_m < last_pt_m and last_jerk - jerk > -20:
                    last_pt_m = pt_m
                    bak_model = True
                if jerk < last_jerk and last_pt_m - pt_m > -0.5:
                    last_jerk = jerk
                    bak_model = True
            if bak_model:
                model_count = 0
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.data-00000-of-00001'),
                    os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.data-00000-of-00001'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.index'),
                    os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.index'))
                copyfile(
                    os.path.join(args.save_dir, args.exp_name, str(epoch) + '.cptk.meta'),
                    os.path.join(args.save_dir, args.exp_name, 'test_best.cptk.meta'))
            else:
                model_count += 1
        if epoch % 5 == 4:
            args.actor_lr = args.actor_lr * 0.9
            args.critic_lr = args.critic_lr * 0.9
    txt_writer.close()
    sess.close()


def test():
    agent1_ddpg_test = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                              nb_other_aciton=args.o_agent_num, num_units=args.num_units)
    saver = tf.train.Saver()
    # summary
    # col = tf.Variable(0, dtype=tf.float32)
    # collisions_op = tf.summary.scalar('collisions', col)
    # etsm_col = tf.Variable(0, dtype=tf.float32)
    # etsm_collisions_op = tf.summary.scalar('estimate_collisions', etsm_col)
    # v_mean = tf.Variable(0, dtype=tf.float32)
    # v_mean_op = tf.summary.scalar('v_mean', v_mean)
    # acc_mean = tf.Variable(0, dtype=tf.float32)
    # acc_mean_op = tf.summary.scalar('acc_mean', acc_mean)
    # reward_1000 = tf.Variable(0, dtype=tf.float32)
    # reward_1000_op = tf.summary.scalar('agent1_reward_l1000_mean', reward_1000)
    # summary_writer = tf.summary.FileWriter(os.path.join(args.save_dir, args.exp_name), graph=tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, './three_ma_weight/40.cptk')
    # csv_file = open(args.csv_name + ".csv", "w", newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(["exp_name", "collision_rate", "pT-m", "jerk"])
    # visible = Visible(lane_w=5, control_dis=150, l_mode="actual", c_mode=args.c_mode)
    # mat_paths = ["arvTimeNewVeh_new_300.mat",
    #              "arvTimeNewVeh_new_600.mat",
    #              "arvTimeNewVeh_new_900.mat",
    #              "arvTimeNewVeh_new_1200.mat",
    #              "arvTimeNewVeh_new_1500.mat",
    #              "arvTimeNewVeh_new_1800.mat",
    #              "arvTimeNewVeh_new_2100.mat"
    #              ]
    mat_paths = ["arvTimeNewVeh_new_2100.mat"]
    # thrs = [3, 6, 9]
    thrs = [1]
    # densities = ["300", "900", "1500", "2100", "random"]
    densities = ["rule"]
    rows = []
    for thr in thrs:
        for density in densities:
            # args.exp_name = "train_only_model_%s_0419" % (density)
            model_path = os.path.join(args.save_dir, args.exp_name, "test_best.cptk")
            print("load cptk file from " + model_path)
            if not os.path.exists(model_path + ".meta"):
                model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.exp_name))
            print("load cptk file from " + model_path)
            saver.restore(sess, model_path)
            modes = [-1, 3]  # -1表示纯模型，thr表示结合规则
            for mode in modes:
                for mat_path in mat_paths:
                    # data = scio.loadmat(args.mat_path)  # 加载.mat数据
                    print("mat:", mat_path, "thr: ", mode)
                    data = scio.loadmat(mat_path)  # 加载.mat数据
                    arrive_time = data["arvTimeNewVeh"]
                    collisions_count = 0
                    time_total = []
                    # for epoch in range(1000):
                    # lens = [95, 120, 150, 170, 195]
                    lens = [150]
                    for l in lens:
                        # print("len: ", l)
                        env = TrafficInteraction(arrive_time, l, args, show_col=False, virtual_l=not args.actual_lane)
                        # env = TrafficInteraction(arrive_time, 150, args, vm=6, vM=20, v0=12)
                        size = (640, 480)
                        fps = 20
                        video_writer = cv2.VideoWriter()
                        if args.video_name != "":
                            video_writer = cv2.VideoWriter(os.path.join("results_img", args.video_name + ".avi"),
                                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
                        for i in range(6000):
                            for lane in range(4):
                                for ind, veh in enumerate(env.veh_info[lane]):
                                    o_n = veh["state"]
                                    agent1_action = [[0]]
                                    if veh["control"]:
                                        temp_t = time.time()
                                        agent1_action = get_agents_action(o_n[0], sess, agent1_ddpg_test,
                                                                          noise_range=0)  # 模型根据当前状态进行预测
                                        time_total.append(time.time() - temp_t)
                                    # agent1_action = [[0]]
                                    env.step(lane, ind, agent1_action[0][0], mode)  # 环境根据输入的动作返回下一时刻的状态和奖励
                                    # env.step(lane, ind, 0)  # 环境根据输入的动作返回下一时刻的状态和奖励
                            ids, state_next, reward, actions, collisions, estm_collisions, collisions_per_veh = env.scene_update()
                            for k in range(len(actions)):
                                if collisions_per_veh[k][0] > 0:
                                    collisions_count += 1
                            if i % 2000 == 0:
                                print("i: %s collisions_rate: %s reward std: %s reward mean: %s" % (
                                    i, float(collisions_count) / env.id_seq, np.std(reward), np.mean(reward)))
                            # if env.passed_veh >= 200:
                            #     print("mat_path:%s; passed_veh num:%s; pT-m: %0.4f s" % (args.mat_path,
                            #                                                              env.passed_veh,
                            #                                                              float(env.passed_veh_step_total) / (
                            #                                                                      env.passed_veh + 0.0001) * env.deltaT))
                            #     break
                            if (args.visible or args.video_name != "") and i > 4400:
                                visible.show(env, i)
                                img = cv2.imread("results_img/%s.png" % i)
                                cv2.putText(img, "frame: " + str(i), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                            (0, 0, 0),
                                            1)
                                cv2.putText(img, "veh: " + str(env.id_seq), (100, 120), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                            (0, 0, 0), 1)
                                cv2.putText(img, "c-veh: %s" % collisions_count, (100, 140), cv2.FONT_HERSHEY_COMPLEX,
                                            0.5,
                                            (0, 0, 255),
                                            1)
                                cv2.putText(img, "c-r: %0.4f" % (float(collisions_count) / env.id_seq), (100, 160),
                                            cv2.FONT_HERSHEY_COMPLEX,
                                            0.5, (0, 0, 255), 1)
                                cv2.putText(img, "p_veh: " + str(env.passed_veh), (100, 180), cv2.FONT_HERSHEY_COMPLEX,
                                            0.5,
                                            (0, 0, 0),
                                            1)
                                cv2.putText(img,
                                            "pT-m: %0.4f s" % (
                                                    float(env.passed_veh_step_total) / (
                                                    env.passed_veh + 0.0001) * env.deltaT),
                                            (100, 200), cv2.FONT_HERSHEY_COMPLEX,
                                            0.5, (0, 0, 0), 1)
                                if args.visible:
                                    cv2.imshow("result", img)
                                    cv2.waitKey(1)
                                if args.video_name != "":
                                    video_writer.write(img)
                            env.delete_vehicle()
                            # if i < 2000:
                            #     scio.savemat("test_mat.mat", {"veh_info": env.veh_info_record})
                        video_writer.release()
                        cv2.destroyAllWindows()
                        choose_veh_visible = False
                        if choose_veh_visible:
                            choose_veh_info = [np.array(item) for item in env.choose_veh_info]
                            plt.figure(0)
                            color = ['r', 'g', 'b', 'y']
                            y_units = ['distance [m]', 'velocity [m/s]', 'accelerate speed [m/s^2]']
                            titles = ["The distance of the vehicle varies with the time",
                                      "The velocity of the vehicle varies with the time",
                                      "The accelerate spped of the vehicle varies with the time"]
                            for m in range(len(y_units)):
                                for n in range(len(choose_veh_info)):
                                    if len(choose_veh_info[n]) > 0:
                                        plt.plot(choose_veh_info[n][:, 0], choose_veh_info[n][:, m + 1])
                                # plt.legend(["lane-0", "lane-1", "lane-2", "lane-3"])
                                plt.xlabel("time [s]")
                                plt.ylabel(y_units[m])
                                plt.title(titles[m], fontsize='small')
                                plt.savefig("exp_result_imgs/%s.png" % (y_units[m].split(" ")[0] + "_c_m"), dpi=600)
                                plt.close()
                        print(
                            "vehicle number: %s; collisions occurred number: %s; collisions rate: %s, MOE: %0.4f,"
                            " pT-m: %0.4f s avg_jerk:%0.4f bad_scene:%0.4f" % (
                                env.id_seq, collisions_count, float(collisions_count) / env.id_seq,
                                env.total_moe / (env.passed_veh + 0.00001),
                                float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT,
                                env.jerk_total / (env.passed_veh + 0.00001),
                                env.bad_scene / (env.passed_veh + 0.00001)))
                        exp_log = "%s_%s_%s" % (args.exp_name, str(mode == -1), mat_path.split(".")[0].split("_")[-1])
                        rows.append([exp_log, float(collisions_count) / env.id_seq,
                                     float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT,
                                     env.jerk_total / (env.passed_veh + 0.00001)])
    # csv_writer.writerows(rows)
    # csv_file.close()
    sess.close()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("results_img"):
        os.makedirs("results_img")
    if not os.path.exists("exp_result_imgs"):
        os.makedirs("exp_result_imgs")
    if not os.path.exists(os.path.join(args.save_dir, args.exp_name)):
        os.makedirs(os.path.join(args.save_dir, args.exp_name))
    with open(os.path.join(args.save_dir, args.exp_name, "args.txt"), "w") as fw:
        fw.write(str(args))
    if args.type == "train":
        train()
    else:
        test()
