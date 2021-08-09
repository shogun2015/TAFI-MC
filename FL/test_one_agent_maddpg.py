# 模型训练的主代码
import numpy as np
import tensorflow as tf

import scipy.io as scio
import time
from traffic_interaction_scene import TrafficInteraction, Visible

from model_agent_maddpg import MADDPG
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                     zip(online_var, target_var)]  # 按照比例用online更新target

    return target_init, target_update


def get_agents_action(o_n, sess, noise_rate=0):
    agent1_action = agent1_ddpg.action(state=[o_n], sess=sess) + np.random.randn(1) * noise_rate
    return agent1_action


# 建立Agent，Agent对应两个DDPG结构，一个是eval-net，一个是target-net
agent1_ddpg = MADDPG('agent1', actor_lr=1e-4, critic_lr=1e-3)
agent1_ddpg_target = MADDPG('agent1_target', actor_lr=1e-4, critic_lr=1e-3)

saver = tf.train.Saver()  # 为存储模型预备

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

if __name__ == '__main__':
    mat_path = "./arvTimeNewVeh_600.mat"

    data = scio.loadmat(mat_path)  # 加载.mat数据
    arrive_time = data["arvTimeNewVeh"]
    diff = 99
    last = 0
    for i in arrive_time:
        if i[1] > 0:
            dis = i[0] - last
            last = i[0]
            if dis < diff:
                diff = dis
                print(diff)
    # exit(0)
    count_n = 0

    col = tf.Variable(0, dtype=tf.float32)
    collisions_op = tf.summary.scalar('collisions', col)
    etsm_col = tf.Variable(0, dtype=tf.float32)
    etsm_collisions_op = tf.summary.scalar('estimate_collisions', etsm_col)
    v_mean = tf.Variable(0, dtype=tf.float32)
    v_mean_op = tf.summary.scalar('v_mean', v_mean)
    acc_mean = tf.Variable(0, dtype=tf.float32)
    acc_mean_op = tf.summary.scalar('acc_mean', acc_mean)
    reward_1000 = tf.Variable(0, dtype=tf.float32)
    reward_1000_op = tf.summary.scalar('agent1_reward_l1000_mean', reward_1000)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init])

    summary_writer = tf.summary.FileWriter('./three_ma_summary_test_4_6_1', graph=tf.get_default_graph())
    # saver.restore(sess, './three_ma_weight/40.cptk')
    saver.restore(sess, 'D:\\maddpg_in_traffic_interaction\\model_data\\04_08_test_2\\37.cptk')

    # 设置经验池最大空间
    reward_1000_list = []
    state_now = []
    statistic_count = 0
    visible = Visible(lane_w=5, control_dis=150)
    collisions_count = 0
    veh_count = 0
    # for epoch in range(1000):
    env = TrafficInteraction(arrive_time, 150, show_col=False)
    for i in range(6000):
        state_now.clear()
        for lane in range(4):
            for ind, veh in enumerate(env.veh_info[lane]):
                o_n = veh["state"]
                agent1_action = [[0]]
                if veh["control"]:
                    count_n += 1
                    agent1_action = get_agents_action(o_n[0], sess, noise_rate=0)  # 模型根据当前状态进行预测
                    state_now.append(o_n)
                env.step(lane, ind, agent1_action[0][0])  # 环境根据输入的动作返回下一时刻的状态和奖励
                # env.step(lane, ind, 0)  # 环境根据输入的动作返回下一时刻的状态和奖励
        statistic_count += 1
        state_next, reward, actions, collisions, estm_collisions, collisions_per_veh = env.scene_update()
        reward_1000_list += reward
        reward_1000_list = reward_1000_list[-100:]
        for k in range(len(actions)):
            if collisions_per_veh[k][0] > 0:
                collisions_count += 1
        if i % 100 == 0:
            print("i: %s collisions_rate: %s" % (i, float(collisions_count) / env.id_seq))
        # visible.show(env, i)
        # cv2.imshow("result", cv2.imread("results_img/%s.png" % i))
        # cv2.waitKey(1)
    print("vehicle number: %s; collisions occurred number: %s; collisions rate: %s" % (
        env.id_seq, collisions_count, float(collisions_count) / env.id_seq))
    # summary_writer.add_summary(sess.run(collisions_op, {col: collisions}), statistic_count)
    # summary_writer.add_summary(sess.run(etsm_collisions_op, {etsm_col: estm_collisions}), statistic_count)
    # summary_writer.add_summary(sess.run(v_mean_op, {v_mean: np.mean(np.array(state_next)[:, 0, 1])}),
    #                            statistic_count)
    # summary_writer.add_summary(sess.run(acc_mean_op, {acc_mean: np.mean(np.array(state_next)[:, 0, 2])}),
    #                            statistic_count)
    #
    # summary_writer.add_summary(sess.run(reward_1000_op, {reward_1000: np.mean(reward_1000_list)}),
    #                                statistic_count)
    sess.close()
