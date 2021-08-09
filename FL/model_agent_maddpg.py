# 定义了单个Agent的DDPG结构，及一些函数

import tensorflow as tf
import tensorflow.contrib as tc


class MADDPG():
    def __init__(self, name, actor_lr, critic_lr, layer_norm=True, nb_actions=1, nb_other_aciton=3,
                 num_units=64, model="MADDPG"):
        nb_input = 4 * (nb_actions + nb_other_aciton)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_aciton], dtype=tf.float32)
        if model == "DDPG":
            other_action_input = tf.placeholder(shape=[None, 0], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # 输入是一个具体的状态state，经过两层的全连接网络输出选择的动作action
        def actor_network(name, state_input):
            with tf.variable_scope(name) as scope:
                x = state_input
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))  # 全连接层
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                # x = tf.nn.softmax(x)
                # x = tf.arg_max(x, 1)
                # x = tf.cast(tf.reshape(x, [-1, 1]), dtype=tf.float32)
                # bias = tf.constant(-30, dtype=tf.float32)
                w_ = tf.constant(3, dtype=tf.float32)
                # x = tf.multiply(tf.add(x, bias), w_)
                x = tf.multiply(tf.nn.tanh(x), w_)
            return x

        # 输入时 state，所有Agent当前的action信息
        def critic_network(name, state_input, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                x = state_input
                # x = tf.concat([x, action_input], axis=-1)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action_input], axis=-1)
                x = tf.layers.dense(x, num_units,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x

        self.state_input = state_input
        self.action_input = action_input
        self.other_action_input = other_action_input
        self.reward = reward
        self.action_output = actor_network(name + "actor", state_input=self.state_input)
        self.critic_output = critic_network(name + '_critic',
                                            action_input=tf.concat([self.action_input, self.other_action_input],
                                                                   axis=1), state_input=self.state_input)

        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        # 最大化Q值
        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic',
                           action_input=tf.concat([self.action_output, self.other_action_input], axis=1),
                           reuse=True, state_input=self.state_input))  # reduce_mean 为求均值，即为期望
        online_var = [i for i in tf.trainable_variables() if name + "actor" in i.name]
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss, var_list=online_var)
        # self.actor_train = self.actor_optimizer.minimize(self.actor_loss)
        self.actor_loss_op = tf.summary.scalar("actor_loss", self.actor_loss)
        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))  # 目标Q 与 真实Q 之间差的平方的均值
        self.critic_loss_op = tf.summary.scalar("critic_loss", self.critic_loss)
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)
        self.count = 0

    def train_actor(self, state, other_action, sess, summary_writer, lr):
        self.count += 1
        self.actor_lr = lr
        summary_writer.add_summary(
            sess.run(self.actor_loss_op, {self.state_input: state, self.other_action_input: other_action}), self.count)
        sess.run(self.actor_train, {self.state_input: state, self.other_action_input: other_action})

    def train_critic(self, state, action, other_action, target, sess, summary_writer, lr):
        self.critic_lr = lr
        summary_writer.add_summary(
            sess.run(self.critic_loss_op, {self.state_input: state, self.action_input: action,
                                           self.other_action_input: other_action,
                                           self.target_Q: target}), self.count)
        sess.run(self.critic_train,
                 {self.state_input: state, self.action_input: action, self.other_action_input: other_action,
                  self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action, self.other_action_input: other_action})


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    end_points = {}

    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device("/cpu:0"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        # should be [?, vocab_size+1]

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points

