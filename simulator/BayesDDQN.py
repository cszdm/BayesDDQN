import numpy as np
import pandas as pd
import random
import csv
import heapq
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import itertools
from functools import reduce
from bayes_opt import BayesianOptimization
import operator
import copy
import math
from scipy.linalg import solve
from sklearn.preprocessing import MinMaxScaler

class DoubleDeepQNetwork():
    def __init__(self, n_actions, n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 e_greedy_increment=None,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 output_graph=False,
                 double_q=True,
                 sess=None
                 ):
        self.n_actions = n_actions
        print(n_actions)
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q

        self.__build_net()

        # Get Parameters to be Updated
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

       
        self.learn_step_counter = 0
        self.cost_his = []

        # Start Session
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def __build_net(self):
        n_h1 = 1000  # Hidden Layer
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(
            0.1)  # config of layers

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_h1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_h1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_h1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # ------------------ loss & optimizer ------------------
        with tf.variable_scope('loss'):
            # ('bb',self.q_target)
            # print('cc',self.q_eval)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            # print('dd',self.loss)
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input

        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_h1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_h1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_h1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)  # Returns the indices of the maximum values along an axis

        if not hasattr(self, 'q'):  # 记录选的 Qmax 值
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)

        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # replace the old memory with new memory
        transition = np.hstack((s, [a, r], s_))
        print(transition)
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
        print('当前存储的信息:', self.memory)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\n' + 'target_params_replaced' + '\n')

      
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,
                                            size=self.batch_size)  
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:], 
                       self.s: batch_memory[:, -self.n_features:]})  
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_action_index = batch_memory[:, self.n_features].astype(
            int) 
        reward = batch_memory[:, self.n_features + 1]
        # Update Q Matrix
        if self.double_q:  
            max_act4next = np.argmax(q_eval4next, axis=1)  
            selected_q_next = q_next[batch_index, max_act4next]  
        else:  
            selected_q_next = np.max(q_next, axis=1)  

        q_target[batch_index, eval_action_index] = reward + self.gamma * selected_q_next

        # train eval network
        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1





