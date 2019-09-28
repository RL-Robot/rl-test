# CartPole-v0 with tensorflow

import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

# hyperparameters
H = 10 # number of hidden layer neurons
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward

# model initialization
D = 4 # input dimensionality
C = 2 # class number

def policy_gradient():
    with tf.variable_scope("policy"):
        state = tf.placeholder(tf.float32, [None, D])
        actions = tf.placeholder(tf.int32, [None, 1])
        advantages = tf.placeholder(tf.float32, [None, 1])

        params_w1 = tf.get_variable("policy_parameters_w1",[D, H])
        params_b1 = tf.get_variable("policy_parameters_b1", [H])
        params_w2 = tf.get_variable("policy_parameters_w2",[H, C])
        params_b2 = tf.get_variable("policy_parameters_b2", [C])

        hidden = tf.nn.relu(tf.matmul(state, params_w1) + params_b1)
        probabilities = tf.nn.softmax(tf.matmul(hidden, params_w2) + params_b2)

        prob_given_state = tf.reduce_sum(-tf.log(probabilities) * tf.one_hot(actions, C), axis=0)
        loss = tf.reduce_mean(prob_given_state * advantages)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

def choose_action(prob):
    action = np.random.choice(range(len(prob)), p=prob)  # select action w.r.t the actions prob
    return action

env = gym.make("CartPole-v0")
policy_grad = policy_gradient()

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

reward_sum = 0
reward_trend = []
for episode_number in range(300):

    observation = env.reset()
    feed_states, feed_actions, feed_reward = [], [], []
    reward_sum = 0
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad

    for _ in range(300):
        state = observation #shape (D,)

        aprob = sess.run(pl_calculated, feed_dict={pl_state: np.reshape(state, (1, D))}) # aprob's shape: 1 * C
        action = choose_action(aprob[0]) # select an action based on policy gradient
        feed_states.append(state)

        feed_actions.append(action)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        feed_reward.append(reward)

        if done: # an episode finished
            if episode_number % 10 == 0:
                print ("episode is done")
                print ("reward_sum: {}".format(reward_sum))

            reward_trend.append(reward_sum)
            feed_advantages = discount_rewards(feed_reward) # compute discounted and normalized rewards

            sess.run(pl_optimizer, feed_dict={pl_state: np.vstack(feed_states), pl_advantages: np.vstack(feed_advantages), pl_actions: np.vstack(feed_actions)})
            break
plt.plot(reward_trend)
plt.show()