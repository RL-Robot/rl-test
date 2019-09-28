import gym
from dqn.RL_brain_DQN import DeepQNetwork

class CartPolev_0():
    def run_cart():
        restore_model
        RL.restore_model() # Restore save files
        total_steps = 0 # Recording steps
        for i_episode in range(50):
            # 獲取回合 i_episode 第一個 observation
            observation = env.reset()
            ep_r = 0
            while True:
                env.render()    # Refresh environment

                action = RL.choose_action(observation)  # Choose Action

                observation_, reward, done, info = env.step(action) # get next state

                x, x_dot, theta, theta_dot = observation_   # 细分开, 为了修改原配的 reward

                # x 是車的水平位移, 所以 r1 是車越偏離中心, 分越少
                # theta 是棒子離垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直,分越高

                x, x_dot, theta, theta_dot = observation_
                r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
                reward = r1 + r2   # The total reward is a combination of r1 and r2, considering both 
                                # position and angle, so DQN learning is more efficient

                # Save this set of memories
                RL.store_transition(observation, action, reward, observation_)

                if total_steps > 1000:
                    RL.learn()  # Learn

                ep_r += reward
                if done:
                    print('episode: ', i_episode,
                        'ep_r: ', round(ep_r, 2),
                        ' epsilon: ', round(RL.epsilon, 2))
                    break

                observation = observation_
                total_steps += 1

if __name__ == '__main__':
    env = gym.make('CartPole-v0')   # Declare what environment will be used
    env = env.unwrapped # 不做這個會有很多限制
    print(env.action_space) # 查看這個環境中可用的 action 有多少個
    print(env.observation_space) # 查看這個環境中可用的 state 的 observation 有多少個
    print(env.observation_space.high) # 查看 observation 最高取值
    print(env.observation_space.low) # 查看 observation 最低取值

    # 定義使用 DQN 的算法
    RL = DeepQNetwork(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.01, e_greedy=0.9,
                    replace_target_iter=100, memory_size=2000,
                    e_greedy_increment=0.0008,) 
    # 
    CartPolev_0.run_cart()          
    # outout cost figure
    RL.plot_cost()


