import gym
import random
from RL_brain import DeepQNetwork
class Simulation():
        
    def run():
        step = 0
        RL.restore_model()
        for episode in range(10):
            observation = env.reset()
            ep_r = 0
            while True:
                
                env.render()
                
                action = RL.choose_action(observation)
                
                observation_, reward, done, info = env.step(action)
                
                x,x_dot,theta,theat_dot = observation_
                
                reward1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
                reward2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
                reward = reward1+reward2 
                ep_r+= reward
                
                RL.store_transition(observation, action, reward, observation_)
                
                if step > 1000:
                    RL.learn()
                
                observation = observation_
                
                
                if done:
                    print('episode :',episode,
                      'ep_r:',round(ep_r,2),
                      "RL's epsilon",round(RL.epsilon,3))
                    break  
                step += 1
                
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    RL = DeepQNetwork(n_actions = env.action_space.n,
                        n_features = env.observation_space.shape[0],
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    Simulation.run()
    RL.plot_cost()