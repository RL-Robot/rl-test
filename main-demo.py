from maze_env import Maze
from dqn.RL_brain import DeepQNetwork


### RL is the class name of DeepQNetwork
### env is the class name of Maze
def run_maze():
    #step count
    step = 0
    # restore_model
    # RL.restore_model()
    for episode in range(300):
        # initial observation from maze
        observation = env.reset()
        while True:
            # refresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            #save step into memory as experience for future learning
            RL.store_transition(observation, action, reward, observation_)
           
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()