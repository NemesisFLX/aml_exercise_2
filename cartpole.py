#
# cartpole.py, exercise sheet 2, Advanced Machine Learning course, RWTH Aachen University, summer term 2019, Jonathon Luiten
#

import gym
# render a dummy environment before importing tensorflow to circumvent tensorflow/openai-gym integration bug
g_env = gym.make('CartPole-v0')
g_env.render()

import tensorflow as tf
import random
import numpy as np

num_training_episodes = 1000
episode_length = 200

# Init Q Matrix 
Qt = np.full((10,10,10,10,3), 0)

def run_episode( env, sess ):
    
    observation = env.reset()
    episode_return = 0




    for _ in range( episode_length ):
        
        # random policy
        #action = 0 if random.uniform(0,1) < 0.5 else 1
        #observation, reward, done, info = env.step(action)
        
        # Q policy
        action
        observation, reward, done, info = env.step(action)
        
        episode_return += reward

        # disable rendering for faster training
        env.render()
        
        if done:
            print("episode ended early")
            break
        
    print("episode return: %f"%(episode_return,))

    return episode_return

def calc

env = gym.make('CartPole-v0')
env.render()
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range( num_training_episodes ):
    
    episode_return = run_episode( env, sess )

    
monitor.close()

