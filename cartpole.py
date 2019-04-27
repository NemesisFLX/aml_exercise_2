#
# cartpole.py, exercise sheet 2, Advanced Machine Learning course, RWTH Aachen University, summer term 2019, Jonathon Luiten
#

import gym
# render a dummy environment before importing tensorflow to circumvent tensorflow/openai-gym integration bug
g_env = gym.make('CartPole-v0')
g_env.render()

import tensorflow as tf
from math import floor
import random
import numpy as np

num_training_episodes = 1000
episode_length = 200
e = 0.1
alpha = 0.1
discount = 0.1

# Init Q Matrix 
Qt = np.full((10,10,10,10,2), 0, dtype=np.float64)

def run_episode( env, sess ):
    
    observation = env.reset()
    episode_return = 0




    for _ in range( episode_length ):
        
        # random policy
        #action = 0 if random.uniform(0,1) < 0.5 else 1
        #observation, reward, done, info = env.step(action)
        
        # Q policy
        cart_position_t0 = floor((observation[0] + 2.4)/0.48)
        cart_velocity_t0 = floor((observation[1] + 5)/1)
        pole_angle_t0 = floor((observation[2] + 0.20943951)/0.041887902)
        pole_velocity_t0 = floor((observation[3] + 5)/1)
        
        possibleActions = Qt[cart_position_t0][cart_velocity_t0][pole_angle_t0][pole_velocity_t0].tolist()
        if random.uniform(0,1) < e:
            action = 0 if random.uniform(0,1) < 0.5 else 1
        else:
            action = possibleActions.index(max(possibleActions))
            
        observation, reward, done, info = env.step(action)
        
        cart_position_t1 = floor((observation[0] + 2.4)/0.48)
        cart_velocity_t1 = floor((observation[1] + 5)/1)
        pole_angle_t1 = floor((observation[2] + 0.20943951)/0.041887902)
        pole_velocity_t1 = floor((observation[3] + 5)/1)

        Qt[cart_position_t0][cart_velocity_t0][pole_angle_t0][pole_velocity_t0][action] = (1 - alpha) * possibleActions[action] + alpha * (reward + discount * Qt[cart_position_t1][cart_velocity_t1][pole_angle_t1][pole_velocity_t1].max() - Qt[cart_position_t0][cart_velocity_t0][pole_angle_t0][pole_velocity_t0][action]) 
        episode_return += reward

        # disable rendering for faster training
        env.render()
        
        if done:
            print("episode ended early")
            break
        
    print("episode return: %f"%(episode_return,))

    return episode_return



env = gym.make('CartPole-v0')
env.render()
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range( num_training_episodes ):
    
    episode_return = run_episode( env, sess )

    
monitor.close()

