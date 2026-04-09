import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from env.environment import RocketLander
from env.constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT

import gym
from gym import spaces
from gym.utils import seeding

simulation_settings = {
    'Side Engines': True,
    'Clouds': False,
    'Vectorized Nozzle': True,
    'Graph': False,
    'Render': True,
    'Starting Y-Pos Constant': 1.2,
    'Initial Force': 'random',
    'Rows': 1,
    'Columns': 2
}

LOG_DIR = "./ppo_rocketlander_tensorboard/"

test_env = RocketLander(settings=simulation_settings)
loaded_model = PPO.load("models/ppo_rocketlander_model")

episodes_to_run = 10
epsilon = 0.05
left_or_right_barge_movement = np.random.randint(0, 2)

for episode in range(episodes_to_run):
    obs = test_env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, info = test_env.step(action)
        steps = steps + 1
        done = terminated
        episode_reward += reward

        test_env._render()
        test_env.draw_marker(test_env.landing_coordinates[0], test_env.landing_coordinates[1])
        test_env.refresh(render=False)

        if obs[LEFT_GROUND_CONTACT] == 0 and obs[RIGHT_GROUND_CONTACT] == 0:
            test_env.move_barge_randomly(epsilon, left_or_right_barge_movement)
            test_env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
            test_env.apply_random_y_disturbance(epsilon=0.005)

        if done:
            print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, episode_reward))
            episode_reward = 0
            test_env.reset()
            break

test_env.close()