import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT
import numpy as np
from env.environment import RocketLander

import gym
from gym import spaces
from gym.utils import seeding

original_step = RocketLander.step
def patched_step(self, action):
    obs, reward, done, info = original_step(self, action)
    return obs, reward, done, info

RocketLander.step = patched_step

if __name__ == "__main__":
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}

    env = RocketLander(settings)
    s = env.reset()

    left_or_right_barge_movement = np.random.randint(0, 2)
    epsilon = 0.05
    total_reward = 0
    episode_number = 5

    for episode in range(episode_number):
        while (1):
            a = env.action_space.sample()
            s, r, done, info = env.step(a)
            total_reward += r
            env._render()
            env.draw_marker(env.landing_coordinates[0], env.landing_coordinates[1])
            env.refresh(render=False)

            if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
                env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            if done:
                print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, total_reward))
                total_reward = 0
                env.reset()
                break