import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from env.environment import RocketLander

LOG_DIR = "./ppo_rocketlander_tensorboard/"

class RocketLanderEnv(RocketLander):
    def __init__(self):
        default_settings = {
            'Starting Y-Pos Constant': 1.2,
            'Side Engines': True,
            'Vectorized Nozzle': True,
            'Initial Force': 'random'
        }
        super().__init__(default_settings)

    def seed(self, seed=None):
        return self._seed(seed)

def make_env():
    env = RocketLanderEnv()
    env = GymV21CompatibilityV0(env=env)
    return env

LOG_DIR = "./ppo_rocketlander_tensorboard/"

train_env = make_vec_env(make_env, n_envs=8, seed=0, vec_env_cls=DummyVecEnv)
test_env = Monitor(make_env())

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=0.0001,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=LOG_DIR
)

model.learn(total_timesteps=1_000_000)
model.save("models/ppo_rocketlander_model")
print("Training complete. Model saved to models/")

print("\nTesting the trained agent...")

loaded_model = PPO.load("models/ppo_rocketlander_model")

for episode in range(10):
    obs, _ = test_env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = loaded_model.predict(np.array(obs), deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        episode_reward += reward

    print(f"Episode {episode + 1} | Reward: {episode_reward:.2f}")

test_env.close()
train_env.close()