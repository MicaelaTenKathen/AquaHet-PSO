
import  os

import gym
import numpy as np
from stable_baselines3 import PPO
from Policies.CNNs import CustomCNNforSensing
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
from utils import SaveOnBestTrainingRewardCallback
import torch as th

import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

from GymEnvironment.GymPSO import GymPSO


def make_env(sd):


    def _init() -> gym.Env:
        env = GymPSO(1, 150, 1, reward_function='inc_mse')
        env.reset()

        return env

    set_random_seed(sd)
    return _init


if __name__ == '__main__':

    policy_kwargs = dict(
        features_extractor_class=CustomCNNforSensing,
        features_extractor_kwargs=dict(features_dim=1024),
        activation_fn=th.nn.LeakyReLU,
        net_arch=[256, 256, 256, dict(pi=[128, 128, 128], vf=[128, 128, 128])],

    )

    # Create log dir
    log_dir = "Experiments/Results/PPO_Results/"
    os.makedirs(log_dir, exist_ok=True)

    navigation_map = np.ones((100, 100))
    initial_position = np.array([50, 50])
    extra_positions = None

    vec_env = SubprocVecEnv([make_env(sd=i) for i in range(7)])

    monitor_env = VecMonitor(vec_env, log_dir)

    model = PPO("CnnPolicy", monitor_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./example_tensorboard/",
                n_steps = 5,
                learning_rate=1e-4,
                target_kl = 5,
                ent_coef=0.01

                )

    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

    model.learn(total_timesteps=int(100000), tb_log_name="ExperimentPPO", callback=callback)

