import gym
import threading
import numpy as np
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import get_arg_parser, get_logger, str2bool
from learner import create_learner, learner_collection
from common import get_args

args = get_args()
env = make_env(args)

def continuous_run():
    while True:
        env.render()


sim_thread = threading.Thread(target=continuous_run)
sim_thread.start()


obs = env.reset()



print(gym.__file__)

counter = 0
obs = env.reset()
action_test = [0,0,0,1]
while True:
    action_test = env.action_space.sample()
    tmp = env.step(action_test)
    obs = env.reset()

