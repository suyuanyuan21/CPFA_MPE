import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.attention_sac import AttentionSAC
from tensorboardX import SummaryWriter

def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    attention_sac = AttentionSAC.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=attention_sac.discrete_action)
    attention_sac.prep_rollouts(device='cpu')
    ifi = 3 / config.fps  # inter-frame interval
  


    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        env.render('human')
        for t_i in range(config.episode_length):
            if (t_i % 100 ==0):
                print("t_i:",t_i)
            calc_start = time.time()
            #print("iiiii:",t_i)
            action_space = np.array([[0.0,1.0],[-1.0,0.0],[0.0,-1.0],[1.0,0.0]])
            actions = []
            for i in range(len(env.agents)):
                actions.append(action_space)
            #print("actions:",actions[0][0])
            
            obs, rewards, dones, infos = env.step(actions)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)
