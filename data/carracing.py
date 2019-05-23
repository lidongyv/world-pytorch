# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-05-09 11:11:05
# @Last Modified by:   yulidong
# @Last Modified time: 2019-05-17 16:18:40

import argparse
from os.path import join, exists
import gym
import numpy as np
from utils.misc import sample_continuous_policy
MAX_FRAMES = 1000 # max length of carracing
MAX_TRIALS = 200 # just use this to extract one trial. 
def generate_data(data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."
    env = gym.make("CarRacing-v0")
    #seq_len = 1000
    for j in range(MAX_TRIALS):

        for i in range(MAX_FRAMES):
            env.reset()
            env.env.viewer.window.dispatch_events()
            if noise_type == 'white':
                a_rollout = [env.action_space.sample() for _ in range(MAX_FRAMES)]
            elif noise_type == 'brown':
                a_rollout = sample_continuous_policy(env.action_space, MAX_FRAMES, 1. / 50)

            s_rollout = []
            r_rollout = []
            d_rollout = []

            t = 0
            while True:
                action = a_rollout[t]
                t += 1

                s, r, done, _ = env.step(action)
                env.render("rgb_array")
                env.env.viewer.window.dispatch_events()
                s_rollout += [s]
                r_rollout += [r]
                d_rollout += [done]
                if done:
                    print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                    np.savez_compressed(join(data_dir, 'rollout_{}'.format(i+j*1000)),
                             observations=np.array(s_rollout),
                             rewards=np.array(r_rollout),
                             actions=np.array(a_rollout),
                             terminals=np.array(d_rollout))
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help="Where to place MAX_FRAMES")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.dir, args.policy)
