#!/usr/bin/env python3
# coding: utf-8
from environment import *
import numpy as np
import rospy
import argparse
from agent import SacAgent
import os

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='grab_ball')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()

    configs = {
        'num_steps': 3000000,
        'batch_size': 256,
        'lr': 0.0003,
        'hidden_units': [256, 256],
        'memory_size': 1e6,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'multi_step': 1,
        'per': False,  # prioritized experience replay
        'alpha': 0.6,  # It's ignored when per=False.
        'beta': 0.4,  # It's ignored when per=False.
        'beta_annealing': 0.0001,  # It's ignored when per=False.
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'log_interval': 1000,
        'target_update_interval': 1,
        'eval_interval': 10000,
        'cuda': 'cuda:0',
        'seed': args.seed
    }
    env = Op3_Walking()
    env.set_max_sim_count(configs['log_interval'])

    log_dir = os.path.join('logs', args.env_id)
    env.reset()
    agent = SacAgent(env = env, log_dir= log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    print("Conection started...")
    rospy.init_node('op3_walking')
    run()