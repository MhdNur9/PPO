import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import time
import gym
import pickle
# import roboschool
import driving
import argparse
from PPO import PPO


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--goalx', type=int, default=15)
    parser.add_argument('--goaly', type=int, default=38)
    parser.add_argument('--env', type=str, default='ContinuousFastRandom-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--use-sleep', action='store_true')
    parser.add_argument('--optimal', action='store_true')
    parser.add_argument('--suboptimal', action='store_true')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_num_samples', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--dontsave', action='store_true')
    args = parser.parse_args()
    # args.env = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # args.env = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # args.env = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    # delay = True               # add delay b/w frames to make video like real time
    # render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    # total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001 
              # learning rate for critic
    run_best_model = False
    if args.optimal:
        run_best_model = True
          # load and run the best saved model
    #####################################################

    env = gym.make(args.env)
    env.set_goal(args.goalx, args.goaly)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + args.env + '/'
    data_directory = "new_data"
    if not os.path.exists(directory):
        print("No directory found")
        exit()
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    data_directory = data_directory + '/' + args.env + '/'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    if not run_best_model:
        checkpoint_path = directory + "PPO_{}_{}_{}_{}_{}.pth".format(args.env, random_seed, args.goalx, args.goaly, run_num_pretrained)
    else:
        checkpoint_path = directory + "PPO_{}_{}_{}_{}_{}best.pth".format(args.env, random_seed, args.goalx, args.goaly, run_num_pretrained)

    optimal_data_path = data_directory + "optimal_data_{}_{}_{}_{}_{}.pkl".format(args.env, random_seed, args.goalx, args.goaly, run_num_pretrained)
    suboptimal_data_path = data_directory + "suboptimal_data_{}_{}_{}_{}_{}.pkl".format(args.env, random_seed, args.goalx, args.goaly, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    dataload = []
    num_optimal = 1
    num_suboptimal = 1
    for ep in range(1, args.num_episodes+1):
        ep_reward = 0
        state = env.reset()
        state_dict = {'state' : [], 'action': [], 'reward': [], 'optimal': []}
        state_dict['state'].append(state)
        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            print(action.dtype)
            state, reward, done, _, _= env.step(action)
            ep_reward += reward
            state_dict['state'].append(state)
            state_dict['action'].append(action)
            state_dict['reward'].append(reward)
            if args.render:
                env.render()
                # time.sleep(frame_delay)
            if args.use_sleep:
                time.sleep(0.05)
            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()
        if num_optimal > args.max_num_samples and args.optimal and not args.suboptimal:
            break
        if num_suboptimal > args.max_num_samples and args.suboptimal and not args.optimal:
            break
        if num_optimal > args.max_num_samples and num_suboptimal > args.max_num_samples and args.optimal and args.suboptimal:
            break
        if ep_reward > args.threshold and args.optimal:
            state_dict['optimal'] = [True] * len(state_dict['action'])
            num_optimal += 1
            dataload.append(state_dict)
        if ep_reward <= args.threshold and args.suboptimal:
            state_dict['optimal'] = [False] * len(state_dict['action'])
            num_suboptimal += 1
            dataload.append(state_dict)
        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / args.num_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    if not args.dontsave and args.optimal:
        with open(optimal_data_path, 'wb') as file:
            pickle.dump(dataload, file)
        print("optimal data saved at : " + optimal_data_path + " with " + str(num_optimal) + " samples")
    if not args.dontsave and args.suboptimal:
        with open(suboptimal_data_path, 'wb') as file:
            pickle.dump(dataload, file)
        print("suboptimal data saved at : " + suboptimal_data_path + " with " + str(num_suboptimal) + " samples")
    print("============================================================================================")
    print(type(dataload))
    for i in range(args.num_episodes):
        accumulator = 0
        accumulator2 = 0
        initial_state = dataload[i]['state'][0]
        env.reset(goal = initial_state[7:9])
        print('goal location', initial_state[7:9])
        env.reset_with_obs(initial_state)
        # print("initial state ", initial_state)
        # print("initial state as per the model ", env.get_obs())
        assert np.allclose(env.get_obs(), initial_state)
        if args.render:
            env.render()
        for step in range(len(dataload[i]['reward'])):
            action = dataload[i]['action'][step]
            print(action.dtype)
            next_state, reward, done, _, info= env.step(action)
            print("next state as per the model ", next_state)
            print("next state as per the data ", dataload[i]['state'][step+1])
            try:
                assert np.allclose(next_state, dataload[i]['state'][step+1])
            except:
                print("the two states are not the same ", next_state, dataload[i]['state'][step+1])
            accumulator += dataload[i]['reward'][step]
            accumulator2 += reward
            if args.render:
                env.render()
            if args.use_sleep: 
                time.sleep(0.05)
        time.sleep(0.1)
        print("episode {} done : reward {}, actual reward {} ".format(i, accumulator, accumulator2))
    env.close()


if __name__ == '__main__':

    test()
