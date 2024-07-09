import gym
import gym.wrappers
import reacher
import driving
import time
from gym import make 
import numpy as np
import argparse
import pickle
parser = argparse.ArgumentParser(description='Test the model')
parser.add_argument('--num-episodes', type=int, default=10)
parser.add_argument('--seed', type=int, default=1001)
parser.add_argument('--use-sleep', action='store_true')
parser.add_argument('--env', type=str, default='ContinuousFastRandom-v0')
parser.add_argument('--data_path', type=str, default='new_data/ContinuousFastRandom-v0/optimal_data_ContinuousFastRandom-v0_0_15_38_0.pkl')
parser.add_argument('--render', action='store_true')
parser.add_argument('--goalx', type=int, default=15)
parser.add_argument('--goaly', type=int, default=38)

paths = {'optimalfast1038' : '/home/smart/PPO-PyTorch/new_data/ContinuousFastRandom-v0/optimal_data_ContinuousFastRandom-v0_0_10_38_0.pkl',
         'suboptimalfast1038' : '/home/smart/PPO-PyTorch/new_data/ContinuousFastRandom-v0/suboptimal_data_ContinuousFastRandom-v0_0_10_38_0.pkl',
         'optimalfast1538' : '/home/smart/PPO-PyTorch/new_data/ContinuousFastRandom-v0/optimal_data_ContinuousFastRandom-v0_0_15_38_0.pkl',
         'suboptimalfast1538' : '/home/smart/PPO-PyTorch/new_data/ContinuousFastRandom-v0/suboptimal_data_ContinuousFastRandom-v0_0_15_38_0.pkl',
         'optimalfast2038' : '/home/smart/PPO-PyTorch/new_data/ContinuousFastRandom-v0/optimal_data_ContinuousFastRandom-v0_0_20_38_0.pkl',
         'suboptimalfast2038' : '/home/smart/PPO-PyTorch/new_data/ContinuousFastRandom-v0/suboptimal_data_ContinuousFastRandom-v0_0_20_38_0.pkl',
         'optimalslow1038' : '/home/smart/PPO-PyTorch/new_data/ContinuousSlowRandom-v0/optimal_data_ContinuousSlowRandom-v0_0_10_38_0.pkl',
         'suboptimalslow1038' : '/home/smart/PPO-PyTorch/new_data/ContinuousSlowRandom-v0/suboptimal_data_ContinuousSlowRandom-v0_0_10_38_0.pkl',
         'optimalslow1538' : '/home/smart/PPO-PyTorch/new_data/ContinuousSlowRandom-v0/optimal_data_ContinuousSlowRandom-v0_0_15_38_0.pkl',
         'suboptimalslow1538' : '/home/smart/PPO-PyTorch/new_data/ContinuousSlowRandom-v0/suboptimal_data_ContinuousSlowRandom-v0_0_15_38_0.pkl',
         'optimalslow2038' : '/home/smart/PPO-PyTorch/new_data/ContinuousSlowRandom-v0/optimal_data_ContinuousSlowRandom-v0_0_20_38_0.pkl',
         'suboptimalslow2038' : '/home/smart/PPO-PyTorch/new_data/ContinuousSlowRandom-v0/suboptimal_data_ContinuousSlowRandom-v0_0_20_38_0.pkl'}
args = parser.parse_args()
with open(args.data_path, 'rb') as f:
    episodes = pickle.load(f)
print(len(episodes))
print(episodes[1]['reward'])
args = parser.parse_args()


env1 = gym.make(args.env)
num_inputs = env1.observation_space.shape[0]
num_actions = env1.action_space.shape[0]
print(num_inputs, num_actions)
for i in range(args.num_episodes):
    accumulator = 0
    accumulator2 = 0
    initial_state = episodes[i]['state'][0]
    env1.reset(goal = initial_state[7:9])
    print('goal location', initial_state[7:9])
    env1.reset_with_obs(initial_state)
    # print("initial state ", initial_state)
    # print("initial state as per the model ", env1.get_obs())
    assert np.allclose(env1.get_obs(), initial_state)
    if args.render:
        env1.render()
    for step in range(len(episodes[i]['reward'])):
        action = episodes[i]['action'][step]
        print(action.dtype)
        next_state, reward, done, _, info= env1.step(action)
        print("next state as per the model ", next_state)
        print("next state as per the data ", episodes[i]['state'][step+1])
        try:
            assert np.allclose(next_state, episodes[i]['state'][step+1])
        except:
            print("the two states are not the same ", next_state, episodes[i]['state'][step+1])
        accumulator += episodes[i]['reward'][step]
        accumulator2 += reward
        if args.render:
            env1.render()
        if args.use_sleep: 
            time.sleep(0.05)
    time.sleep(0.1)
    print("episode {} done : reward {}, actual reward {} ".format(i, accumulator, accumulator2))
env1.close()

# with open(paths['optimalfast1538'], 'rb') as f:
#     episodes = pickle.load(f)
# print(len(episodes))
# print(episodes[1]['reward'])
# environment = args.env
# env1 = gym.make(environment)
# # env1.set_goal(args.goalx, args.goaly)
# # env1.reset(goal = [args.goalx, args.goaly])
# # env1.set_goal(args.goalx, args.goaly)
# num_inputs = env1.observation_space.shape[0]
# num_actions = env1.action_space.shape[0]
# print(num_inputs, num_actions)
# for i in range(args.num_episodes):
#     accumulator = 0
#     accumulator2 = 0
#     initial_state = episodes[i]['state'][0]
#     env1.reset(goal = initial_state[7:9])
#     print('goal location', initial_state[7:9])
#     # env1.set_goal(initial_state[7], initial_state[8])
#     env1.reset_with_obs(initial_state)
#     if args.render:
#         env1.render()
#     for step in range(len(episodes[i]['reward'])):
#         # accumulator += episodes[i]['reward'][step]
#         action = episodes[i]['action'][step]
#         # print(episodes[i]['reward'][step])
#         next_state, reward, done, _, info= env1.step(action)
#         # print("next state as per the model ", next_state)
#         # print("next state as per the data ", episodes[i]['state'][step+1])
#         # env1.reset_with_obs(episodes[i]['state'][step+1])
#         accumulator += episodes[i]['reward'][step]
#         accumulator2 += reward
#         if args.render:
#             env1.render()
#         if args.use_sleep: 
#             time.sleep(0.05)
#     time.sleep(0.1)
#     print("episode {} done : reward {}, actual reward {} ".format(i, accumulator, accumulator2))
# env1.close()


# import pickle as pkl
# with open('new_data/ContinuousFastRandom-v0/optimal_data_ContinuousFastRandom-v0_0_15_38_0.pkl', 'rb') as f:
#     data = pkl.load(f)
# print(len(data[0]['state']))
# print((data[0]['state'][0]))
# print(len(data[0]['state'][0]))
