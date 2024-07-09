
import pickle
import random
import os

# Define the paths to the pickle files
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


all_list = []
# Load the lists from pickle files


def select_random_elements(input_list, percentage):
    num_elements = int(len(input_list) * percentage / 100)
    return random.sample(input_list, num_elements)

def combine(path1, path2, path3):
    with open(path1, 'rb') as f:
        list1 = pickle.load(f)

    with open(path2, 'rb') as f:
        list2 = pickle.load(f)

    with open(path3, 'rb') as f:
        list3 = pickle.load(f)
    print(len(list1), len(list2), len(list3))
    selected_list1 = select_random_elements(list1, 33)
    selected_list2 = select_random_elements(list2, 33)
    selected_list3 = select_random_elements(list3, 34)
    
    combined_list = selected_list1 + selected_list2 + selected_list3
    random.shuffle(combined_list)
    print(len(combined_list))
    # Print the combined list
    return combined_list

combined_list1 = combine(paths['optimalfast1038'], paths['optimalfast1538'], paths['optimalfast2038'])
combined_list2 = combine(paths['suboptimalfast1038'], paths['suboptimalfast1538'], paths['suboptimalfast2038'])
combined_list3 = combine(paths['optimalslow1038'], paths['optimalslow1538'], paths['optimalslow2038'])
combined_list4 = combine(paths['suboptimalslow1038'], paths['suboptimalslow1538'], paths['suboptimalslow2038'])



directory1 = 'new_data/ContinuousFastRandom-v0'
directory2 = 'new_data/ContinuousSlowRandom-v0'

if not os.path.exists(directory1):
    os.makedirs(directory1)
if not os.path.exists(directory2):
    os.makedirs(directory2)

path1 = directory1 + '/combined_list_fast_optimal.pkl'
path2 = directory1 + '/combined_list_fast_suboptimal.pkl'
path3 = directory2 + '/combined_list_slow_optimal.pkl'
path4 = directory2 + '/combined_list_slow_suboptimal.pkl'

with open(path1, 'wb') as f:
    pickle.dump(combined_list1, f)
with open(path2, 'wb') as f:
    pickle.dump(combined_list2, f)
with open(path3, 'wb') as f:
    pickle.dump(combined_list3, f)
with open(path4, 'wb') as f:
    pickle.dump(combined_list4, f)
print('done')



