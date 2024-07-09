# python3 train.py --goalx 15 --goaly 38 --env ContinuousSlowRandom-v0 
# python3 train.py --goalx 10 --goaly 38 --env ContinuousSlowRandom-v0
# python3 train.py --goalx 20 --goaly 38 --env ContinuousSlowRandom-v0
# # python3 train.py --goalx 15 --goaly 38 --env ContinuousFastRandom-v0
# # python3 train.py --goalx 10 --goaly 38 --env ContinuousFastRandom-v0
# # python3 train.py --goalx 20 --goaly 38 --env ContinuousFastRandom-v0
# python3 test.py --goalx 15 --goaly 38 --env ContinuousSlowRandom-v0 --num_episodes 2000 --max_num_samples 1000 --optimal --threshold -1000
# python3 test.py --goalx 10 --goaly 38 --env ContinuousSlowRandom-v0 --num_episodes 2000 --max_num_samples 1000 --optimal --threshold -1000
# python3 test.py --goalx 20 --goaly 38 --env ContinuousSlowRandom-v0 --num_episodes 2000 --max_num_samples 1000 --optimal --threshold -1000
# python3 test.py --goalx 15 --goaly 38 --env ContinuousFastRandom-v0 --num_episodes 2000 --max_num_samples 1000 --optimal
# python3 test.py --goalx 10 --goaly 38 --env ContinuousFastRandom-v0 --num_episodes 2000 --max_num_samples 1000 --optimal
# python3 test.py --goalx 20 --goaly 38 --env ContinuousFastRandom-v0 --num_episodes 2000 --max_num_samples 1000 --optimal
python3 test.py --goalx 15 --goaly 38 --env ContinuousSlowRandom-v0 --num_episodes 2000 --max_num_samples 1000 --suboptimal --threshold -1000
# python3 test.py --goalx 10 --goaly 38 --env ContinuousSlowRandom-v0 --num_episodes 2000 --max_num_samples 1000 --suboptimal --threshold -1000
python3 test.py --goalx 20 --goaly 38 --env ContinuousSlowRandom-v0 --num_episodes 2000 --max_num_samples 1000 --suboptimal --threshold -1000
# python3 test.py --goalx 15 --goaly 38 --env ContinuousFastRandom-v0 --num_episodes 2000 --max_num_samples 1000 --suboptimal
# python3 test.py --goalx 10 --goaly 38 --env ContinuousFastRandom-v0 --num_episodes 2000 --max_num_samples 1000 --suboptimal
# python3 test.py --goalx 20 --goaly 38 --env ContinuousFastRandom-v0 --num_episodes 2000 --max_num_samples 1000 --suboptimal
