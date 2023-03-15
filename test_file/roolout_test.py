import common.arguments as ar
import rollout
from smac.env import StarCraft2Env
from controller.share_params import SharedMAC
import numpy as np




args = ar.get_common_args()
args = ar.get_mixer_args(args)
time_steps, train_steps, evaluate_steps = 0, 0, -1
env = StarCraft2Env(map_name="8m",
                    replay_dir="../replay_dir")
env_info = env.get_env_info()
args.n_actions = env_info["n_actions"]
args.n_agents = env_info["n_agents"]
args.state_shape = env_info["state_shape"]
args.obs_shape = env_info["obs_shape"]
args.episode_limit = env_info["episode_limit"]
args.epsilon = 0
args.anneal_epsilon = 0
args.min_epsilon = 0
args.epsilon_anneal_scale = " "

mac = SharedMAC(args)
RolloutWorker = rollout.RolloutWorker(env, mac, args)
while time_steps < args.n_steps:   # 总训练步数
    episodes, reward, win_tag, step = RolloutWorker.generate_episodes(2)
    break