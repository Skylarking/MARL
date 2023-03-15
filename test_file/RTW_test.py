from network.RWT import RTWAgent
from smac.env import StarCraft2Env
from rollout import RolloutWorker
from common.replaybuffer import ReplayBuffer
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
from controller.share_params import SharedMAC, SeparatedMAC, SharedMACWithState
import torch as th
import numpy as np

def get_max_episode_len(args, batch):
    # calculate the max_episode_len of n episodes for cutting redundant transitions
    n_episodes = batch["o"].shape[0]
    max_episode_len = 0
    for episode_idx in range(n_episodes):
        for transition_idx in range(args.episode_limit):
            if batch['terminated'][episode_idx][transition_idx][0] == 1:
                if transition_idx + 1 >= max_episode_len:
                    max_episode_len = transition_idx + 1
                break
    if max_episode_len == 0:  # 防止所有episode都是最大长度
        max_episode_len = args.episode_limit

    # cut redundant transitions
    for key in batch.keys():
        batch[key] = th.tensor(batch[key][:, :max_episode_len]).cuda()  # 截取max_episode_len前面的数据

    return batch, max_episode_len

def run(env, args):

    mac = SharedMAC(args)

    buffer = ReplayBuffer(args)
    rolloutWorker = RolloutWorker(env, mac, args)

    # generate n episodes for training
    episodes, rewards, win_tags, steps = rolloutWorker.generate_episodes(n_episodes=2, random_select=True)  # episodes中每个key的value的shape为(n_episodes, episode_limit, n_agents, features)


    # store n episodes into buffer
    buffer.store_episode(episodes)

    mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))  # sample mini batch
    batch, max_len = get_max_episode_len(args, mini_batch)


    agent = RTWAgent(mac., args).cuda()
    agent(batch['o'], batch['u'], th.zeros(batch['o'].size(0), args.n_agents, args.n_actions), 1)

    # # sample batch from buffer and train
    # for train_step in range(args.train_steps):
    #     mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))  # sample mini batch
    #     loss = learner.train(mini_batch, train_steps)  # train_steps用于更新target网络
    #     # print('loss is ', loss)
    #     train_steps += 1
    #
    # # save model for k episodes
    # if train_steps > 0 and train_steps % args.save_cycle == 0:
    #     learner.save_models(train_steps)
    return

if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)

    env = StarCraft2Env(map_name=args.map,
                        step_mul=args.step_mul,
                        difficulty=args.difficulty,
                        game_version=args.game_version,
                        replay_dir=args.replay_dir)

    env_info = env.get_env_info()  # env info for other args
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    args.alg = 'qmix'
    args.hidden_dim = 64

    run(env, args)
