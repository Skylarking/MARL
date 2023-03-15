from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_RTW_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
from env.single_state_matrix_game import TwoAgentsMatrixGame
from utils.logging import Logger, get_logger

if __name__ == '__main__':
    for i in range(8):
        # initial args
        args = get_common_args()
        get_mixer_args(args)
        get_RTW_args(args)

        # initial env and other args
        if args.env == 'smac':
            env = StarCraft2Env(map_name=args.map,
                                step_mul=args.step_mul,
                                difficulty=args.difficulty,
                                game_version=args.game_version,
                                replay_dir=args.replay_dir)
        else:
            raise ValueError("env not found")

        env_info = env.get_env_info()   # env info for other args
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        log = Logger()

        # initial runner
        runner = Runner(env, log, args)

        # train (if not eval)
        if not args.evaluate:
            runner.run(i)   # 训练n次
        else:   # eval
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break

        env.close()
