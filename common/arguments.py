import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # TODO 新加的
    parser.add_argument('--RTW', type=bool, default=True, help='whether to predict state')

    # the environment setting
    # TODO 可换成Matrix Game
    parser.add_argument('--env', type=str, default='smac', help='which env to choose')
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='2s3z', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    # TODO 默认2000000
    parser.add_argument('--n_steps', type=int, default=800000, help='total time steps（总训练步数）')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    # TODO 修改多agent不共用网络
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents（若是，则在输入增加一个n_agents数目的维度，one hot值表示其ID）')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model')
    # TODO 默认32
    parser.add_argument('--evaluate_epoch', type=int, default=0, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    # TODO load model 默认 False
    parser.add_argument('--load_model', type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args

# arguments of RTW
def get_RTW_args(args):
    args.world_loss_weight = 1
    args.teammate_loss_weight = 1
    args.hidden_dim = 64
    args.attn_dim = 64
    args.not_self_model = True

# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    # TODO 原始为5000
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran、 qplex
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    # TODO,原来是5e-4
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1      # TODO 默认1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001

    # QPLEX
    ''' labmda net '''
    args.adv_hypernet_embed = 64  # hidden dim of lambda net
    args.num_kernel = 10  # heads num of lambda net
    args.adv_hypernet_layers = 3  # hidden layers of lambda Net
    args.adv_hypernet_embed = 64  # hidden dim of lambda Net

    ''' Transforation net '''
    args.weighted_head = True  # whether to use Transformation Net
    args.hypernet_embed = 64  # hidden dim of Transformation Net

    ''' Mix net '''
    args.is_minus_one = True    # lambda - 1 or lambda

    args.mixing_embed_dim = 32  # no use

    # q_learning args
    args.double_q = True    # whether to use double q architecture
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args

