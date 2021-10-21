import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


str_to_initializer = {
    'uniform': nn.init.uniform_,
    'normal': nn.init.normal_,
    'eye': nn.init.eye_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'he': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_
    }


str_to_activation = {
    'elu': nn.ELU(),
    'hardshrink': nn.Hardshrink(),
    'hardtanh': nn.Hardtanh(),
    'leakyrelu': nn.LeakyReLU(),
    'logsigmoid': nn.LogSigmoid(),
    'prelu': nn.PReLU(),
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'rrelu': nn.RReLU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'logsoftmax': nn.LogSoftmax(),
    'softshrink': nn.Softshrink(),
    'softsign': nn.Softsign(),
    'tanh': nn.Tanh(),
    'tanhshrink': nn.Tanhshrink(),
    'softmin': nn.Softmin(),
    'softmax': nn.Softmax(dim=1),
    'none': None
    }


def initialize_weights(initializer):
    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            initializer(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    return initialize


def create_linear_network(input_dim, output_dim, hidden_units=[],
                          hidden_activation='relu', output_activation=None,
                          initializer='xavier_uniform'):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(str_to_activation[hidden_activation])
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(str_to_activation[output_activation])

    return nn.Sequential(*model).apply(
        initialize_weights(str_to_initializer[initializer]))


def create_dqn_base(num_channels, initializer='xavier_uniform'):
    return nn.Sequential(
        nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
    ).apply(initialize_weights(str_to_initializer[initializer]))

#自作SAC用
def create_dq_network1(num_channels, input_actions , output_dim, hidden_units, initializer = 'xavier_unifrom', ):
    return nn.Sequential(
        # 画像サイズの変化84*84→20*20
        nn.Conv2d(num_channels , 32, kernel_size=8, stride=4),
        # stackするflameは4画像なのでinput_dim=4である、出力は32とする、
        # sizeの計算  size = (Input_size - Kernel_size + 2*Padding_size)/ Stride_size + 1
        nn.ReLU(),
        # 画像サイズの変化20*20→9*9
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 画像サイズの変化9*9→7*7
        nn.ReLU(),
        nn.Flatten(),  # 画像形式を1次元に変換
        nn.Linear(64 * 7 * 7, 16),  # 64枚の7×7の画像を、256次元のoutputへ
        nn.ReLU(),
    ).apply(initialize_weights(str_to_initializer[initializer]))

def create_dq_network2(num_channels, input_actions , output_dim, hidden_units, initializer = 'xavier_unifrom', ):
    return nn.Sequential(
        nn.Linear(16 + input_actions ,128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    ).apply(initialize_weights(str_to_initializer[initializer]))

def create_policy_network(input_dim, output_dim, hidden_units, initializer = 'xavier_unifrom', ):
    return nn.Sequential(
        # 画像サイズの変化84*84→20*20
        nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
        # stackするflameは4画像なのでinput_dim=4である、出力は32とする、
        # sizeの計算  size = (Input_size - Kernel_size + 2*Padding_size)/ Stride_size + 1
        nn.ReLU(),
        # 画像サイズの変化20*20→9*9
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 画像サイズの変化9*9→7*7
        nn.ReLU(),
        nn.Flatten(),  # 画像形式を1次元に変換
        nn.Linear(64 * 7 * 7, 256),  # 64枚の7×7の画像を、256次元のoutputへ
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    ).apply(initialize_weights(str_to_initializer[initializer]))



def conv_net(env, in_dim, hidden, depth, act_layer):
    # this assumes image shape == (1, 28, 28)
    act_layer = getattr(nn, act_layer)
    n_acts = env.action_space.n
    conv_modules = [
        nn.Conv2d(1, 10, kernel_size=5, stride=2),
        nn.Conv2d(10, 20, kernel_size=5, stride=2),
        nn.Flatten(),
    ]
    fc_modules = []
    fc_modules.append(nn.Linear(320 + in_dim, hidden))
    for _ in range(depth - 1):
        fc_modules += [act_layer(), nn.Linear(hidden, hidden)]
    fc_modules.append(act_layer())
    conv_modules = nn.Sequential(*conv_modules)
    fc_modules = nn.Sequential(*fc_modules)
    return fc_modules, conv_modules


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
