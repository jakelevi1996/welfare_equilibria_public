import torch

def linear(x):
    return x

def relu(x):
    return torch.relu(x)

def sigmoid(x):
    return torch.sigmoid(x)

def gaussian(x):
    return torch.exp(-torch.square(x))

def cauchy(x):
    return 1.0 / (1.0 + torch.square(x))

act_func_dict = {
    act_func.__name__: act_func
    for act_func in [torch.relu, torch.sigmoid, linear, gaussian, cauchy]
}

def add_parser_args(parser, default="relu"):
    parser.add_argument(
        "--act_func",
        default=default,
        type=str,
        choices=act_func_dict.keys(),
    )

def get_act_func_from_args(args):
    return act_func_dict[args.act_func]
