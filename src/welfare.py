import torch

welfare_dict = {
    "greedy":           [lambda vx, vy:  vx, lambda vx, vy:  vy],
    "selfless":         [lambda vx, vy:  vy, lambda vx, vy:  vx],
    "self_destructive": [lambda vx, vy: -vx, lambda vx, vy: -vy],
    "malicious":        [lambda vx, vy: -vy, lambda vx, vy: -vx],
    "egalitarian":      [lambda vx, vy: torch.minimum(vx, vy)  ] * 2,
    "utilitarian":      [lambda vx, vy:  0.5 * (vx + vy)       ] * 2,
    "fairness":         [lambda vx, vy: -0.5 * abs(vx - vy)    ] * 2,
}

def add_parser_args(parser, default="greedy"):
    parser.add_argument(
        "--welfare",
        default=default,
        type=str,
        choices=welfare_dict.keys(),
    )

def get_welfare_from_args(args):
    return welfare_dict[args.welfare]
