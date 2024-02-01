import argparse
import os
import torch
import numpy as np
from jutility import plotting, util
import __init__
import games
import marl
import activation
import train_marl

def main(args):
    results_dict = dict()
    store_table(args, "ExactLola", "ExactLola",   0, 1, results_dict)
    store_table(args, "ExactLola", "Naive",       0, 1, results_dict)
    for seed in range(args.num_seeds):
        we = "WelFuSeELola"
        nl = "Naive"
        num_episodes = args.num_welfuse_episodes
        store_table(args, we, we, seed, num_episodes, results_dict)
        store_table(args, we, nl, seed, num_episodes, results_dict)

    plot_results(args, results_dict)

def store_table(args, a0_type, a1_type, seed, num_episodes, results_dict):
    print("\n%s vs %s, seed = %i\n" % (a0_type, a1_type, seed))
    args.a0_type        = a0_type
    args.a1_type        = a1_type
    args.seed           = seed
    args.num_episodes   = num_episodes
    a0, _, _, table      = train_marl.main(args)
    results_dict[a0_type, a1_type, seed] = table
    if a0_type == "WelFuSeELola":
        results_dict[a1_type, seed] = a0.get_welfare_history()

def plot_results(args, results_dict):
    args.num_episodes = 1
    for a1_type in ["ExactLola", "Naive"]:
        table = results_dict["ExactLola", a1_type, 0]
        args.a0_type = "ExactLola"
        args.a1_type = a1_type
        train_marl.plot_rewards_summary(
            args,
            np.array(table.get_data("r0")),
            np.array(table.get_data("r1")),
            "ExactLola vs %s in %s" % (a1_type, args.game)
        )

    args.num_episodes = args.num_welfuse_episodes
    for a1_type in ["WelFuSeELola", "Naive"]:
        table_list = [
            results_dict["WelFuSeELola", a1_type, seed]
            for seed in range(args.num_seeds)
        ]
        r0_array = np.concatenate(
            [np.array(table.get_data("r0")) for table in table_list],
            axis=1,
        )
        r1_array = np.concatenate(
            [np.array(table.get_data("r1")) for table in table_list],
            axis=1,
        )
        args.a0_type = "WelFuSeELola"
        args.a1_type = a1_type
        train_marl.plot_rewards_summary(
            args,
            r0_array,
            r1_array,
            "WelFuSeELola vs %s in %s" % (a1_type, args.game)
        )

        welfare_history_list = [
            results_dict[a1_type, seed]
            for seed in range(args.num_seeds)
        ]
        welfare_name_list = ["greedy", "egalitarian", "fairness"]
        noisy_data_dict = {
            name: util.NoisyData()
            for name in welfare_name_list
        }
        for welfare_history in welfare_history_list:
            for name in welfare_name_list:
                for i, num_samples in enumerate(welfare_history[name]):
                    noisy_data_dict[name].update(i, num_samples)
        cp = plotting.ColourPicker(len(welfare_name_list))
        lines = [
            line
            for i, name in enumerate(welfare_name_list)
            for line in plotting.get_noisy_data_lines(
                noisy_data_dict[name],
                colour=cp(i),
                name=name,
            )
        ]
        plot_name = (
            "Welfare function history for WelFuSeELola vs %s in %s"
            % (a1_type, args.game)
        )
        plotting.plot(
            *lines,
            axis_properties=plotting.AxisProperties(
                "Episode",
                "Number of samples",
            ),
            legend=True,
            plot_name=plot_name,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    games.add_parser_args(  parser, default="ChickenGame")
    activation.add_parser_args(parser, default="sigmoid")
    parser.add_argument("--num_welfuse_episodes",   type=int,   default=3)
    parser.add_argument("--num_steps",              type=int,   default=1000)
    parser.add_argument("--num_seeds",              type=int,   default=5)
    parser.add_argument("--batch_size",             type=int,   default=100)
    parser.add_argument("--learning_rate",          type=float, default=0.1)
    parser.add_argument("--a0_args", nargs="*",     type=float, default=[])
    parser.add_argument("--a1_args", nargs="*",     type=float, default=[])
    parser.add_argument("--downsample_ratio",       type=int,   default=1)
    parser.add_argument("--gpu",                    action="store_true")
    args = parser.parse_args()

    with util.Timer("main function"):
        main(args)
