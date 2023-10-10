import argparse
import os
import torch
import numpy as np
from jutility import plotting, util
import __init__
import games
import marl
import activation

def main(args, agents=None):
    torch.manual_seed(args.seed)
    device = 0 if args.gpu else None
    game = games.get_game_from_args(args, device)
    act_func = activation.get_act_func_from_args(args)
    if agents is not None:
        agent_0, agent_1 = agents
    else:
        agent_0, agent_1 = marl.get_agents_from_args(
            args=args,
            game=game,
            batch_size=args.batch_size,
            device=device,
            learning_rate=args.learning_rate,
            act_func=act_func,
        )

    table = util.Table(
        util.TimeColumn("t"),
        util.Column("episode"),
        util.Column("r0_episode_mean", ".5f"),
        util.Column("r1_episode_mean", ".5f"),
        util.Column("step"),
        util.Column("r0_mean", ".5f", width=10),
        util.Column("r1_mean", ".5f", width=10),
        util.Column("r0_std",  ".5f", width=10),
        util.Column("r1_std",  ".5f", width=10),
        util.SilentColumn("r0"),
        util.SilentColumn("r1"),
        util.SilentColumn("a0"),
        util.SilentColumn("a1"),
        print_interval=util.TimeInterval(num_seconds=1),
    )

    r0_episode_mean = 0
    r1_episode_mean = 0

    for episode in range(args.num_episodes):

        a0 = act_func(agent_0.reset(r0_episode_mean))
        a1 = act_func(agent_1.reset(r1_episode_mean))
        r0_sum = 0
        r1_sum = 0

        for step in range(args.num_steps):
            state   = torch.cat([a0, a1], dim=1).detach()
            a0_raw  = agent_0.get_raw_action(state)
            a1_raw  = agent_1.get_raw_action(state)
            a0      = act_func(a0_raw)
            a1      = act_func(a1_raw)
            r0, r1  = game.step(a0, a1)
            agent_0.set_update(r0, r1, a1_raw)
            agent_1.set_update(r1, r0, a0_raw)
            with torch.no_grad():
                agent_0.update()
                agent_1.update()

            r0_sum += r0.detach()
            r1_sum += r1.detach()

            table.update(
                step=step,
                r0_mean=r0.mean().item(),
                r1_mean=r1.mean().item(),
                r0_std=r0.std().item(),
                r1_std=r1.std().item(),
                r0=r0.detach().cpu().clone().numpy(),
                r1=r1.detach().cpu().clone().numpy(),
                a0=a0.detach().cpu().clone().numpy(),
                a1=a1.detach().cpu().clone().numpy(),
            )

        r0_episode_mean = r0_sum / args.num_steps
        r1_episode_mean = r1_sum / args.num_steps

        table.update(
            episode=episode,
            r0_episode_mean=r0_episode_mean.mean().item(),
            r1_episode_mean=r1_episode_mean.mean().item(),
            level=1,
        )

    return agent_0, agent_1, game, table

def plot_results(args, agent_0, agent_1, game, table):
    r0_array = np.array(table.get_data("r0"))
    r1_array = np.array(table.get_data("r1"))
    a0_array = np.array(table.get_data("a0"))
    a1_array = np.array(table.get_data("a1"))
    action_dim    = game.get_action_dim()
    action_labels = game.get_action_labels()
    title_suffix  = ", ".join(
        "%s = %s" % (key, vars(args)[key])
        for key in ["game", "a0_type", "a1_type"]
    )

    summarise_batch_list = []
    if args.batch_size >= 2:
        summarise_batch_list.append(True)
        a0_rewards_str = "%s in %s" % (args.a0_type, title_suffix)
        a1_rewards_str = "%s in %s" % (args.a1_type, title_suffix)
        plot_rewards_summary(args, r0_array, r1_array, title_suffix)
        plot_actions_summary(args, a0_array, action_labels, a0_rewards_str)
        plot_actions_summary(args, a1_array, action_labels, a1_rewards_str)
    if args.batch_size <= 500:
        summarise_batch_list.append(False)
    summarise_episodes_list = []
    if args.num_episodes <= 10:
        summarise_episodes_list.append(False)
    if args.num_episodes >= 5:
        summarise_episodes_list.append(True)
        plot_mean_episode_rewards(args, table, title_suffix)

    for summarise_episodes in summarise_episodes_list:
        for summarise_batch in summarise_batch_list:
            plot_rewards_actions(
                args,
                r0_array,
                r1_array,
                a0_array,
                a1_array,
                action_dim,
                action_labels,
                title_suffix,
                summarise_batch=summarise_batch,
                summarise_episodes=summarise_episodes,
            )

    if (action_dim == 1) and (args.num_episodes == 1):
        plot_phase_portrait(a0_array, a1_array, title_suffix)

    for i, agent in enumerate([agent_0, agent_1]):
        if isinstance(agent, marl.WelFuSeELola):
            plot_name = (
                "Welfare function history for agent %i, %s"
                % (i, title_suffix)
            )
            plot_welfare_function_history(agent, plot_name)

def plot_mean_episode_rewards(args, table, title_suffix):
    plotting.plot(
        plotting.Line(
            table.get_data("r0_episode_mean"),
            label="Agent 0 mean reward",
            c="r",
        ),
        plotting.Line(
            table.get_data("r1_episode_mean"),
            label="Agent 1 mean reward",
            c="b",
        ),
        axis_properties=plotting.AxisProperties("Episode", "Mean reward"),
        legend=True,
        plot_name=(
            "Mean rewards per episode for %s, seed = %i"
            % (title_suffix, args.seed)
        ),
    )

def confidence_bounds(data_list, n_sigma=1):
    mean = np.array([np.mean(x) for x in data_list])
    std  = np.array([np.std( x) for x in data_list])
    ucb = mean + (n_sigma * std)
    lcb = mean - (n_sigma * std)
    return mean, ucb, lcb

def plot_rewards_summary(args, r0_array, r1_array, title_suffix):
    num_steps = args.num_episodes * args.num_steps
    num_split = int(num_steps / args.downsample_ratio)
    t = np.arange(0, num_steps, args.downsample_ratio)
    r0_split = np.split(r0_array, num_split, axis=0)
    r1_split = np.split(r1_array, num_split, axis=0)
    r0_mean, r0_ucb, r0_lcb = confidence_bounds(r0_split)
    r1_mean, r1_ucb, r1_lcb = confidence_bounds(r1_split)
    mean_kwargs_0 = {"c": "b", "zorder": 20, "label": "Agent 0 mean reward"}
    mean_kwargs_1 = {"c": "r", "zorder": 20, "label": "Agent 1 mean reward"}
    std_kwargs    = {"zorder": 10, "label": "$\\pm\\sigma$", "alpha": 0.3}
    std_kwargs_0  = std_kwargs.copy()
    std_kwargs_1  = std_kwargs.copy()
    std_kwargs_0["color"] = "b"
    std_kwargs_1["color"] = "r"
    plotting.plot(
        plotting.Line(          t, r0_mean,         **mean_kwargs_0),
        plotting.FillBetween(   t, r0_lcb, r0_ucb,  **std_kwargs_0 ),
        plotting.Line(          t, r1_mean,         **mean_kwargs_1),
        plotting.FillBetween(   t, r1_lcb, r1_ucb,  **std_kwargs_1 ),
        axis_properties=plotting.AxisProperties("Step", "Reward"),
        legend=True,
        plot_name="Reward summaries for %s" % title_suffix,
    )

def plot_actions_summary(args, action_array, labels, title_suffix):
    num_steps, batch_size, action_dim = action_array.shape
    cp  = plotting.ColourPicker(action_dim)
    std_kwargs  = {"zorder": 10, "alpha": 0.3}

    num_split = int(num_steps / args.downsample_ratio)
    t = np.arange(0, num_steps, args.downsample_ratio)
    action_bounds = [
        confidence_bounds(np.split(action_array[:, :, i], num_split, axis=0))
        for i in range(action_dim)
    ]
    action_lines = [
        line
        for i, [mean, ucb, lcb] in enumerate(action_bounds)
        for line in [
            plotting.Line(t, mean, label=labels[i], color=cp(i), zorder=20),
            plotting.FillBetween(t, lcb, ucb, color=cp(i), **std_kwargs),
        ]
    ]
    plotting.plot(
        *action_lines,
        axis_properties=plotting.AxisProperties("Step", "Action"),
        legend=True,
        plot_name="Action summaries for %s" % title_suffix,
    )

def plot_rewards_actions(
    args,
    r0_array,
    r1_array,
    a0_array,
    a1_array,
    action_dim,
    action_labels,
    title_suffix,
    summarise_batch,
    summarise_episodes,
):
    cp  = plotting.ColourPicker(action_dim)
    cr0 = "b"
    cr1 = "r" if (action_dim > 1) else "b"
    xlabel = "Episode" if summarise_episodes else "Step"

    if summarise_episodes or summarise_batch:
        if summarise_episodes:
            num_split = args.num_episodes
        else:
            num_split = args.num_episodes * args.num_steps

        x = np.arange(num_split)
        r0_split = np.split(r0_array, num_split, axis=0)
        r1_split = np.split(r1_array, num_split, axis=0)
        r0_mean, r0_ucb, r0_lcb = confidence_bounds(r0_split)
        r1_mean, r1_ucb, r1_lcb = confidence_bounds(r1_split)
        a0_bounds = [
            confidence_bounds(np.split(a0_array[:, :, i], num_split, axis=0))
            for i in range(action_dim)
        ]
        a1_bounds = [
            confidence_bounds(np.split(a1_array[:, :, i], num_split, axis=0))
            for i in range(action_dim)
        ]

        mean_kwargs = {"zorder": 20}
        std_kwargs  = {"zorder": 10, "alpha": 0.3}
        r0_lines = [
            plotting.Line(       x, r0_mean,        color=cr0, **mean_kwargs),
            plotting.FillBetween(x, r0_lcb, r0_ucb, color=cr0, **std_kwargs ),
        ]
        r1_lines = [
            plotting.Line(       x, r1_mean,        color=cr1, **mean_kwargs),
            plotting.FillBetween(x, r1_lcb, r1_ucb, color=cr1, **std_kwargs ),
        ]
        a0_lines = [
            line
            for i, [mean, ucb, lcb] in enumerate(a0_bounds)
            for line in [
                plotting.Line(       x, mean,     color=cp(i), **mean_kwargs),
                plotting.FillBetween(x, lcb, ucb, color=cp(i), **std_kwargs ),
            ]
        ]
        a1_lines = [
            line
            for i, [mean, ucb, lcb] in enumerate(a1_bounds)
            for line in [
                plotting.Line(       x, mean,     color=cp(i), **mean_kwargs),
                plotting.FillBetween(x, lcb, ucb, color=cp(i), **std_kwargs ),
            ]
        ]

    else:
        r0_lines = [plotting.Line(r0_array, c=cr0)]
        r1_lines = [plotting.Line(r1_array, c=cr1)]
        a0_lines = [
            plotting.Line(a0_array[:, :, i], c=cp(i))
            for i in range(action_dim)
        ]
        a1_lines = [
            plotting.Line(a1_array[:, :, i], c=cp(i))
            for i in range(action_dim)
        ]

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *r0_lines,
            axis_properties=plotting.AxisProperties(
                xlabel=xlabel,
                ylabel="Reward",
                title="Player 0 rewards",
            ),
        ),
        plotting.Subplot(
            *r1_lines,
            axis_properties=plotting.AxisProperties(
                xlabel=xlabel,
                ylabel="Reward",
                title="Player 1 rewards",
            ),
        ),
        plotting.Empty(),
        plotting.Subplot(
            *a0_lines,
            axis_properties=plotting.AxisProperties(
                xlabel=xlabel,
                ylabel="Action",
                title="Player 0 actions",
            ),
        ),
        plotting.Subplot(
            *a1_lines,
            axis_properties=plotting.AxisProperties(
                xlabel=xlabel,
                ylabel="Action",
                title="Player 1 actions",
            ),
        ),
        plotting.Legend(
            *[
                plotting.Line(c=cp(i), label=action_labels[i])
                for i in range(action_dim)
            ],
        ),
        figure_properties=plotting.FigureProperties(
            num_rows=2,
            num_cols=3,
            width_ratios=[1, 1, 0.2],
            title=title_suffix,
            constrained_layout=True,
        ),
    )
    mp.save(
        plot_name=(
            "%s %s %s"
            % (summarise_batch, summarise_episodes, vars(args))
        ),
    )

def plot_phase_portrait(a0_array, a1_array, title_suffix):
    plotting.plot(
        plotting.Line(a0_array.squeeze(-1), a1_array.squeeze(-1), c="r"),
        axis_properties=plotting.AxisProperties("x", "y"),
        plot_name="Phase portrait for %s" % title_suffix,
    )

def plot_welfare_function_history(agent, plot_name):
    welfare_history_dict = agent.get_welfare_history()
    cp = plotting.ColourPicker(len(welfare_history_dict))
    plotting.plot(
        *[
            plotting.Line(num_samples_list, c=cp(i), label=name)
            for i, [name, num_samples_list]
            in enumerate(welfare_history_dict.items())
        ],
        axis_properties=plotting.AxisProperties(
            xlabel="Episode",
            ylabel="Number of samples",
        ),
        legend=True,
        plot_name=plot_name,
    )

def plot_br(agent, plot_name, n=500, xmin=-3, xmax=3):
    x_raw = torch.linspace(xmin, xmax, n, dtype=torch.float32).reshape(n, 1)
    if args.gpu:
        x_raw = x_raw.cuda()
    y_raw = agent.get_br(x_raw)
    act_func = activation.get_act_func_from_args(args)
    x = act_func(x_raw)
    y = act_func(y_raw)
    plotting.plot(
        plotting.Line(x.cpu().numpy(), y.cpu().numpy(), c="b"),
        axis_properties=plotting.AxisProperties(
            xlabel="Self-action",
            ylabel="Approximate learned opponent BR",
        ),
        legend=True,
        plot_name=plot_name,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    games.add_parser_args(  parser, default="IteratedPrisonersDilemma")
    marl.add_parser_args(   parser, default="Naive")
    activation.add_parser_args(parser, default="sigmoid")
    parser.add_argument("--num_episodes",       type=int,   default=1)
    parser.add_argument("--num_steps",          type=int,   default=1000)
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--batch_size",         type=int,   default=100)
    parser.add_argument("--learning_rate",      type=float, default=1)
    parser.add_argument("--downsample_ratio",   type=int,   default=1)
    parser.add_argument("--gpu",                action="store_true")
    args = parser.parse_args()

    with util.Timer("main function"):
        agent_0, agent_1, game, table = main(args)
    with util.Timer("plot_results"):
        plot_results(args, agent_0, agent_1, game, table)
