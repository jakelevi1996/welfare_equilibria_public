import os
import argparse
import torch
import numpy as np
from jutility import plotting, util, latex
import __init__
import games
import marl
import activation

def main(args):
    torch.manual_seed(args.seed)
    device = 0 if args.gpu else None
    game = games.get_game_from_args(args, device)
    if game.get_action_dim() != 1:
        raise ValueError("Phase portraits only available for 1D action space")

    act_func = activation.get_act_func_from_args(args)
    batch_size = args.num_x * args.num_x
    agent_0, agent_1 = marl.get_agents_from_args(
        args=args,
        game=game,
        batch_size=batch_size,
        device=device,
        learning_rate=args.learning_rate,
        act_func=act_func,
    )

    table = util.Table(
        util.TimeColumn("t"),
        util.Column("step"),
        util.Column("r0_mean", ".5f", width=10),
        util.Column("r1_mean", ".5f", width=10),
        util.Column("r0_std",  ".5f", width=10),
        util.Column("r1_std",  ".5f", width=10),
        util.SilentColumn("r0"),
        util.SilentColumn("r1"),
        util.SilentColumn("a0"),
        util.SilentColumn("a1"),
        print_interval=util.TimeInterval(1),
    )

    x = np.linspace(*args.xlim, args.num_x)
    x_pairs = [
        [x0, x1]
        for x0 in x
        for x1 in x
    ]
    x0, x1 = zip(*x_pairs)
    shape = [batch_size, 1]
    kwargs = {"dtype": torch.float32, "device": device}
    agent_0.set_raw_action(torch.tensor(x0, **kwargs).reshape(*shape))
    agent_1.set_raw_action(torch.tensor(x1, **kwargs).reshape(*shape))

    a0 = act_func(agent_0.get_raw_action(None))
    a1 = act_func(agent_1.get_raw_action(None))

    table.update(
        step=0,
        a0=a0.detach().cpu().clone().numpy(),
        a1=a1.detach().cpu().clone().numpy(),
    )

    for step in range(1, args.num_steps + 1):
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

        table.update(
            step=step,
            r0_mean=r0.mean().item(),
            r1_mean=r1.mean().item(),
            r0_std=r0.std().item(),
            r1_std=r1.std().item(),
            r0=r0.detach().cpu().clone().numpy(),
            r1=r1.detach().cpu().clone().numpy(),
            a0=act_func(a0_raw).detach().cpu().clone().numpy(),
            a1=act_func(a1_raw).detach().cpu().clone().numpy(),
        )

    table.print_last()

    return agent_0, agent_1, game, table

def reward_bounds(reward_array, n_sigma=1):
    mean = reward_array.mean(axis=1)
    std  = reward_array.std( axis=1)
    ucb = mean + (n_sigma * std)
    lcb = mean - (n_sigma * std)
    return mean, ucb, lcb

def plot_results(
    args,
    agent_0,
    agent_1,
    game,
    table,
    return_phase_portrait_mp=False,
    title_suffix=None,
    xlim=None,
    ylim=None,
):
    r0_array = np.array(table.get_data("r0"))
    r1_array = np.array(table.get_data("r1"))
    a0_array = np.array(table.get_data("a0"))
    a1_array = np.array(table.get_data("a1"))
    if title_suffix is None:
        title_suffix = (
            "%s vs %s in %s" % (args.a0_type, args.a1_type, args.game)
        )

    x0      = a0_array[:, :, 0]
    x1      = a1_array[:, :, 0]
    x0_init = a0_array[0, :, 0]
    x1_init = a1_array[0, :, 0]
    dx0     = a0_array[1, :, 0] - x0_init
    dx1     = a1_array[1, :, 0] - x1_init
    quiver_kwargs = {"zorder": 20, "normalise": True}
    if return_phase_portrait_mp:
        save_close = False
    else:
        save_close = True

    mp = plotting.plot(
        plotting.Quiver(x0_init, x1_init, dx0, dx1, **quiver_kwargs),
        plotting.Line(x0, x1, alpha=0.5, color="r", zorder=10),
        axis_properties=plotting.AxisProperties(
            xlabel="Agent 0 strategy",
            ylabel="Agent 1 strategy",
            xlim=xlim,
            ylim=ylim,
            wrap_title=False,
        ),
        plot_name="Phase portrait for %s" % title_suffix,
        save_close=save_close,
    )
    if return_phase_portrait_mp:
        return mp

    step = list(range(args.num_steps))
    r0_mean, r0_ucb, r0_lcb = reward_bounds(r0_array)
    r1_mean, r1_ucb, r1_lcb = reward_bounds(r1_array)
    label = "Agent %i mean reward"
    std_kwargs  = {"zorder": 10, "alpha": 0.3, "label": "$\\pm \\sigma$"}
    plotting.plot(
        plotting.Line(step, r0_mean, color="b", zorder=20, label=label % 0),
        plotting.FillBetween(step, r0_ucb, r0_lcb, color="b", **std_kwargs),
        plotting.Line(step, r1_mean, color="r", zorder=20, label=label % 1),
        plotting.FillBetween(step, r1_ucb, r1_lcb, color="r", **std_kwargs),
        axis_properties=plotting.AxisProperties("Step", "Reward"),
        legend=True,
        plot_name="Rewards for %s" % title_suffix,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    games.add_parser_args(    parser, default="StagHunt")
    marl.add_parser_args(     parser, default="Naive")
    activation.add_parser_args(  parser, default="sigmoid")
    parser.add_argument("--num_steps",          type=int,   default=1000)
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--learning_rate",      type=float, default=1)
    parser.add_argument("--num_x",              type=int,   default=20)
    parser.add_argument("--xlim", nargs=2,      type=float, default=[-3, 3])
    parser.add_argument("--gpu",                action="store_true")
    args = parser.parse_args()

    with util.Timer("main function"):
        agent_0, agent_1, game, table = main(args)
    with util.Timer("plot_results"):
        plot_results(args, agent_0, agent_1, game, table)
