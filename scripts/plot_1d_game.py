import argparse
import os
import torch
import numpy as np
from jutility import plotting, util, latex
import __init__
import games
import welfare

def main(args):
    env = games.get_game_from_args(args)
    wx, wy = welfare.get_welfare_from_args(args)

    x = torch.linspace(*args.xlim, args.num_x)
    y = torch.linspace(*args.ylim, args.num_y)

    rx, ry = env.step(
        x.reshape(1, args.num_x, 1),
        y.reshape(args.num_y, 1, 1),
    )

    assert rx.shape == (args.num_y, args.num_x)
    assert ry.shape == (args.num_y, args.num_x)

    x_br        = np.argmax(rx,             axis=1)
    y_br        = np.argmax(ry,             axis=0)
    x_br_y_br   = np.argmax(rx[y_br, :],    axis=1)
    y_br_x_br   = np.argmax(ry[:, x_br],    axis=0)

    x_ind = np.arange(args.num_x)
    ry_y_br         = ry[y_br, x_ind    ]
    rx_y_br         = rx[y_br, x_ind    ]
    rx_x_br_y_br    = rx[y_br, x_br_y_br]

    y_ind = np.arange(args.num_y)
    rx_x_br         = rx[y_ind,     x_br]
    ry_x_br         = ry[y_ind,     x_br]
    ry_y_br_x_br    = ry[y_br_x_br, x_br]

    wx_x = wx(rx[y_br,  x_ind], ry[y_br,  x_ind])
    wy_y = wy(rx[y_ind, x_br ], ry[y_ind, x_br ])

    wx_x_argmax = torch.argmax(wx_x)
    wy_y_argmax = torch.argmax(wy_y)

    x_star  =  x[wx_x_argmax]
    y_star  =  y[wy_y_argmax]
    rx_star = rx[wy_y_argmax, wx_x_argmax]
    ry_star = ry[wy_y_argmax, wx_x_argmax]

    vmin = min(rx.min(), ry.min())
    vmax = max(rx.max(), ry.max())
    contour = {"zorder": -20, "levels": 20, "vmin": vmin, "vmax": vmax}
    bg      = {"zorder": -10, "alpha": 0.4, "lw": 12}

    caption_data = (
        "$x^* = %.3f, y^* = %.3f, R^x = %.3f, R^y = %.3f$"
        % (x_star, y_star, rx_star, ry_star)
    )
    print(caption_data)

    if args.no_title:
        title = None
    else:
        title = (
            "%s welfare equilibrium for %s\n"
            "$(x^*, y^*) = (%.3f, %.3f), (R^x, R^y) = (%.3f, %.3f)$"
            % (args.welfare, args.game, x_star, y_star, rx_star, ry_star)
        )

    mp = plotting.MultiPlot(
        plotting.Subplot(
            plotting.ContourFilled(x, y, rx,            **contour),
            plotting.HVLine(h=y_star, v=x_star, c="m",  **bg),
            axis_properties=plotting.AxisProperties(
                xlabel="x",
                ylabel="y",
                title="$V^x(x, y)$",
            ),
        ),
        plotting.ColourBar(vmin=vmin, vmax=vmax),
        plotting.Subplot(
            plotting.ContourFilled(x, y, ry,            **contour),
            plotting.HVLine(h=y_star, v=x_star, c="m",  **bg),
            axis_properties=plotting.AxisProperties(
                xlabel="x",
                ylabel="y",
                title="$V^y(x, y)$",
            ),
        ),
        plotting.ColourBar(vmin=vmin, vmax=vmax),
        plotting.Subplot(
            plotting.Line(x, x,             c="b"),
            plotting.Line(x, y[y_br],       c="r"),
            plotting.Line(x, x[x_br_y_br],  c="g", **bg),
            plotting.HVLine(v=x_star,       c="m", **bg),
            axis_properties=plotting.AxisProperties(
                xlabel="x",
                ylabel="Action",
                title="Best responses as a function of $x$",
            ),
        ),
        plotting.Legend(
            plotting.Line(c="b", label="$x$"),
            plotting.Line(c="r", label="$y^*(x)$"),
            plotting.Line(c="g", label="$x^*(y^*(x))$", **bg),
            plotting.Line(c="m", label="$x*$",          **bg),
        ),
        plotting.Subplot(
            plotting.Line(y, y,             c="b"),
            plotting.Line(y, x[x_br],       c="r"),
            plotting.Line(y, y[y_br_x_br],  c="g", **bg),
            plotting.HVLine(v=y_star,       c="m", **bg),
            axis_properties=plotting.AxisProperties(
                xlabel="y",
                ylabel="Action",
                title="Best responses as a function of $y$",
            ),
        ),
        plotting.Legend(
            plotting.Line(c="b", label="$y$"),
            plotting.Line(c="r", label="$x^*(y)$"),
            plotting.Line(c="g", label="$y^*(x^*(y))$", **bg),
            plotting.Line(c="m", label="$y*$",          **bg),
        ),
        plotting.Subplot(
            plotting.Line(x, rx_y_br,               c="b"),
            plotting.Line(x, ry_y_br,               c="r"),
            plotting.Line(x, rx_x_br_y_br,          c="g", **bg),
            plotting.Line(x, wx_x,                  c="c", **bg),
            plotting.HVLine(h=wx_x[wx_x_argmax],    c="y", **bg),
            plotting.HVLine(v=x_star,               c="m", **bg),
            axis_properties=plotting.AxisProperties(
                xlabel="x",
                ylabel="Reward",
                title="Rewards as a function of $x$",
            ),
        ),
        plotting.Legend(
            plotting.Line(c="b", label="$V^x(x, y^*(x))$"),
            plotting.Line(c="r", label="$V^y(x, y^*(x))$"),
            plotting.Line(c="g", label="$V^x(x^*(y^*(x)), y^*(x))$", **bg),
            plotting.Line(c="c", label="$w^x(x, y^*(x))$",           **bg),
            plotting.Line(c="y", label="$w^x(x^*, y^*(x^*))$",       **bg),
            plotting.Line(c="m", label="$x*$",                       **bg),
        ),
        plotting.Subplot(
            plotting.Line(y, ry_x_br,               c="b"),
            plotting.Line(y, rx_x_br,               c="r"),
            plotting.Line(y, ry_y_br_x_br,          c="g", **bg),
            plotting.Line(y, wy_y,                  c="c", **bg),
            plotting.HVLine(h=wy_y[wy_y_argmax],    c="y", **bg),
            plotting.HVLine(v=y_star,               c="m", **bg),
            axis_properties=plotting.AxisProperties(
                xlabel="y",
                ylabel="Reward",
                title="Rewards as a function of $y$",
            ),
        ),
        plotting.Legend(
            plotting.Line(c="b", label="$V^y(x^*(y), y)$"),
            plotting.Line(c="r", label="$V^x(x^*(y), y)$"),
            plotting.Line(c="g", label="$V^y(x^*(y), y^*(x^*(y)))$", **bg),
            plotting.Line(c="c", label="$w^y(x^*(y), y)$",           **bg),
            plotting.Line(c="y", label="$w^y(x^*(y^*), y^*)$",       **bg),
            plotting.Line(c="m", label="$y*$",                       **bg),
        ),
        figure_properties=plotting.FigureProperties(
            num_rows=3,
            num_cols=4,
            figsize=[15, 10],
            width_ratios=[1, 0.1, 1, 0.1],
            title=title,
            wrap_title=False,
            constrained_layout=True,
        ),
    )
    mp.save("Plot game %s" % vars(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    games.add_parser_args(    parser, default="AwkwardGame")
    welfare.add_parser_args(  parser, default="greedy")
    parser.add_argument("--num_x", type=int,    default=2001)
    parser.add_argument("--num_y", type=int,    default=2001)
    parser.add_argument("--xlim", nargs=2, type=float, default=[0, 1])
    parser.add_argument("--ylim", nargs=2, type=float, default=[0, 1])
    parser.add_argument("--no_title", action="store_true")
    args = parser.parse_args()

    with util.Timer("main function"):
        main(args)
