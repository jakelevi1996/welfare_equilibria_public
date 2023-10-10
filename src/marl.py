import torch
import activation
import welfare
from jutility import util

class _MultiPlayerAgent:
    def __init__(
        self,
        game,
        is_agent_0,
        *extra_args,
        batch_size=1,
        device=None,
        learning_rate=0.1,
        act_func=None,
    ):
        if act_func is None:
            act_func = activation.linear

        self._game          = game
        self._is_agent_0    = is_agent_0
        self._batch_size    = batch_size
        self._device        = device
        self._action_dim    = game.get_action_dim()
        self._state_dim     = 2 * self._action_dim
        self._lr            = learning_rate
        self._act_func      = act_func
        self._action_raw = torch.empty(
            [self._batch_size, self._action_dim],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        self._set_extra_args(*extra_args)

    def _set_extra_args(self):
        return

    def reset(self, reward_episode_mean):
        with torch.no_grad():
            self._action_raw.normal_()
        return self._action_raw

    def set_raw_action(self, raw_action):
        with torch.no_grad():
            self._action_raw.copy_(raw_action)

    def get_raw_action(self, state=None):
        return self._action_raw

    def set_update(self, r_self, r_opponent, opponent_action_raw):
        self._update = None
        raise NotImplementedError()

    def update(self):
        self._action_raw += self._update

    def _step(self, self_action_raw, opponent_action_raw):
        action_self     = self._act_func(    self_action_raw)
        action_opponent = self._act_func(opponent_action_raw)
        if self._is_agent_0:
            r_self, r_opponent = self._game.step(action_self, action_opponent)
        else:
            r_opponent, r_self = self._game.step(action_opponent, action_self)

        return r_self, r_opponent

class Naive(_MultiPlayerAgent):
    def set_update(self, r_self, r_opponent, opponent_action_raw):
        [dr0da0] = torch.autograd.grad(
            r_self.sum(),
            self._action_raw,
            retain_graph=True,
        )
        self._update = self._lr * dr0da0

class Lola(_MultiPlayerAgent):
    def _set_extra_args(self, inner_lr=25.0):
        self._inner_lr = inner_lr

    def set_update(self, r_self, r_opponent, opponent_action_raw):
        [dr1da1] = torch.autograd.grad(
            r_opponent.sum(),
            opponent_action_raw,
            create_graph=True,
        )
        [dr0da1] = torch.autograd.grad(
            r_self.sum(),
            opponent_action_raw,
            create_graph=True,
        )
        dr0 = r_self + self._inner_lr * (dr1da1 * dr0da1).sum(dim=1)
        [ddr0da0] = torch.autograd.grad(
            dr0.sum(),
            self._action_raw,
            retain_graph=True,
        )
        self._update = self._lr * ddr0da0

class ExactLola(Lola):
    def set_update(self, r_self, r_opponent, opponent_action_raw):
        [dr1da1] = torch.autograd.grad(
            r_opponent.sum(),
            opponent_action_raw,
            create_graph=True,
        )
        da1 = opponent_action_raw + self._inner_lr * dr1da1
        dr0, _ = self._step(self._action_raw, da1)
        [ddr0da0] = torch.autograd.grad(
            dr0.sum(),
            self._action_raw,
            retain_graph=True,
        )
        self._update = self._lr * ddr0da0

class Lookahead(Lola):
    def set_update(self, r_self, r_opponent, opponent_action_raw):
        [dr1da1] = torch.autograd.grad(
            r_opponent.sum(),
            opponent_action_raw,
            retain_graph=True,
        )
        da1 = opponent_action_raw + self._inner_lr * dr1da1
        dr0, _ = self._step(self._action_raw, da1.detach())
        [ddr0da0] = torch.autograd.grad(
            dr0.sum(),
            self._action_raw,
            retain_graph=True,
        )
        self._update = self._lr * ddr0da0

class Saga(_MultiPlayerAgent):
    def _set_extra_args(self, cobra_num_samples=200, cobra_std=1.0):
        self._cobra_std = cobra_std
        self._opponent_action_noise = torch.empty(
            [int(cobra_num_samples), self._batch_size, self._action_dim],
            dtype=torch.float32,
            device=self._device,
        )

    def set_update(self, r_self, r_opponent, opponent_action_raw):
        self._opponent_action_noise.normal_(std=self._cobra_std)
        da1 = opponent_action_raw.detach() + self._opponent_action_noise
        dr0, dr1    = self._step(self._action_raw, da1)
        br_inds     = torch.argmax(dr1, dim=0, keepdim=False)
        r0br        = dr0[br_inds, range(self._batch_size)]
        [dr0brda0] = torch.autograd.grad(
            r0br.sum(),
            self._action_raw,
            retain_graph=True,
        )
        self._update = self._lr * dr0brda0

class Sasa(Saga):
    def _set_extra_args(self, num_samples=15, noise_std=1.0):
        self._num_samples = int(num_samples)
        self._std         = noise_std
        self._opponent_action_noise = torch.empty(
            [1, self._num_samples, self._batch_size, self._action_dim],
            dtype=torch.float32,
            device=self._device,
        )
        self._self_action_noise = torch.empty(
            [self._num_samples, 1, self._batch_size, self._action_dim],
            dtype=torch.float32,
            device=self._device,
        )

    def set_update(self, r_self, r_opponent, opponent_action_raw):
        self._opponent_action_noise.normal_(std=self._std)
        self._self_action_noise.normal_(    std=self._std)
        da1 = opponent_action_raw.detach() + self._opponent_action_noise
        da0 =    self._action_raw.detach() +     self._self_action_noise
        dr0, dr1 = self._step(da0, da1)
        opponent_br_inds = torch.argmax(dr1, dim=1, keepdim=False)
        dr0_br = dr0[
            torch.arange(self._num_samples).reshape(self._num_samples, 1),
            opponent_br_inds,
            torch.arange(self._batch_size).reshape(1, self._batch_size),
        ]
        self_br_inds = torch.argmax(dr0_br, dim=0, keepdim=False)
        da0_br = da0[self_br_inds, :, range(self._batch_size), :].squeeze(1)
        self._update = self._lr * (da0_br - self._action_raw)

class WelFuSeELola(ExactLola):
    def _set_extra_args(self, inner_lr=25.0):
        self._inner_lr = inner_lr
        self._welfare_function_names = [
            "greedy",
            "egalitarian",
            "fairness",
        ]
        self._welfare_functions = [
            welfare.welfare_dict[name][0]
            for name in self._welfare_function_names
        ]
        self._welfare_batch_inds = None
        self._welfare_history_dict = {
            welfare_name: []
            for welfare_name in self._welfare_function_names
        }

    def _prune_welfare_functions(self, welfare_reward_list):
        valid_welfare_inds = [
            i for i, welfare_reward in enumerate(welfare_reward_list)
            if torch.numel(welfare_reward) > 0
        ]
        welfare_reward_list = [
            welfare_reward_list[i]          for i in valid_welfare_inds
        ]
        self._welfare_function_names = [
            self._welfare_function_names[i] for i in valid_welfare_inds
        ]
        self._welfare_functions = [
            self._welfare_functions[i]      for i in valid_welfare_inds
        ]
        return welfare_reward_list

    def reset(self, reward_episode_mean):
        if self._welfare_batch_inds is None:
            self._welfare_batch_inds = torch.randint(
                len(self._welfare_functions),
                size=[self._batch_size],
            )
        else:
            welfare_reward_list = [
                reward_episode_mean[self._welfare_batch_inds == i]
                for i in range(len(self._welfare_functions))
            ]
            welfare_reward_list = self._prune_welfare_functions(
                welfare_reward_list,
            )
            welfare_reward_sample_inds = [
                torch.randint(
                    torch.numel(welfare_reward_list[i]),
                    size=[self._batch_size],
                )
                for i in range(len(self._welfare_functions))
            ]
            welfare_reward_sample_tensor = torch.stack(
                [
                    welfare_reward[sample_inds]
                    for welfare_reward, sample_inds
                    in zip(welfare_reward_list, welfare_reward_sample_inds)
                ],
                dim=0,
            )
            self._welfare_batch_inds = torch.argmax(
                welfare_reward_sample_tensor,
                dim=0,
                keepdim=False,
            )

        self._update_welfare_history()
        with torch.no_grad():
            self._action_raw.normal_()
        return self._action_raw

    def _update_welfare_history(self):
        for i, welfare_name in enumerate(self._welfare_function_names):
            num_samples = (self._welfare_batch_inds == i).sum().item()
            self._welfare_history_dict[welfare_name].append(num_samples)

    def get_welfare_history(self):
        return self._welfare_history_dict

    def _welfare(self, r_self, r_opponent):
        welfare_tensor = torch.zeros_like(r_self)
        for i in range(len(self._welfare_functions)):
            welfare_function_i = self._welfare_functions[i]
            welfare_i_inds = (self._welfare_batch_inds == i)
            welfare_tensor[welfare_i_inds] = welfare_function_i(
                r_self[welfare_i_inds],
                r_opponent[welfare_i_inds],
            )
        return welfare_tensor

    def set_update(self, r_self, r_opponent, opponent_action_raw):
        [dr1da1] = torch.autograd.grad(
            r_opponent.sum(),
            opponent_action_raw,
            create_graph=True,
        )
        da1 = opponent_action_raw + self._inner_lr * dr1da1
        dr0, dr1 = self._step(self._action_raw, da1)
        [ddr0da0] = torch.autograd.grad(
            self._welfare(dr0, dr1).sum(),
            self._action_raw,
            retain_graph=True,
        )
        self._update = self._lr * ddr0da0

agent_type_dict = {
    agent_type_dict.__name__: agent_type_dict
    for agent_type_dict in [
        Naive,
        Lola,
        ExactLola,
        Lookahead,
        Saga,
        Sasa,
        WelFuSeELola,
    ]
}

def add_parser_args(parser, default="Naive"):
    parser.add_argument(
        "--a0_type",
        default=default,
        type=str,
        choices=agent_type_dict.keys(),
    )
    parser.add_argument(
        "--a1_type",
        default=default,
        type=str,
        choices=agent_type_dict.keys(),
    )
    parser.add_argument("--a0_args", nargs="*", type=float, default=[])
    parser.add_argument("--a1_args", nargs="*", type=float, default=[])

def get_agents_from_args(
    args,
    game,
    batch_size,
    device,
    learning_rate,
    act_func,
):
    agent_0_type = agent_type_dict[args.a0_type]
    agent_1_type = agent_type_dict[args.a1_type]
    agent_0 = agent_0_type(
        game,
        True,
        *args.a0_args,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        act_func=act_func,
    )
    agent_1 = agent_1_type(
        game,
        False,
        *args.a1_args,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        act_func=act_func,
    )
    return agent_0, agent_1
