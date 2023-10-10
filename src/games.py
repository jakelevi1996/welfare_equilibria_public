import torch

class Game:
    def __init__(self, device=None):
        return

    def get_action_dim(self):
        raise NotImplementedError()

    def step(self, x, y):
        raise NotImplementedError()

    def get_action_labels(self):
        action_dim = self.get_action_dim()
        label_list = ["a%i" % i for i in range(action_dim)]
        return label_list

    def __repr__(self):
        return type(self).__name__

class Game1D(Game):
    def get_action_dim(self):
        return 1

class MatrixGame(Game1D):
    def __init__(self, device=None):
        self._set_reward_matrix()
        self._reward_matrix = self._reward_matrix.to(
            dtype=torch.float32,
            device=device,
        )

    def _set_reward_matrix(self):
        self._reward_matrix = None
        raise NotImplementedError()

    def step(self, p, q):
        joint_probabilities = torch.cat(
            [p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)],
            dim=-1,
        )
        rewards = joint_probabilities @ self._reward_matrix
        reward_0, reward_1 = torch.split(rewards, [1, 1], dim=-1)

        return reward_0.squeeze(dim=-1), reward_1.squeeze(dim=-1)

    def get_payoff_tables(self):
        r0, r1 = torch.split(self._reward_matrix, [1, 1], dim=1)
        return r0.reshape(2, 2), r1.reshape(2, 2)

    def __repr__(self):
        s = "%s(\n%s,\n%s,\n)" % (
            type(self).__name__,
            *self.get_payoff_tables(),
        )
        return s

class PrisonersDilemma(MatrixGame):
    def _set_reward_matrix(self):
        r0_matrix = torch.tensor([[-1, -3], [0, -2]])
        r0 = r0_matrix.reshape(4)
        r1 = r0_matrix.T.reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class ChickenGame(MatrixGame):
    def _set_reward_matrix(self):
        r0_matrix = torch.tensor([[0, -1], [1, -100]])
        r0 = r0_matrix.reshape(4)
        r1 = r0_matrix.T.reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class BabyChickenGame(MatrixGame):
    def _set_reward_matrix(self):
        r0_matrix = torch.tensor([[0, -1], [1, -3]])
        r0 = r0_matrix.reshape(4)
        r1 = r0_matrix.T.reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class MatchingPennies(MatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([1, -1, -1, 1])
        self._reward_matrix = torch.stack([r0, -r0], dim=1)

class AwkwardGame(MatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([[3, 1], [2, 4]]).reshape(4)
        r1 = torch.tensor([[1, 3], [5, 2]]).reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class ElusiveGame(MatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([[1, 3], [0, 2]]).reshape(4)
        r1 = torch.tensor([[1, 0], [0, 1]]).reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class CoordinationGame(MatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([[2, 0], [0, 1]]).reshape(4)
        r1 = torch.tensor([[1, 0], [0, 2]]).reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class StagHunt(MatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([[10, 1], [8, 5]]).reshape(4)
        r1 = torch.tensor([[10, 8], [1, 5]]).reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class EnforcementGame(MatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([[3, 0], [5, 1]]).reshape(4)
        r1 = torch.tensor([[3, 5], [0, 1]]).reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class EagleGame(MatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([[4, -4], [-2, 2]]).reshape(4)
        r1 = torch.tensor([[1, -1], [-3, 3]]).reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class Tandem(Game1D):
    def step(self, x, y):
        s = x + y
        reward_0 = -s * s + 2 * x
        reward_1 = -s * s + 2 * y
        return reward_0.squeeze(dim=-1), reward_1.squeeze(dim=-1)

class ImpossibleMarket(Game1D):
    def step(self, x, y):
        interaction = (
            x * y
            + pow(y, 4) / (4 * (1 + x*x))
            - pow(x, 4) / (4 * (1 + y*y))
        )
        reward_0 = -(pow(x, 6) / 6 - (x * x / 2) + interaction)
        reward_1 = -(pow(y, 6) / 6 - (y * y / 2) - interaction)
        return reward_0.squeeze(dim=-1), reward_1.squeeze(dim=-1)

class IpdTftAlldMix(Game1D):
    def __init__(self, device=None, gamma=0.96):
        self._ipd = IteratedPrisonersDilemma(device, gamma)
        tensor_kwargs = {"dtype": torch.float32, "device": device}
        self._alld  = torch.zeros(5,                **tensor_kwargs)
        self._tft_0 = torch.tensor([1, 1, 0, 1, 0], **tensor_kwargs)
        self._tft_1 = torch.tensor([1, 1, 1, 0, 0], **tensor_kwargs)

    def step(self, x, y):
        p = x * self._tft_0 + (1 - x) * self._alld
        q = y * self._tft_1 + (1 - y) * self._alld
        return self._ipd.step(p, q)

class UltimatumGame(Game1D):
    def step(self, x, y):
        reward_0 = (5 * x) + (8 * (1 - x) * y)
        reward_1 = (5 * x) + (2 * (1 - x) * y)
        return reward_0.squeeze(dim=-1), reward_1.squeeze(dim=-1)

class IteratedMatrixGame(Game):
    def __init__(self, device=None, gamma=0.96):
        self._set_reward_matrix()
        tensor_kwargs = {"dtype": torch.float32, "device": device}
        self._reward_matrix = self._reward_matrix.to(**tensor_kwargs)
        self._I = torch.eye(4, **tensor_kwargs)
        self._gamma = gamma

    def _set_reward_matrix(self):
        self._reward_matrix = None
        raise NotImplementedError()

    def get_action_dim(self):
        return 5

    def step(self, p, q, normalise=True):
        joint_probabilities = torch.stack(
            [p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)],
            dim=-1,
        )
        init_distribution, transition_matrix = torch.split(
            joint_probabilities,
            [1, 4],
            dim=-2,
        )
        value_matrix = torch.linalg.solve(
            self._I - self._gamma * transition_matrix,
            self._reward_matrix,
        )
        returns = (init_distribution @ value_matrix).squeeze(dim=-2)
        if normalise:
            returns *= (1 - self._gamma)
        g0, g1 = torch.split(returns, [1, 1], dim=-1)
        return g0.squeeze(dim=-1), g1.squeeze(dim=-1)

class IteratedPrisonersDilemma(IteratedMatrixGame):
    def _set_reward_matrix(self):
        r0_matrix = torch.tensor([[-1, -3], [0, -2]], dtype=torch.float32)
        r0 = r0_matrix.reshape(4)
        r1 = r0_matrix.T.reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

    def get_action_labels(self):
        return ["p(C|--)", "p(C|CC)", "p(C|CD)", "p(C|DC)", "p(C|DD)"]

class IteratedChickenGame(IteratedMatrixGame):
    def _set_reward_matrix(self):
        r0_matrix = torch.tensor([[0, -1], [1, -100]], dtype=torch.float32)
        r0 = r0_matrix.reshape(4)
        r1 = r0_matrix.T.reshape(4)
        self._reward_matrix = torch.stack([r0, r1], dim=1)

class IteratedMatchingPennies(IteratedMatrixGame):
    def _set_reward_matrix(self):
        r0 = torch.tensor([1, -1, -1, 1],   dtype=torch.float32)
        self._reward_matrix = torch.stack([r0, -r0], dim=1)

game_dict = {
    game.__name__: game
    for game in [
        PrisonersDilemma,
        ChickenGame,
        BabyChickenGame,
        MatchingPennies,
        AwkwardGame,
        ElusiveGame,
        CoordinationGame,
        StagHunt,
        EnforcementGame,
        EagleGame,
        Tandem,
        ImpossibleMarket,
        IpdTftAlldMix,
        UltimatumGame,
        IteratedPrisonersDilemma,
        IteratedChickenGame,
        IteratedMatchingPennies,
    ]
}

def add_parser_args(parser, default="MatchingPennies"):
    parser.add_argument(
        "--game",
        default=default,
        type=str,
        choices=game_dict.keys(),
    )

def get_game_from_args(args, device=None):
    game_type = game_dict[args.game]
    game = game_type(device)
    return game
