import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class Player:
    def __init__(self):
        self.position = None
        self.level = None
        self.field_size = None
        self.reward = 0
        self.current_step = None
        self.has_reached = False

    def setup(self, position, level, field_size):
        self.position = position
        self.level = level
        self.field_size = field_size

    @property
    def name(self):
        return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        task,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)
        self.length = self.field.shape[0]
        assert (field_size == (7, 7) or field_size == (9, 9))
        assert (players == 3)
        assert (max_food == 1)
        assert (sight == 1)
        assert (force_coop == True)
        assert (task in list(range(1, 11)))

        self.max_food = max_food
        self.sight = sight
        self.force_coop = force_coop
        self.task = task
        if self.task == 1:
            self.food_row = 1
            self.food_col = 1
        elif self.task == 2:
            self.food_row = 1
            self.food_col = self.length - 2
        elif self.task == 3:
            self.food_row = self.length - 2
            self.food_col = 1
        elif self.task == 4:
            self.food_row = self.length - 2
            self.food_col = self.length - 2
        elif self.task == 5:
            self.food_row = 1
            self.food_col = (self.length - 1) // 2
        elif self.task == 6:
            self.food_row = (self.length - 1) // 2
            self.food_col = 1
        elif self.task == 7:
            self.food_row = (self.length - 1) // 2
            self.food_col = self.length - 2
        elif self.task == 8:
            self.food_row = self.length - 2
            self.food_col = (self.length - 1) // 2
        elif self.task == 9:
            self.food_row = (self.length - 1) // 2
            self.food_col = (self.length - 1) // 2
        elif self.task == 10:
            self.food_row = np.random.randint(1, self.length - 1)
            self.food_col = np.random.randint(1, self.length - 1)
        self._game_over = None

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.players)))

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]

        max_food = self.max_food

        min_obs = [-1, -1] * max_food + [0, 0] * len(self.players)
        max_obs = [field_x, field_y] * max_food + [
            field_x,
            field_y,
        ] * len(self.players)

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0): min(row + distance + 1, self.rows),
                max(col - distance, 0): min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0): min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0): min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):
        food_count = 0
        min_level = max_level if self.force_coop else 1

        self.field[self.food_row, self.food_col] = (
            min_level
            if min_level == max_level
            else self.np_random.randint(min_level, max_level)
        )

    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self):
        for player in self.players:
            player.reward = 0
            player.has_reached = False

        self.players[0].setup(
            (self.length - 1, 0),
            1,
            self.field_size,
        )

        self.players[1].setup(
            (self.length - 1, self.length - 1),
            1,
            self.field_size,
        )

        self.players[2].setup(
            (0, (self.length - 1) // 2),
            1,
            self.field_size,
        )

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error(
            "Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=(a.position[0], a.position[1]),
                    level=a.level,
                    is_self=a == player,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if abs(player.position[0] - a.position[0]) <= self.sight and abs(player.position[1] - a.position[1]) <= self.sight
            ],
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[2 * i] = -1
                obs[2 * i + 1] = -1

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                frow, fcol = np.nonzero(self.field)
                obs[2 * i] = frow[0] - seen_players[0].position[0] + self.sight
                obs[2 * i + 1] = fcol[0] - \
                    seen_players[0].position[1] + self.sight

            for i in range(len(self.players)):
                obs[self.max_food * 2 + 2 * i] = -1
                obs[self.max_food * 2 + 2 * i + 1] = -1

            for i, p in enumerate(seen_players):
                obs[self.max_food * 2 + 2 * i] = p.position[0] - \
                    seen_players[0].position[0] + self.sight
                obs[self.max_food * 2 + 2 * i + 1] = p.position[1] - \
                    seen_players[0].position[1] + self.sight

            obs[self.max_food * 2] = seen_players[0].position[0]
            obs[self.max_food * 2 + 1] = seen_players[0].position[1]

            return obs

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        ninfo = {}

        return nobs, nreward, ndone, ninfo

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players()
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3])
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        return nobs

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1,
                            player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1,
                            player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0],
                            player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0],
                            player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # first reach food
        frow, fcol = np.nonzero(self.field)
        adj_players = self.adjacent_players(frow[0], fcol[0])
        for p in adj_players:
            if p.has_reached is False:
                p.reward += 1 / self.n_agents
                p.has_reached = True

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward += 1 / self.n_agents
            # and the food is removed
            self.field[frow, fcol] = 0

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        return self._make_gym_obs(observations)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
