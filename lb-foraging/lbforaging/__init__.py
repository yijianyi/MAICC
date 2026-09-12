from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(5, 10)
players = range(2, 5)
foods = range(1, 3)
coop = [True, False]
partial_obs = [True, False]
tasks = range(1, 11)

for s, p, f, c, po, task in product(sizes, players, foods, coop, partial_obs, tasks):
    register(
        id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}-task{5}-v1".format(
            s, p, f, "-coop" if c else "", "-1s" if po else "", task),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "field_size": (s, s),
            "max_food": f,
            "sight": 1 if po else s,
            "max_episode_steps": 50,
            "force_coop": c,
            "task": task
        },
    )
