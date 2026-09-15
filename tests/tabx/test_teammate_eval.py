from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from tabx.online_wrapper import TabxOnlineWrapper
from tabx.teammate_modes import load_teammate_modes, validate_policies, condition_metadata, add_baseline_deltas
from tools.eval_tabx_online import prepare_resume, atomic_write_json, append_episode, evaluate_task


class FakeEnv:
    ally_keys = ('a0', 'a1', 'a2', 'padding')
    unit_keys = ally_keys + ('enemy',)
    max_n_ally, max_n_enemy, max_n_zone = 4, 1, 1
    num_agents = 4  # Enemy wrapper exposes only ally count.
    max_episode_steps = 1
    action_spaces = {k: NS(n=8) for k in ally_keys}
    observation_spaces = {k: NS(shape=(2,)) for k in ally_keys}

    def reset(self, key, params):
        self.keys = [key]
        self.raw = {k: NS(status=NS(max_health=0 if k == 'padding' else 10,
                                   is_disabled=k == 'padding', is_alive=True))
                    for k in self.ally_keys}
        return {k: np.array([i, 2.0]) for i, k in enumerate(self.ally_keys)}, {
            'state': self.raw, 'physics_params': 'physics'}

    def get_avail_actions(self, state):
        return {k: np.ones(8) for k in self.ally_keys}

    def step(self, key, state, actions):
        self.keys.append(key)
        self.actions = dict(actions)
        return ({k: np.array([i, 3.0]) for i, k in enumerate(self.ally_keys)},
                state, {'__all__': 1., 'a0': 1.}, {'__all__': True}, {'is_win': np.array([1, 0])})


def fake_runtime():
    def heuristic(key, obs, target, n_agents, n_zones, params, physics):
        assert n_agents == 5  # Must pass total units, not wrapper.num_agents.
        assert float(obs[1]) == 2.0  # Raw observation, not normalized.
        return np.array(4), target + 1
    return {
        'jax': NS(random=NS(PRNGKey=lambda s: s, split=lambda k: (k * 2 + 1, k * 2 + 2),
                            fold_in=lambda k, v: k * 31 + v), numpy=np),
        'jax.numpy': np,
        # EpisodeBatch does not use FAISS, but its module imports it eagerly.
        'faiss': NS(),
        'src.tabx.heuristic_policy': NS(
            heuristic_policy=heuristic, LastVisibleTarget=lambda: 0,
            build_batched_heuristic_params=lambda name: name),
    }


class TeammateTests(unittest.TestCase):
    def test_config_and_deltas(self):
        modes = load_teammate_modes(ROOT / 'src/config/tabx/teammate_modes.yaml')
        self.assertEqual(len(modes), 7)
        for bad in ({-1: 'novice'}, {True: 'expert'}, {0: 'invalid'}, {0: 'novice', '0': 'expert'}):
            with self.assertRaises(ValueError):
                validate_policies(bad)
        rows = [dict(task_id='t', teammate_mode='expert_0', win_rate=.2, mean_return=.5),
                dict(task_id='t', teammate_mode='all_model', win_rate=.6, mean_return=1.)]
        add_baseline_deltas(rows)
        self.assertAlmostEqual(rows[0]['delta_win_rate'], -.4)
        self.assertEqual(rows[0]['delta_mean_return'], -.5)

    def test_replacement_rng_isolation_and_reset(self):
        with patch.dict(sys.modules, fake_runtime()):
            original = TabxOnlineWrapper(FakeEnv(), {}, seed=7)
            mixed = TabxOnlineWrapper(FakeEnv(), {}, seed=7)
            mixed.configure_teammates({0: 'expert'})
            original.reset()
            mixed.reset()
            original.step([1, 1, 1, 1])
            mixed.step([1, 1, 1, 1])
            self.assertEqual(original.env.keys, mixed.env.keys)
            np.testing.assert_array_equal(mixed.last_executed_actions, [4, 1, 1, 7])
            self.assertEqual(mixed._teammate_targets[0], 1)
            mixed.reset()
            self.assertEqual(mixed._teammate_targets[0], 0)
            self.assertIsNone(mixed.last_executed_actions)
            mixed.configure_teammates({3: 'novice'})
            with self.assertRaises(ValueError):
                mixed.reset()
            mixed.configure_teammates({0: 'expert', 1: 'expert', 2: 'expert'})
            with self.assertRaises(ValueError):
                mixed.reset()

    def test_actual_actions_reach_context_and_online_store(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        context_actions, stored_actions = [], []
        mac = NS(agent=object(), init_hidden=lambda n: None,
                 select_actions=lambda *a, **kw: torch.tensor([[1, 1, 1, 1]]),
                 update_context=lambda batch, t: context_actions.append(batch['actions'][0, t].cpu().numpy().reshape(-1)))
        store = NS(offline_probability=1.,
                   add_completed_online_episode=lambda agent, obs, actions, *a: stored_actions.append(actions))
        task = NS(task_id='t', n_agents=4, obs_dim=2, action_dim=8, return_scale=1.)
        definition = NS(path=Path('task.json'), physics_name='default', enemy_heuristic_name='expert')
        condition = condition_metadata('t', {'name': 'expert_0', 'policies': {0: 'expert'}})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, fake_runtime()):
            env = TabxOnlineWrapper(FakeEnv(), {}, seed=10)
            env.configure_teammates({0: 'expert'})
            result = evaluate_task(task, definition, env, mac, store, NS(apply=lambda x: x * .1),
                                   'cpu', 2, 10, Path(directory) / 'episodes.csv', condition)
        self.assertEqual(result['wins'], 2)
        for actions in context_actions:
            np.testing.assert_array_equal(actions, [4, 1, 1, 7])
        for actions in stored_actions:
            np.testing.assert_array_equal(actions, [[4, 1, 1, 7]])

    def test_main_resets_memory_per_mode_and_reuses_checkpoint(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        from contextlib import ExitStack, redirect_stdout
        from io import StringIO
        from unittest.mock import Mock
        from tools import eval_tabx_online as ev
        modes = [{'name': 'all_model', 'policies': {}},
                 {'name': 'expert_0', 'policies': {0: 'expert'}}]
        resets, memory, sizes = [], [], []
        def reset_memory(seed):
            resets.append(seed)
            memory.clear()
        def add_episode(*args):
            sizes.append(len(memory))
            memory.append(args[2])
        store = NS(reset_online_memory=reset_memory, close=Mock(), offline_probability=1.,
                   add_completed_online_episode=add_episode)
        agent = torch.nn.Linear(1, 1)
        mac = NS(agent=agent, load_models=Mock(), init_hidden=Mock(),
                 select_actions=lambda *a, **kw: torch.tensor([[1, 1, 1, 1]]),
                 update_context=lambda *a: None)
        task = NS(task_id='t', n_agents=4, obs_dim=2, action_dim=8, return_scale=1., target_return=1.)
        definition = NS(task={'scenario': {'teams': [0, 0, 0, 1]}},
                        path=Path('task.json'), physics_name='default', enemy_heuristic_name='expert')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / 'agent.th').write_bytes(b'test checkpoint')
            cli = NS(episodes=2, seed=10, episode_limit=1, teammate_config=None,
                     config=root / 'config.yaml', task_ids=None, raw_data_root=root,
                     task_package_root=[], max_allies=4, max_enemies=1, max_zones=1,
                     output_dir=root / 'out', checkpoint=root, tabx_root=ROOT / 'TABX',
                     resume=False, disable_online_memory=False)
            config = {'data': {'manifest': 'manifest.json', 'normalization': 'normalization.npz'},
                      'retrieval': {'index_manifest': 'index.json'}}
            replacements = {
                'parse_args': lambda: cli, 'load_yaml_config': lambda p: config,
                'load_teammate_modes': lambda p: modes,
                'TabxManifest': lambda p: NS(tasks=[task], obs_dim=2, path=p),
                'infer_raw_data_root': lambda *a: root,
                'load_task_definitions': lambda **kw: {'t': definition},
                'ObservationNormalizer': NS(load=lambda *a, **kw: NS(apply=lambda x: x)),
                'select_device': lambda x: 'cpu',
                'create_model_and_learner': lambda *a: (NS(k=2, l=4), mac, None, None),
                'TabxRetrievalStore': lambda **kw: store,
                'build_environment': lambda **kw: TabxOnlineWrapper(FakeEnv(), {}, seed=kw['seed']),
            }
            stack.enter_context(patch.dict(sys.modules, fake_runtime()))
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(ev, name, replacement))
            stack.enter_context(redirect_stdout(StringIO()))
            self.assertEqual(ev.main(), 0)
            self.assertEqual(resets, [10, 10])
            self.assertEqual(sizes, [0, 1, 0, 1])
            mac.load_models.assert_called_once()
            report = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(report['completed_conditions'], 2)
            self.assertEqual(report['tasks'][1]['delta_win_rate'], 0.)
            cli.resume = True
            self.assertEqual(ev.main(), 0)
            self.assertEqual(resets, [10, 10])  # Completed conditions were skipped.

    def test_resume_discards_only_interrupted_condition(self):
        conditions = [condition_metadata('t', m) for m in load_teammate_modes(None)]
        conditions.append(condition_metadata('t', {'name': 'expert_0', 'policies': {0: 'expert'}}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {'seed': 10}
            self.assertEqual(prepare_resume(root, conditions, 2, 10, False, spec), ([], set()))
            for c, count in zip(conditions, [2, 1]):
                for i in range(count):
                    append_episode(root / 'episodes.csv', 't', i + 1, 10 + i, True, 1., 1, c)
            result = dict(conditions[0], task_id='t', episodes=2, seed_start=10)
            atomic_write_json(root / 'summary.json', {'tasks': [result]})
            kept, completed = prepare_resume(root, conditions, 2, 10, True, spec)
            self.assertEqual(kept, [result])
            self.assertEqual(completed, {conditions[0]['condition_id']})
            with (root / 'episodes.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            with self.assertRaises(SystemExit):
                prepare_resume(root, conditions, 2, 10, True, {'seed': 11})
            # Missing episode evidence invalidates an otherwise completed summary.
            (root / 'episodes.csv').unlink()
            self.assertEqual(prepare_resume(root, conditions, 2, 10, True, spec), ([], set()))


if __name__ == '__main__':
    unittest.main()
