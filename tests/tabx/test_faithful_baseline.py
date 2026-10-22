"""Server-side regression checks for faithful MAICC/TABX semantics."""
from types import SimpleNamespace as NS
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from modules.agents.transformer_agent import CustomGPT2Model
from tabx.evaluation_protocol import resolve_return_target
from tabx.retrieval_store import (
    TabxRetrievalStore, RetrievalBucket, training_scores, offline_scores,
    online_scores, normalized_utility,
)


def test_dense_cem_preserves_original_mask_and_padding_isolation():
    torch.manual_seed(14)
    config = GPT2Config(n_positions=20, n_embd=16, n_layer=2, n_head=2,
                        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., use_cache=False)
    config._attn_implementation = "eager"
    model = CustomGPT2Model(config, False).eval()
    x = torch.randn(2, 3, 5, 16)
    present = torch.ones(2, 3, 2, dtype=torch.bool)
    with torch.no_grad():
        original = model.custom_forward(x.flatten(1, 2), 2).last_hidden_state
        padded = model.custom_forward(x.flatten(1, 2), 2, present).last_hidden_state
        torch.testing.assert_close(original, padded)
        present[1, :, 1] = False
        baseline = model.custom_forward(x.flatten(1, 2), 2, present).last_hidden_state
        changed = x.clone()
        changed[1, :, 2:4] *= 100
        torch.testing.assert_close(baseline,
            model.custom_forward(changed.flatten(1, 2), 2, present).last_hidden_state)
        changed = x.clone()
        changed[:, 1, [1, 3, 4]] *= 20
        changed[:, 2] *= 20
        after = model.custom_forward(changed.flatten(1, 2), 2, present).last_hidden_state
        torch.testing.assert_close(baseline.reshape(2, 3, 5, 16)[:, 1, 0],
                                   after.reshape(2, 3, 5, 16)[:, 1, 0])
        changed = x.clone()
        changed[:, 0, 1] *= 20
        past = model.custom_forward(changed.flatten(1, 2), 2, present).last_hidden_state
        assert not torch.allclose(baseline[:, 5], past[:, 5])


def test_original_score_formulas_and_fixed_bounds():
    np.testing.assert_allclose(training_scores([.9, .8], [3, 1], 2, .98), [.95, 1.9])
    np.testing.assert_allclose(offline_scores([.9, .5], [0, .4], (0, 1)), [.95, 1.07])
    # Actual active agents, not padded slot capacity, scale individual return.
    assert online_scores(.5, .4, .1, 3, (0, 1)) == pytest.approx(1.13)
    np.testing.assert_allclose(normalized_utility([-1, 2], (0, 1)), [-1, 2])


def test_training_context_order_varies_and_excludes_query_episode():
    import faiss
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [.9, .1], [.8, .2]], np.float32))
    bucket = RetrievalBucket(0, index, np.zeros(3), np.arange(3), np.zeros(3),
                             np.array([10, 20, 30]), np.zeros(3), np.zeros(3))
    store = TabxRetrievalStore.__new__(TabxRetrievalStore)
    store.k, store.search_k, store.max_similarity = 2, 3, .98
    store.rng = np.random.default_rng(7)
    orders = {tuple(store._select_candidates(bucket, np.array([[1, 0]], np.float32),
        np.array([0.]), query_record_hashes=np.array([10]))[0]) for _ in range(12)}
    assert orders == {(1, 2), (2, 1)}


def test_online_memory_is_local_and_requires_distinct_episodes():
    import faiss
    store = TabxRetrievalStore.__new__(TabxRetrievalStore)
    store.faiss, store.embedding_dim, store.k, store.search_k = faiss, 2, 2, 4
    store.test_nepisode, store.use_utility = 10, True
    store.weight, store.ind_weight, store.return_bounds = .8, .2, (0, 1)
    store.reset_online_memory(7)
    for agent_id in (0, 1):
        key = (0, agent_id)
        store._online_indexes[key] = faiss.IndexFlatIP(2)
        store._online_indexes[key].add(np.array([[1, 0], [.9, .1]], np.float32))
        store._online_contexts[key] = [f"agent-{agent_id}-episode-0", f"agent-{agent_id}-episode-1"]
        store._online_returns[key] = [(0., 0., 2, 0), (1., .5, 2, 1)]
    query = np.array([[1, 0]], np.float32)
    store.completed_online_episodes = 1
    assert store.offline_probability == 1
    assert store._online_mips(query, 0, 0) is None
    store.completed_online_episodes = 2
    assert all(item.startswith("agent-0-") for item in store._online_mips(query, 0, 0))
    # Two windows of one episode cannot satisfy the two-episode requirement.
    store._online_returns[(0, 0)] = [(0., 0., 2, 0), (1., .5, 2, 0)]
    assert store._online_mips(query, 0, 0) is None
    assert store._online_mips(query, 0, 2) is None


def test_held_out_targets_do_not_depend_on_held_out_returns():
    train = NS(split="train", target_return=4., return_scale=2.)
    held_out = NS(split="test", target_return=999., return_scale=999.)
    manifest = NS(tasks_for_split=lambda split: [train])
    before = resolve_return_target(manifest, held_out, {})
    held_out.target_return, held_out.return_scale = -999., .0001
    assert resolve_return_target(manifest, held_out, {}) == before
    assert before["target_return"] == 4. and before["return_scale"] == 2.


def test_accumulation_updates_only_after_full_effective_batch():
    from learners.ic_learner import ICLearner
    from torch.cuda.amp import GradScaler
    learner = ICLearner.__new__(ICLearner)
    parameter = torch.nn.Parameter(torch.tensor(2.))
    learner.params, learner.args = [parameter], NS(grad_norm_clip=0.)
    learner.optimiser = torch.optim.SGD([parameter], lr=.1)
    learner.scaler = GradScaler(enabled=False)
    learner._accumulating, learner._loss_weight, learner._finish_update = False, .5, False
    learner._optimise(parameter.square())
    assert parameter.item() == 2.
    learner._finish_update = True
    learner._optimise((3 * parameter).square())
    assert parameter.item() == pytest.approx(0., abs=1e-6)
    assert not learner._accumulating
