"""Behavioral tests for the TABX baseline corrections (CPU, no real dataset)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from modules.agents.tabx_transformer_agent import FactorizedCEM
from tabx.observation_layout import ObservationLayout
from tabx.episode_utility import episode_cover
from tabx.retrieval_store import hybrid_scores, TabxRetrievalStore, RetrievalBucket

torch.set_num_threads(1)


def test_zone_layout_rejects_silent_floor_division():
    with pytest.raises(ValueError, match="incompatible"):
        ObservationLayout.resolve(378, 10, 4)
    layout = ObservationLayout.resolve(378, 10, 10, 10)
    assert layout.other_blocks == 19 and layout.zone_start == 318
    assert max(layout.semantic_indices()) == 377


def test_cem_history_and_causality():
    torch.manual_seed(4)
    model = FactorizedCEM(16, 4, 2, 0.0, 2, 4, False).eval()
    x = torch.randn(1, 4, 5, 16)
    def output(value):
        return model.custom_forward(value.flatten(1, 2), 2)["last_hidden_state"].reshape(1, 4, 5, 16)
    with torch.no_grad():
        baseline = output(x)
        # Past actions and rewards must affect current observation embeddings.
        for slot in (1, 4):
            changed = x.clone()
            changed[:, 0, slot] = torch.randn_like(changed[:, 0, slot]) * 3
            assert not torch.allclose(output(changed)[:, 2, 0], baseline[:, 2, 0])
        # Current actions/posts and every future token must remain invisible.
        changed = x.clone()
        changed[:, 2, [1, 3, 4]] = torch.randn_like(changed[:, 2, [1, 3, 4]]) * 5
        changed[:, 3] = torch.randn_like(changed[:, 3]) * 5
        torch.testing.assert_close(output(changed)[:, 2, [0, 2]], baseline[:, 2, [0, 2]], atol=1e-6, rtol=1e-5)


def test_cem_ignores_nonexistent_agents():
    torch.manual_seed(8)
    model = FactorizedCEM(16, 2, 2, 0.0, 2, 3, False).eval()
    x = torch.randn(1, 3, 5, 16)
    present = torch.tensor([[[1, 0]] * 3], dtype=torch.bool)
    with torch.no_grad():
        a = model.custom_forward(x.flatten(1, 2), 2, present)["last_hidden_state"]
        x[:, :, 2:4] = 100 * torch.randn_like(x[:, :, 2:4])
        b = model.custom_forward(x.flatten(1, 2), 2, present)["last_hidden_state"]
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b)


def test_overlap_cover_counts_every_reward_once(tmp_path):
    windows = [{"row": i, "source_record": "r", "source_segment_start": 0,
                "source_segment_end": 6, "source_start": start, "source_end": end}
               for i, (start, end) in enumerate(((0, 4), (2, 6)))]
    path = tmp_path / "index.jsonl"
    path.write_text("".join(json.dumps(w) + "\n" for w in windows))
    groups, plan, count = episode_cover(path, 2)
    assert count == 1 and groups.tolist() == [0, 0]
    assert plan == [(0, 0, 0, 4), (1, 0, 2, 4)]
    assert sum(end - start for _, _, start, end in plan) == 6


def test_utility_retrieval_training_exclusion_and_online_ranking():
    import faiss
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [.95, np.sqrt(1 - .95**2)], [.9, np.sqrt(1-.9**2)]], np.float32))
    store = TabxRetrievalStore.__new__(TabxRetrievalStore)
    store.faiss = faiss
    store.embedding_dim = 2
    store.k = 1
    store.search_k = store.online_search_k = 3
    store._query_record_hashes = np.array([10], np.uint64)
    store._offline_search_calls = 0
    store.trace_search = False
    store.use_utility = True
    store.utility_alpha = .8
    store._combined_utility_bounds = {"team": [0, 1], "individual": [0, 1]}
    bucket = RetrievalBucket(0, index, np.zeros(3), np.arange(3), np.zeros(3),
                             np.array([10, 20, 30]), np.array([1, 0, 1]), np.array([1, 0, 1]))
    query = np.array([[1, 0]], np.float32)
    assert store._select_candidates(bucket, query, None) == [[1]]
    assert store._select_candidates(bucket, query, None, test_mode=True) == [[2]]
    scores = hybrid_scores([.9, .95], [1, 0], [1, 0], .8, store._combined_utility_bounds)
    assert scores[0] > scores[1]


def test_two_stage_pipeline_and_index_identity(tmp_path, monkeypatch):
    """Real optimizers, masked HDF5 windows, FAISS build and online memory."""
    import h5py
    import yaml
    from tabx.training import run_stage0, run_stage1, create_model_and_learner
    from tabx.manifest import TabxManifest
    from tabx.normalization import ObservationNormalizer
    from tabx.episode_adapter import TabxEpisodeSource
    from tabx.provenance import file_sha256, encoder_sha256
    from tools import build_tabx_index

    rng = np.random.default_rng(3)
    data = {k: [] for k in ("obs", "actions", "reward", "rtg", "terminated", "completion", "truncation", "present", "filled")}
    provenance = []
    for record in range(3):
        obs = rng.normal(size=(6, 2, 68)).astype(np.float32)
        actions = rng.integers(0, 8, size=(6, 2, 1))
        reward = np.arange(1, 7, dtype=np.float32)[:, None] / 100
        rtg = np.cumsum(reward[::-1], axis=0)[::-1].copy()
        for start in (0, 2):
            row = len(provenance)
            provenance.append(dict(row=row, source_record=f"record-{record}",
                source_segment_start=0, source_segment_end=6, source_start=start, source_end=start+4))
            for name, values in (("obs", obs), ("actions", actions), ("reward", reward), ("rtg", rtg)):
                data[name].append(np.concatenate([values[start:start+4], np.zeros_like(values[:1])]))
            term = np.zeros((5, 1), np.uint8)
            if start == 2:
                term[3] = 1
            data["terminated"].append(term)
            data["completion"].append(term.copy())
            data["truncation"].append(np.zeros_like(term))
            data["present"].append(np.ones((5, 2, 1), np.uint8))
            data["filled"].append(np.array([[1], [1], [1], [1], [0]], np.uint8))
    with h5py.File(tmp_path / "task.h5", "w") as h5:
        for name, values in data.items():
            h5[name] = np.stack(values)
    (tmp_path / "task.jsonl").write_text("".join(json.dumps(p)+"\n" for p in provenance))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(dict(schema="maicc_tabx_manifest_v2", sequence_length=4,
        global_schema=dict(obs_dim=68, action_dim=8, max_agents=2), tasks=[dict(task_id="task",
        n_agents=2, obs_dim=68, action_dim=8, n_trajectories=6, sequence_length=4,
        h5_file="task.h5", index_file="task.jsonl", return_stats=dict(min=0, max=1))])))
    norm_path = tmp_path / "norm.npz"
    np.savez(norm_path, mean=np.zeros(68, np.float32), scale=np.ones(68, np.float32),
             manifest_sha256=file_sha256(manifest_path), n_zones=1, max_enemies=2)
    config = dict(data=dict(manifest=str(manifest_path), normalization=str(norm_path)),
        model=dict(agent="tabx_transformer", max_agents=2, max_enemies=2, n_zones=1,
                   n_embd=16, n_layer=2, n_head=2, hidden_dim=16, dropout=.1, train_cem=True),
        training=dict(total_steps=2, max_batch_size=2, seed=7, save_interval=2,
                      output_dir=str(tmp_path / "stage0"), device="cpu"),
        retrieval=dict(k=2, search_k=8, index_factory="Flat", history_buckets=[1, 4],
                       index_batch_rows=2, index_dir=str(tmp_path / "index")))
    c0 = run_stage0(config, manifest_path, norm_path, "cpu")
    config["retrieval"].update(stage0_checkpoint=str(c0), index_manifest=str(tmp_path / "index/retrieval_manifest.json"))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["build_tabx_index", "--config", str(config_file)])
    assert build_tabx_index.main() == 0
    with h5py.File(tmp_path / "index/bucket_0003.h5") as h5:
        np.testing.assert_allclose(h5["episode_return"][:], .21, atol=1e-6)
        assert np.isfinite(h5["individual_return"][:]).all()
    config["training"]["output_dir"] = str(tmp_path / "stage1")
    index_path = tmp_path / "index/retrieval_manifest.json"
    c1 = run_stage1(config, manifest_path, norm_path, index_path, c0, "cpu")
    a = torch.load(c0 / "agent.th", weights_only=True)
    b = torch.load(c1 / "agent.th", weights_only=True)
    assert encoder_sha256(a) == encoder_sha256(b)
    assert not torch.equal(a["DT_pred_actions.weight"], b["DT_pred_actions.weight"])
    # Modern torch defaults must not break a saved NumPy RNG-state resume.
    from tabx.training import load_checkpoint
    manifest = TabxManifest(manifest_path)
    normalizer = ObservationNormalizer.load(norm_path, 68, manifest_path=manifest_path)
    _, mac, learner, _ = create_model_and_learner(config, manifest, "cpu")
    assert load_checkpoint(c1, learner, "cpu") == 2
    mac.agent.load_state_dict(b)
    kwargs = dict(index_manifest=index_path, data_manifest=manifest, normalizer=normalizer,
                  k=2, search_k=8, device="cpu", enable_online_memory=True,
                  normalization_path=norm_path, expected_encoder_sha256=encoder_sha256(b))
    from tabx.retrieval_store import RetrievalError
    with pytest.raises(RetrievalError, match="encoder hash"):
        TabxRetrievalStore(**{**kwargs, "expected_encoder_sha256": "wrong"})
    store = TabxRetrievalStore(**kwargs)
    source = TabxEpisodeSource(manifest, normalizer=normalizer)
    batch = source.load_rows(manifest.tasks[0], [0, 2]).batch
    store.begin_query_batch("task", [0, 2])
    learner.train(batch, 3, 3, store, train_mode=1)
    assert not mac.agent.DEM.training and mac.agent.DT.training
    seq = learner._sequences(batch)
    with torch.no_grad():
        key1 = mac.agent.get_dem_traj_embedding(seq[0], seq[1][:, :, 0], seq[2][:, :, 0], seq[3], seq[5])[0]
        key2 = mac.agent.get_dem_traj_embedding(seq[0], seq[1][:, :, 0], seq[2][:, :, 0], seq[3], seq[5])[0]
    torch.testing.assert_close(key1, key2, atol=0, rtol=0)
    mac.agent.eval()
    store.add_completed_online_episode(mac.agent, obs, actions[:, :, 0], reward,
        np.array([[0]]*5+[[1]]), np.array([[0]]*5+[[1]]), np.array([1, 1]), rtg)
    assert store.completed_online_episodes == 1
    assert all(abs(t-.21) < 1e-6 for pairs in store._online_returns.values() for t, _ in pairs)
    assert store._online_mips(key1[:1].numpy(), 3) is not None
    store.reset_online_memory()
    assert store.completed_online_episodes == 0
    assert all(not pairs for pairs in store._online_returns.values())
    source.close()
    store.close()
