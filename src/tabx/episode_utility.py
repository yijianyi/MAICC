"""Full-episode utility from windowed data without double-counting overlaps.

Individual rewards are DEM predictions, never the dataset's individual labels.
The encoder has a bounded history: each covering window resets its positions.
"""
import json
import numpy as np


def episode_cover(index_path, n_rows):
    groups = {}
    row_group = np.full(n_rows, -1, dtype=np.int64)
    with open(index_path, encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            key = (item["source_record"], int(item["source_segment_start"]),
                   int(item["source_segment_end"]))
            groups.setdefault(key, []).append(item)
    plan = []
    for group_id, ((_, start, end), windows) in enumerate(groups.items()):
        cursor = start
        for window in sorted(windows, key=lambda w: (w["source_start"], -w["source_end"])):
            row = int(window["row"])
            if not 0 <= row < n_rows or row_group[row] != -1:
                raise ValueError("Invalid or duplicate provenance row")
            row_group[row] = group_id
            lo, hi = int(window["source_start"]), int(window["source_end"])
            if lo > cursor:
                raise ValueError("Episode windows have a coverage gap; rebuild converted data")
            if hi > cursor:
                plan.append((row, group_id, cursor - lo, hi - lo))
                cursor = hi
        if cursor != end:
            raise ValueError("Cannot recover a complete episode from these windows")
    if np.any(row_group < 0):
        raise ValueError("Provenance does not cover every HDF5 row")
    return row_group, plan, len(groups)


def compute_episode_utilities(source, task, agent, device, batch_size):
    import torch as th
    from .episode_adapter import process_episode_batch
    from .retrieval_store import encode_dem_prefixes

    row_group, plan, count = episode_cover(task.index_path, task.n_trajectories)
    team = np.zeros(count, dtype=np.float64)
    individual = np.zeros((count, task.n_agents), dtype=np.float64)
    with th.no_grad():
        for offset in range(0, len(plan), batch_size):
            items = plan[offset:offset + batch_size]
            batch = source.load_rows(task, [item[0] for item in items]).batch
            batch.to(device)
            rtg, obs, action, post, _, mask = process_episode_batch(batch, max_return=1.0)
            _, predictions = encode_dem_prefixes(agent, rtg, obs, action, post, return_rewards=True)
            predictions = predictions * batch["present"][:, :-1, :, 0] * mask
            predictions = predictions.cpu().numpy()
            rewards = (post[:, :, 0] * mask[:, :, 0]).cpu().numpy()
            for i, (_, group, start, end) in enumerate(items):
                team[group] += rewards[i, start:end].sum(dtype=np.float64)
                individual[group] += predictions[i, start:end].sum(axis=0, dtype=np.float64)
    if not np.isfinite(team).all() or not np.isfinite(individual).all():
        raise ValueError("Non-finite episode utility")
    return team[row_group].astype(np.float32), individual[row_group].astype(np.float32)
