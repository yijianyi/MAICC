# Offline sequence sample for baseline data interface

This sample is generated from `diversity_test` commit:

`b0f8524 Fix obs NaN from padded-unit ratios and dump truncation/is_win in generate records.`

## Purpose

This is a small interface-testing sample for baseline model data loading.

## Content

- task index: 59
- records: 1
- timesteps: 64
- agent-centric dirs: 10 units

## New fields

- `truncation.npy`: whether the step ended because max episode steps were reached.
- `is_win.npy`: whether team 0 won. Non-terminal steps are 0.

## Layout

```text
records/
  task_xxx/record-xxxxxx/
    actions_behavior.npy
    actions_reference.npy
    reward_team.npy
    reward_individual.npy
    done.npy
    reset.npy
    truncation.npy
    is_win.npy
    episode_id.npy
    behavior_policy_id.npy
    visible_matrix.npy
    obs_flat.npy
    unit_*.npy
    agent_centric/
      unit_00/
        obs_static.npy
        obs_dynamic.npy
        obs_dynamic_mask.npy
        behavior_action.npy
        reference_action.npy
        reward_team.npy
        reward_individual.npy
        done.npy
        reset.npy
        truncation.npy
        is_win.npy
        episode_id.npy
        behavior_policy_id.npy
Notes
All arrays within the same record share the same time index.
done=1 marks a terminal transition.
truncation=1 marks timeout caused by max episode steps.
is_win=1 marks team 0 win. Non-terminal steps are 0.
In this 64-step smoke sample, done/truncation/is_win may all be 0 because the episode may not terminate within 64 steps.
check_summary.txt contains file completeness and NaN/Inf checks.
