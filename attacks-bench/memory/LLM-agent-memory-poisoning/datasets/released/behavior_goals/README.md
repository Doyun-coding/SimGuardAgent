# Behavior Goals

This folder contains the released LLM behavior goal bank.

## Files

- `behavior_goals_700.json`: combined released behavior-goal bank
- `behavior_goals_eval_source_500.json`: the 500 unique goals embedded in the released behavior source dataset
- `behavior_goals_ood_200.json`: the disjoint 200-goal OOD bank
- `manifest.json`: provenance for the files above

## Provenance

The combined `behavior_goals_700.json` file is the union of:

- the 500 unique goals embedded in `datasets/released/behaviour_eval_in_with_true_optimized_goals.json`
- the 200-goal OOD bank previously stored at `goals_v2/datasets_actual_split/goals_ood.json`

This folder is the data surface for released behavior goals.

The corresponding source/procedure files live under `goals_generation/`.
