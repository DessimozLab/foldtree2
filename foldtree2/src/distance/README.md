# Distance and contact supervision

Pairwise-distance and contact supervision for training geometry models.

[Source map](../README.md) | [Project](../../../README.md) |
[Training guide](../../../docs/training.md)

- [distance_restraints](./distance_restraints.py): the hub. `PairwiseDistanceHead`,
  `distance_labels`, `eligible_pairs`, `validate_edges` and the `DISTANCE_EDGES`
  constants. Use this for all distance/contact supervision.
- [distance_energy](./distance_energy.py): energy/loss layer built on
  `distance_restraints`; composes the restraints into a training objective.
- [contact_dot_experiments](./contact_dot_experiments.py): thin experimental
  contact-head variant; research ablation of the base head.
- [contact_levers](./contact_levers.py): thin experimental contact-head variant;
  research ablation of the base head.
- [contact_bias_experiment](./contact_bias_experiment.py): thin experimental
  contact-head variant; research ablation of the base head.
- [contact_cross_dot_perturbed](./contact_cross_dot_perturbed.py): extends
  `contact_dot_experiments` with perturbed cross-dot terms; research ablation.
- [contact_lever_metrics](./contact_lever_metrics.py): metrics for the lever
  variants. Imports the top-level `train_distance_restraints` driver, so it
  couples to that training entry point.

## When to use this

Use `distance_restraints` (with `distance_energy`) as the production
distance/contact supervision. The `contact_*` modules are research variants of
the same base head: pick one only for an ablation study and do not stack them.
Note that `contact_lever_metrics` imports the top-level `train_distance_restraints`
driver directly, so it is not a standalone loss layer. The main entry point is
[train_distance_restraints](../../train_distance_restraints.py); the
`scripts/contact/train_contact_levers.py` and `scripts/contact/report_contact_objectives.py`
helpers drive the lever ablations. See the [training guide](../../../docs/training.md)
and [experiments](../../../docs/experiments.md) for configured runs.