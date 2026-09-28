# ForkScope

a tool for finding the optimum hyperparameters for world model rollouts at inference.

works for setups that use test-time, action-conditioned planning rollouts (planner generates a bunch
of actions, world model scores those actions based on outcome, planner evaluates and selects the best
one to execute).

## Runnable example

The [point-mass example](examples/point_mass/README.md) includes a small environment, a planner,
zero-argument factories for the CLI, a reference YAML config, and a programmatic example. Install
ForkScope in editable mode from the repository root with `python -m pip install -e .`, then run:

```sh
PYTHONPATH=. forkscope run \
  --env-factory examples.point_mass.example:make_environment \
  --planner-factory examples.point_mass.example:make_planner \
  --seeds 0,1,2 \
  --horizons 1,2,4 \
  --max-steps 10 \
  --output outputs/point_mass/results.jsonl
```

The same factories can be passed to `RolloutBenchmarker` from Python. See the example README for
that usage and details about the JSON Lines output. The YAML file documents the matching settings;
the CLI currently accepts options directly and does not load YAML configs.
