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

`--rollout-counts` and `--temperatures` are optional sweep dimensions. ForkScope evaluates every
combination of horizon, supplied rollout count, supplied temperature, and seed. For a planner that
supports both settings, add `--rollout-counts 64,128,256` and
`--temperatures 0.1,0.5,1.0`. Omit either option
to leave that planner setting unset. When a range is supplied, the planner must accept the matching
`rollout_count` or `temperature` keyword in `plan`; planners that do not use these settings can
continue to run without those options. The point-mass example does not use sampling, so run it
without the two optional options above.

The same factories can be passed to `RolloutBenchmarker` from Python. See the example README for
that usage and details about the JSON Lines output. The YAML file documents the matching settings;
the CLI currently accepts options directly and does not load YAML configs.

## Array backends

The Python API uses NumPy arrays by default. To use JAX or PyTorch, install the corresponding extra
and select the backend when creating a benchmark:

```sh
python -m pip install -e '.[jax]'
python -m pip install -e '.[torch]'
```

```python
from forkscope import RolloutBenchmarker

benchmarker = RolloutBenchmarker(
    env_factory,
    planner_factory,
    seeds=[0, 1],
    horizons=[1, 2],
    max_steps=10,
    backend="jax",  # or "torch"; omit for NumPy
)
records = benchmarker.run()
benchmarker.write_results("results.jsonl")
```

The environment and planner must produce arrays from the selected framework. For example, a JAX
environment can return `{"position": jax.numpy.asarray([position])}` from `reset` and `step`,
and its planner can return a `jax.numpy.asarray(action)` action. The equivalent PyTorch values are
`{"position": torch.tensor([position])}` and `torch.tensor(action)`. Python scalar actions remain
supported for environments that use them. ForkScope rejects arrays from a different backend at
the environment and planner boundaries; it does not convert them during a run.

`run_episode`, `run_sweep`, and the tree utilities also accept `backend="jax"` or
`backend="torch"`. Tree operations retain the framework's array type and device. JSON Lines output
converts arrays to host values when writing; PyTorch tensors are detached first. The CLI currently
uses the default NumPy backend.

Each JSON Lines episode record includes `rollout_count` and `temperature` so it can be associated
with the configuration used. These fields are `null` when the corresponding sweep range is omitted.
