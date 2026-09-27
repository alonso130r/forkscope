# ForkScope

a tool for finding the optimum hyperparameters for world model rollouts at inference.

works for setups that use test-time, action-conditioned planning rollouts (planner generates a bunch
of actions, world model scores those actions based on outcome, planner evaluates and selects the best
one to execute).

## Programmatic use

Create zero-argument factories for a Gymnasium-compatible environment and a planner implementing
`reset(seed=...)` and `plan(observation, horizon=...)`, then pass them to the public benchmarker:

```python
from forkscope import RolloutBenchmarker

benchmarker = RolloutBenchmarker(
    make_environment,
    make_planner,
    seeds=[1, 2, 3],
    horizons=[1, 4, 8],
    max_steps=500,
)
records = benchmarker.run()
benchmarker.write_results("outputs/results.jsonl")
```

The CLI uses the same benchmarker. Factories must be importable as `package.module:callable`:

```sh
forkscope run \
  --env-factory my_project.environment:make_environment \
  --planner-factory my_project.planner:make_planner \
  --seeds 1,2,3 \
  --horizons 1,4,8 \
  --max-steps 500 \
  --output outputs/results.jsonl
```
