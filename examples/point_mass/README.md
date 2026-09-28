# Point-mass example

This example compares planning horizons on a deterministic one-dimensional task. The environment starts at `-seed`, has a goal at position `5`, and moves by the action selected by the planner. The planner moves toward the goal by at most the current horizon. This makes it easy to see how the same environment and seed behave across horizon settings.

Run commands from the repository root after installing ForkScope and its NumPy dependency:

```sh
python -m pip install -e .
```

## CLI

The example exposes zero-argument factories for the environment and planner. Set `PYTHONPATH=.` so the CLI can import the example package from the repository root, then pass the settings from [`config.yaml`](config.yaml):

```sh
PYTHONPATH=. forkscope run \
  --env-factory examples.point_mass.example:make_environment \
  --planner-factory examples.point_mass.example:make_planner \
  --seeds 0,1,2 \
  --horizons 1,2,4 \
  --max-steps 10 \
  --output outputs/point_mass/results.jsonl
```

The CLI accepts factory paths and individual options; it does not read YAML files directly. `config.yaml` records the same settings for reference and for use in your own launcher or workflow.

## Python API

The script runs the same sweep using `RolloutBenchmarker`:

```sh
PYTHONPATH=. python -m examples.point_mass.example
```

It writes JSON Lines records to `outputs/point_mass/programmatic-results.jsonl`. You can also import its factories into your own program:

```python
from examples.point_mass.example import make_environment, make_planner
from forkscope import RolloutBenchmarker

benchmarker = RolloutBenchmarker(
    make_environment,
    make_planner,
    seeds=[0, 1, 2],
    horizons=[1, 2, 4],
    max_steps=10,
)
records = benchmarker.run()
benchmarker.write_results("outputs/point_mass/results.jsonl")
```

Each output line is one episode record, including its seed, horizon, total reward, termination status, and step-level observations, actions, rewards, and environment info.
