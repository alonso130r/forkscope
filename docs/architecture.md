# Architecture notes

Forkscope evaluates test-time, action-conditioned planning rollouts. The runner compares planning horizons while holding scenario seeds and other configured settings fixed.

## Intended component boundaries

- **Environment adapter:** resets a scenario and advances the real environment with actions.
- **Dynamics model adapter:** predicts the next observation from observations and actions.
- **Objective adapter:** scores predicted outcomes for a task.
- **Planner:** generates candidate action sequences and selects an action from predicted outcomes.
- **Evaluator:** runs matched scenarios across configured rollout horizons and records outcomes.
- **Backend adapter:** loads and executes a model using a selected framework, then returns NumPy-compatible results to the framework-neutral core.

The first backend implementation should run in one process per framework. A later implementation may use isolated worker processes for side-by-side runtime comparisons.

