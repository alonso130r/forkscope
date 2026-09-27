# ForkScope

a tool for finding the optimum hyperparameters for world model rollouts at inference.

works for setups that use test-time, action-conditioned planning rollouts (planner generates a bunch
of actions, world model scores those actions based on outcome, planner evaluates and selects the best
one to execute).
