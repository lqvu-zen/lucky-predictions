"""Machine-learning experiments for lottery prediction.

Everything here is an honest engineering exercise: because lottery draws
are uniform and independent, models are expected to land on the random
baseline. The value is the pipeline — leakage-safe features, versioned
models, walk-forward evaluation, and a predict->score loop.
"""
